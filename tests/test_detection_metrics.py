import contextlib
import copy
import io
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detection_metrics import precision_recall, coco_box_metrics


def ground_truth():
    return {'info':{},'images':[{'id':0,'width':100,'height':100},{'id':1,'width':100,'height':100}],
            'categories':[{'id':1,'name':'kemasan'},{'id':2,'name':'pil'}],
            'annotations':[{'id':1,'image_id':0,'category_id':1,'bbox':[10,10,20,20],'area':400,'iscrowd':0}]}


def prediction(image=0,category=1,score=0.9,box=None):
    return {'image_id':image,'category_id':category,'bbox':box or [10,10,20,20],'score':score}


class MatchingTests(unittest.TestCase):
    def evaluate(self,predictions):
        return precision_recall(ground_truth(),predictions,['kemasan','pil'])
    def test_perfect_detection_and_absent_class(self):
        result=self.evaluate([prediction()])
        self.assertEqual(result['micro']['precision'],1)
        self.assertEqual(result['micro']['recall'],1)
        self.assertIsNone(result['per_class']['pil']['recall'])
    def test_duplicates_and_negative_image_count_as_false_positives(self):
        result=self.evaluate([prediction(),prediction(score=.8),prediction(image=1)])['micro']
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,0))
        self.assertAlmostEqual(result['precision'],1/3)
    def test_wrong_class_and_no_predictions(self):
        result=self.evaluate([prediction(category=2)])['micro']
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
        self.assertEqual(self.evaluate([])['micro']['recall'],0)
    def test_confidence_does_not_affect_ground_truth_count(self):
        result=self.evaluate([prediction(score=.2)])['micro']
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,0,1))
    def test_matching_uses_best_unmatched_ground_truth(self):
        gt=ground_truth(); second=copy.deepcopy(gt['annotations'][0]);second['id']=2
        gt['annotations'].append(second)
        result=precision_recall(gt,[prediction(),prediction(score=.8)],['kemasan','pil'])['micro']
        self.assertEqual((result['tp'],result['fp'],result['fn']),(2,0,0))


class CocoTests(unittest.TestCase):
    def evaluate(self,predictions):
        with contextlib.redirect_stdout(io.StringIO()):
            return coco_box_metrics(ground_truth(),predictions)
    def test_perfect_coco_ap(self):
        result=self.evaluate([prediction()])
        self.assertAlmostEqual(result['mAP50'],1)
        self.assertAlmostEqual(result['mAP50_95'],1)
        self.assertAlmostEqual(result['per_class']['kemasan']['AP50'],1)
        self.assertIsNone(result['per_class']['pil']['AP50'])
    def test_empty_predictions_have_zero_ap(self):
        result=self.evaluate([])
        self.assertEqual(result['mAP50'],0)
        self.assertEqual(result['mAP50_95'],0)
    def test_localization_quality_is_measured_across_iou_thresholds(self):
        result=self.evaluate([prediction(box=[10,10,25,25])])
        self.assertAlmostEqual(result['mAP50'],1)
        self.assertEqual(result['mAP75'],0)
        self.assertGreater(result['mAP50_95'],0)
        self.assertLess(result['mAP50_95'],1)

if __name__=='__main__':unittest.main()
