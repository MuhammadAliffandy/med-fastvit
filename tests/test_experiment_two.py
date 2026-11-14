import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from training_config import parse_training_args, validate_settings
from compare_experiments import compare_reports

ROOT=Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def parser(self):
        parser=argparse.ArgumentParser()
        parser.add_argument('--config')
        defaults=json.loads((ROOT/'configs/exp02_fastvit_384.json').read_text())
        for name,value in defaults.items():
            parser.add_argument('--'+name.replace('_','-'),type=type(value),default=value)
        return parser
    def test_recipe_and_explicit_cli_override(self):
        args=parse_training_args(self.parser(),['--config',str(ROOT/'configs/exp02_fastvit_384.json'),'--workers','2'])
        self.assertEqual(args.image_size,384)
        self.assertEqual(args.epochs,100)
        self.assertEqual(args.val_map_every,5)
        self.assertEqual(args.workers,2)
        self.assertEqual(args.lr,2e-4)
    def test_unknown_settings_and_invalid_values_rejected(self):
        for settings in ({'image_szie':384},{'image_size':True},{'epochs':0},{'lr':float('nan')},
                         {'val_map_every':-1},{'annotation_format':'unknown'}):
            with self.assertRaises(ValueError):validate_settings(settings)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.before=self.root/'before';self.after=self.root/'after'
        self.before.mkdir();self.after.mkdir()
        pr={'score_threshold':.5,'iou_threshold':.5,'matching':'same',
            'per_class':{'pil':{'tp':103,'fp':22,'fn':2,'precision':.824,'recall':.981,'f1':.896}}}
        self.report={'split':'valid','classes':['pil'],'dataset_audit':{'images':216},
                     'inference':{'score_floor':.001,'max_detections_per_image':100,'precision':'float32','torch':'2.5.1'},
                     'image_size':320,'checkpoint':'before/best.pth','precision_recall':pr,
                     'coco_bbox':{'mAP50_95':.688,'mAP50':.977,'mAP75':.824,'per_class':{'pil':{'AP50_95':.602}}}}
        self.gt={'images':[{'id':0}],'annotations':[{'id':1,'bbox':[1,1,10,10]}],'categories':[{'id':1,'name':'pil'}]}
        for folder in (self.before,self.after):
            (folder/'metrics.json').write_text(json.dumps(self.report))
            (folder/'ground_truth.json').write_text(json.dumps(self.gt))
    def tearDown(self):self.tmp.cleanup()
    def test_map_gain_and_fp_delta_recorded(self):
        report=copy.deepcopy(self.report)
        report['image_size']=384;report['coco_bbox']['mAP50_95']=.7
        report['precision_recall']['per_class']['pil']['fp']=15
        (self.after/'metrics.json').write_text(json.dumps(report))
        result=compare_reports(self.before,self.after,'minimum_loss')
        self.assertAlmostEqual(result['metrics']['mAP50_95']['delta_percentage_points'],1.2)
        self.assertEqual(result['per_class']['pil']['false_positive_delta'],-7)
    def test_changed_threshold_or_annotations_rejected(self):
        report=copy.deepcopy(self.report);report['precision_recall']['score_threshold']=.7
        (self.after/'metrics.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'threshold'):compare_reports(self.before,self.after,'minimum_loss')
        (self.after/'metrics.json').write_text(json.dumps(self.report))
        self.gt['annotations'][0]['bbox'][2]=30
        (self.after/'ground_truth.json').write_text(json.dumps(self.gt))
        with self.assertRaisesRegex(ValueError,'ground-truth'):compare_reports(self.before,self.after,'minimum_loss')


class ValidationMapTests(unittest.TestCase):
    def test_perfect_predictions_and_model_state_restored(self):
        import torch
        from validation_map import evaluate_validation_map
        class Detector(torch.nn.Module):
            def __init__(self,fail=False):
                super().__init__();self.roi_heads=SimpleNamespace(score_thresh=.05);self.fail=fail
            def forward(self,images):
                if self.fail:raise RuntimeError('fixture failure')
                assert not self.training and self.roi_heads.score_thresh==.001
                return [{'boxes':torch.tensor([[10.,10.,30.,30.]]),'labels':torch.tensor([1]),'scores':torch.tensor([.9])}]
        images=[torch.zeros(3,64,128)]
        targets=[{'image_id':torch.tensor([42]),'boxes':torch.tensor([[10.,10.,30.,30.]]),'labels':torch.tensor([1])}]
        loader=[(images,targets)]
        model=Detector();model.train()
        result=evaluate_validation_map(model,loader,['pil'],'cpu')
        self.assertAlmostEqual(result['mAP50_95'],1)
        self.assertTrue(model.training);self.assertEqual(model.roi_heads.score_thresh,.05)
        failing=Detector(fail=True)
        with self.assertRaisesRegex(RuntimeError,'fixture failure'):
            evaluate_validation_map(failing,loader,['pil'],'cpu')
        self.assertTrue(failing.training);self.assertEqual(failing.roi_heads.score_thresh,.05)


class DetectorTests(unittest.TestCase):
    def test_fastvit384_inference_and_checkpoint_state_compatibility(self):
        import torch
        from train_fastvit_medpack import build_detector
        torch.set_num_threads(2)
        model=build_detector('fastvit_sa12.apple_in1k',3,384,pretrained=False).eval()
        state=model.state_dict()
        restored=build_detector('fastvit_sa12.apple_in1k',3,384,pretrained=False).eval()
        restored.load_state_dict(state,strict=True)
        with torch.inference_mode():
            prediction=restored([torch.rand(3,384,256)])[0]
        self.assertEqual(prediction['boxes'].shape[1],4)
        self.assertTrue(torch.isfinite(prediction['boxes']).all())
        self.assertTrue((prediction['boxes'][:,[0,2]] <= 256).all())
        self.assertTrue((prediction['boxes'][:,[1,3]] <= 384).all())
        self.assertEqual(restored.transform.min_size,(384,))
        self.assertEqual(restored.transform.max_size,384)

if __name__=='__main__':unittest.main()
