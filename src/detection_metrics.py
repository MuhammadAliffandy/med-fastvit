"""COCO box AP and score-thresholded precision/recall for MedPackID."""
from collections import defaultdict


def iou_xywh(a, b):
    intersection = max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0])) * max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return intersection/union if union > 0 else 0.0


def precision_recall(ground_truth, predictions, classes, score_threshold=0.5, iou_threshold=0.5):
    grouped_gt=defaultdict(list); grouped_pred=defaultdict(list)
    for ann in ground_truth['annotations']:
        grouped_gt[(ann['image_id'],ann['category_id'])].append(ann['bbox'])
    for det in predictions:
        if det['score'] >= score_threshold:
            grouped_pred[(det['image_id'],det['category_id'])].append(det)
    counts={i:{'tp':0,'fp':0,'fn':0} for i in range(1,len(classes)+1)}
    for key in grouped_gt.keys() | grouped_pred.keys():
        boxes=grouped_gt[key]; detections=sorted(grouped_pred[key],key=lambda d:d['score'],reverse=True)
        used=set(); row=counts[key[1]]
        for det in detections:
            available=[(iou_xywh(det['bbox'],box),i) for i,box in enumerate(boxes) if i not in used]
            overlap,index=max(available,default=(0.0,-1))
            if index >= 0 and overlap >= iou_threshold:
                row['tp']+=1; used.add(index)
            else:
                row['fp']+=1
        row['fn']+=len(boxes)-len(used)
    def result(count):
        tp,fp,fn=count['tp'],count['fp'],count['fn']
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        return {**count,'precision':precision,'recall':recall,'f1':f1}
    per_class={name:result(counts[i]) for i,name in enumerate(classes,1)}
    total={name:sum(c[name] for c in counts.values()) for name in ('tp','fp','fn')}
    return {'score_threshold':score_threshold,'iou_threshold':iou_threshold,
            'matching':'score-descending, same class, one-to-one, best available IoU',
            'micro':result(total),'per_class':per_class}


def coco_box_metrics(ground_truth, predictions):
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt=COCO(); coco_gt.dataset=ground_truth; coco_gt.createIndex()
    if predictions:
        coco_dt=coco_gt.loadRes(predictions)
    else:
        # COCO.loadRes indexes predictions[0]; construct an empty result explicitly.
        coco_dt=COCO()
        coco_dt.dataset={'images':ground_truth['images'], 'categories':ground_truth['categories'], 'annotations':[]}
        coco_dt.createIndex()
    evaluator=COCOeval(coco_gt,coco_dt,'bbox')
    evaluator.params.imgIds=[im['id'] for im in ground_truth['images']]
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    def number(value): return float(value) if value >= 0 else None
    stats=evaluator.stats
    result={'mAP50_95':number(stats[0]),'mAP50':number(stats[1]),'mAP75':number(stats[2]),
            'AR100':number(stats[8]),'per_class':{}}
    precision=evaluator.eval['precision']  # IoU, recall, class, area, maxDets
    for k,category_id in enumerate(evaluator.params.catIds):
        name=next(c['name'] for c in ground_truth['categories'] if c['id']==category_id)
        def average(values):
            values=values[values >= 0]
            return float(values.mean()) if values.size else None
        values=precision[:,:,k,0,-1]
        result['per_class'][name]={'AP50_95':average(values),
            'AP50':average(values[np.isclose(evaluator.params.iouThrs,0.5),:]),
            'AP75':average(values[np.isclose(evaluator.params.iouThrs,0.75),:])}
    return result
