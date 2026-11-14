"""Validation-only COCO evaluation during detector training."""
import contextlib
import io

import torch
from detection_metrics import coco_box_metrics


@torch.inference_mode()
def evaluate_validation_map(model, loader, classes, device):
    previous_training=model.training
    previous_floor=model.roi_heads.score_thresh
    model.eval()
    model.roi_heads.score_thresh=0.001
    ground_truth={'info':{},'images':[],'annotations':[],
                  'categories':[{'id':i,'name':name} for i,name in enumerate(classes,1)]}
    predictions=[];annotation_id=1
    try:
        for images,targets in loader:
            outputs=model([image.to(device,non_blocking=True) for image in images])
            for image,target,prediction in zip(images,targets,outputs):
                image_id=int(target['image_id'].item())
                _,height,width=image.shape
                ground_truth['images'].append({'id':image_id,'width':width,'height':height})
                def xywh(box):
                    x1,y1,x2,y2=box
                    return [float(x1),float(y1),float(x2-x1),float(y2-y1)]
                for box,label in zip(target['boxes'].tolist(),target['labels'].tolist()):
                    bbox=xywh(box)
                    ground_truth['annotations'].append({'id':annotation_id,'image_id':image_id,
                        'category_id':label,'bbox':bbox,'area':bbox[2]*bbox[3],'iscrowd':0})
                    annotation_id+=1
                for box,label,score in zip(prediction['boxes'].cpu().tolist(),prediction['labels'].cpu().tolist(),
                                           prediction['scores'].cpu().tolist()):
                    predictions.append({'image_id':image_id,'category_id':label,'bbox':xywh(box),'score':score})
        with contextlib.redirect_stdout(io.StringIO()):
            metrics=coco_box_metrics(ground_truth,predictions)
        return metrics
    finally:
        model.roi_heads.score_thresh=previous_floor
        model.train(previous_training)
