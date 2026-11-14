#!/usr/bin/env python3
"""Evaluate a trained FastViT detector on validation or test images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm import tqdm

from medpack_annotations import build_annotation_index, read_class_names
from train_fastvit_medpack import MedPackDataset, build_detector, collate_fn, find_split
from detection_metrics import coco_box_metrics, precision_recall


def xywh(box):
    x1,y1,x2,y2=box
    return [float(x1),float(y1),float(x2-x1),float(y2-y1)]


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def save_preview(path, target, prediction, classes, threshold, destination):
    with Image.open(path) as source:
        image=source.convert('RGB')
    width,height=image.size
    image.thumbnail((800,800))
    sx,sy=image.width/width,image.height/height
    left=image.copy(); right=image.copy()
    for panel,boxes,labels,scores,color in (
        (left,target['boxes'].tolist(),target['labels'].tolist(),None,'lime'),
        (right,prediction['boxes'].tolist(),prediction['labels'].tolist(),prediction['scores'].tolist(),'red')):
        draw=ImageDraw.Draw(panel)
        for i,(box,label) in enumerate(zip(boxes,labels)):
            if scores is not None and scores[i] < threshold: continue
            x1,y1,x2,y2=box
            draw.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline=color,width=3)
            text=classes[label-1]+(f' {scores[i]:.2f}' if scores is not None else '')
            draw.text((x1*sx,max(0,y1*sy-12)),text,fill=color)
    combined=Image.new('RGB',(2*image.width,image.height+30),'white')
    combined.paste(left,(0,30));combined.paste(right,(image.width,30))
    draw=ImageDraw.Draw(combined)
    draw.text((5,8),'Ground truth',fill='black')
    draw.text((image.width+5,8),f'Prediction (score >= {threshold:.2f})',fill='black')
    combined.save(destination)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/MedPackID')
    parser.add_argument('--checkpoint',default='runs/fastvit_medpack_fixed/best.pth')
    parser.add_argument('--split',choices=['valid','test'],default='valid')
    parser.add_argument('--out',default=None)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--score-threshold',type=float,default=0.5,help='Threshold for precision/recall and previews only')
    parser.add_argument('--map-score-floor',type=float,default=0.001,help='Inference floor before COCO AP evaluation')
    parser.add_argument('--iou-threshold',type=float,default=0.5,help='Matching IoU for precision/recall only')
    parser.add_argument('--preview-count',type=int,default=12)
    args=parser.parse_args()
    if not 0 <= args.map_score_floor <= args.score_threshold <= 1:
        parser.error('Require 0 <= map-score-floor <= score-threshold <= 1')
    if not 0 < args.iou_threshold <= 1: parser.error('Require 0 < iou-threshold <= 1')
    if args.batch_size < 1 or args.workers < 0: parser.error('Invalid batch size or workers')
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; activate fastvit_medpack or pass --device cpu')
    checkpoint_path=Path(args.checkpoint).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    classes=checkpoint['classes']; image_size=checkpoint['image_size']
    data=Path(args.data).resolve()
    if read_class_names(data) != classes:
        raise ValueError('Dataset class names/order differ from checkpoint')
    split_dir=find_split(data,['valid','validation','val'] if args.split=='valid' else ['test','testing'])
    annotation_format=checkpoint.get('args',{}).get('annotation_format','yolo')
    dataset=MedPackDataset(split_dir,build_annotation_index(data),classes,train=False,preferred=annotation_format)
    out=Path(args.out or f'runs/eval_{args.split}').resolve();out.mkdir(parents=True,exist_ok=True)
    previews=out/'previews';previews.mkdir(exist_ok=True)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                      pin_memory=args.device=='cuda',collate_fn=collate_fn)
    model=build_detector(checkpoint['backbone'],len(classes)+1,image_size,pretrained=False,
                         score_threshold=args.map_score_floor)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(args.device).eval()
    ground_truth={'info':{'description':'MedPackID '+args.split},'images':[], 'annotations':[],
                  'categories':[{'id':i,'name':name} for i,name in enumerate(classes,1)]}
    predictions=[]; image_index=[]; preview_index=[]; annotation_id=1
    with torch.inference_mode():
        for images,targets in tqdm(loader,desc=f'Evaluating {args.split}'):
            # Full precision for repeatable evaluation; resizing/normalization match training.
            outputs=model([image.to(args.device) for image in images])
            for image,target,output in zip(images,targets,outputs):
                image_id=int(target['image_id'].item())
                image_path=dataset.paths[image_id]
                _,height,width=image.shape
                ground_truth['images'].append({'id':image_id,'file_name':image_path.name,'width':width,'height':height})
                image_index.append({'image_id':image_id,'path':str(image_path)})
                for box,label in zip(target['boxes'].tolist(),target['labels'].tolist()):
                    bbox=xywh(box)
                    ground_truth['annotations'].append({'id':annotation_id,'image_id':image_id,'category_id':label,
                        'bbox':bbox,'area':bbox[2]*bbox[3],'iscrowd':0})
                    annotation_id+=1
                pred={k:v.detach().cpu() for k,v in output.items()}
                for box,label,score in zip(pred['boxes'].tolist(),pred['labels'].tolist(),pred['scores'].tolist()):
                    predictions.append({'image_id':image_id,'category_id':label,'bbox':xywh(box),'score':score})
                if len(preview_index) < args.preview_count:
                    preview=previews/f'{image_id:04d}.jpg'
                    save_preview(image_path,target,pred,classes,args.score_threshold,preview)
                    preview_index.append({'image_id':image_id,'image':str(image_path),'preview':str(preview)})
    coco=coco_box_metrics(ground_truth,predictions)
    pr=precision_recall(ground_truth,predictions,classes,args.score_threshold,args.iou_threshold)
    report={'checkpoint':str(checkpoint_path),'checkpoint_sha256':sha256_file(checkpoint_path),
            'checkpoint_epoch':checkpoint['epoch'],'split':args.split,'image_size':image_size,
            'dataset_audit':dataset.audit,'classes':classes,
            'inference':{'score_floor':args.map_score_floor,'max_detections_per_image':100,'precision':'float32',
                         'torch':torch.__version__},
            'coco_bbox':coco,'precision_recall':pr}
    for filename,value in [('metrics.json',report),('predictions.json',predictions),('ground_truth.json',ground_truth),
                            ('image_index.json',image_index),('preview_index.json',preview_index)]:
        (out/filename).write_text(json.dumps(value,indent=2,allow_nan=False))
    rows=[{'class':name,**coco['per_class'][name],**pr['per_class'][name]} for name in classes]
    with open(out/'per_class.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2,allow_nan=False))
    print(f'Saved metrics, per_class.csv and prediction previews to {out}',flush=True)


if __name__=='__main__': main()
