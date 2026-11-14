#!/usr/bin/env python3
"""FastViT + FPN + Faster R-CNN training for MedPackID."""
import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import functional as TF
import timm

from medpack_annotations import IMAGE_EXTS, read_class_names, build_annotation_index, audit_split
from training_config import parse_training_args
from validation_map import evaluate_validation_map


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def find_split(root, options):
    wanted = {x.lower() for x in options}
    matches = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() in wanted
               and any(q.suffix.lower() in IMAGE_EXTS for q in p.rglob("*"))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one image split {options}, found {matches}")
    return matches[0]


class MedPackDataset(Dataset):
    def __init__(self, split_dir, ann_index, class_names, train=False, preferred="yolo"):
        self.root, self.ann_index, self.class_names, self.train = split_dir, ann_index, class_names, train
        self.paths = sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not self.paths: raise RuntimeError(f"No images found in {split_dir}")
        self.targets, self.audit = audit_split(self.paths, ann_index, class_names, preferred)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        image = Image.open(path).convert("RGB"); width, height = image.size
        boxes, labels = self.targets[i]
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1,4)
        labels = torch.tensor(labels, dtype=torch.int64)
        if self.train and random.random() < .5:
            image = TF.hflip(image)
            if len(boxes): boxes[:,[0,2]] = width - boxes[:,[2,0]]
        image = TF.to_tensor(image)
        target = {"boxes":boxes, "labels":labels, "image_id":torch.tensor([i]),
                  "area":(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),
                  "iscrowd":torch.zeros(len(boxes), dtype=torch.int64)}
        return image, target


class FastViTFPN(torch.nn.Module):
    def __init__(self, model_name, out_channels=192, pretrained=True):
        super().__init__()
        self.body = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        self.fpn = FeaturePyramidNetwork(self.body.feature_info.channels(), out_channels)
        self.out_channels = out_channels
    def forward(self, x):
        return self.fpn({str(i): feat for i, feat in enumerate(self.body(x))})


def build_detector(backbone_name, num_classes, image_size, pretrained=True, score_threshold=0.05):
    backbone = FastViTFPN(backbone_name, pretrained=pretrained)
    anchors = AnchorGenerator(((16,32),(32,64),(64,128),(128,256)), ((0.5,1.0,2.0),)*4)
    return FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchors,
                      min_size=image_size, max_size=image_size, box_detections_per_img=100,
                      box_score_thresh=score_threshold)


def collate_fn(batch): return tuple(zip(*batch))


def train_epoch(model, loader, optimizer, scaler, device):
    model.train(); total = 0.0
    for images, targets in loader:
        images = [x.to(device, non_blocking=True) for x in images]
        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets); loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss_dict}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += float(loss.detach())
    return total / max(1, len(loader))


@torch.no_grad()
def validation_loss(model, loader, device):
    # Torchvision detectors return component losses in training mode.
    model.train()
    model.backbone.eval()
    # Keep batch-normalization statistics and dropout fixed during validation loss.
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)):
            module.eval()
    total = 0.0
    for images, targets in loader:
        images = [x.to(device, non_blocking=True) for x in images]
        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            total += float(sum(model(images, targets).values()))
    model.eval()
    return total / max(1, len(loader))


def save_target_previews(dataset, destination):
    count = 0
    for path,(boxes,labels) in zip(dataset.paths,dataset.targets):
        if not boxes:
            continue
        with Image.open(path) as source:
            image=source.convert("RGB")
        width,height=image.size
        image.thumbnail((1024,1024))
        sx,sy=image.width/width,image.height/height
        draw=ImageDraw.Draw(image)
        for box,label in zip(boxes,labels):
            x1,y1,x2,y2=box
            draw.rectangle((x1*sx,y1*sy,x2*sx,y2*sy),outline="red",width=3)
            draw.text((x1*sx,max(0,y1*sy-12)),dataset.class_names[label-1],fill="red")
        image.save(destination/f"train_target_{count}.jpg")
        count += 1
        if count == 4:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MEDPACK_DATA_ROOT", "data/MedPackID"))
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "runs/fastvit_medpack"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="fastvit_sa12.apple_in1k")
    ap.add_argument("--annotation-format", choices=["yolo","voc"], default="yolo")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiment-name", default="fastvit_medpack")
    ap.add_argument("--val-map-every", type=int, default=0,
                    help="Compute validation mAP every N epochs and save best_map.pth; 0 disables it")
    args = parse_training_args(ap)
    data_root = Path(args.data).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.audit_only and any((out/name).exists() for name in ("best.pth","last.pth","history.json")):
        raise FileExistsError(f"Output already has training results: {out}. Use a new output directory.")
    train_dir = find_split(data_root, ["train", "training"])
    valid_dir = find_split(data_root, ["valid", "validation", "val"])
    names = read_class_names(data_root)
    if not names: raise RuntimeError("Could not read class names from data.yaml or labels.txt")
    anns = build_annotation_index(data_root)
    train_ds = MedPackDataset(train_dir, anns, names, train=True, preferred=args.annotation_format)
    valid_ds = MedPackDataset(valid_dir, anns, names, train=False, preferred=args.annotation_format)
    audit={"classes":names,"train":train_ds.audit,"valid":valid_ds.audit}
    (out/"dataset_audit.json").write_text(json.dumps(audit,indent=2))
    save_target_previews(train_ds,out)
    print(json.dumps(audit,indent=2),flush=True)
    if args.audit_only:
        return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not available; run this on the DGX GPU node.")
    set_seed(args.seed); device = torch.device("cuda")
    if args.val_map_every:
        from pycocotools.cocoeval import COCOeval  # Check dependency before starting the run.
    try:
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        git_status=subprocess.check_output(['git','status','--porcelain'],text=True)
    except (OSError,subprocess.CalledProcessError):
        git_commit=None;git_status=None
    run_config={"experiment":args.experiment_name,"started_at_utc":datetime.now(timezone.utc).isoformat(),
        "args":vars(args),"git_commit":git_commit,"git_status":git_status,
        "versions":{"torch":torch.__version__,"timm":timm.__version__},
        "gpu":torch.cuda.get_device_name(0),"initialization":"ImageNet pretrained backbone; random detection head",
        "warmup_epochs":0,"dataset_audit":audit,
        "checkpoint_selection":{"best.pth":"minimum validation loss",
            "best_map.pth":"maximum sampled validation mAP50-95" if args.val_map_every else "disabled"}}
    (out/"run_config.json").write_text(json.dumps(run_config,indent=2))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_fn)
    model = build_detector(args.backbone,len(names)+1,args.image_size).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,args.epochs*2//3), gamma=.1)
    scaler = torch.amp.GradScaler("cuda")
    print(f"GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__} | timm={timm.__version__}")
    print(f"train={len(train_ds)} valid={len(valid_ds)} classes={names} epochs={args.epochs} image={args.image_size}")
    best = float("inf"); best_map=-1.0; history=[]
    for epoch in range(1,args.epochs+1):
        start=time.time(); tr=train_epoch(model,train_loader,optimizer,scaler,device)
        va=validation_loss(model,valid_loader,device)
        map_metrics=None
        if args.val_map_every and (epoch % args.val_map_every == 0 or epoch == args.epochs):
            # Metric inference must not change the RNG trajectory of future training batches.
            py_rng=random.getstate();np_rng=np.random.get_state()
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
            try:
                map_metrics=evaluate_validation_map(model,valid_loader,names,device)
            finally:
                random.setstate(py_rng);np.random.set_state(np_rng)
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        scheduler.step()
        row={"epoch":epoch,"train_loss":tr,"val_loss":va,"lr":optimizer.param_groups[0]["lr"]}
        if map_metrics is not None: row['validation_coco']=map_metrics
        history.append(row)
        map_log=f" mAP50-95={map_metrics['mAP50_95']:.4f} mAP50={map_metrics['mAP50']:.4f}" if map_metrics else ""
        print(f"epoch {epoch:03d}/{args.epochs} train={tr:.6e} val={va:.6e} lr={row['lr']:.2g}{map_log} time={time.time()-start:.1f}s", flush=True)
        state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "classes":names,"image_size":args.image_size,"backbone":args.backbone,"args":vars(args),
               "git_commit":git_commit,"validation_coco":map_metrics}
        torch.save(state,out/"last.pth")
        if va < best:
            best=va;torch.save(state,out/"best.pth")
        if map_metrics is not None and map_metrics['mAP50_95'] > best_map:
            best_map=map_metrics['mAP50_95'];torch.save(state,out/"best_map.pth")
            (out/"best_map_metrics.json").write_text(json.dumps({"epoch":epoch,"coco_bbox":map_metrics},indent=2))
        (out/"history.json").write_text(json.dumps(history,indent=2))
    print(f"Training complete. Best validation loss={best:.4f}; outputs={out}")
    if args.val_map_every: print(f"Best sampled validation mAP50-95={best_map:.4f}; checkpoint=best_map.pth")

if __name__ == "__main__": main()
