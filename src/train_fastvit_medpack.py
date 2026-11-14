#!/usr/bin/env python3
"""FastViT + FPN + Faster R-CNN training for MedPackID."""
import argparse
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms import functional as TF
import timm
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def find_split(root, options):
    wanted = {x.lower() for x in options}
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in wanted and any(q.suffix.lower() in IMAGE_EXTS for q in p.rglob("*")):
            return p
    raise FileNotFoundError(f"Cannot find image split {options} below {root}")


def read_class_names(root):
    for p in root.rglob("data.yaml"):
        data = yaml.safe_load(p.read_text()) or {}
        names = data.get("names")
        if isinstance(names, dict): return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        if isinstance(names, list): return [str(x) for x in names]
    for name in ("labels.txt", "classes.txt", "obj.names"):
        matches = list(root.rglob(name))
        for p in matches:
            lines = [x.strip() for x in p.read_text(errors="ignore").splitlines() if x.strip()]
            if lines: return lines
    return None


def build_annotation_index(root):
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".xml", ".txt"} and p.name.lower() not in {"labels.txt", "classes.txt", "obj.names"}:
            index.setdefault(p.stem, []).append(p)
    # Prefer Pascal VOC XML when both LabelImg exports are included.
    for paths in index.values(): paths.sort(key=lambda p: (p.suffix.lower() != ".xml", str(p)))
    return index


def parse_annotation(path, width, height, class_names):
    boxes, labels = [], []
    if path is None: return boxes, labels
    if path.suffix.lower() == ".xml":
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            b = obj.find("bndbox")
            if b is None: continue
            x1, y1 = float(b.findtext("xmin", "0")), float(b.findtext("ymin", "0"))
            x2, y2 = float(b.findtext("xmax", "0")), float(b.findtext("ymax", "0"))
            label = class_names.index(name) + 1 if name in class_names else None
            if label and x2 > x1 and y2 > y1:
                boxes.append([max(0,x1), max(0,y1), min(width,x2), min(height,y2)]); labels.append(label)
    else:
        for line in path.read_text(errors="ignore").splitlines():
            v = line.split()
            if len(v) < 5: continue
            try: cid = int(float(v[0])); cx, cy, bw, bh = map(float, v[1:5])
            except ValueError: continue
            if cid < 0 or cid >= len(class_names): continue
            x1, y1 = (cx-bw/2)*width, (cy-bh/2)*height
            x2, y2 = (cx+bw/2)*width, (cy+bh/2)*height
            if x2 > x1 and y2 > y1:
                boxes.append([max(0,x1), max(0,y1), min(width,x2), min(height,y2)]); labels.append(cid+1)
    return boxes, labels


class MedPackDataset(Dataset):
    def __init__(self, split_dir, ann_index, class_names, train=False):
        self.root, self.ann_index, self.class_names, self.train = split_dir, ann_index, class_names, train
        self.paths = sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not self.paths: raise RuntimeError(f"No images found in {split_dir}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        path = self.paths[i]
        image = Image.open(path).convert("RGB"); width, height = image.size
        ann = self.ann_index.get(path.stem, [None])[0]
        boxes, labels = parse_annotation(ann, width, height, self.class_names)
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
    def __init__(self, model_name, out_channels=192):
        super().__init__()
        self.body = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=(0,1,2,3))
        self.fpn = FeaturePyramidNetwork(self.body.feature_info.channels(), out_channels)
        self.out_channels = out_channels
    def forward(self, x):
        return self.fpn({str(i): feat for i, feat in enumerate(self.body(x))})


def collate_fn(batch): return tuple(zip(*batch))


def train_epoch(model, loader, optimizer, scaler, device):
    model.train(); total = 0.0
    for images, targets in loader:
        images = [x.to(device, non_blocking=True) for x in images]
        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets); loss = sum(loss_dict.values())
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        total += float(loss.detach())
    return total / max(1, len(loader))


@torch.no_grad()
def validation_loss(model, loader, device):
    # Torchvision detectors return component losses in training mode.
    model.train(); total = 0.0
    for images, targets in loader:
        images = [x.to(device, non_blocking=True) for x in images]
        targets = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in targets]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            total += float(sum(model(images, targets).values()))
    model.eval()
    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("MEDPACK_DATA_ROOT", "data/MedPackID"))
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "runs/fastvit_medpack"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 4))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", default="fastvit_sa12.apple_in1k")
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU not available; run this on the DGX GPU node.")
    set_seed(args.seed); device = torch.device("cuda")
    data_root = Path(args.data).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    train_dir = find_split(data_root, ["train", "training"])
    valid_dir = find_split(data_root, ["valid", "validation", "val"])
    names = read_class_names(data_root)
    if not names: raise RuntimeError("Could not read class names from data.yaml or labels.txt")
    anns = build_annotation_index(data_root)
    train_ds = MedPackDataset(train_dir, anns, names, train=True)
    valid_ds = MedPackDataset(valid_dir, anns, names, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=max(1,args.batch_size//2), shuffle=False, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_fn)
    backbone = FastViTFPN(args.backbone)
    anchors = AnchorGenerator(((16,32),(32,64),(64,128),(128,256)), ((0.5,1.0,2.0),)*4)
    model = FasterRCNN(backbone, num_classes=len(names)+1, rpn_anchor_generator=anchors,
                       min_size=args.image_size, max_size=args.image_size, box_detections_per_img=100).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,args.epochs*2//3), gamma=.1)
    scaler = torch.amp.GradScaler("cuda")
    print(f"GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__} | timm={timm.__version__}")
    print(f"train={len(train_ds)} valid={len(valid_ds)} classes={names} epochs={args.epochs} image={args.image_size}")
    best = float("inf"); history=[]
    for epoch in range(1,args.epochs+1):
        start=time.time(); tr=train_epoch(model,train_loader,optimizer,scaler,device)
        va=validation_loss(model,valid_loader,device); scheduler.step()
        row={"epoch":epoch,"train_loss":tr,"val_loss":va,"lr":optimizer.param_groups[0]["lr"]}; history.append(row)
        print(f"epoch {epoch:03d}/{args.epochs} train={tr:.4f} val={va:.4f} lr={row['lr']:.2g} time={time.time()-start:.1f}s", flush=True)
        state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "classes":names,"image_size":args.image_size,"backbone":args.backbone,"args":vars(args)}
        torch.save(state,out/"last.pth")
        if va < best:
            best=va; torch.save(state,out/"best.pth")
        (out/"history.json").write_text(json.dumps(history,indent=2))
    print(f"Training complete. Best validation loss={best:.4f}; outputs={out}")

if __name__ == "__main__": main()
