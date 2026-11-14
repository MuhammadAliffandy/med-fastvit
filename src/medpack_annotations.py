"""Match MedPackID image files to YOLO/VOC annotations without dropping errors."""
from collections import Counter
from pathlib import Path
import math
import unicodedata
import xml.etree.ElementTree as ET

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train":"train", "training":"train", "valid":"valid", "val":"valid",
          "validation":"valid", "test":"test", "testing":"test"}


def split_name(path):
    for part in reversed(Path(path).parts):
        if part.lower() in SPLITS:
            return SPLITS[part.lower()]
    return None


def image_key(path):
    stem = Path(path).stem
    # LabelImg exports can be named photo.jpg.txt or photo.jpg.xml.
    if Path(path).suffix.lower() in {".xml", ".txt"} and Path(stem).suffix.lower() in IMAGE_EXTS:
        stem = Path(stem).stem
    return stem.casefold()


def read_class_names(root):
    import yaml
    for p in sorted(root.rglob("data.yaml")):
        data = yaml.safe_load(p.read_text()) or {}
        names = data.get("names")
        if isinstance(names, dict):
            if sorted(int(k) for k in names) != list(range(len(names))):
                raise ValueError(f"Non-contiguous class IDs in {p}")
            return [str(names[k]).strip() for k in sorted(names, key=lambda x:int(x))]
        if isinstance(names, list):
            return [str(x).strip() for x in names]
    for filename in ("labels.txt", "classes.txt", "obj.names"):
        for p in sorted(root.rglob(filename)):
            lines = [x.strip() for x in p.read_text().splitlines() if x.strip()]
            if lines:
                return lines
    raise ValueError("Class names missing: expected data.yaml, labels.txt or classes.txt")


def build_annotation_index(root):
    index = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".txt", ".xml"}:
            continue
        if p.name.lower() in {"labels.txt", "classes.txt", "obj.names", "license.txt"}:
            continue
        keys = {image_key(p)}
        if p.suffix.lower() == ".xml":
            try:
                filename = ET.parse(p).getroot().findtext("filename")
                if filename:
                    keys.add(image_key(filename.replace("\\", "/")))
            except ET.ParseError as exc:
                raise ValueError(f"Invalid XML {p}: {exc}") from exc
        for key in keys:
            index.setdefault(key, []).append(p)
    return index


def resolve_annotation(image, index, preferred="yolo"):
    candidates = list(dict.fromkeys(index.get(image_key(image), [])))
    split = split_name(image)
    same_split = [p for p in candidates if split_name(p) == split]
    candidates = same_split or [p for p in candidates if split_name(p) is None]
    if not candidates:
        raise ValueError(f"No matching annotation for image {image} (key={image_key(image)!r})")
    suffix = ".txt" if preferred == "yolo" else ".xml"
    selected = [p for p in candidates if p.suffix.lower() == suffix]
    selected = selected or candidates
    if len(selected) != 1:
        raise ValueError(f"Ambiguous annotation for {image}: {selected}")
    return selected[0]


def normalized_name(name):
    return unicodedata.normalize("NFC", name).strip().casefold()


def parse_annotation(path, width, height, class_names):
    boxes, labels = [], []
    name_map = {normalized_name(name):i+1 for i,name in enumerate(class_names)}
    def append_box(box, label):
        if not all(math.isfinite(v) for v in box):
            raise ValueError(f"Non-finite box in {path}: {box}")
        x1, y1, x2, y2 = box
        clipped = [max(0,min(width,x1)), max(0,min(height,y1)),
                   max(0,min(width,x2)), max(0,min(height,y2))]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError(f"Degenerate box in {path}: {box}")
        boxes.append(clipped); labels.append(label)
    if path.suffix.lower() == ".xml":
        root = ET.parse(path).getroot()
        xml_w = float(root.findtext("size/width", str(width)))
        xml_h = float(root.findtext("size/height", str(height)))
        if xml_w <= 0 or xml_h <= 0:
            raise ValueError(f"Invalid XML image size in {path}")
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            label = name_map.get(normalized_name(name))
            if label is None and name.isdigit() and 0 <= int(name) < len(class_names):
                label = int(name)+1
            if label is None:
                raise ValueError(f"Unknown VOC class {name!r} in {path}; expected {class_names}")
            b = obj.find("bndbox")
            if b is None:
                raise ValueError(f"Object without bndbox in {path}")
            values = [float(b.findtext(k)) for k in ("xmin","ymin","xmax","ymax")]
            append_box([values[0]*width/xml_w, values[1]*height/xml_h,
                        values[2]*width/xml_w, values[3]*height/xml_h], label)
    else:
        for number,line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f"Expected 5 YOLO columns at {path}:{number}, got {line!r}")
            raw_id, cx, cy, bw, bh = map(float, values)
            cid = int(raw_id)
            if raw_id != cid or not 0 <= cid < len(class_names):
                raise ValueError(f"Unknown YOLO class ID {raw_id} at {path}:{number}")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx,cy,bw,bh)):
                raise ValueError(f"YOLO coordinates must be normalized to [0,1] at {path}:{number}")
            append_box([(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height],cid+1)
    return boxes, labels


def audit_split(paths, index, class_names, preferred="yolo"):
    from PIL import Image
    targets, errors = [], []
    counts = Counter(); formats = Counter(); positive = 0
    for image in paths:
        try:
            annotation = resolve_annotation(image,index,preferred)
            with Image.open(image) as im:
                boxes, labels = parse_annotation(annotation,*im.size,class_names)
            targets.append((boxes,labels))
            counts.update(class_names[label-1] for label in labels)
            formats[annotation.suffix.lower()] += 1
            positive += bool(boxes)
        except (ValueError, OSError, TypeError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError(f"Annotation audit failed for {len(errors)}/{len(paths)} images. Examples:\n"+"\n".join(errors[:8]))
    if not positive:
        samples = [str(p) for values in list(index.values())[:3] for p in values[:1]]
        raise ValueError(f"Annotation audit found ZERO bounding boxes in {len(paths)} images. Annotation examples: {samples}")
    report={"images":len(paths),"images_with_boxes":positive,"empty_images":len(paths)-positive,
            "boxes":sum(counts.values()),"boxes_by_class":dict(counts),"annotation_formats":dict(formats)}
    return targets,report
