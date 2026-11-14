import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from medpack_annotations import build_annotation_index, resolve_annotation, parse_annotation, image_key, audit_split

class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def write(self,name,content):
        path=self.root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content); return path
    def test_double_extension_and_split_pairing(self):
        train=self.write('labels/train/a.jpg.txt','0 0.5 0.5 0.4 0.2')
        self.write('labels/valid/a.jpg.txt','1 0.1 0.1 0.1 0.1')
        index=build_annotation_index(self.root)
        self.assertEqual(resolve_annotation(self.root/'images/train/a.jpg',index),train)
        boxes,labels=parse_annotation(train,100,200,['kemasan','pil'])
        self.assertEqual(boxes,[[30,80,70,120]]); self.assertEqual(labels,[1])
    def test_voc_filename_alias_and_numeric_class(self):
        xml=self.write('annotations/train/renamed.xml','<annotation><filename>original.jpg</filename><object><name>1</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>')
        self.assertEqual(resolve_annotation(self.root/'images/train/original.jpg',build_annotation_index(self.root)),xml)
        self.assertEqual(parse_annotation(xml,100,100,['kemasan','pil'])[1],[2])
    def test_missing_and_unknown_labels_are_errors(self):
        with self.assertRaisesRegex(ValueError,'No matching'):
            resolve_annotation(self.root/'images/train/a.jpg',{})
        xml=self.write('a.xml','<annotation><object><name>unmapped</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>')
        with self.assertRaisesRegex(ValueError,'Unknown VOC class'):
            parse_annotation(xml,100,100,['kemasan','pil'])
        txt=self.write('a.txt','3 0.5 0.5 0.4 0.2')
        with self.assertRaisesRegex(ValueError,'Unknown YOLO class'):
            parse_annotation(txt,100,100,['kemasan','pil'])
    def test_clipped_degenerate_box_rejected(self):
        xml=self.write('a.xml','<annotation><object><name>pil</name><bndbox><xmin>200</xmin><ymin>200</ymin><xmax>220</xmax><ymax>220</ymax></bndbox></object></annotation>')
        with self.assertRaisesRegex(ValueError,'Degenerate'):
            parse_annotation(xml,100,100,['kemasan','pil'])
    def test_zero_boxes_stop_training_and_nonempty_audit_counts(self):
        from PIL import Image
        image=self.root/'images/train/a.jpg'; image.parent.mkdir(parents=True)
        Image.new('RGB',(100,200)).save(image)
        txt=self.write('labels/train/a.jpg.txt','')
        index=build_annotation_index(self.root)
        with self.assertRaisesRegex(ValueError,'ZERO bounding boxes'):
            audit_split([image],index,['kemasan','pil'])
        txt.write_text('1 0.5 0.5 0.4 0.2')
        targets,report=audit_split([image],index,['kemasan','pil'])
        self.assertEqual(report['images_with_boxes'],1)
        self.assertEqual(report['boxes_by_class'],{'pil':1})
        self.assertEqual(targets[0][1],[2])
    def test_ambiguous_annotation_not_arbitrarily_selected(self):
        self.write('labels/train/one/a.txt','0 0.5 0.5 0.4 0.2')
        self.write('labels/train/two/a.txt','0 0.5 0.5 0.4 0.2')
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            resolve_annotation(self.root/'images/train/a.jpg',build_annotation_index(self.root))

if __name__=='__main__': unittest.main()
