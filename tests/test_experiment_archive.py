import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('archive',Path(__file__).resolve().parents[1]/'scripts/archive_experiment.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.training=self.root/'training';self.training.mkdir()
        self.evaluation=self.root/'evaluation';self.evaluation.mkdir()
        (self.training/'best.pth').write_bytes(b'best checkpoint fixture')
        (self.training/'last.pth').write_bytes(b'last checkpoint fixture')
        (self.training/'train.pid').write_text('123')
        history=[{'epoch':1,'val_loss':.1},{'epoch':2,'val_loss':.2}]
        (self.training/'history.json').write_text(json.dumps(history))
        self.metrics={'split':'valid','checkpoint_sha256':archive.sha256(self.training/'best.pth'),
                      'checkpoint_epoch':1,'classes':['kemasan','pil'],'image_size':320,
                      'coco_bbox':{'mAP50':.97},'precision_recall':{}}
        (self.evaluation/'metrics.json').write_text(json.dumps(self.metrics))
    def tearDown(self):self.tmp.cleanup()
    def test_matching_checkpoint_passes_and_changed_checkpoint_fails(self):
        archive.validate_run(self.training,self.evaluation)
        (self.training/'best.pth').write_bytes(b'different checkpoint')
        with self.assertRaisesRegex(ValueError,'differs from the checkpoint'):
            archive.validate_run(self.training,self.evaluation)
    def test_epoch_mismatch_is_rejected(self):
        self.metrics['checkpoint_epoch']=2
        (self.evaluation/'metrics.json').write_text(json.dumps(self.metrics))
        with self.assertRaisesRegex(ValueError,'epoch differs'):
            archive.validate_run(self.training,self.evaluation)
    def test_complete_archive_and_corruption_detection(self):
        out=self.root/'out'
        def git_archive(command,**kwargs):
            destination=next(x.split('=',1)[1] for x in command if x.startswith('--output='))
            Path(destination).write_bytes(b'code snapshot fixture')
        with patch.object(archive.subprocess,'check_output',side_effect=['a'*40,'','example==1\n']), \
             patch.object(archive.subprocess,'run',side_effect=git_archive), \
             patch.object(archive.shutil,'which',return_value=None):
            archive.prepare_archive(self.root,self.training,self.evaluation,out)
        self.assertTrue((out/'training/last.pth').exists())
        self.assertFalse((out/'training/train.pid').exists())
        self.assertEqual(json.loads((out/'experiment.json').read_text())['best_checkpoint_epoch'],1)
        archive.prepare_archive(self.root,self.training,self.evaluation,out)
        (out/'training/best.pth').write_bytes(b'corrupted copy')
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            archive.verify_archive(out)

if __name__=='__main__':unittest.main()
