import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import archive_exp02 as archive


class Exp02ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.run=self.root/'run';self.run.mkdir()
        self.write_run()
    def tearDown(self):self.tmp.cleanup()
    def write_run(self):
        (self.run/'best.pth').write_bytes(b'loss-selected checkpoint')
        (self.run/'best_map.pth').write_bytes(b'map-selected checkpoint')
        (self.run/'last.pth').write_bytes(b'last checkpoint')
        history=[{'epoch':1,'val_loss':.1,'validation_coco':{'mAP50_95':.6}},
                 {'epoch':2,'val_loss':.2,'validation_coco':{'mAP50_95':.69}}]
        (self.run/'history.json').write_text(json.dumps(history))
        (self.run/'run_config.json').write_text(json.dumps({'experiment':'exp02_fastvit_384','args':{'epochs':2}}))
        for folder,name,epoch in [('eval_valid_loss','best.pth',1),('eval_valid_map','best_map.pth',2)]:
            path=self.run/folder;path.mkdir(exist_ok=True)
            report={'split':'valid','image_size':384,'classes':['kemasan','pil'],'dataset_audit':{'boxes':234},
                    'checkpoint_epoch':epoch,'checkpoint_sha256':archive.sha256(self.run/name),
                    'coco_bbox':{'mAP50_95':.69},'precision_recall':{}}
            (path/'metrics.json').write_text(json.dumps(report))
            (path/'comparison_exp01.json').write_text('{}')
    def test_validation_tracks_both_selection_criteria(self):
        _,_,loss,metric=archive.validate_exp02(self.run)
        self.assertEqual(loss['checkpoint_epoch'],1);self.assertEqual(metric['checkpoint_epoch'],2)
        (self.run/'best_map.pth').write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError,'differs from its evaluation'):
            archive.validate_exp02(self.run)
    def test_incomplete_run_is_rejected(self):
        history=json.loads((self.run/'history.json').read_text())[:1]
        (self.run/'history.json').write_text(json.dumps(history))
        with self.assertRaisesRegex(ValueError,'incomplete'):
            archive.validate_exp02(self.run)
    def test_selected_epoch_mismatch_is_rejected(self):
        path=self.run/'eval_valid_map/metrics.json';report=json.loads(path.read_text());report['checkpoint_epoch']=1
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'selection epochs'):
            archive.validate_exp02(self.run)
    def test_portable_bundle_and_full_archive(self):
        import torch
        from train_fastvit_medpack import build_detector
        torch.set_num_threads(2)
        detector=build_detector('fastvit_sa12.apple_in1k',3,384,pretrained=False)
        state=detector.state_dict()
        for filename,epoch in [('best.pth',1),('best_map.pth',2),('last.pth',2)]:
            torch.save({'model':state,'epoch':epoch,'classes':['kemasan','pil'],'image_size':384,
                        'backbone':'fastvit_sa12.apple_in1k','optimizer':{'example':torch.ones(1024)}},self.run/filename)
        for folder,filename in [('eval_valid_loss','best.pth'),('eval_valid_map','best_map.pth')]:
            path=self.run/folder/'metrics.json';report=json.loads(path.read_text())
            report['checkpoint_sha256']=archive.sha256(self.run/filename);path.write_text(json.dumps(report))
        def capture(project,staging):
            (staging/'code').mkdir();(staging/'code/source_snapshot.tar.gz').write_bytes(b'source fixture')
            (staging/'environment').mkdir();(staging/'environment/pip-freeze.txt').write_text('timm==1.0.31\n')
            return 'a'*40,''
        out=self.root/'archive'
        with patch.object(archive,'capture_source_environment',side_effect=capture):
            archive.prepare_exp02(ROOT,self.run,out)
        manifest=json.loads((out/'experiment.json').read_text())
        self.assertEqual(manifest['best_map']['epoch'],2)
        self.assertEqual(manifest['best_loss']['epoch'],1)
        self.assertTrue((out/'training/last.pth').is_file())
        self.assertFalse((out/'training/eval_valid_map').exists())
        self.assertTrue((out/'validation_map/metrics.json').is_file())
        bundle=out/manifest['deployment_candidate']['bundle'];unpacked=self.root/'unpacked'
        with zipfile.ZipFile(bundle) as zf:zf.extractall(unpacked)
        config=json.loads((unpacked/'model_config.json').read_text())
        self.assertEqual(config['input']['max_size'],384)
        self.assertEqual(config['class_ids'],{'0':'background','1':'kemasan','2':'pil'})
        portable=torch.load(unpacked/'exp02_inference_weights.pth',map_location='cpu',weights_only=True)
        self.assertNotIn('optimizer',portable)
        self.assertEqual(portable['epoch'],2)
        detector.load_state_dict(portable['model'],strict=True)
        self.assertEqual(set(state),set(portable['model']))
        self.assertTrue(all(torch.equal(value,portable['model'][key]) for key,value in state.items()))
        archive.verify_archive(unpacked)
        archive.prepare_exp02(ROOT,self.run,out)

if __name__=='__main__':unittest.main()
