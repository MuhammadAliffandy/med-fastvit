#!/usr/bin/env python3
"""Archive Experiment 02 and create a portable PyTorch bundle for Core ML work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from archive_experiment import sha256, verify_archive, capture_source_environment


def validate_exp02(run):
    required=['best.pth','best_map.pth','last.pth','history.json','run_config.json',
              'eval_valid_loss/metrics.json','eval_valid_map/metrics.json',
              'eval_valid_loss/comparison_exp01.json','eval_valid_map/comparison_exp01.json']
    for relative in required:
        if not (run/relative).is_file():raise FileNotFoundError(f'Missing Experiment 02 result: {run/relative}')
    history=json.loads((run/'history.json').read_text())
    config=json.loads((run/'run_config.json').read_text())
    if not history or [h['epoch'] for h in history] != list(range(1,config['args']['epochs']+1)):
        raise ValueError('Experiment 02 training history is incomplete or non-contiguous')
    loss=json.loads((run/'eval_valid_loss/metrics.json').read_text())
    metric=json.loads((run/'eval_valid_map/metrics.json').read_text())
    for filename,report in [('best.pth',loss),('best_map.pth',metric)]:
        if report['split']!='valid' or report['image_size']!=384:
            raise ValueError(f'Expected 384px validation evaluation for {filename}')
        if sha256(run/filename)!=report['checkpoint_sha256']:
            raise ValueError(f'{filename} differs from its evaluation checkpoint')
    if loss['classes']!=metric['classes'] or loss['dataset_audit']!=metric['dataset_audit']:
        raise ValueError('Loss and mAP checkpoints were evaluated with different datasets/classes')
    best_loss=min(history,key=lambda h:h['val_loss'])
    measured=[h for h in history if h.get('validation_coco') is not None]
    if not measured:raise ValueError('History does not contain validation mAP measurements')
    best_map=max(measured,key=lambda h:h['validation_coco']['mAP50_95'])
    if best_loss['epoch']!=loss['checkpoint_epoch'] or best_map['epoch']!=metric['checkpoint_epoch']:
        raise ValueError('Checkpoint selection epochs differ from training history')
    return config,history,loss,metric


def export_deployment_bundle(project,checkpoint_path,report,staging):
    import torch
    if str(project/'src') not in sys.path:sys.path.insert(0,str(project/'src'))
    from train_fastvit_medpack import build_detector
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    for key in ('epoch','classes','image_size'):
        report_key='checkpoint_epoch' if key=='epoch' else key
        if checkpoint[key]!=report[report_key]:raise ValueError(f'Checkpoint metadata mismatch: {key}')
    model=build_detector(checkpoint['backbone'],len(checkpoint['classes'])+1,checkpoint['image_size'],
                         pretrained=False,score_threshold=.001)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    portable={key:checkpoint[key] for key in ('model','epoch','classes','image_size','backbone')}
    portable.update({'format_version':1,'checkpoint_selection':'maximum_sampled_validation_mAP50_95',
                     'source_checkpoint_sha256':report['checkpoint_sha256']})
    downloads=staging/'download';downloads.mkdir()
    with tempfile.TemporaryDirectory(prefix='exp02-deployment-',dir=staging) as temporary:
        folder=Path(temporary)
        torch.save(portable,folder/'exp02_inference_weights.pth')
        model_config={'architecture':'FastViT-SA12 + FPN + Faster R-CNN',
            'backbone':checkpoint['backbone'],'checkpoint_epoch':checkpoint['epoch'],
            'classes':checkpoint['classes'],'class_ids':{'0':'background',**{str(i):n for i,n in enumerate(checkpoint['classes'],1)}},
            'input':{'color_space':'RGB','tensor_range':[0,1],'resize_preserves_aspect_ratio':True,
                'min_size':list(model.transform.min_size),'max_size':model.transform.max_size,
                'padding_size_divisible':model.transform.size_divisible,
                'normalization_mean':model.transform.image_mean,'normalization_std':model.transform.image_std,
                'normalization_location':'inside Torchvision detector; do not apply twice'},
            'fpn':{'out_channels':model.backbone.out_channels},
            'anchors':{'sizes':model.rpn.anchor_generator.sizes,'aspect_ratios':model.rpn.anchor_generator.aspect_ratios},
            'postprocess':{'score_floor':model.roi_heads.score_thresh,'nms_iou':model.roi_heads.nms_thresh,
                'max_detections_per_image':model.roi_heads.detections_per_img,'display_confidence':.5},
            'source_checkpoint_sha256':report['checkpoint_sha256'],
            'portable_weights_sha256':sha256(folder/'exp02_inference_weights.pth'),
            'conversion_status':'PyTorch source bundle; Core ML conversion and numerical validation pending'}
        (folder/'model_config.json').write_text(json.dumps(model_config,indent=2))
        (folder/'validation_metrics.json').write_text(json.dumps(report,indent=2))
        shutil.copy2(staging/'code/source_snapshot.tar.gz',folder/'source_snapshot.tar.gz')
        shutil.copytree(staging/'environment',folder/'environment')
        (folder/'README.md').write_text(
            '# Experiment 02: input for Core ML conversion\n\n'
            'Candidate: best_map.pth, selected by validation mAP50-95.\n'
            'exp02_inference_weights.pth contains the complete detector state and metadata without optimizer state.\n'
            'The model detects kemasan and pil. It does not identify medicine names.\n'
            'Reconstruct build_detector from the bundled source with pretrained=False and load state strictly.\n'
            'Follow model_config.json for RGB input, aspect ratio, padding, normalization and postprocessing.\n'
            'This is not a .mlpackage. Full Faster R-CNN operator support, numerical parity and device inference must be checked during conversion.\n')
        files=sorted(p for p in folder.rglob('*') if p.is_file())
        (folder/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(folder).as_posix()}\n' for p in files))
        archive=downloads/'exp02_fastvit_384_coreml_source.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as zf:
            for path in sorted(folder.rglob('*')):
                if path.is_file():zf.write(path,path.relative_to(folder))
    return {'candidate':'training/best_map.pth','epoch':checkpoint['epoch'],
            'checkpoint_sha256':report['checkpoint_sha256'],'bundle':archive.relative_to(staging).as_posix(),
            'bundle_sha256':sha256(archive),'conversion_status':model_config['conversion_status']}


def prepare_exp02(project,run,out):
    if out.exists():
        verify_archive(out)
        existing=json.loads((out/'experiment.json').read_text())
        if existing.get('experiment')!='exp02_fastvit_384':
            raise ValueError('Existing archive is not Experiment 02')
        bundle=existing['deployment_candidate']
        if sha256(out/bundle['bundle'])!=bundle['bundle_sha256']:
            raise ValueError('Existing deployment bundle checksum mismatch')
        return
    config,history,loss,metric=validate_exp02(run)
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=out.name+'.building-',dir=out.parent))
    try:
        shutil.copytree(run,staging/'training',ignore=shutil.ignore_patterns('*.pid','eval_valid_loss','eval_valid_map'))
        shutil.copytree(run/'eval_valid_loss',staging/'validation_loss')
        shutil.copytree(run/'eval_valid_map',staging/'validation_map')
        commit,status=capture_source_environment(project,staging)
        bundle=export_deployment_bundle(project,staging/'training/best_map.pth',metric,staging)
        # Check both full checkpoint copies against the corresponding evaluations.
        for name,report in [('best.pth',loss),('best_map.pth',metric)]:
            if sha256(staging/'training'/name)!=report['checkpoint_sha256']:
                raise ValueError(f'Copied {name} checksum mismatch')
        manifest={'experiment':'exp02_fastvit_384','archived_at_utc':datetime.now(timezone.utc).isoformat(),
            'dataset':{'doi':'10.17632/d23kndp3sv.1','version':1},'image_size':384,'classes':metric['classes'],
            'completed_epochs':history[-1]['epoch'],'run_config':config,
            'best_loss':{'epoch':loss['checkpoint_epoch'],'checkpoint_sha256':loss['checkpoint_sha256'],'validation':loss['coco_bbox']},
            'best_map':{'epoch':metric['checkpoint_epoch'],'checkpoint_sha256':metric['checkpoint_sha256'],'validation':metric['coco_bbox'],
                        'precision_recall':metric['precision_recall']},
            'deployment_candidate':bundle,'source_snapshot_commit':commit,
            'source_snapshot_note':'Code captured at archive time. Training revision is recorded separately in run_config.',
            'environment_note':'Environment captured at archive time.','test_evaluation':'not included'}
        (staging/'experiment.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
        paths=sorted(p for p in staging.rglob('*') if p.is_file())
        (staging/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(staging).as_posix()}\n' for p in paths))
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    verify_archive(out)
    print('Download for Core ML preparation:',out/bundle['bundle'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/exp02_fastvit_384'))
    parser.add_argument('--out',type=Path,default=Path('runs/archives/exp02_fastvit_384'))
    args=parser.parse_args()
    prepare_exp02(Path(__file__).resolve().parents[1],args.run.resolve(),args.out.resolve())
