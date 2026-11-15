#!/usr/bin/env python3
"""Prepare a verified experiment archive before copying it to mounted Drive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def validate_run(training, evaluation):
    for path in [training/'best.pth',training/'last.pth',training/'history.json',evaluation/'metrics.json']:
        if not path.is_file(): raise FileNotFoundError(f'Required result missing: {path}')
    metrics=json.loads((evaluation/'metrics.json').read_text())
    history=json.loads((training/'history.json').read_text())
    if metrics['split'] != 'valid': raise ValueError('Experiment 01 expects validation evaluation')
    actual=sha256(training/'best.pth')
    if actual != metrics['checkpoint_sha256']:
        raise ValueError('best.pth differs from the checkpoint used for evaluation')
    if not history: raise ValueError('Training history is empty')
    best=min(history,key=lambda row:row['val_loss'])
    if best['epoch'] != metrics['checkpoint_epoch']:
        raise ValueError('Best checkpoint epoch differs from training history')
    return metrics,history,best


def verify_archive(out, checksums=None):
    reference=checksums or out/'SHA256SUMS'
    if checksums is not None and sha256(out/'SHA256SUMS') != sha256(reference):
        raise ValueError('Destination SHA256SUMS differs from the source archive')
    lines=reference.read_text().splitlines()
    if not lines: raise ValueError('Archive checksum list is empty')
    for line in lines:
        expected,relative=line.split('  ',1)
        if sha256(out/relative) != expected: raise ValueError(f'Archive checksum mismatch: {relative}')
    print(f'Archive contents verified: {out}',flush=True)


def capture_source_environment(project, staging):
    code=staging/'code';code.mkdir()
    commit=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','-C',str(project),'archive','--format=tar.gz',
                    f'--output={code / "source_snapshot.tar.gz"}','HEAD'],check=True)
    status=subprocess.check_output(['git','-C',str(project),'status','--porcelain'],text=True)
    (code/'git_status.txt').write_text(status)
    env=staging/'environment';env.mkdir()
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (env/'pip-freeze.txt').write_text(freeze)
    if shutil.which('conda'):
        exported=subprocess.run(['conda','env','export','--no-builds'],capture_output=True,text=True)
        if exported.returncode == 0: (env/'conda-environment.yml').write_text(exported.stdout)
    return commit,status


def prepare_archive(project, training, evaluation, out):
    if out.exists():
        verify_archive(out)
        return
    metrics,history,best=validate_run(training,evaluation)
    out.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=out.name+'.building-',dir=out.parent))
    try:
        shutil.copytree(training,staging/'training',ignore=shutil.ignore_patterns('*.pid'))
        shutil.copytree(evaluation,staging/'validation')
        commit,status=capture_source_environment(project,staging)
        manifest={'experiment':'exp01_fastvit_320','archived_at_utc':datetime.now(timezone.utc).isoformat(),
            'dataset':{'doi':'10.17632/d23kndp3sv.1','version':1},
            'classes':metrics['classes'],'image_size':metrics['image_size'],
            'completed_epochs':history[-1]['epoch'],'best_checkpoint_epoch':metrics['checkpoint_epoch'],
            'best_validation_loss':best['val_loss'],'best_checkpoint_sha256':metrics['checkpoint_sha256'],
            'evaluation_split':metrics['split'],'validation_metrics':metrics['coco_bbox'],
            'precision_recall':metrics['precision_recall'],
            'source_snapshot_commit':commit,
            'source_snapshot_note':'Tracked code at archive time; original training Git revision was not recorded by trainer.',
            'environment_note':'Environment captured at archive time.',
            'test_evaluation':'not included'}
        (staging/'experiment.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
        # Detect source changes during the copy rather than archiving the wrong checkpoint.
        if sha256(staging/'training/best.pth') != metrics['checkpoint_sha256']:
            raise ValueError('Copied checkpoint hash differs from validation evaluation')
        paths=sorted(p for p in staging.rglob('*') if p.is_file())
        (staging/'SHA256SUMS').write_text(''.join(f'{sha256(p)}  {p.relative_to(staging).as_posix()}\n' for p in paths))
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    verify_archive(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',default='runs/fastvit_medpack_fixed')
    parser.add_argument('--evaluation',default='runs/eval_valid_best')
    parser.add_argument('--out',default='runs/archives/exp01_fastvit_320')
    parser.add_argument('--verify-dir',type=Path,default=None)
    parser.add_argument('--checksums',type=Path,default=None)
    args=parser.parse_args()
    if args.verify_dir:
        verify_archive(args.verify_dir.expanduser().resolve(),args.checksums)
        sys.exit(0)
    project=Path(__file__).resolve().parents[1]
    prepare_archive(project,Path(args.training).resolve(),Path(args.evaluation).resolve(),Path(args.out).resolve())
