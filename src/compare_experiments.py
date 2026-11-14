#!/usr/bin/env python3
"""Compare two validation reports after verifying data and evaluation settings."""
import argparse
import json
from pathlib import Path


def compare_reports(baseline_dir, candidate_dir, selection):
    baseline=json.loads((baseline_dir/'metrics.json').read_text())
    candidate=json.loads((candidate_dir/'metrics.json').read_text())
    if baseline['split'] != 'valid' or candidate['split'] != 'valid':
        raise ValueError('Development comparisons require validation reports')
    for key in ('classes','dataset_audit','inference'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'Evaluation mismatch: {key}')
    for key in ('score_threshold','iou_threshold','matching'):
        if baseline['precision_recall'][key] != candidate['precision_recall'][key]:
            raise ValueError(f'Evaluation threshold/matching mismatch: {key}')
    def targets(directory):
        gt=json.loads((directory/'ground_truth.json').read_text())
        return {key:gt[key] for key in ('images','annotations','categories')}
    if targets(baseline_dir) != targets(candidate_dir):
        raise ValueError('Validation images or ground-truth annotations differ')
    values={}
    for metric in ('mAP50_95','mAP50','mAP75'):
        old,new=baseline['coco_bbox'][metric],candidate['coco_bbox'][metric]
        values[metric]={'exp01':old,'exp02':new,'delta_percentage_points':100*(new-old)}
    per_class={}
    for name in candidate['classes']:
        before=baseline['precision_recall']['per_class'][name]
        after=candidate['precision_recall']['per_class'][name]
        per_class[name]={'exp01':before,'exp02':after,'false_positive_delta':after['fp']-before['fp'],
                         'AP50_95_exp01':baseline['coco_bbox']['per_class'][name]['AP50_95'],
                         'AP50_95_exp02':candidate['coco_bbox']['per_class'][name]['AP50_95']}
    return {'split':'valid','exp01_checkpoint':baseline['checkpoint'],'exp02_checkpoint':candidate['checkpoint'],
            'exp01_image_size':baseline['image_size'],'exp02_image_size':candidate['image_size'],
            'exp02_checkpoint_selection':selection,
            'comparison_note':'Same validation data and thresholds. MAP-selected checkpoints are a separate selection analysis.'
                if selection=='maximum_sampled_map' else 'Both checkpoints selected by minimum validation loss.',
            'metrics':values,'per_class':per_class}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',default='runs/eval_valid_best',type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--selection',choices=['minimum_loss','maximum_sampled_map'],default='minimum_loss')
    args=parser.parse_args()
    report=compare_reports(args.baseline,args.candidate,args.selection)
    (args.candidate/'comparison_exp01.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))
