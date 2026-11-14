"""Validated JSON defaults for training experiment recipes."""
import json
import math
from pathlib import Path

FIELDS = {
    'experiment_name': str, 'epochs': int, 'image_size': int, 'batch_size': int,
    'workers': int, 'lr': (int,float), 'weight_decay': (int,float), 'seed': int,
    'backbone': str, 'annotation_format': str, 'val_map_every': int,
}


def validate_settings(settings):
    for key,value in settings.items():
        if key not in FIELDS:
            raise ValueError(f'Unknown training setting: {key}')
        if isinstance(value,bool) or not isinstance(value,FIELDS[key]):
            raise ValueError(f'Invalid type for training setting {key}: {value!r}')
        if isinstance(value,(int,float)) and not math.isfinite(value):
            raise ValueError(f'Non-finite training setting {key}')
    for key in ('epochs','image_size','batch_size'):
        if key in settings and settings[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('workers','weight_decay','val_map_every'):
        if key in settings and settings[key] < 0:
            raise ValueError(f'{key} cannot be negative')
    if 'lr' in settings and settings['lr'] <= 0:
        raise ValueError('lr must be positive')
    if 'annotation_format' in settings and settings['annotation_format'] not in ('yolo','voc'):
        raise ValueError('annotation_format must be yolo or voc')
    for key in ('experiment_name','backbone'):
        if key in settings and not settings[key].strip():
            raise ValueError(f'{key} cannot be empty')
    return settings


def parse_training_args(parser, argv=None):
    early,_=parser.parse_known_args(argv)
    if early.config:
        config=json.loads(Path(early.config).read_text())
        if not isinstance(config,dict):
            raise ValueError('Training configuration must be a JSON object')
        parser.set_defaults(**validate_settings(config))
    args=parser.parse_args(argv)
    validate_settings({key:getattr(args,key) for key in FIELDS})
    return args
