"""Resolve explicit paths from one dataset YAML; paths are relative to that file."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from data_pipeline.sources.audio import DATASETS


@dataclass
class Settings:
    dataset: str
    root: Path
    output: Path
    retarget: dict
    tracker: dict
    audio: dict
    quality: dict

    @classmethod
    def read(cls, config, output=None):
        config = Path(config).resolve()
        values = yaml.safe_load(config.read_text())
        def path(value):
            return str((config.parent / Path(value).expanduser()).resolve())
        dataset = values['dataset']
        if dataset not in DATASETS:
            raise ValueError(f'Unsupported dataset: {dataset}')
        retarget = dict(values['retarget'])
        for key in ('gmr_root', 'smpl_model'):
            retarget[key] = path(retarget[key])
        tracker = dict(values['tracker'])
        tracker['onnx'] = path(tracker['onnx'])
        audio = dict(values['audio'])
        audio['model'] = path(audio['model'])
        destination = Path(output).resolve() if output else Path(path(values['output']))
        root = Path(path(values['root']))
        if destination == root or destination.is_relative_to(root):
            raise ValueError('Output must be outside the source dataset')
        return cls(dataset, root, destination, retarget, tracker, audio, values.get('quality', {}))
