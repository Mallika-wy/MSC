"""Dataset-centric discovery; no model is initialized here."""
from importlib import import_module

DATASETS = ('aistpp', 'finedance', 'gdance', 'omg')


def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name!r}; choose from {DATASETS}')
    return import_module(f'data_pipeline.sources.{name}')


def discover(name, root, *, audio=True, text=True, video=False, sequence=None):
    for source in load_dataset(name).discover(root, audio=audio, text=text, video=video, sequence=sequence):
        if sequence is None or source.sequence_id == sequence or any(m.motion_id == sequence for m in source.motions):
            yield source
