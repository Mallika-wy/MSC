"""Audio-motion datasets. Each dataset owns its file discovery and motion decoder."""
from importlib import import_module

DATASETS = ('aistpp', 'finedance', 'gdance')


def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown audio dataset {name!r}; choose from {DATASETS}')
    return import_module(f'data_pipeline.sources.audio.{name}')
