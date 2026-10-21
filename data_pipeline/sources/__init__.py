"""Dataset-centric discovery; no model is initialized here."""
from importlib import import_module
from itertools import groupby
from .base import SourceAlignment, SourceAsset, SourceMotion, SourceSequence

DATASETS = ('aistpp', 'finedance', 'gdance', 'omg')


def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {name!r}; choose from {DATASETS}')
    return import_module(f'data_pipeline.sources.{name}')


def discover(name, root, *, audio=True, text=True, video=False, sequence=None):
    if name == 'omg':
        yield from load_dataset(name).discover(root, text=text, sequence=sequence)
        return
    # GDance index may not be grouped; retain only lightweight discovery metadata.
    samples = load_dataset(name).samples(root, require_audio=audio)
    if name == 'gdance':
        samples = sorted(samples, key=lambda s: (s.sequence, s.name))
    for sequence_id, group in groupby(samples, key=lambda s: s.sequence):
        members = list(group)
        if len({s.split for s in members}) != 1:
            raise ValueError(f'Conflicting splits for {sequence_id}')
        assets, alignments = {}, []
        for sample in members:
            if audio and sample.audio is not None:
                identity = str(sample.audio.resolve())
                asset = assets.setdefault(identity, SourceAsset(identity, sample.audio, 'audio'))
                alignments.append(SourceAlignment(sample.name, asset.asset_id))
        yield SourceSequence(sequence_id, name, sequence_id,
            [SourceMotion(s.name, s.motion, s.actor) for s in members],
            list(assets.values()), alignments=alignments, split=members[0].split,
            provenance={'source_sequence': sequence_id})
