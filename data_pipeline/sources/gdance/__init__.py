"""AIOZ-GDance: actor motions grouped by sequence, with shared audio."""
import csv
from pathlib import Path

from data_pipeline.sources.base import SourceMotion, paired_sequence
from .motion import decode_motion


def discover(root, *, audio=True, text=True, video=False, sequence=None):
    groups = {}
    with (root / 'index.csv').open() as stream:
        for row in csv.DictReader(stream):
            groups.setdefault(row['source_sequence'], []).append(row)
    for identity, rows in sorted(groups.items()):
        if len({row['split'] for row in rows}) != 1:
            raise ValueError(f'Conflicting splits for {identity}')
        pairs = []
        for row in sorted(rows, key=lambda row: row['motion_id']):
            wav = Path(row['audio_path']) if audio else None
            if wav is not None and not wav.is_absolute():
                wav = root / wav
            motion = SourceMotion(row['motion_id'], root / row['motion_path'],
                                  f"person_{int(row['actor_index']):02d}")
            pairs.append((motion, wav))
        yield paired_sequence('gdance', identity, pairs, rows[0]['split'])
