"""Discover AIST++ SMPL sequences and optional matching WAV excerpts."""
from data_pipeline.sources.base import SourceMotion, paired_sequence
from .motion import decode_motion


def discover(root, *, audio=True, text=True, video=False, sequence=None):
    if (root / 'edge_aistpp').is_dir():
        root = root / 'edge_aistpp'
    ignore = root / 'archives/ignore_list.txt'
    ignored = set(ignore.read_text().split()) if ignore.is_file() else set()
    for motion in sorted((root / 'motions').glob('*.pkl')):
        if motion.stem not in ignored:
            wav = root / 'wavs' / f'{motion.stem}.wav' if audio else None
            yield paired_sequence('aistpp', motion.stem, [(SourceMotion(motion.stem, motion), wav)])
