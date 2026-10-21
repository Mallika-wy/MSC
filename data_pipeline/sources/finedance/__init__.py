"""Discover FineDance SMPL-H motions and optional matching WAV files."""
from data_pipeline.sources.base import SourceMotion, paired_sequence
from .motion import decode_motion


def discover(root, *, audio=True, text=True, video=False, sequence=None):
    if (root / 'finedance').is_dir():
        root = root / 'finedance'
    for motion in sorted((root / 'motion').glob('*.npy')):
        wav = root / 'music_wav' / f'{motion.stem}.wav' if audio else None
        yield paired_sequence('finedance', motion.stem, [(SourceMotion(motion.stem, motion), wav)])
