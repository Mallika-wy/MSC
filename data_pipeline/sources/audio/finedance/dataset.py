"""List the paired FineDance motion and WAV files."""
from data_pipeline.sources.sample import Sample


def samples(root):
    if (root / 'finedance').is_dir():
        root = root / 'finedance'
    for motion in sorted((root / 'motion').glob('*.npy')):
        yield Sample(motion.stem, motion.stem, motion, root / 'music_wav' / f'{motion.stem}.wav').validate()
