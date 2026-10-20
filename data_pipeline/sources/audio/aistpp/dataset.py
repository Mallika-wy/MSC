"""List AIST++ sequences without loading SMPL or GMR."""
from data_pipeline.sources.sample import Sample


def samples(root):
    if (root / 'edge_aistpp').is_dir():
        root = root / 'edge_aistpp'
    ignore = root / 'archives/ignore_list.txt'
    ignored = set(ignore.read_text().split()) if ignore.is_file() else set()
    for motion in sorted((root / 'motions').glob('*.pkl')):
        if motion.stem not in ignored:
            yield Sample(motion.stem, motion.stem, motion, root / 'wavs' / f'{motion.stem}.wav').validate()
