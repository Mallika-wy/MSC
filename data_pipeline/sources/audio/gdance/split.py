"""Split GDance's multi-person SMPL sequences into individual motion files."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def split_dataset(root: Path, output: Path):
    root, output = Path(root), Path(output)
    splits = {}
    for split in ('train', 'val', 'test'):
        for name in (root / f'{split}_split_sequence_names.txt').read_text().splitlines():
            if name in splits:
                raise ValueError(f'Duplicate split assignment: {name}')
            splits[name] = split
    files = sorted((root / 'motions_smpl').glob('*.pkl'))
    if not files:
        raise ValueError('No GDance motions found')
    output.mkdir(parents=True, exist_ok=True)
    motions = output / 'motions_smpl'
    motions.mkdir(exist_ok=True)
    rows = []
    for path in files:
        name = path.stem
        audio = root / 'musics' / f'{name}.wav'
        if name not in splits or not audio.is_file():
            raise ValueError(f'Missing split or audio for {name}')
        with path.open('rb') as stream:
            data = pickle.load(stream)
        poses, trans, betas = (np.asarray(data[k]) for k in ('smpl_poses', 'root_trans', 'smpl_betas'))
        if (poses.ndim != 3 or poses.shape[-1] != 72 or
                trans.shape != (*poses.shape[:2], 3) or betas.shape != (*poses.shape[:2], 10)):
            raise ValueError(f'Invalid SMPL shapes in {path}')
        for actor in range(len(poses)):
            actor_name = f'{name}__person_{actor:02d}'
            target = motions / f'{actor_name}.npz'
            np.savez(target, smpl_poses=poses[actor], root_trans=trans[actor], smpl_betas=betas[actor],
                     source_sequence=name, actor_index=actor, source_meta=json.dumps(data['meta']))
            rows.append((actor_name, name, actor, splits[name], f'motions_smpl/{target.name}',
                         str(audio.resolve()), poses.shape[1]))
    with (output / 'index.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('motion_id', 'source_sequence', 'actor_index', 'split', 'motion_path', 'audio_path', 'frames'))
        writer.writerows(rows)
    return len(files), len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    sequences, actors = split_dataset(args.root, args.output)
    print(f'Split {sequences} sequences into {actors} individual motions in {args.output}')
