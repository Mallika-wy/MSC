"""Dataset discovery preserves identities without importing motion models."""
import numpy as np
import pickle

from data_pipeline.sources import discover
from data_pipeline.sources.gdance.split import split_dataset
from data_pipeline.tests.test_pipeline import make_wav


def test_aist_pairs_do_not_depend_on_camera_files(tmp_path):
    root = tmp_path / 'edge_aistpp'
    (root / 'motions').mkdir(parents=True)
    (root / 'motions/one.pkl').touch()
    make_wav(root / 'wavs/one.wav')
    sequences = list(discover('aistpp', tmp_path))
    assert len(sequences) == 1
    assert sequences[0].motions[0].motion_id == sequences[0].sequence_id == 'one'
    assert sequences[0].split == 'train'


def test_gdance_split_preserves_each_actor_and_shares_audio(tmp_path):
    root = tmp_path / 'raw'
    (root / 'motions_smpl').mkdir(parents=True)
    poses = np.arange(2 * 3 * 72, dtype=np.float32).reshape(2, 3, 72)
    data = dict(smpl_poses=poses, root_trans=np.zeros((2, 3, 3)),
                smpl_betas=np.zeros((2, 3, 10)), meta={'n_persons': 2})
    with (root / 'motions_smpl/group.pkl').open('wb') as stream:
        pickle.dump(data, stream)
    make_wav(root / 'musics/group.wav')
    for split in ('train', 'val', 'test'):
        (root / f'{split}_split_sequence_names.txt').write_text('group\n' if split == 'val' else '')
    output = tmp_path / 'single_person'
    assert split_dataset(root, output) == (1, 2)
    sequences = list(discover('gdance', output))
    assert len(sequences) == 1 and sequences[0].sequence_id == 'group'
    sequence = sequences[0]
    assert sequence.split == 'val' and len(sequence.motions) == 2
    assert sequence.motions[0].actor_id != sequence.motions[1].actor_id
    assert len(sequence.assets) == 1
    assert len({link.asset_id for link in sequence.alignments}) == 1
    for actor, motion in enumerate(sequence.motions):
        with np.load(motion.path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['smpl_poses'], poses[actor])
