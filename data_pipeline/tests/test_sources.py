"""Dataset discovery preserves identities without importing motion models."""
import numpy as np
import pickle

from data_pipeline.sources.audio import load_dataset
from data_pipeline.sources.audio.gdance.split import split_dataset
from data_pipeline.tests.test_pipeline import make_wav


def test_aist_pairs_do_not_depend_on_camera_files(tmp_path):
    root = tmp_path / 'edge_aistpp'
    (root / 'motions').mkdir(parents=True)
    (root / 'motions/one.pkl').touch()
    make_wav(root / 'wavs/one.wav')
    samples = list(load_dataset('aistpp').samples(tmp_path))
    assert len(samples) == 1
    assert samples[0].name == samples[0].sequence == 'one'
    assert samples[0].split == 'train'


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
    samples = list(load_dataset('gdance').samples(output))
    assert samples[0].actor != samples[1].actor
    assert samples[0].audio == samples[1].audio
    assert {s.sequence for s in samples} == {'group'}
    assert {s.split for s in samples} == {'val'}
    for actor, sample in enumerate(samples):
        with np.load(sample.motion, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['smpl_poses'], poses[actor])
