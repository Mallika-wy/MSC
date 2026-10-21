"""SONIC's 50 Hz boundary and upstream policy observation contract."""
import numpy as np
import yaml
import pytest

from data_pipeline.motion.sonic import (TO_ISAAC, decoder_observation,
                                        encoder_observation, sample_reference)


def test_reference_covers_original_duration_without_changing_30hz_labels():
    reference = np.zeros((31, 36), dtype=np.float32)
    reference[:, 0] = np.arange(31) / 30
    reference[:, 3] = 1
    resampled, times = sample_reference(reference)
    assert resampled.shape == (51, 36)
    np.testing.assert_allclose(resampled[::5, 0], reference[::3, 0], atol=1e-7)
    np.testing.assert_allclose(times[-1], 1)


def test_v11_observation_layout_and_joint_order():
    from pathlib import Path
    config = yaml.safe_load((Path(__file__).resolve().parents[2] /
        'third_party/sonic/observation_config.yaml').read_text())
    reference = np.zeros((60, 36), dtype=np.float32)
    reference[:, 3] = 1
    reference[:, 7:] = np.arange(29)
    velocity = np.ones((60, 29), dtype=np.float32)
    encoded = encoder_observation(reference, velocity, 0, reference[0, 3:7], config['encoder'])
    assert encoded.shape == (1751,)
    np.testing.assert_array_equal(encoded[:4], 0)
    np.testing.assert_array_equal(encoded[4:33], np.arange(29)[TO_ISAAC])
    np.testing.assert_array_equal(encoded[584:590], [1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(encoded[644:], 0)
    snapshot = (np.arange(3), np.arange(29), np.ones(29), np.zeros(29), np.array([0, 0, -1]))
    decoded = decoder_observation(np.zeros((1, 64)), [snapshot] * 10, config['observations'])
    assert decoded.shape == (994,)
    np.testing.assert_array_equal(decoded[64:67], [0, 1, 2])


def test_cached_robot_starts_each_motion_from_fresh_state(tmp_path):
    from pathlib import Path
    from data_pipeline.config import Settings
    from data_pipeline.motion.sonic import DEFAULT, run
    root = Path(__file__).resolve().parents[2]
    settings = Settings.read(root / 'configs/data_pipeline/omg_text.yaml')
    if not Path(settings.tracker['decoder']).is_file():
        pytest.skip('Requires local SONIC weights')
    pytest.importorskip('onnxruntime')
    pytest.importorskip('mujoco')
    reference = np.zeros((7, 36), dtype=np.float32)
    reference[:, 2:4] = [.8, 1.]
    reference[:, 7:] = DEFAULT
    run(reference, settings.tracker, tmp_path / 'first.npz')
    run(reference, settings.tracker, tmp_path / 'second.npz')
    with np.load(tmp_path / 'first.npz') as first, np.load(tmp_path / 'second.npz') as second:
        np.testing.assert_array_equal(first['executed_qpos_36'], second['executed_qpos_36'])
