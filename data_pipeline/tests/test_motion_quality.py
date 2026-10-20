import numpy as np
import pytest

from data_pipeline.stages.motion_quality import MotionScreeningConfig, resample_motion_at_times, screen_executed_motion


def states(n):
    q = np.zeros((n, 36)); q[:, 2] = .8; q[:, 3] = 1
    return q


def screen(q, **kwargs):
    return screen_executed_motion(q, fps=50, joint_lower=np.full(29, -1), joint_upper=np.ones(29), **kwargs)


def test_rejects_ten_consecutive_falls_but_not_separate_runs():
    q = states(30); q[:9, 2] = .1; q[10:19, 2] = .1
    assert screen(q)['accepted']
    q[9, 2] = .1
    assert 'persistent_fall' in screen(q)['rejection_reasons']


def test_yaw_is_not_tilt_and_combined_height_tilt_rule():
    q = states(10); q[:, 3:7] = [0, 0, 0, 1]
    assert screen(q)['accepted']
    q[:, 2] = .3
    angle = np.deg2rad(65) / 2
    q[:, 3:7] = [np.cos(angle), np.sin(angle), 0, 0]
    assert not screen(q)['accepted']


def test_nonfinite_and_joint_limit_failures_are_not_silently_accepted():
    q = states(3); q[1, 7] = np.nan
    assert screen(q)['rejection_reasons'] == ['nonfinite_state']
    q = states(3); q[1, 7] = 1.1
    assert not screen(q)['accepted']
    assert screen(q, config=MotionScreeningConfig(strict_joint_limits=False))['accepted']


def test_poststep_timestamps_require_initial_state_and_slerp_handles_sign():
    q = states(3); q[:, 0] = [0, .02, .04]; q[1, 3] = -1
    result = resample_motion_at_times(q, [0, .02, .04], [0, 1/30])
    np.testing.assert_allclose(result[:, 0], [0, 1/30], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1)
    with pytest.raises(ValueError, match='Extrapolation'):
        resample_motion_at_times(q[1:], [.02, .04], [0])
    with pytest.raises(ValueError, match='strictly increase'):
        resample_motion_at_times(q, [0, 0, .04], [0])


from data_pipeline.stages.motion import validate_qpos, execute_and_screen
from data_pipeline.io import write_json
from data_pipeline.video.media import nearest_frames


def test_nearest_frames_uses_pts_not_every_second_frame():
    pts = np.array([0, .019, .037, .05, .071])
    np.testing.assert_array_equal(nearest_frames(pts, np.array([0, 1/30, 2/30])), [0, 2, 4])
    with pytest.raises(ValueError, match='coverage'):
        nearest_frames(pts, np.array([.1]))



def test_invalid_motion_quaternion():
    with pytest.raises(ValueError, match='quaternion'):
        validate_qpos(np.zeros((2, 36)), np.array([0, 1/30]))


def test_execution_cache_preserves_quality_and_creates_video(tmp_path, monkeypatch):
    from data_pipeline.stages import execution_video
    calls = []
    monkeypatch.setattr(execution_video, 'save_execution_video',
                        lambda rollout: calls.append(rollout) or tmp_path / 'execution.mp4')
    (tmp_path / 'paired_motion.npz').touch()
    path = tmp_path / 'quality.json'
    quality = {'accepted': False, 'rejection_reasons': ['persistent_fall']}
    write_json(path, quality)
    before = path.read_bytes()
    result = execute_and_screen(None, {}, tmp_path)
    assert result == {**quality, 'execution_video': str(tmp_path / 'execution.mp4')}
    assert calls == [tmp_path / 'tracker_rollout.npz']
    assert path.read_bytes() == before
