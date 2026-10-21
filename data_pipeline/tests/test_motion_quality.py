import numpy as np
import pytest

from data_pipeline.motion.quality import MotionScreeningConfig, resample_motion_at_times, screen_executed_motion


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


from data_pipeline.motion.validation import validate_qpos
from data_pipeline.motion.tracking import track
from data_pipeline.video.media import nearest_frames


def test_nearest_frames_uses_pts_not_every_second_frame():
    pts = np.array([0, .019, .037, .05, .071])
    np.testing.assert_array_equal(nearest_frames(pts, np.array([0, 1/30, 2/30])), [0, 2, 4])
    with pytest.raises(ValueError, match='coverage'):
        nearest_frames(pts, np.array([.1]))



def test_invalid_motion_quaternion():
    with pytest.raises(ValueError, match='quaternion'):
        validate_qpos(np.zeros((2, 36)), np.array([0, 1/30]))


@pytest.mark.parametrize('diagnostic', [False, True])
def test_tracking_returns_trimmed_arrays_and_only_renders_diagnostics(tmp_path, monkeypatch, diagnostic):
    import sys
    from types import ModuleType, SimpleNamespace
    from data_pipeline.motion import tracking_video
    reference = states(31).astype(np.float32)
    sampled = states(50).astype(np.float32)
    xml = tmp_path / 'robot.xml'
    xml.write_text('<mujoco><option timestep="0.002"/><worldbody><body><freejoint/>'
                   '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    runs, videos = [], []
    def execute(config):
        runs.append(config)
        np.savez(config.output, executed_qpos_36=sampled, reference_qpos_36=sampled,
                 fps=50., robot_xml=str(xml))
    for name in ('omg', 'omg.tracking', 'omg.tracking.holomotion'):
        module = ModuleType(name); module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, 'omg.tracking.holomotion.reference',
                        SimpleNamespace(resample_qpos=lambda *args: sampled))
    monkeypatch.setitem(sys.modules, 'omg.tracking.holomotion.runner',
                        SimpleNamespace(TrackerRunConfig=SimpleNamespace, run_holomotion_tracker=execute))
    monkeypatch.setattr(tracking_video, 'save_execution_video', lambda path: videos.append(path))
    labels, quality = track(reference, {'onnx':'mock'}, {'strict_joint_limits':False},
                            tmp_path / 'work', diagnostic=diagnostic)
    assert len(runs) == 1 and len(videos) == int(diagnostic)
    np.testing.assert_allclose(labels, reference[1:])
    assert quality['accepted'] and quality['tracking']['joint_rmse_rad'] == pytest.approx(0.)
    assert not (tmp_path / 'work/paired_motion.npz').exists()
    assert not (tmp_path / 'work/quality.json').exists()
