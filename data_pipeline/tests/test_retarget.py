import numpy as np
import pytest

from data_pipeline.robot import G1_JOINT_NAMES
from data_pipeline.stages.retarget import HumanMotion, run_gmr


@pytest.mark.parametrize('gmr_source', ['smplx', 'bvh_lafan1'])
def test_shared_gmr_stage_preserves_frames_and_jump_height(monkeypatch, gmr_source):
    import mujoco
    import data_pipeline.stages.retarget as stage
    calls, options = [], []

    class Solver:
        def __init__(self, **kwargs):
            options.append(kwargs)
            joints = ''.join(f'<body><joint name="{name}"/><geom type="sphere" size=".01" mass=".01"/></body>'
                             for name in G1_JOINT_NAMES)
            self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
                '<geom type="sphere" size=".1" mass="1"/>' + joints + '</body></worldbody></mujoco>')

        def retarget(self, frame):
            calls.append(frame)
            qpos = np.zeros(36)
            qpos[:3], qpos[3:7] = frame['pelvis']
            return qpos

    monkeypatch.setattr(stage, 'load_gmr', lambda root: Solver)
    positions = np.array([[[0., 0., .8]], [[.1, 0., .9]], [[.2, 0., .8]]])
    motion = HumanMotion(('pelvis',), positions.copy(), np.tile([1., 0, 0, 0], (3, 1, 1)),
                         np.arange(3)/30, 1.8, gmr_source)
    qpos, report = run_gmr(motion, {'gmr_root': '/unused/mock'})
    assert qpos.shape == (3, 36)
    assert len(calls) == 11  # Eight initialization solves do not add time steps.
    assert options[0]['src_human'] == gmr_source
    assert options[0]['tgt_robot'] == 'unitree_g1'
    np.testing.assert_allclose(qpos[:, :2], positions[:, 0, :2])
    np.testing.assert_allclose(qpos[:, 2] - qpos[0, 2], [0., .1, 0.], atol=1e-7)
    np.testing.assert_array_equal(motion.positions, positions)
    assert np.isfinite(report['ground_constant_shift_m'])


def test_invalid_human_timing_is_rejected_before_loading_gmr(monkeypatch):
    import data_pipeline.stages.retarget as stage
    monkeypatch.setattr(stage, 'load_gmr', lambda _: pytest.fail('Invalid input must fail before GMR loads'))
    motion = HumanMotion(('pelvis',), np.zeros((2, 1, 3)), np.tile([1., 0, 0, 0], (2, 1, 1)),
                         np.array([0., .5]), 1.8, 'smplx')
    with pytest.raises(ValueError, match='30 Hz'):
        run_gmr(motion, {'gmr_root': '/unused'})


def test_aist_decoder_preserves_scale_axes_and_source_times(tmp_path, monkeypatch):
    import inspect
    import pickle
    import sys
    from types import SimpleNamespace
    import torch
    from data_pipeline.sources.aistpp import decode_motion
    from data_pipeline.schemas.records import SourceMotion
    before_aliases = {name: np.__dict__.get(name) for name in ('bool', 'int', 'float', 'complex', 'object', 'unicode', 'str')}
    before_argspec = getattr(inspect, 'getargspec', None)

    class SMPL:
        parents = torch.tensor([-1] + list(range(23)))
        def __init__(self, **kwargs):
            pass
        def __call__(self, transl=None, **kwargs):
            if transl is None:
                return SimpleNamespace(vertices=torch.tensor([[[0., 0., 0.], [0., 2., 0.]]]))
            joints = torch.zeros((len(transl), 24, 3))
            joints[:, :, 1] = torch.arange(24)/10
            return SimpleNamespace(joints=joints + transl[:, None, :])

    monkeypatch.setitem(sys.modules, 'smplx', SimpleNamespace(SMPL=SMPL))
    path = tmp_path / 'source.pkl'
    path.write_bytes(pickle.dumps({'smpl_poses': np.zeros((3, 72)), 'smpl_trans': np.array([[2.,0,0],[3.,0,0],[4.,0,0]]),
                                  'smpl_scaling': 2.}))
    source = SourceMotion('test:smpl', 'test:one', 'source.pkl', 'aist_smpl', 60.)
    motion = decode_motion(source, path, {'smpl_model': '/mock/model.pkl', 'gender': 'male'})
    np.testing.assert_allclose(motion.timestamps, [0., 1/30])
    np.testing.assert_allclose(motion.positions[:, 0, :], [[0.,0.,0.], [1.,0.,0.]], atol=1e-7)
    np.testing.assert_allclose(motion.positions[:, 1, 2], .1, atol=1e-7)  # Y-up became Z-up.
    assert motion.height_m == 2.
    assert motion.positions.shape == (2, 24, 3)
    assert {name: np.__dict__.get(name) for name in before_aliases} == before_aliases
    assert getattr(inspect, 'getargspec', None) is before_argspec
