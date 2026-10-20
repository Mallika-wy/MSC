from pathlib import Path

import numpy as np

from data_pipeline.visualization.execution import prepare_execution_video


def test_execution_replay_uses_saved_states_and_native_rate(tmp_path, monkeypatch):
    import mujoco
    from omg.tracking.holomotion.runtime import resolve_robot_xml
    seen = []

    class FakeRenderer:
        def __init__(self, model, height, width):
            assert (height, width) == (512, 512)

        def update_scene(self, state, camera):
            seen.append(state.qpos[:7].copy())

        def render(self):
            return np.zeros((512, 512, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(mujoco, 'Renderer', FakeRenderer)
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 2], qpos[:, 3] = .8, 1.
    qpos[:, 0] = [0., .01, .02]
    np.savez(tmp_path / 'tracker_rollout.npz', executed_qpos_36=qpos,
             fps=np.array([50.]), robot_xml=np.array([str(resolve_robot_xml(None))]))
    output = prepare_execution_video(tmp_path)
    assert output == tmp_path / 'execution.mp4'
    np.testing.assert_array_equal(seen, qpos[:, :7])
    prepare_execution_video(tmp_path)  # 只复用视频，不再次渲染或仿真。
    assert len(seen) == 3
    prepare_execution_video(tmp_path, force=True)
    assert len(seen) == 6


def test_rejected_motion_still_gets_execution_video(tmp_path, monkeypatch):
    import data_pipeline.core.runner as runner
    from data_pipeline.core.pairs import Pair
    rendered = []
    monkeypatch.setattr(runner, 'prepare_reference', lambda *a, **kw: Path('reference.npz'))
    monkeypatch.setattr(runner, 'screen_reference', lambda *a, **kw: {'accepted': True})
    (tmp_path / 'sequences/fixture/one/reference').mkdir(parents=True)
    monkeypatch.setattr(runner, 'execute_and_screen', lambda *a, **kw:
                        {'accepted': False, 'rejection_reasons': ['persistent_fall']})
    monkeypatch.setattr(runner, 'prepare_execution_video', lambda directory, **kw: rendered.append(directory))
    pair = Pair('fixture', 'one', 'front', 'video.mp4', 'motion.npz', 'g1_qpos_wxyz')
    result, entry = runner.process_group([pair], {}, tmp_path)
    assert result['status'] == 'rejected' and entry is None
    assert rendered == [tmp_path / 'sequences/fixture/one']
