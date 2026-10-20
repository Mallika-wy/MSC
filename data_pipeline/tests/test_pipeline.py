from pathlib import Path

import numpy as np
import pytest

from data_pipeline.core.io import read_json, write_json
from data_pipeline.exporters.omg import assign_split, export_release, validate_release, windows
from data_pipeline.stages.motion import validate_qpos
from data_pipeline.stages.video import nearest_frames


def test_nearest_frames_uses_pts_not_every_second_frame():
    pts = np.array([0, .019, .037, .05, .071])
    np.testing.assert_array_equal(nearest_frames(pts, np.array([0, 1/30, 2/30])), [0, 2, 4])
    with pytest.raises(ValueError, match='coverage'):
        nearest_frames(pts, np.array([.1]))


def test_windows_and_group_splits():
    assert windows(359, 240, 120) == [(0, 240), (119, 359)]
    assert windows(80, 240, 120) == [(0, 80)]
    assert windows(1, 240, 120) == []
    assert assign_split('same/group', [1, 0, 0], '0') == 'train'
    assert assign_split('same/group', [.8, .1, .1], '0') == assign_split('same/group', [.8, .1, .1], '0')
    with pytest.raises(ValueError):
        windows(100, 10, 11)
    with pytest.raises(ValueError):
        assign_split('x', [.8, .1, .2], '0')


def test_invalid_motion():
    with pytest.raises(ValueError, match='quaternion'):
        validate_qpos(np.zeros((2, 36)), np.array([0, 1/30]))


def test_execution_cache_is_read_only(tmp_path, monkeypatch):
    from data_pipeline.stages import motion

    (tmp_path / 'paired_motion.npz').touch()
    quality_path = tmp_path / 'quality.json'
    quality = {'accepted': False, 'rejection_reasons': ['persistent_fall']}
    write_json(quality_path, quality)
    before = quality_path.read_bytes()

    def unexpected_execution(*args, **kwargs):
        pytest.fail('Cache hit must not execute motion')

    monkeypatch.setattr(motion, '_execute_motion', unexpected_execution)
    assert motion.execute_and_screen(None, {}, tmp_path, None) == quality
    assert quality_path.read_bytes() == before


@pytest.mark.parametrize('force,cached', [(False, False), (True, True)])
def test_execution_dispatch_forwards_force(tmp_path, monkeypatch, force, cached):
    from data_pipeline.stages import motion

    if cached:
        (tmp_path / 'paired_motion.npz').touch()
        write_json(tmp_path / 'quality.json', {'accepted': False})
    calls = []

    def fake_execute(reference, config, output, *, force):
        calls.append((reference, config, output, force))
        return {'accepted': True}

    monkeypatch.setattr(motion, '_execute_motion', fake_execute)
    result = motion.execute_and_screen('reference.npz', {}, tmp_path, None, force=force)
    assert result['accepted']
    assert calls == [('reference.npz', {}, tmp_path, force)]


def make_entry(tmp_path: Path, accepted=True):
    directory = tmp_path / 'sequence'
    directory.mkdir()
    video = tmp_path / 'source.mp4'
    video.touch()  # Export validation checks path/indices; decoding is a prior stage.
    qpos = np.zeros((8, 36), dtype=np.float32)
    qpos[:, 2], qpos[:, 3] = .8, 1
    qpos[:, 0] = np.arange(8) * .01
    times = np.arange(8) / 30
    np.savez_compressed(directory / 'paired_motion.npz', qpos_36=qpos, timestamps=times,
                        executed_state_mask=np.arange(8) > 0, fps=np.float32(30))
    write_json(directory / 'quality.json', {'accepted': accepted, 'production_ready': False})
    write_json(directory / 'video_index.json', {'views': [{'uri': str(video), 'camera_id': 'front',
                'frame_indices': list(range(8)), 'frame_pts_s': times.tolist()}]})
    return {'item': {'source': 'fixture', 'sequence_id': 'one', 'split_group': 'fixture/one'},
            'directory': str(directory), 'video_index': str(directory / 'video_index.json')}


def test_export_excludes_initialization_and_preserves_actions(tmp_path):
    import pyarrow.parquet as pq
    entry = make_entry(tmp_path)
    root = tmp_path / 'release'
    result = export_release([entry], {'split': {'ratios': [1, 0, 0]},
                            'export': {'episode_frames': 4, 'stride_frames': 3}}, root)
    assert result['episodes'] == 2
    rows = pq.read_table(root / 'data/chunk-000/file-000.parquet').to_pylist()
    assert rows[0]['pipeline.source_timestamp'] == pytest.approx(1/30)
    assert rows[3]['pipeline.action_valid'] is False
    assert rows[3]['action'] == rows[3]['observation.state']
    assert rows[2]['action'] == rows[3]['observation.state']
    mapping = read_json(root / 'meta/video_index.json')
    assert mapping[0]['views'][0]['frame_indices'] == [1, 2, 3, 4]
    assert validate_release(root)['valid']
    assert export_release([entry], {'split': {'ratios': [1, 0, 0]},
                          'export': {'episode_frames': 4, 'stride_frames': 3}}, root)['episodes'] == 2
    write_json(root / 'meta/video_index.json', [])
    with pytest.raises(ValueError, match='count mismatch'):
        validate_release(root)


def test_rejected_sequence_never_exports(tmp_path):
    entry = make_entry(tmp_path, accepted=False)
    with pytest.raises(ValueError, match='No quality-accepted'):
        export_release([entry], {}, tmp_path / 'release')


def test_native_motion_loader_can_read_export(tmp_path):
    from omg.data.lerobot_dataset import LeRobotG1MotionDataset
    entry = make_entry(tmp_path)
    root = tmp_path / 'release'
    info = export_release([entry], {'split': {'ratios': [1, 0, 0]}}, root)
    dataset = LeRobotG1MotionDataset(
        root, 'train', repo_id=info['repo_id'], revision='0' * 40,
        manifest_sha256=info['manifest_sha256'], use_text=False,
        sequence_duration=2 / 30, num_prev_states=1,
    )
    sample = dataset[0]
    assert sample['motion_features'].shape[-1] == 125


def test_workers_must_be_positive():
    from data_pipeline.core.runner import run
    with pytest.raises(ValueError, match='workers'):
        run(Path('not-read.yaml'), workers=0)


@pytest.mark.parametrize('workers', [1, 3])
def test_group_dispatch_returns_each_sequence_once(tmp_path, monkeypatch, workers):
    from concurrent.futures import ThreadPoolExecutor
    import data_pipeline.core.runner as runner
    # 用轻量线程池测试调度契约；真实spawn/CUDA另做真实视频小批量验证。
    monkeypatch.setattr(runner, 'ProcessPoolExecutor',
                        lambda max_workers, mp_context: ThreadPoolExecutor(max_workers=max_workers))
    calls = []

    def fake_process(group, config, root, force):
        calls.append(group[0])
        return {'sequence_id': group[0]}, {'directory': str(root / group[0])}

    monkeypatch.setattr(runner, 'process_group', fake_process)
    results = list(runner.iter_processed([['a'], ['b'], ['c']], {}, tmp_path, False, workers))
    assert sorted(calls) == ['a', 'b', 'c']
    assert sorted(result['sequence_id'] for result, _ in results) == ['a', 'b', 'c']


def test_group_failure_is_reported_without_export_entry(tmp_path):
    from dataclasses import replace
    from data_pipeline.core.pairs import Pair
    from data_pipeline.core.runner import process_group
    pair = Pair('fixture', 'one', 'front', 'video.mp4', 'motion.npz', 'g1_qpos_wxyz')
    other = replace(pair, camera_id='back', motion_path='different.npz')
    result, entry = process_group([pair, other], {}, tmp_path)
    assert result['status'] == 'failed'
    assert 'share motion' in result['error']
    assert entry is None


def test_aist_backend_runs_as_pipeline_module(tmp_path, monkeypatch):
    import data_pipeline.stages.motion as motion
    from data_pipeline.core.pairs import Pair
    calls = []
    monkeypatch.setattr(motion.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    pair = Pair('aistpp', 'one', 'c01', 'video.mp4', 'motions.zip', 'aist_smpl',
                motion_fps=60, motion_member='motions/one.pkl', source_root='/source')
    config = {'retarget': {'gmr_root': '/gmr', 'smpl_model': '/smpl.pkl', 'gender': 'neutral',
                           'allow_nonofficial_model': True}}
    motion.decode_aist(pair, config, tmp_path / 'reference', tmp_path)
    args, options = calls[0]
    assert args[1:3] == ['-m', 'data_pipeline.backends.aist_smpl_gmr']
    assert options['cwd'] == tmp_path and options['check']
    assert '--allow-nonofficial-model' in args
