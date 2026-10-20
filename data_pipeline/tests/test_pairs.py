import csv
from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest

from data_pipeline.adapters.aistpp import build_csv
from data_pipeline.core.io import read_json, write_json
from data_pipeline.core.pairs import Pair, read_pairs, write_pairs
from data_pipeline.core.runner import run


def test_aist_directory_to_csv_without_audit_or_unpacking(tmp_path):
    raw = tmp_path / 'AIST'
    (raw / 'archives').mkdir(parents=True)
    (raw / 'videos').mkdir()
    sequences = ['gBR_sBM_cAll_d04_mBR0_ch01', 'gBR_sBM_cAll_d04_mBR0_ch02']
    archive = raw / 'archives/motions_complete.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        for sequence in sequences:
            stream.writestr(f'motions/{sequence}.pkl', b'not-decoded-by-indexer')
    (raw / 'archives/ignore_list.txt').write_text(sequences[1])
    video = raw / 'videos' / (sequences[0].replace('_cAll_', '_c01_')+'.mp4')
    video.touch()
    output = raw / 'pairs.csv'
    build_csv(raw, output)
    pairs = read_pairs(output)
    assert len(pairs) == 1
    assert pairs[0].video_path == str(video)
    assert pairs[0].motion_path == str(archive)
    assert pairs[0].motion_member == f'motions/{sequences[0]}.pkl'
    assert not (raw / 'metadata').exists()
    assert not (raw / 'motions').exists()


def test_csv_relative_paths_duplicates_and_unsafe_names(tmp_path):
    (tmp_path / 'video.mp4').touch()
    (tmp_path / 'motion.npz').touch()
    pair = Pair('custom', 'clip', 'front', 'video.mp4', 'motion.npz', 'g1_qpos_wxyz')
    path = tmp_path / 'pairs.csv'
    write_pairs(path, [pair])
    assert read_pairs(path)[0].video_path == str(tmp_path / 'video.mp4')
    write_pairs(path, [pair, pair])
    with pytest.raises(ValueError, match='Duplicate'):
        read_pairs(path)
    with pytest.raises(ValueError, match='Unsafe'):
        replace(pair, sequence_id='../escape')


def test_csv_driven_pipeline_groups_views_and_exports(tmp_path, monkeypatch):
    import data_pipeline.core.runner as runner
    raw = tmp_path / 'raw'
    raw.mkdir()
    qpos = np.zeros((8, 36), np.float32)
    qpos[:, 2], qpos[:, 3] = .8, 1
    motion = raw / 'motion.npz'
    np.savez_compressed(motion, qpos_36=qpos, timestamps=np.arange(8)/30)
    pairs = []
    for camera in ['front', 'side']:
        video = raw / f'{camera}.mp4'
        video.touch()
        pairs.append(Pair('custom', 'clip', camera, str(video), str(motion), 'g1_qpos_wxyz'))
    csv_path = tmp_path / 'pairs.csv'
    write_pairs(csv_path, pairs)
    config = tmp_path / 'config.yaml'
    config.write_text(f'input_csv: {csv_path}\noutput:\n  root: {tmp_path / "output"}\nsplit:\n  ratios: [1, 0, 0]\n')
    calls = []

    def fake_execute(reference, config, directory, project, force=False):
        assert read_json(directory / 'reference/quality.json')['accepted']
        calls.append(reference)
        np.savez_compressed(directory / 'paired_motion.npz', qpos_36=qpos, timestamps=np.arange(8)/30,
                            executed_state_mask=np.arange(8)>0, fps=np.float32(30))
        write_json(directory / 'quality.json', {'accepted': True})
        return {'accepted': True}

    def fake_video(pair, times, config, output, force=False):
        return {'uri': pair.video_path, 'camera_id': pair.camera_id,
                'frame_indices': list(range(8)), 'frame_pts_s': times.tolist()}

    monkeypatch.setattr(runner, 'execute_and_screen', fake_execute)
    monkeypatch.setattr(runner, 'prepare_execution_video', lambda directory, **kw: directory / 'execution.mp4')
    monkeypatch.setattr(runner, 'prepare_video_view', fake_video)
    output = run(config)
    assert len(calls) == 1  # Two videos share one motion processing step.
    assert (output / 'sequences/custom/clip/reference/motion.npz').is_file()
    views = read_json(output / 'release/meta/video_index.json')[0]['views']
    assert len(views) == 2
    with (output / 'results.csv').open() as stream:
        assert list(csv.DictReader(stream))[0]['status'] == 'complete'
    assert not (output / 'inputs.json').exists()
    assert not (output / 'inventory.json').exists()


def test_unknown_motion_format_fails_clearly(tmp_path):
    from data_pipeline.stages.motion import prepare_reference
    pair = Pair('custom', 'clip', 'front', 'video.mp4', 'motion.bvh', 'bvh')
    with pytest.raises(ValueError, match='add a decoder'):
        prepare_reference(pair, {}, tmp_path, tmp_path)
