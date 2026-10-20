"""Exercise orchestration and real publication without loading neural models."""
from pathlib import Path
import wave

import numpy as np
import pytest
import yaml

from data_pipeline.pipeline import run
from data_pipeline.schema import Inventory
from data_pipeline.release.export import read_motion


def make_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b'\0' * 48000)


@pytest.mark.parametrize('diagnostic', [False, True])
def test_batch_keeps_failures_out_of_release_and_preserves_execution_timing(tmp_path, monkeypatch, diagnostic):
    from data_pipeline.motion import retarget, tracking, arm_twist
    from data_pipeline.audio import cache
    raw = tmp_path / 'raw'
    (raw / 'motions').mkdir(parents=True)
    for name in ('accepted', 'rejected', 'broken'):
        (raw / 'motions' / f'{name}.pkl').touch()
        make_wav(raw / 'wavs' / f'{name}.wav')
    onnx = tmp_path / 'tracker.onnx'
    onnx.write_bytes(b'synthetic-controller')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(
        dataset='aistpp', root='raw', output='processed',
        retarget=dict(gmr_root='gmr', smpl_model='smpl.pkl'),
        tracker=dict(onnx='tracker.onnx'), audio=dict(model='mert'))))
    qpos = np.zeros((6, 36), dtype=np.float32)
    qpos[:, 0] = np.arange(6) / 30
    qpos[:, 2:4] = [.8, 1.]

    def solve(sample, dataset, settings, output):
        if sample.name == 'broken':
            raise ValueError('broken source')
        output.mkdir(parents=True)
        reference = output / 'motion.npz'
        np.savez(reference, qpos_36=qpos, source_timestamps=np.arange(6)/30)
        return reference, 'synthetic-gmr', {}

    def screen(reference, policy):
        return {'accepted': 'rejected' not in str(reference), 'rejection_reasons': []}

    calls = []
    def execute(reference, version, execution, quality, controller, output):
        calls.append(reference)
        output.mkdir(parents=True)
        paired = output / 'paired_motion.npz'
        np.savez(paired, qpos_36=qpos, executed_state_mask=np.arange(6)>0)
        return dict(accepted=True, path=str(paired), version=controller)

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def extract(self, wav, output):
            from data_pipeline.io import sha256_file
            output.mkdir(exist_ok=True)
            np.save(output / 'values.npy', np.zeros((75, 768), dtype=np.float16))
            np.save(output / 'times.npy', np.arange(75, dtype=np.float64)/75)
            return dict(encoder='synthetic', checkpoint_hash='synthetic', preprocess_hash='synthetic',
                        input_checksum=sha256_file(wav), shape=[75, 768],
                        values=str(output/'values.npy'), timestamps=str(output/'times.npy'))

    monkeypatch.setattr(retarget, 'retarget', solve)
    monkeypatch.setattr(arm_twist, 'screen_reference', screen)
    monkeypatch.setattr(tracking, 'track', execute)
    monkeypatch.setattr(cache, 'AudioCache', Encoder)
    summary = run(config, diagnostic=diagnostic)
    assert (summary['accepted'], summary['rejected'], summary['errors']) == (1, 1, 1)
    assert len(calls) == (2 if diagnostic else 1)
    release = tmp_path / 'processed/release'
    inventory = Inventory.read(release)
    assert len(inventory.motions) == 2 and not inventory.annotations
    ref, executed = inventory.motions
    assert executed.parent_motion_id == ref.motion_id
    assert executed.time_origin_sec == pytest.approx(1/30)
    np.testing.assert_array_equal(read_motion(release, executed)[0], qpos[1:])
    link = next(l for l in inventory.links if l.motion_id == executed.motion_id)
    assert link.time_offset_sec == pytest.approx(1/30)
    assert link.alignment_status == 'estimated'
    assert (tmp_path / 'processed/samples/broken/error.json').is_file()


def test_dry_run_lists_real_pairs_without_creating_output(tmp_path):
    (tmp_path / 'raw/motion').mkdir(parents=True)
    (tmp_path / 'raw/motion/001.npy').touch()
    make_wav(tmp_path / 'raw/music_wav/001.wav')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(
        dataset='finedance', root='raw', output='processed',
        retarget=dict(gmr_root='missing', smpl_model='missing'),
        tracker=dict(onnx='missing'), audio=dict(model='missing'))))
    assert run(config, dry_run=True)['count'] == 1
    assert not (tmp_path / 'processed').exists()
    with pytest.raises(ValueError, match='No matching'):
        run(config, sequence='missing', dry_run=True)
