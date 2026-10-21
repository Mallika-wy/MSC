"""Exercise independent orchestration and real publication without neural models."""
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


def make_config(tmp_path, names=('accepted', 'rejected', 'broken')):
    raw = tmp_path / 'raw'
    (raw / 'motions').mkdir(parents=True)
    for name in names:
        (raw / 'motions' / f'{name}.pkl').write_bytes(name.encode())
        make_wav(raw / 'wavs' / f'{name}.wav')
    (tmp_path / 'tracker.onnx').write_bytes(b'synthetic-controller')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(source=dict(dataset='aistpp', root='raw'),
        motion=dict(retarget=dict(gmr_root='gmr', smpl_model='smpl.pkl'), tracking=dict(onnx='tracker.onnx')),
        release=dict(output='processed'),
        audio=dict(model='mert'))))
    return config


@pytest.mark.parametrize('diagnostic', [False, True])
def test_batch_keeps_failures_out_of_release_and_preserves_execution_timing(tmp_path, monkeypatch, diagnostic):
    from data_pipeline.motion import retarget, tracking, arm_twist
    from data_pipeline.audio import processor as cache
    from data_pipeline.motion import processor as motion
    config = make_config(tmp_path)
    qpos = np.zeros((6, 36), dtype=np.float32)
    qpos[:, 0] = np.arange(6) / 30
    qpos[:, 2:4] = [.8, 1.]
    def solve(source, dataset, settings):
        if source.motion_id == 'broken':
            raise ValueError('broken source')
        return qpos.copy(), {}
    # Source identity is explicit, not inferred from temporary output names.
    original_process = motion.process_motion
    current = []
    def process(source, *args, **kwargs):
        current[:] = [source.motion_id]
        return original_process(source, *args, **kwargs)
    def screen(reference, policy):
        return {'accepted': current[0] != 'rejected', 'rejection_reasons': []}
    calls = []
    def execute(reference, execution, quality, output, *, diagnostic=False):
        calls.append(reference)
        return qpos[1:].copy(), dict(accepted=True)
    class Encoder:
        def __init__(self, *args, **kwargs): pass
        def extract(self, wav, output):
            from data_pipeline.io import sha256_file
            from data_pipeline.storage.store import Store
            arrays = Store(output / 'features.sqlite').put('fake', {},
                values=np.zeros((75,768), dtype=np.float16), timestamps=np.arange(75,dtype=np.float64)/75)
            return dict(encoder='synthetic', checkpoint_hash='synthetic', preprocess_hash='synthetic',
                        input_checksum=sha256_file(wav), shape=[75,768], arrays=arrays)
    monkeypatch.setattr(motion, 'environment_key', lambda *a: 'test')
    monkeypatch.setattr(motion, 'process_motion', process)
    monkeypatch.setattr(retarget, 'retarget', solve)
    monkeypatch.setattr(arm_twist, 'screen_reference', screen)
    monkeypatch.setattr(tracking, 'track', execute)
    monkeypatch.setattr(cache, 'AudioCache', Encoder)
    summary = run(config, diagnostic=diagnostic)
    assert (summary['accepted'], summary['rejected'], summary['errors']) == (1,1,1)
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
    from data_pipeline.storage.store import Store
    rows = list(Store(tmp_path / 'processed/work/results.sqlite').results())
    assert rows[1]['motions'][0]['error'] == 'broken source'
    assert not (tmp_path / 'processed/samples').exists()
    assert ref.storage_uri == executed.storage_uri
    assert executed.row_start == ref.num_frames


def test_dry_run_lists_real_pairs_without_creating_output(tmp_path):
    config = make_config(tmp_path, ['one'])
    assert run(config, dry_run=True)['count'] == 1
    assert not (tmp_path / 'processed').exists()
    with pytest.raises(ValueError, match='No matching'):
        run(config, sequence='missing', dry_run=True)


def _parallel_fake_sequence(source):
    import time
    time.sleep(.05 if source.sequence_id == 'accepted' else .01)
    name = source.sequence_id
    from data_pipeline.pipeline import _record
    return dict(name=name, sequence=_record(source), motions=[], assets=[], state='error' if name=='broken' else name)


def _parallel_fake_init(*args):
    from data_pipeline import pipeline
    pipeline._worker_process = _parallel_fake_sequence


def test_spawn_workers_keep_errors_and_stable_publication_order(tmp_path, monkeypatch):
    from data_pipeline import pipeline
    config = make_config(tmp_path)
    monkeypatch.setattr(pipeline, '_init_worker', _parallel_fake_init)
    published = []
    def publish(work, output, config):
        from data_pipeline.storage.store import Store
        published.extend(Store(work / 'results.sqlite').results())
    monkeypatch.setattr(pipeline, 'publish_work', publish)
    summary = pipeline.run(config, workers=2, diagnostic=True)
    assert (summary['accepted'], summary['rejected'], summary['errors']) == (1,1,1)
    assert [r['name'] for r in published] == ['accepted','broken','rejected']


@pytest.mark.parametrize('workers', [0,-1,True,1.5])
def test_invalid_worker_count(workers):
    with pytest.raises(ValueError, match='workers'):
        run('/not/read.yaml', workers=workers)
