from dataclasses import asdict, replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from data_pipeline.tests.fixtures import fixture_records, make_fixture
from data_pipeline.schemas import Inventory, TimeMap
from data_pipeline.schemas.manifest import resolve_uri
from data_pipeline.release.export import export_release, read_motion
from data_pipeline.release.package import package_release
from data_pipeline.release.validate import validate_release


def test_four_tasks_and_multi_actor_camera_roundtrip(tmp_path):
    inv, _ = fixture_records(tmp_path / 'raw')
    inv.write(tmp_path / 'inventory')
    loaded = Inventory.read(tmp_path / 'inventory')
    assert asdict(loaded) == asdict(inv)
    assert len(loaded.motions) == 5
    assert len(loaded.assets) == 3
    assert len(loaded.features) == 3
    combined = [l for l in loaded.links if 'combined' in l.motion_id]
    assert len([l for l in combined if l.modality == 'video']) == 4


@pytest.mark.parametrize('mutation,match', [
    (lambda i: i.sequences.append(i.sequences[0]), 'Duplicate'),
    (lambda i: setattr(i.links[0], 'motion_id', 'missing'), 'foreign key'),
    (lambda i: setattr(i.assets[0], 'uri', '../escape.wav'), 'Unsafe URI'),
    (lambda i: setattr(i.assets[0], 'uri', '/absolute.wav'), 'Unsafe URI'),
    (lambda i: setattr(i.motions[0], 'time_origin_sec', float('nan')), 'Invalid type'),
    (lambda i: setattr(i.motions[0], 'num_frames', 3.5), 'Invalid type'),
    (lambda i: setattr(i.features[0], 'shape', [3., 8]), 'integers'),
    (lambda i: setattr(i.features[0], 'input_asset_checksum', 'different'), 'identity'),
    (lambda i: setattr(i.annotations[0], 'end_sec', 4.), 'coverage'),
    (lambda i: setattr(i.links[0], 'time_offset_sec', 100.), 'asset coverage'),
    (lambda i: setattr(i.links[0], 'time_scale', -1.), 'positive scale'),
    (lambda i: setattr(i.sequences[-1], 'split', 'test'), 'leakage'),
])
def test_invalid_records(tmp_path, mutation, match):
    inv, _ = fixture_records(tmp_path)
    mutation(inv)
    with pytest.raises(ValueError, match=match):
        inv.validate(tmp_path)


@pytest.mark.parametrize('use_alias', [False, True])
def test_shared_audio_and_group_allowed_across_splits(tmp_path, use_alias):
    inv, _ = fixture_records(tmp_path)
    inv.sequences[1].split = 'test'  # Audio and combined sequences still share a group.
    if use_alias:
        alias = replace(inv.assets[0], asset_id='alias:audio')
        inv.assets.append(alias)
        for link in inv.links:
            if 'combined' in link.motion_id and link.modality == 'audio':
                link.asset_id = alias.asset_id
    inv.validate()


def test_shared_video_still_rejects_split_leakage(tmp_path):
    inv, _ = fixture_records(tmp_path)
    inv.sequences[-1].split = 'test'
    with pytest.raises(ValueError, match='Shared asset split leakage'):
        inv.validate()


def test_same_source_sequence_still_requires_consistent_split(tmp_path):
    inv, _ = fixture_records(tmp_path)
    inv.sequences.append(replace(inv.sequences[0], sequence_id='alias:sequence', split='test'))
    with pytest.raises(ValueError, match='Source sequence split leakage'):
        inv.validate()


@pytest.mark.parametrize('uri', ['../x', '/x', '//server/x', 'root://unknown/x', 'root://raw/../x', '%2e%2e/x', 'x\\y', 'x?query=1'])
def test_unsafe_uris(tmp_path, uri):
    with pytest.raises(ValueError):
        resolve_uri(tmp_path, uri, {'raw': str(tmp_path)})


def test_symlink_escape(tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = tmp_path / 'release'
    root.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='escapes'):
        resolve_uri(root, 'escape/motion.parquet')


def test_portable_package_and_long_tracks(tmp_path):
    root = make_fixture(tmp_path / 'release')
    inv = Inventory.read(root)
    assert inv.external_roots == {}
    report = validate_release(root, deep_sample=100)
    assert report['modality_combinations'] == {'text': 1, 'audio': 1, 'video': 1, 'audio+text+video': 2}
    assert len(list((root / 'motion').glob('*.parquet'))) <= len(inv.motions)
    assert len({a.uri for a in inv.assets}) == 3
    qpos, times = read_motion(root, inv.motions[0])
    np.testing.assert_allclose(times, np.arange(90)/30, atol=1e-12)
    assert qpos.shape == (90, 36)
    copied = tmp_path / 'bundle'
    package_release(root, copied)
    moved = tmp_path / 'moved'
    copied.rename(moved)
    assert validate_release(moved, deep_sample=100)['valid']
    with pytest.raises(FileExistsError):
        package_release(root, moved)
    reference = tmp_path / 'reference'
    assert not package_release(root, reference, mode='reference')['portable']
    assert validate_release(reference, deep_sample=100)['valid']


def test_deep_check_catches_corrupt_feature_time(tmp_path):
    root = make_fixture(tmp_path / 'release')
    inv = Inventory.read(root)
    times = root / inv.features[0].timestamp_uri
    values = np.load(times)
    values[1] = values[0]
    np.save(times, values)
    assert validate_release(root)['valid']  # Metadata-only deliberately avoids payload reads.
    with pytest.raises(ValueError, match='strictly increase'):
        validate_release(root, deep_sample=100)


def test_export_is_atomic_on_failed_quality(tmp_path):
    inv, tracks = fixture_records(tmp_path / 'source')
    q = resolve_uri(tmp_path, inv.motions[0].quality_ref, inv.external_roots)
    q.write_text(json.dumps({'motion_id': inv.motions[0].motion_id, 'accepted': False}))
    with pytest.raises(ValueError, match='Quality'):
        export_release(inv, tracks, tmp_path / 'release')
    assert not (tmp_path / 'release').exists()
    assert not list(tmp_path.glob('.release-*'))


def test_aist_discovery_does_not_require_video(tmp_path):
    from data_pipeline.sources.aistpp import discover
    root = tmp_path / 'aist'
    (root / 'archives').mkdir(parents=True)
    with zipfile.ZipFile(root / 'archives/motions_complete.zip', 'w') as z:
        z.writestr('motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl', b'not decoded during discovery')
    out = tmp_path / 'inventory'
    report = discover(root, out)
    assert report['source_motions'] == 1 and report['videos'] == 0
    inv = Inventory.read(out)
    assert len(inv.sequences) == 1 and not inv.motions
    rows = json.loads((out / 'meta/source_motions.json').read_text())
    assert rows[0]['format'] == 'aist_smpl'
    assert rows[0]['fps'] == 60


def test_affine_75hz_tokens_and_offset_sign():
    times = np.arange(1500, dtype=np.float64)/75
    mapping = TimeMap(1., 13.4)
    chosen = times[mapping.select(times, 1., 2.)]
    assert chosen[0] == pytest.approx(14.4)
    assert chosen[-1] < 15.4
    shifted = times[TimeMap(1., 13.9).select(times, 1., 2.)]
    assert shifted.mean() - chosen.mean() == pytest.approx(.5, abs=1/75)
    old_offset, origin = .7, 1.2
    convert = TimeMap.from_aist_offset(old_offset, origin)
    query = np.array([0., 1., 2.])
    np.testing.assert_allclose(convert.to_asset(query) + old_offset, query + origin)
    np.testing.assert_allclose(convert.to_motion(convert.to_asset(query)), query)


def test_tubelets_preserve_support_and_detect_invalid_times():
    ranges = np.array([[0., .067], [.067, .134], [.2, .267]])
    np.testing.assert_array_equal(TimeMap().select(ranges, .1, .19, semantics='interval'), [False, True, False])
    assert not TimeMap().select(ranges, .15, .19, semantics='interval').any()
    with pytest.raises(ValueError, match='Invalid interval'):
        TimeMap().select(np.array([[0., 0.]]), 0, 1, semantics='interval')


def test_tracker_post_control_timing_excludes_initialization():
    from data_pipeline.stages.motion import resample_executed
    reference = np.zeros((31, 36), dtype=np.float32)
    reference[:, 3] = 1.
    executed = np.zeros((50, 36), dtype=np.float32)
    executed[:, 3] = 1.
    executed[:, 0] = np.arange(1, 51)/50  # x(t)=t, first saved execution is t=.02.
    qpos, mask = resample_executed(reference, executed)
    np.testing.assert_allclose(qpos[:, 0], np.arange(31)/30, atol=1e-7)
    assert not mask[0] and mask[1:].all()
    assert qpos[mask][0, 0] == pytest.approx(1/30)
    with pytest.raises(ValueError, match='Extrapolation'):
        resample_executed(reference, executed[:10])


def test_native_build_has_no_video_or_tracker_dependency(tmp_path):
    from dataclasses import asdict
    from data_pipeline.schemas.records import SourceMotion
    from data_pipeline.runner import build
    root = tmp_path / 'raw'
    root.mkdir()
    qpos = np.zeros((60, 36), dtype=np.float32)
    qpos[:, 3] = 1.
    np.savez(root / 'motion.npz', qpos_36=qpos, timestamps=np.arange(60)/30)
    inv = Inventory(sequences=[__import__('data_pipeline.schemas', fromlist=['SequenceRecord']).SequenceRecord(
        'test:one', 'test', 'one', 'test:group', 'unassigned', {'test': True})], external_roots={'test': str(root)})
    inventory_root = tmp_path / 'inventory'
    inv.write(inventory_root)
    source = SourceMotion('test:one:g1', 'test:one', 'root://test/motion.npz', 'g1_qpos_wxyz', 30.)
    (inventory_root / 'meta/source_motions.json').write_text(json.dumps([asdict(source)]))
    config = tmp_path / 'build.yaml'
    config.write_text('inventory: inventory\noutput: release\ntracker:\n  enabled: false\nsplit_assignments:\n  test:group: val\n  test:one: train\n')
    assert build(config)['motions'] == 1
    result = Inventory.read(tmp_path / 'release')
    assert result.sequences[0].split == 'train'
    assert result.motions[0].label_type == 'source_g1'
    assert result.motions[0].tracker_version == 'not_applicable'
    np.testing.assert_array_equal(read_motion(tmp_path / 'release', result.motions[0])[0], qpos)
    assert 'build_quality' not in result.external_roots


def test_aist_build_retains_reference_and_execution_identity(tmp_path, monkeypatch):
    from dataclasses import asdict
    from data_pipeline.schemas.records import SourceMotion
    from data_pipeline.schemas import SequenceRecord
    from data_pipeline.runner import build
    import data_pipeline.stages.retarget as retarget
    import data_pipeline.stages.motion as motion
    source_root = tmp_path / 'raw'
    source_root.mkdir()
    root = tmp_path / 'inventory'
    inv = Inventory(sequences=[SequenceRecord('aistpp:one', 'aistpp', 'one', 'aistpp:song', 'train', {'synthetic': True})],
                    external_roots={'aistpp': str(source_root)})
    inv.write(root)
    source = SourceMotion('aistpp:one:smpl', 'aistpp:one', 'root://aistpp/motion.pkl', 'aist_smpl', 60.)
    (root / 'meta/source_motions.json').write_text(json.dumps([asdict(source)]))
    qpos = np.zeros((60, 36), dtype=np.float32)
    qpos[:, 3] = 1.
    qpos[:, 0] = np.arange(60)/30
    reference = tmp_path / 'reference.npz'
    np.savez(reference, qpos_36=qpos, source_timestamps=np.arange(60)/30, fps=np.float32(30))
    monkeypatch.setattr(retarget, 'retarget', lambda *a: (reference, 'test:retarget', {'synthetic': True}))
    onnx = tmp_path / 'tracker.onnx'
    onnx.write_bytes(b'synthetic identity only')
    def execute(ref, config, directory):
        np.savez(directory / 'paired_motion.npz', qpos_36=qpos, timestamps=np.arange(60)/30,
                 executed_state_mask=np.arange(60)>0)
        return {'accepted': True}
    monkeypatch.setattr(motion, 'execute_and_screen', execute)
    config = tmp_path / 'build.yaml'
    config.write_text('inventory: inventory\noutput: release\nretarget: {}\ntracker:\n  enabled: true\n  version: synthetic-v1\n  onnx: tracker.onnx\n')
    report = build(config)
    assert report['motions'] == 2
    assert json.loads((reference.parent / 'reference_quality.json').read_text())['accepted']
    result = Inventory.read(tmp_path / 'release')
    ref, executed = result.motions
    assert ref.label_type == 'retarget_reference'
    assert executed.label_type == 'tracker_executed'
    assert executed.parent_motion_id == ref.motion_id
    assert executed.time_origin_sec == pytest.approx(1/30)
    assert executed.num_frames == ref.num_frames - 1
    np.testing.assert_array_equal(read_motion(tmp_path / 'release', executed)[0], qpos[1:])


def test_omg_decode_checks_source_timestamps_across_row_groups(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_pipeline.stages.motion import decode, read_global_rows
    from data_pipeline.schemas.records import SourceMotion
    directory = tmp_path / 'data/chunk-000'
    directory.mkdir(parents=True)
    qpos = np.zeros((8, 36), dtype=np.float32)
    qpos[:, 3] = 1.
    qpos[:, 0] = np.arange(8)
    times = np.arange(8, dtype=np.float32)/30
    def write():
        pq.write_table(pa.table({'observation.state': pa.array(qpos.tolist(), type=pa.list_(pa.float32(),36)),
                                 'timestamp': times}), directory / 'file-000.parquet', row_group_size=3)
    write()
    np.testing.assert_array_equal(read_global_rows(directory.parent, 2, 4)[0], qpos[2:6])
    source = SourceMotion('omg:g1', 'omg:one', 'root://omg/data', 'omg_lerobot_g1', 30.,
                          metadata={'row_start': 2, 'num_frames': 4})
    inv = Inventory(external_roots={'omg': str(tmp_path)})
    decoded, actual = decode(source, inv, tmp_path)
    np.testing.assert_array_equal(decoded, qpos[2:6])
    np.testing.assert_allclose(actual, times[2:6])
    times[4] += .01
    write()
    with pytest.raises(ValueError, match='Nonuniform'):
        decode(source, inv, tmp_path)


def test_retarget_cache_fingerprint_and_motion_only_dispatch(tmp_path, monkeypatch):
    import data_pipeline.stages.retarget as retarget
    from data_pipeline.schemas import SequenceRecord
    from data_pipeline.schemas.records import SourceMotion
    raw = tmp_path / 'raw'
    (raw / 'archives').mkdir(parents=True)
    archive = raw / 'archives/motions_complete.zip'
    archive.write_bytes(b'synthetic source revision one')
    model = tmp_path / 'smpl.pkl'
    model.write_bytes(b'synthetic model identity')
    source = SourceMotion('aistpp:one:smpl', 'aistpp:one', 'root://aistpp/archives/motions_complete.zip',
                          'aist_smpl', 60., member='motions/one.pkl')
    inv = Inventory(sequences=[SequenceRecord('aistpp:one', 'aistpp', 'one', 'aistpp:group', source_provenance={'synthetic':True})],
                    external_roots={'aistpp': str(raw)})
    def git(args, **kwargs):
        if 'rev-parse' in args:
            return 'f' * 40
        if 'diff' in args:
            return b'synthetic local solver diff'
        return b''
    calls = []
    from data_pipeline.sources import aistpp
    from data_pipeline.stages.retarget import HumanMotion
    motion = HumanMotion(('pelvis',), np.zeros((2, 1, 3)), np.tile([1., 0, 0, 0], (2, 1, 1)),
                         np.arange(2)/30, 1.8, 'smplx')
    monkeypatch.setattr(aistpp, 'decode_motion', lambda *args: motion)
    def solve(human, settings):
        calls.append(human)
        qpos = np.zeros((2, 36), dtype=np.float32)
        qpos[:, 3] = 1.
        return qpos, {'synthetic': True}
    monkeypatch.setattr(retarget.subprocess, 'check_output', git)
    monkeypatch.setattr(retarget, 'run_gmr', solve)
    settings = {'smpl_model': str(model), 'gmr_root': str(tmp_path), 'gender': 'neutral'}
    a = retarget.retarget(source, inv, tmp_path, settings, tmp_path / 'work')
    b = retarget.retarget(source, inv, tmp_path, settings, tmp_path / 'work')
    assert a == b and len(calls) == 1
    assert calls[0] is motion
    assert a[2]['gmr_worktree_diff_sha256']
    archive.write_bytes(b'synthetic source revision two')
    c = retarget.retarget(source, inv, tmp_path, settings, tmp_path / 'work')
    assert c[1] != a[1] and len(calls) == 2


def test_aist_splits_are_independent_of_shared_song_and_choreography(tmp_path):
    from data_pipeline.sources.aistpp import discover
    names = ['gBR_sBM_cAll_d01_mBR0_ch01', 'gBR_sBM_cAll_d02_mBR1_ch01',
             'gBR_sFM_cAll_d03_mBR1_ch02', 'gPO_sBM_cAll_d01_mPO0_ch01']
    root = tmp_path / 'raw'
    (root / 'motions').mkdir(parents=True)
    for name in names:
        (root / 'motions' / f'{name}.pkl').write_bytes(b'not decoded during discovery')
    discover(root, tmp_path / 'full')
    discover(root, tmp_path / 'limited', limit=1)
    inv = Inventory.read(tmp_path / 'full')
    assert len({s.split_group_id for s in inv.sequences}) == len(names)
    assert Inventory.read(tmp_path / 'limited').sequences[0].split_group_id == inv.sequences[0].split_group_id
    for sequence, split in zip(inv.sequences, ['train', 'val', 'test', 'train']):
        sequence.split = split
    inv.validate()


def test_edge_aist_directory_discovers_motion_and_audio_excerpt(tmp_path):
    import wave
    from data_pipeline.sources.aistpp import discover
    root = tmp_path / 'AISTpp/edge_aistpp'
    (root / 'motions').mkdir(parents=True)
    (root / 'wavs').mkdir()
    name = 'gBR_sBM_cAll_d04_mBR0_ch01'
    (root / 'motions' / f'{name}.pkl').write_bytes(b'not decoded during discovery')
    with wave.open(str(root / 'wavs' / f'{name}.wav'), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(b'\0' * 48000 * 2 * 2)
    out = tmp_path / 'inventory'
    report = discover(root.parent, out, limit=1)
    assert report['source_motions'] == report['audios'] == 1
    assert report['videos'] == 0
    inv = Inventory.read(out)
    assert inv.assets[0].modality == 'audio'
    assert inv.assets[0].sample_rate_hz == 48000
    assert inv.assets[0].duration_sec == 1.
    assert not inv.links  # A matching filename is not verified temporal alignment.
    raw = json.loads((out / 'meta/source_motions.json').read_text())[0]
    assert raw['uri'] == f'root://aistpp/motions/{name}.pkl'
    assert raw['member'] == ''
