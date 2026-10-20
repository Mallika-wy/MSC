from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.tests.fixtures import fixture_records, make_fixture
from data_pipeline.schema import Inventory, TimeMap
from data_pipeline.io import resolve_uri
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
    from data_pipeline.motion.tracking import resample_executed
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
