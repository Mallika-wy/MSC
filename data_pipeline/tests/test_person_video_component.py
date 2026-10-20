from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.components.person_video import PersonVideoConfig, preprocess_person_video, validate_person_video
from data_pipeline.components.person_video.geometry import crop_affines
from data_pipeline.components.person_video.processor import select_track
from data_pipeline.stages.video import prepare_video_view
from data_pipeline.core.pairs import Pair


def test_training_size_is_a_hard_contract():
    with pytest.raises(ValueError, match='512'):
        PersonVideoConfig(weights='model.pt', output_size=256)
    for kwargs in ({'fps': float('nan')}, {'margin': float('inf')}, {'max_gap_s': -1}):
        with pytest.raises(ValueError):
            PersonVideoConfig(weights='model.pt', **kwargs)


def test_yaml_integer_and_float_rates_share_identity():
    from dataclasses import asdict
    a = PersonVideoConfig(weights='model.pt', fps=30)
    b = PersonVideoConfig(weights='model.pt', fps=30.0)
    assert asdict(a) == asdict(b)


def test_forward_inverse_affine_and_padding_coordinates():
    crops = np.array([[-20, -10, 180, 190], [10, 20, 410, 420]])
    forward, inverse = crop_affines(crops)
    points = np.array([[30., 50.], [200., 300.]])
    transformed = np.einsum('nij,nj->ni', forward[:, :, :2], points) + forward[:, :, 2]
    restored = np.einsum('nij,nj->ni', inverse[:, :, :2], transformed) + inverse[:, :, 2]
    np.testing.assert_allclose(restored, points)


def test_tracks_are_not_merged_by_framewise_largest_box():
    times = np.arange(3) / 30
    a = {'track_id': 1, 'xyxy': [10, 10, 30, 80], 'confidence': .9}
    b = {'track_id': 2, 'xyxy': [0, 0, 200, 200], 'confidence': .9}
    identity, boxes, valid, *_ = select_track([[a], [a, b], [a]], times, None, .5)
    assert identity == 1 and valid.all()
    np.testing.assert_array_equal(boxes[1], a['xyxy'])
    with pytest.raises(ValueError, match='endpoints'):
        select_track([[a], [a, b], [a]], times, 2, .5)


@pytest.fixture
def synthetic_video(tmp_path, monkeypatch):
    import cv2
    import data_pipeline.components.person_video.processor as processor
    raw = tmp_path / 'raw'
    raw.mkdir()
    video, weights = raw / 'input.mp4', tmp_path / 'test_weights.pt'
    weights.write_bytes(b'test-only fake detector weights')
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (160, 120))
    assert writer.isOpened()
    for index in range(8):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[20:100, 20 + index:70 + index] = (0, 180, 240)
        writer.write(frame)
    writer.release()
    calls = []

    def fake_detect(path, source_indices, config, settings_root):
        calls.append(str(path))
        return {'frames': [[{'track_id': 1, 'xyxy': [20 + int(i), 20, 70 + int(i), 100], 'confidence': .9}]
                           for i in source_indices], 'track_counts': {1: len(source_indices)}}

    monkeypatch.setattr(processor, 'detect_tracks', fake_detect)
    config = PersonVideoConfig(weights=str(weights))
    return video, config, tmp_path / 'processed', calls


def test_real_encoding_cache_and_crop_only_invalidation(synthetic_video):
    video, config, root, calls = synthetic_video
    original = video.read_bytes()
    first = preprocess_person_video(video, root, config)
    assert not first['tracking_cache_hit'] and not first['output_cache_hit']
    assert first['output_size'] == [512, 512]
    assert validate_person_video(Path(first['directory']), decode=True)['frames'] == 8
    second = preprocess_person_video(video, root, config)
    assert second['tracking_cache_hit'] and second['output_cache_hit']
    with pytest.raises(ValueError, match='--force'):
        preprocess_person_video(video, root, replace(config, margin=1.6))
    third = preprocess_person_video(video, root, replace(config, margin=1.6), force=True)
    assert third['tracking_cache_hit'] and not third['output_cache_hit']
    assert len(calls) == 1
    assert third['directory'] == first['directory']
    assert video.read_bytes() == original
    with np.load(first['metadata'], allow_pickle=False) as data:
        payload = dict(data)
    payload['original_to_crop_affine'] += 1
    np.savez_compressed(first['metadata'], **payload)
    with pytest.raises(ValueError, match='coordinate mapping'):
        validate_person_video(Path(first['directory']))


def test_pairing_uses_exact_label_grid_and_retains_coordinate_mapping(synthetic_video):
    video, component, root, _ = synthetic_video
    source = Pair('fixture', 'one', 'front', str(video), component.weights, 'g1_qpos_wxyz',
                  camera_path='/original/camera.json', video_time_offset_s=2.0)
    times = 2 + np.arange(8) / 30
    settings = {'video_preprocess': {'enabled': True, 'weights': component.weights, 'fps': 30}}
    view = prepare_video_view(source, times, settings, root)
    assert view['metadata']['width'] == view['metadata']['height'] == 512
    assert view['frame_indices'] == list(range(8))
    np.testing.assert_allclose(np.array(view['frame_pts_s']) + view['sequence_time_offset_s'], times)
    assert view['source_calibration_uri'] == source.camera_path
    assert Path(view['crop_metadata_uri']).is_file()


def test_decoder_thread_limit_preserves_pixels_and_duplicate_frames(synthetic_video):
    import cv2
    from data_pipeline.components.person_video.media import selected_frames
    video, *_ = synthetic_video
    capture = cv2.VideoCapture(str(video))
    original = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        original.append(frame)
    capture.release()
    indices = np.array([0, 0, 2, 7])
    decoded = list(selected_frames(video, indices))
    for (output_index, frame), source_index in zip(decoded, indices):
        np.testing.assert_array_equal(frame, original[source_index])
    assert [index for index, _ in decoded] == list(range(4))


def test_yolo_weight_reuse_and_per_video_tracker_reset(monkeypatch):
    import sys
    from types import SimpleNamespace
    from data_pipeline.components.person_video.detection import load_model, reset_tracks
    loads, resets = [], []

    def fake_yolo(weights):
        loads.append(weights)
        tracker = SimpleNamespace(reset=lambda: resets.append('reset'))
        return SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=fake_yolo))
    load_model.cache_clear()
    try:
        first = load_model('weights.pt', 123, '0')
        reset_tracks(first)
        second = load_model('weights.pt', 123, '0')
        reset_tracks(second)
        assert first is second
        assert loads == ['weights.pt']
        assert resets == ['reset', 'reset']
        assert load_model('weights.pt', 124, '0') is not first
        reset_tracks(SimpleNamespace(predictor=None))  # 尚未推理时无需重置。
    finally:
        load_model.cache_clear()
