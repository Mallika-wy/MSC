"""人物视频组件：RGB → 固定身份跟踪 → 平滑裁剪 → 512×512视频。

只依赖视频，不依赖人体动作标注。检测结果可复用，输出使用普通可读文件名。
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np

from data_pipeline.video.detection import detect_tracks
from data_pipeline.video.geometry import crop_affines, interpolate_boxes, plan_crops
from data_pipeline.video.media import render, verify_encoded
from data_pipeline.io import read_json, write_json
from data_pipeline.video.media import PTS_TOLERANCE_S, nearest_frames, probe_video


def select_track(records, times, track_id, max_gap_s):
    """选一个固定身份，返回ID、补齐后的框[T,4]、有效掩码[T]和置信度。

    未指定ID时选择出现帧数最多的轨迹，这不保证它一定是语义上的主角。
    不会每帧选最大框换人；只对同一ID的短暂漏检进行插值。
    """
    counts = {}
    if len(records) != len(times):
        raise ValueError('Tracking/timestamp length mismatch')
    for detections in records:
        for detection in detections:
            identity = int(detection['track_id'])
            counts[identity] = counts.get(identity, 0) + 1
    if not counts:
        raise ValueError('No person tracks detected')
    if track_id is None:
        track_id = max(counts, key=lambda identity: (counts[identity], -identity))
    # 框坐标均在原图上，顺序为[x1,y1,x2,y2]；漏检用NaN，不能当作零坐标。
    valid, boxes, scores = np.zeros(len(times), bool), np.full((len(times), 4), np.nan), np.zeros(len(times))
    for index, detections in enumerate(records):
        matches = [d for d in detections if int(d['track_id']) == track_id]
        if len(matches) > 1:
            raise ValueError('Duplicate track identity in one frame')
        if matches:
            valid[index], boxes[index], scores[index] = True, matches[0]['xyxy'], matches[0]['confidence']
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Invalid detection confidence')
    return track_id, interpolate_boxes(boxes, valid, times, max_gap_s), valid, scores, counts


def preprocess_person_video(video: Path, output: Path, config, timestamps=None, force=False):
    """预处理一个连续镜头，返回视频路径、坐标映射路径、帧数和复用状态。

    output是该视频的确切目录；timestamps是源视频PTS，而非输出视频时间。
    传入timestamps时严格跟随动作标签；不传则从原片覆盖范围构建均匀时间轴。
    force允许重写派生结果，调整裁剪参数仍可复用已有检测；不改写原视频。
    """
    video, output = Path(video).resolve(), Path(output).resolve()
    if output.is_relative_to(video.parent):
        raise ValueError('Outputs must be outside the raw video directory')
    if not video.is_file() or not Path(config.weights).is_file():
        raise FileNotFoundError('Video and detector weights are required')
    # 1. 先建立实际取帧表，AIST的59.94FPS视频不能直接当作精确60FPS。
    probe = probe_video(video)
    pts = probe.pop('timestamps')
    times = (pts[0] + np.arange(int(np.floor((pts[-1]-pts[0]+PTS_TOLERANCE_S)*config.fps))+1)/config.fps
             if timestamps is None else np.asarray(timestamps, dtype=np.float64))
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or not np.allclose(np.diff(times), 1/config.fps, atol=1e-7, rtol=0):
        raise ValueError('Need uniform source PTS at configured FPS')
    indices = nearest_frames(pts, times)
    output.mkdir(parents=True, exist_ok=True)
    # 2. 路径、修改时间和参数相同就复用。这里不扫描全文件哈希或环境版本。
    inputs = {'video': str(video), 'video_mtime_ns': video.stat().st_mtime_ns,
              'weights_mtime_ns': Path(config.weights).stat().st_mtime_ns}
    settings = {**inputs, **asdict(config)}
    same = (output / 'settings.json').is_file() and read_json(output / 'settings.json') == settings
    complete = (output / 'person.mp4').is_file() and (output / 'metadata.npz').is_file()
    if complete and not force:
        with np.load(output / 'metadata.npz', allow_pickle=False) as data:
            same = same and np.array_equal(data['source_requested_pts'], times)
        if not same:
            raise ValueError('Video/crop settings changed; use a new output folder or --force')
        validate_person_video(output)
        return result_paths(output, len(times), config.fps, True, True)
    # 检测参数与裁剪参数分开：改变margin/平滑/CRF不需要再次运行YOLO。
    detection_settings = {**inputs, 'weights': config.weights, 'imgsz': config.imgsz,
                          'confidence': config.confidence, 'indices': indices.tolist()}
    cache = output / 'detections.json'
    tracked = read_json(cache) if cache.is_file() else None
    detection_hit = tracked is not None and tracked['settings'] == detection_settings
    if not detection_hit:
        tracked = detect_tracks(video, indices, config, output.parent / '.yolo_settings')
        tracked['settings'] = detection_settings
        write_json(cache, tracked)
    # 3. 固定主体 → 补齐短漏检 → 规划窗口 → 建立原图与512图的正逆变换。
    identity, boxes, valid, scores, counts = select_track(tracked['frames'], times, config.track_id, config.max_gap_s)
    crops = plan_crops(boxes, probe['width'], probe['height'], config.fps, config.margin, config.smoothing_s)
    affine, inverse = crop_affines(crops)
    # 4. 先写partial文件并完整解码验证；成功后才替换正式person.mp4。
    render(video, indices, affine, output, config)
    verify_encoded(output / '.person.partial.mp4', len(times), config.fps, decode=True)
    # 这些是训练所需的时间/空间映射，不是历史下载审计：
    # output_timestamps从0计时；source_requested_pts保留动作对应的源视频时间。
    # source_pts是实际取到的帧时间，两者通常只有最近邻采样带来的小误差。
    np.savez_compressed(output / 'metadata.npz', output_timestamps=np.arange(len(times))/config.fps,
        source_requested_pts=times, source_frame_indices=indices, source_pts=pts[indices], crop_xyxy=crops,
        original_to_crop_affine=affine, crop_to_original_affine=inverse, detection_valid=valid,
        confidence=scores, track_id=np.int32(identity), source_size=[probe['width'], probe['height']],
        output_size=[512, 512], fps=np.float32(config.fps))
    write_json(output / 'settings.json', settings)
    write_json(output / 'quality.json', {'frames': len(times), 'track_id': identity, 'track_counts': counts,
        'detection_coverage': float(valid.mean()), 'mean_confidence': float(scores[valid].mean()),
        'max_sampling_error_s': float(abs(pts[indices]-times).max()), 'source_annotations_used': False})
    (output / '.person.partial.mp4').replace(output / 'person.mp4')
    return result_paths(output, len(times), config.fps, detection_hit, False)


def result_paths(output, count, fps, detection_hit, video_hit):
    """给CLI和数据集pipeline返回相同的轻量接口，不返回视频帧数组。"""
    return {'directory': str(output), 'video': str(output / 'person.mp4'), 'metadata': str(output / 'metadata.npz'),
            'frames': count, 'fps': fps, 'output_size': [512, 512],
            'tracking_cache_hit': detection_hit, 'output_cache_hit': video_hit}


def validate_person_video(directory: Path, decode=False):
    """检查512尺寸、时间轴、仿射映射；decode=True时再做全视频解码检查。"""
    directory = Path(directory)
    with np.load(directory / 'metadata.npz', allow_pickle=False) as data:
        times, fps = data['output_timestamps'], float(data['fps'])
        if len(times) < 2 or not np.array_equal(data['output_size'], [512, 512]) or not np.isfinite(times).all():
            raise ValueError('Invalid 512x512 metadata')
        if not np.allclose(times, np.arange(len(times))/fps, atol=1e-7):
            raise ValueError('Invalid output timestamps')
        forward, inverse = crop_affines(data['crop_xyxy'])
        if not np.allclose(forward, data['original_to_crop_affine']) or not np.allclose(inverse, data['crop_to_original_affine']):
            raise ValueError('Invalid crop coordinate mapping')
        verify_encoded(directory / 'person.mp4', len(times), fps, decode)
    return {'valid': True, 'frames': len(times), 'fps': fps, 'output_size': [512, 512]}
