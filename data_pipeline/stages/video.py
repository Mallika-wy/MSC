"""视频分支：按实际显示时间戳PTS配对，不假定视频帧率是精确整数。"""
import json
import subprocess
from pathlib import Path

import numpy as np

PTS_TOLERANCE_S = 1e-6  # 仅容忍ffprobe时间文本的微秒舍入，不容忍真正的缺帧区间。


def probe_video(path: Path) -> dict:
    """取视频尺寸和每个解码帧的PTS，返回 timestamps[N]，单位秒。"""
    result = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'stream=width,height,avg_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)]))
    pts = np.array([float(frame['best_effort_timestamp_time']) for frame in result['frames']])
    if not len(pts) or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError(f'Invalid video PTS: {path}')
    return {**result['streams'][0], 'timestamps': pts}


def nearest_frames(pts, requested):
    """为每个请求时间找最近的源帧，返回零基帧号；超出覆盖范围直接报错。"""
    if not len(pts) or not len(requested) or not np.isfinite(requested).all():
        raise ValueError('Empty or invalid video timeline')
    if requested.min() < pts[0]-PTS_TOLERANCE_S or requested.max() > pts[-1]+PTS_TOLERANCE_S:
        raise ValueError('Motion timestamps exceed video PTS coverage')
    right = np.searchsorted(pts, requested).clip(0, len(pts)-1)
    left = (right-1).clip(0)
    # 比较请求时间左右两帧的距离；距离相同时取较早帧。
    return np.where(abs(pts[left]-requested) <= abs(pts[right]-requested), left, right)


def prepare_video_view(pair, sequence_times, config, output, force=False):
    """把同一份动作时间轴映射到一个视频机位，返回用于导出的逐帧索引。"""
    # CSV定义：公共时间 = 视频PTS + offset，因此取帧时需要减去offset。
    requested = sequence_times-pair.video_time_offset_s
    settings = dict(config.get('video_preprocess', {}))
    base = {'camera_id': pair.camera_id, 'source_uri': pair.video_path,
            'source_calibration_uri': pair.camera_path or None}
    if settings.pop('enabled', False):
        from data_pipeline.components.person_video import PersonVideoConfig, preprocess_person_video
        component = PersonVideoConfig(**settings)
        if component.fps != 30:
            raise ValueError('OMG paired video output must be 30 FPS')
        result = preprocess_person_video(Path(pair.video_path), output, component, requested, force=force)
        # 裁剪视频从0秒开始计时；公共时间起点另存，避免丢失原来的对齐关系。
        # 原图标定与裁剪变换分别保存，不能直接把原K矩阵当作裁剪图K矩阵。
        return {**base, 'uri': result['video'], 'metadata': {'width': 512, 'height': 512, 'fps': 30},
                'frame_indices': list(range(len(requested))), 'frame_pts_s': (np.arange(len(requested))/30).tolist(),
                'sequence_time_offset_s': float(sequence_times[0]), 'crop_metadata_uri': result['metadata']}
    # 关闭裁剪时直接索引原片。这个分支不保证视频已经是512×512。
    probe = probe_video(Path(pair.video_path))
    pts = probe.pop('timestamps')
    indices = nearest_frames(pts, requested)
    return {**base, 'uri': pair.video_path, 'metadata': probe,
            'frame_indices': indices.tolist(), 'frame_pts_s': pts[indices].tolist(),
            'sequence_time_offset_s': pair.video_time_offset_s}
