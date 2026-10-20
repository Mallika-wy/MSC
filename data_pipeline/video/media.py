"""视频读写：按已选帧号解码、应用裁剪变换，再流式送入ffmpeg编码。"""
from contextlib import contextmanager
from pathlib import Path
import json
import subprocess

import numpy as np


@contextmanager
def encoder(path, width, height, fps, crf=22):
    """提供原始BGR字节输入管道；退出上下文时关闭管道并等待MP4写入完成。"""
    # -y只用于本模块创建的partial文件；正式结果由processor验证后再替换。
    process = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-threads', '2', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-threads', '4', '-crf', str(crf),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
    try:
        yield process.stdin
    finally:
        process.stdin.close()
        result = process.wait()
        if result:
            raise RuntimeError(f'Video encoder exited with {result}')


def selected_frames(path, source_indices):
    """顺序解码原片，逐个yield(输出序号, BGR帧)，避免把整条视频存进内存。"""
    import cv2
    if len(source_indices) < 1 or (np.diff(source_indices) < 0).any() or min(source_indices) < 0:
        raise ValueError('Source frame indices must be nonnegative and nondecreasing')
    # 解码线程必须在open时设置；默认按机器核心数开线程，多个worker会严重超额订阅。
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    wanted, source_index = 0, 0
    try:
        while wanted < len(source_indices):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Video decode ended before frame {source_indices[wanted]}')
            # 多个输出时刻可能选中同一源帧，因此这里允许重复yield同一张图。
            while wanted < len(source_indices) and source_indices[wanted] == source_index:
                yield wanted, frame
                wanted += 1
            source_index += 1
    finally:
        capture.release()


def render(video, source_indices, affine, output, config):
    """Encode the 512×512 training crop."""
    import cv2
    cv2.setNumThreads(2)
    with encoder(output / '.person.partial.mp4', 512, 512, config.fps, config.crf) as stream:
        for index, frame in selected_frames(video, source_indices):
            crop = cv2.warpAffine(frame, affine[index], (512, 512), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            stream.write(crop.tobytes())


def verify_encoded(path: Path, count: int, fps: float, decode=False):
    """检查尺寸、帧数、帧率；decode=True时全片解码，发现损坏即报错。"""
    args = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if decode:
        args += ['-count_frames']
    result = json.loads(subprocess.check_output([
        *args, '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,nb_read_frames', '-of', 'json', str(path)]))['streams'][0]
    numerator, denominator = map(int, result['avg_frame_rate'].split('/'))
    frame_count = int(result['nb_read_frames'] if decode else result['nb_frames'])
    if (result['width'], result['height']) != (512, 512) or frame_count != count or not np.isclose(numerator / denominator, fps):
        raise ValueError('Encoded video violates 512x512 / FPS / frame count contract')
    if decode:
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
                        '-threads', '2', '-f', 'null', '-'], check=True)
    return result


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
