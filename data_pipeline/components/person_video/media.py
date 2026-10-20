"""视频读写：按已选帧号解码、应用裁剪变换，再流式送入ffmpeg编码。"""
from contextlib import contextmanager, ExitStack
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


def render(video, source_indices, crops, boxes, affine, output, config):
    """写512训练视频；可选再写800×320的原图/裁剪诊断对比视频。"""
    import cv2
    cv2.setNumThreads(2)
    with ExitStack() as stack:
        stream = stack.enter_context(encoder(output / '.person.partial.mp4', 512, 512, config.fps, config.crf))
        preview = stack.enter_context(encoder(output / '.preview.partial.mp4', 800, 320, config.fps, 26)) if config.preview else None
        for index, frame in selected_frames(video, source_indices):
            # 一次仿射同时完成平移和等比例缩放；超出原片的部分用黑色补边。
            crop = cv2.warpAffine(frame, affine[index], (512, 512), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            stream.write(crop.tobytes())
            if preview is None:
                continue
            marked = frame.copy()
            cv2.rectangle(marked, tuple(boxes[index, :2].astype(int)), tuple(boxes[index, 2:].astype(int)), (0, 0, 255), 4)
            cv2.rectangle(marked, tuple(crops[index, :2]), tuple(crops[index, 2:]), (0, 255, 0), 4)
            canvas = np.zeros((320, 800, 3), dtype=np.uint8)
            # 预览中的原片保持长宽比，横屏/竖屏都不拉伸；与训练图尺寸无关。
            ratio = min(512 / frame.shape[1], 288 / frame.shape[0])
            w, h = max(1, round(frame.shape[1] * ratio)), max(1, round(frame.shape[0] * ratio))
            x, y = (512 - w) // 2, 32 + (288 - h) // 2
            canvas[y:y+h, x:x+w] = cv2.resize(marked, (w, h))
            canvas[32:, 512:] = cv2.resize(crop, (288, 288))
            cv2.putText(canvas, 'RGB only | red: person | green: crop', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
            cv2.putText(canvas, '512 x 512', (550, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
            preview.write(canvas.tobytes())


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
