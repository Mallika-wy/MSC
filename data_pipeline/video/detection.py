"""人物检测与身份跟踪；只有缺少可复用结果时才加载视觉模型运行环境。"""
from collections import Counter
from functools import lru_cache
import os
from pathlib import Path
import time

from data_pipeline.video.media import selected_frames


@lru_cache(maxsize=1)
def load_model(weights, weights_mtime_ns, device):
    """每个worker复用一个YOLO模型；权重更新或设备变化时重新加载。"""
    from ultralytics import YOLO
    return YOLO(weights)


def reset_tracks(model):
    """只复用网络权重，不复用上一条视频的ByteTrack身份、帧号或运动状态。"""
    for tracker in getattr(getattr(model, 'predictor', None), 'trackers', []):
        tracker.reset()


def detect_tracks(video, source_indices, config, settings_root: Path):
    """逐帧运行YOLO11n+ByteTrack，返回每帧的ID、原图bbox和置信度。

    source_indices是已经按动作时间选好的源视频帧号，不再独立决定取帧间隔。
    """
    settings_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(settings_root.resolve()))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import torch
    torch.set_num_threads(4)
    model = load_model(config.weights, Path(config.weights).stat().st_mtime_ns, config.device)
    reset_tracks(model)
    records, counts, timings = [], Counter(), []
    started = time.perf_counter()
    # 每条视频重置跟踪状态；网络留在当前worker内，减少重复加载和GPU预热。
    for index, frame in selected_frames(video, source_indices):
        # persist=True在当前视频的帧间保留身份；COCO类别0是person。
        # OpenCV输入是BGR，Ultralytics接口会处理通道；xyxy结果已映回原图尺寸。
        result = model.track(frame, tracker='bytetrack.yaml', persist=True, classes=[0],
                             conf=config.confidence, imgsz=config.imgsz, device=config.device, verbose=False)[0]
        boxes, detections = result.boxes, []
        if boxes.id is not None:
            for track_id, box, score in zip(boxes.id.int().cpu().tolist(), boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist()):
                counts[track_id] += 1
                detections.append({'track_id': track_id, 'xyxy': box, 'confidence': score})
        # 即使这一帧没有检测，也保存空列表，保证结果与时间轴长度一致。
        records.append(detections)
        timings.append(result.speed['inference'])
        if index % 200 == 0:
            print(f'person-video: detected {index}/{len(source_indices)}', flush=True)
    return {'frames': records, 'track_counts': dict(counts), 'inference_ms': timings,
            'detection_wall_s': time.perf_counter() - started}
