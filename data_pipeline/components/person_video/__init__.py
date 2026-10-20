"""仅依赖 RGB 视频的人物检测、跟踪和裁剪组件，统一输出 512×512 视频。"""
from data_pipeline.components.person_video.config import PersonVideoConfig
from data_pipeline.components.person_video.processor import preprocess_person_video, validate_person_video

__all__ = ['PersonVideoConfig', 'preprocess_person_video', 'validate_person_video']
