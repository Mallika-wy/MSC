"""人物视频预处理参数；配置对象只管参数，不负责加载模型。"""
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class PersonVideoConfig:
    """输出分辨率固定512×512；检测输入尺寸imgsz与输出尺寸不是一回事。"""
    weights: str
    device: str = 'cpu'
    output_size: int = 512
    fps: float = 30.0  # 输出视频采样频率；配对OMG时必须为30。
    imgsz: int = 640  # YOLO内部推理尺寸，不改变输出512×512的约定。
    confidence: float = .1
    track_id: int | None = None  # None选最长轨迹；多人场景可显式指定主体。
    margin: float = 1.3          # 裁剪边长放大倍数，为手脚留余量。
    smoothing_s: float = .15     # 中心平滑Gaussian标准差，单位秒。
    max_gap_s: float = .5        # 最长允许插值的缺失时长，单位秒。
    crf: int = 22                # H.264质量参数，越小通常质量越高、文件越大。
    preview: bool = False        # 小尺寸诊断对比图，不作为训练输入。

    def __post_init__(self):
        # 统一数值类型，避免YAML中的30与30.0造成不必要的参数差异。
        for name in ('fps', 'confidence', 'margin', 'smoothing_s', 'max_gap_s'):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, 'device', str(self.device))
        object.__setattr__(self, 'weights', str(self.weights))
        if not self.weights or self.output_size != 512:
            raise ValueError('Person video training output must be 512x512; weights required')
        if not isfinite(self.fps) or self.fps <= 0 or self.imgsz < 32:
            raise ValueError('Invalid sampling FPS or inference image size')
        if not isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError('confidence must be in (0,1)')
        if not isfinite(self.margin) or self.margin < 1.1:
            raise ValueError('margin must be finite and >=1.1')
        if not all(isfinite(value) and value >= 0 for value in (self.smoothing_s, self.max_gap_s)):
            raise ValueError('Invalid smoothing or gap duration')
        if not 0 <= self.crf <= 51:
            raise ValueError('Invalid H.264 CRF')

    @classmethod
    def from_mapping(cls, mapping: dict, project: Path):
        """从YAML字典构建配置，权重相对路径以项目根目录为基准。"""
        mapping = dict(mapping)
        path = Path(mapping['weights']).expanduser()
        mapping['weights'] = str((project / path).resolve())
        return cls(**mapping)
