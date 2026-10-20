"""数据源与流水线之间的统一接口：CSV 每行对应一个视频机位和一条动作。"""
import csv
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
import re


@dataclass
class Pair:
    """一行配对数据。这里只描述文件和同步关系，不读取动作数组。

    同一 sequence_id 可以有多个 camera_id；它们在后续共享一次动作处理。
    新格式通过 motion_format 选择解码器，不根据 .pkl/.npy 后缀猜测含义。
    """
    source: str
    sequence_id: str
    camera_id: str
    video_path: str
    motion_path: str
    motion_format: str
    motion_member: str = ''  # 普通文件留空；ZIP 数据填写包内路径，例如 motions/xxx.pkl。
    motion_fps: float = 30.0
    video_time_offset_s: float = 0.0  # 单位秒；公共序列时间 = 视频 PTS + 此偏移。
    motion_time_offset_s: float = 0.0
    split_group: str = ''  # 同组数据必须进入同一个 train/val/test 集合。
    source_root: str = ''   # 可选源资源目录；AIST 中用于定位旧 SMPL 读取依赖。
    camera_path: str = ''   # 可选原图相机标定，不能直接当作裁剪后的标定使用。

    def __post_init__(self):
        """尽早检查 CSV 字段，避免运行很久以后才发现路径名或帧率不合法。"""
        # 这些标识符也用于创建输出子目录，所以拒绝 ../ 等路径穿越形式。
        for name in ('source', 'sequence_id', 'camera_id'):
            value = getattr(self, name)
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
                raise ValueError(f'Unsafe or empty {name}: {value!r}')
        for name in ('video_path', 'motion_path', 'motion_format'):
            if not getattr(self, name):
                raise ValueError(f'{name} is required')
        for name in ('motion_fps', 'video_time_offset_s', 'motion_time_offset_s'):
            setattr(self, name, float(getattr(self, name)))
            if not isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if self.motion_fps <= 0:
            raise ValueError('motion_fps must be positive')
        # 缺省按“数据源/原始序列”分组，确保多机位和重叠切片不会跨集合。
        self.split_group = self.split_group or f'{self.source}/{self.sequence_id}'


COLUMNS = [field.name for field in fields(Pair)]


def write_pairs(path: Path, pairs: list[Pair]):
    """写统一表头；每个数据源 adapter 都调用它，不各自定义另一套 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(pair) for pair in pairs)


def read_pairs(path: Path) -> list[Pair]:
    """读取并检查 CSV。相对资产路径以 CSV 所在目录为基准，而非启动目录。"""
    pairs, seen = [], set()
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        for line, record in enumerate(reader, 2):
            try:
                # 空的可选字段交给 Pair 的默认值处理，例如空 motion_member。
                record = {key: value for key, value in record.items() if value != ''}
                for key in ('video_path', 'motion_path', 'source_root', 'camera_path'):
                    if record.get(key):
                        record[key] = str((path.parent / Path(record[key]).expanduser()).resolve())
                pair = Pair(**record)
                for name in ('video_path', 'motion_path'):
                    if not Path(getattr(pair, name)).is_file():
                        raise FileNotFoundError(getattr(pair, name))
                # 允许一条动作配多个机位，但不允许同一个机位重复出现。
                identity = (pair.source, pair.sequence_id, pair.camera_id)
                if identity in seen:
                    raise ValueError(f'Duplicate pair: {identity}')
                seen.add(identity)
                pairs.append(pair)
            except (TypeError, ValueError, FileNotFoundError) as error:
                raise ValueError(f'{path}:{line}: {error}') from error
    return pairs
