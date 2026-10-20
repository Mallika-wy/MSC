"""AIST++ 专用目录适配器：把原始目录扫描成通用配对 CSV，不转换动作。"""
from pathlib import Path
import zipfile

from data_pipeline.core.pairs import Pair, write_pairs


def build_csv(root: Path, output: Path, cameras=('c01',)) -> list[Pair]:
    """扫描现有视频与完整动作包；每个“序列×指定机位”生成一行。"""
    root = root.resolve()
    archive = root / 'archives/motions_complete.zip'
    # 官方忽略项不进入候选表，但原视频和原动作文件仍原样保留。
    ignored = set((root / 'archives/ignore_list.txt').read_text().split())
    with zipfile.ZipFile(archive) as stream:
        members = sorted(name for name in stream.namelist() if name.startswith('motions/') and name.endswith('.pkl'))
    pairs = []
    for member in members:
        sequence = Path(member).stem
        if sequence in ignored:
            continue
        for camera in cameras:
            # 动作以 cAll 命名；实际视频用 c01/c02 等具体机位名。
            video = root / 'videos' / (sequence.replace('_cAll_', f'_{camera}_') + '.mp4')
            if not video.is_file():
                raise FileNotFoundError(f'Missing paired video: {video}')
            # 动作仍在 ZIP 内：记录 archive + member 即可，不重复解压一份。
            pairs.append(Pair('aistpp', sequence, camera, str(video), str(archive),
                              'aist_smpl', motion_member=member, motion_fps=60,
                              source_root=str(root)))
    write_pairs(output, pairs)
    return pairs
