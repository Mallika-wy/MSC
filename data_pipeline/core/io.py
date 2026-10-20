"""少量JSON读写工具；不包含审计和原始文件哈希功能。"""
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: Path, value) -> None:
    """先在同目录写临时文件，成功后原子替换，避免留下半截JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json(path: Path):
    """读取处理参数、质量结果或视频索引等普通JSON。"""
    return json.loads(path.read_text())
