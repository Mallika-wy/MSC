"""JSON读写、流式文件指纹，以及显式外部根目录的路径解析。"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
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


def sha256_file(path):
    """Hash large media without materializing it in memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_uri(root, uri, external_roots=None):
    if not isinstance(uri, str) or not uri or '\\' in uri or '%' in uri or '\x00' in uri:
        raise ValueError(f'Unsafe URI: {uri!r}')
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment or (parsed.netloc and not parsed.scheme):
        raise ValueError('URI query/fragment not supported')
    base = Path(root).resolve()
    relative = uri
    if parsed.scheme:
        if parsed.scheme != 'root' or parsed.netloc not in (external_roots or {}):
            raise ValueError(f'Unknown external root: {uri}')
        base = Path(external_roots[parsed.netloc]).expanduser().resolve()
        relative = parsed.path.removeprefix('/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or relative in ('.', ''):
        raise ValueError(f'Unsafe URI: {uri}')
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f'URI escapes root: {uri}')
    return target
