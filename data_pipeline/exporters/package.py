"""把带外部视频引用的release打包为可独立移动的数据目录。"""
from pathlib import Path
import shutil
from tempfile import mkdtemp

from data_pipeline.core.io import read_json, write_json
from data_pipeline.exporters.omg import resolve_asset, validate_release


def package_release(source: Path, output: Path) -> dict:
    """复制动作、人物视频、裁剪映射及可用的执行回放；不改写源release。

    输出必须不存在。先在同级临时目录完整构建和校验，再发布；失败时保留临时结果。
    重叠episode共用视频，所有可读取资产使用相对数据集根目录的路径，不使用软链接。
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and packaged output must be separate directories')
    records = read_json(source / 'meta/video_index.json')
    manifest = read_json(source / 'meta/omg_manifest.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f'.{output.name}.pack-', dir=output.parent))
    for name in ('data', 'meta'):
        shutil.copytree(source / name, temporary / name)
    copied = {}
    sequences, executions = set(), set()

    def component(name):
        if not name or name in ('.', '..') or Path(name).name != name or '\\' in name:
            raise ValueError(f'Unsafe source/sequence/camera name: {name!r}')
        return name

    def copy_asset(original, relative):
        original = original.resolve()
        if relative in copied:
            if copied[relative] != original:
                raise ValueError(f'Different assets map to the same output: {relative}')
            return relative.as_posix()
        if not original.is_file():
            raise FileNotFoundError(original)
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if target.stat().st_size != original.stat().st_size:
            raise ValueError(f'Incomplete copied asset: {original}')
        copied[relative] = original
        return relative.as_posix()

    for record in records:
        dataset, sequence = component(record['source']), component(record['sequence_id'])
        sequences.add((dataset, sequence))
        execution = record.get('execution_video_uri')
        execution = resolve_asset(source, execution) if execution else None
        for view in record['views']:
            directory = Path('videos') / dataset / sequence / component(view['camera_id'])
            video = resolve_asset(source, view['uri'])
            metadata = view.get('crop_metadata_uri')
            metadata = resolve_asset(source, metadata) if metadata else None
            view['uri'] = copy_asset(video, directory / 'person.mp4')
            if metadata:
                view['crop_metadata_uri'] = copy_asset(metadata, directory / 'metadata.npz')
                # 现有处理目录为 <sequence>/videos/<camera>/metadata.npz。
                candidate = metadata.parent.parent.parent / 'execution.mp4'
                if execution is None and candidate.is_file():
                    execution = candidate
            # 原片仅保留名字作为说明，不能留下看起来仍可读取的失效绝对路径。
            for field, label in [('source_uri', 'original_video_name'),
                                 ('source_calibration_uri', 'original_calibration_name')]:
                old = view.pop(field, None)
                if old:
                    view[label] = Path(old).name
        if execution is not None:
            record['execution_video_uri'] = copy_asset(execution, Path('execution_videos') / dataset / f'{sequence}.mp4')
            executions.add((dataset, sequence))

    write_json(temporary / 'meta/video_index.json', records)
    manifest['video_storage'] = 'bundled_relative_paths_with_decoded_frame_indices'
    manifest['limitations'] = [item for item in manifest.get('limitations', [])
                               if item != 'External video paths are not portable']
    preprocessing = manifest.get('video_preprocess', {})
    if preprocessing.get('weights'):
        preprocessing['weights'] = Path(preprocessing['weights']).name  # 模型来源说明，不是运行依赖。
    write_json(temporary / 'meta/omg_manifest.json', manifest)
    report = validate_release(temporary)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output appeared while packaging: {output}')
    temporary.rename(output)
    return {**report, 'directory': str(output), 'source_sequences': len(sequences),
            'execution_videos': len(executions)}
