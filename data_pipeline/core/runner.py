"""流水线调度中心：CSV → 参考动作 → 肘腕筛查 → 执行动作 → 512 视频 → OMG episode。

这里负责处理顺序和结果汇总；原始格式解码、视觉裁剪、导出分别放在独立模块。
"""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import groupby
import multiprocessing
from pathlib import Path
import time

import numpy as np
import yaml

from data_pipeline.core.io import read_json, write_json
from data_pipeline.core.pairs import read_pairs
from data_pipeline.exporters.omg import export_release
from data_pipeline.stages.motion import execute_and_screen, prepare_reference
from data_pipeline.stages.arm_twist import screen_reference
from data_pipeline.stages.video import prepare_video_view
from data_pipeline.visualization.execution import prepare_execution_video


PROJECT = Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict:
    """加载参数。YAML 中的相对路径统一相对 OMG 根目录，便于 VS Code 调试。"""
    config = yaml.safe_load(path.read_text())
    for section, keys in [('retarget', ('gmr_root', 'smpl_model', 'dependency_site')),
                          ('execution', ('onnx',)), ('output', ('root',)),
                          ('video_preprocess', ('weights',))]:
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = str((PROJECT / Path(config[section][key]).expanduser()).resolve())
    config['input_csv'] = str((PROJECT / config['input_csv']).resolve())
    return config


def write_results(path, results):
    """更新简单进度表：成功、质检拒绝或运行失败，并记录对应输出位置。"""
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source', 'sequence_id', 'status', 'directory', 'error'])
        writer.writeheader()
        writer.writerows(results)


def process_group(group, config, root, force=False):
    """处理一条动作及其机位；每个进程独占序列目录，不写全局进度或release。"""
    pair = group[0]
    directory = root / 'sequences' / pair.source / pair.sequence_id
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(source=pair.source, sequence_id=pair.sequence_id, status='failed', directory=str(directory), error='')
    entry, started = None, time.perf_counter()
    try:
        if any((p.motion_path, p.motion_member, p.motion_format, p.motion_fps, p.motion_time_offset_s, p.split_group) !=
               (pair.motion_path, pair.motion_member, pair.motion_format, pair.motion_fps, pair.motion_time_offset_s, pair.split_group) for p in group):
            raise ValueError('Views of a sequence must share motion, timeline and split_group')
        print(f'{pair.source}/{pair.sequence_id}', flush=True)
        reference = prepare_reference(pair, config, directory, PROJECT, force=force)
        # 重定向后立即筛查，坏参考不能继续消耗tracker、渲染和视频裁剪资源。
        quality = screen_reference(reference, config.get('quality', {}))
        write_json(directory / 'reference/quality.json', quality)
        if quality['accepted']:
            quality = execute_and_screen(reference, config, directory, PROJECT, force=force)
            # 真正执行过的动作，无论执行质检通过或拒绝，都保留实际回放。
            prepare_execution_video(directory, force=force)
        if not quality['accepted']:
            result.update(status='rejected', error='; '.join(quality['rejection_reasons']))
        else:
            # 同组机位共享动作时间轴；未完成的视频可继续生成，已有结果按原规则校验复用。
            with np.load(directory / 'paired_motion.npz', allow_pickle=False) as data:
                times = data['timestamps'].copy()
            views = [prepare_video_view(p, times, config, directory / 'videos' / p.camera_id, force=force) for p in group]
            write_json(directory / 'video_index.json', {'views': views})
            entry = {'item': {'source': pair.source, 'sequence_id': pair.sequence_id,
                              'split_group': pair.split_group},
                     'directory': str(directory), 'video_index': str(directory / 'video_index.json')}
            result['status'] = 'complete'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        print(f'  FAILED {pair.sequence_id}: {result["error"]}', flush=True)
    print(f'  {result["status"]} {pair.sequence_id}: {time.perf_counter()-started:.1f}s', flush=True)
    return result, entry


def iter_processed(groups, config, root, force, workers):
    """串行便于断点调试；并行用spawn隔离MuJoCo/CUDA状态，不能fork已初始化的CUDA。"""
    if workers == 1:
        for group in groups:
            yield process_group(group, config, root, force)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(process_group, group, config, root, force) for group in groups]
            try:
                for future in as_completed(futures):
                    yield future.result()
            except BaseException:
                # 中断时取消尚未开始的序列，不让队列继续跑完整个数据集。
                for future in futures:
                    future.cancel()
                raise


def run(config_path: Path, until='export', *, video_device=None, csv_path=None, limit=None, force=False, workers=1) -> Path:
    """处理 CSV 中的原始序列，返回结果目录；workers控制同时处理的序列数。

    until：inventory 只读表；process 处理到视频配对；export 继续打包训练数据。
    force：重算已有处理结果；原始视频、原始动作始终不被改写。
    """
    if workers < 1:
        raise ValueError('--workers must be positive')
    # 1. 加载配置及命令行覆盖值；此后所有数据源都通过 Pair 对象进入流程。
    config = load_config(config_path)
    if csv_path:
        config['input_csv'] = str(Path(csv_path).resolve())
    if video_device is not None:
        config['video_preprocess']['device'] = str(video_device)
    pairs = read_pairs(Path(config['input_csv']))
    selected = config.get('sequences', [])
    if selected:
        pairs = [pair for pair in pairs if pair.sequence_id in selected]
    # 一条动作可能配多个机位：先按序列聚合，动作只处理一次，视频各自处理。
    # groupby 只合并相邻元素，因此必须先按同样的键排序。
    groups = [list(group) for _, group in groupby(sorted(pairs, key=lambda p: (p.source, p.sequence_id)),
                                                   key=lambda p: (p.source, p.sequence_id))]
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive')
        groups = groups[:limit]
    if not groups:
        raise ValueError('No selected video/motion pairs')
    print(f'CSV: {len(groups)} sequences, {sum(map(len, groups))} video/motion pairs', flush=True)
    if until == 'inventory':
        return Path(config['input_csv'])
    root = Path(config['output']['root'])
    if limit is not None:
        root = root / f'subset_{limit}'  # 调试子集单独落盘，不能覆盖全量 release。
    for pair in pairs:
        if root.is_relative_to(Path(pair.video_path).parent) or root.is_relative_to(Path(pair.motion_path).parent):
            raise ValueError('Outputs must be outside raw video/motion directories')
    root.mkdir(parents=True, exist_ok=True)
    # 2. 防止在同一输出目录里混用不同参数，造成旧动作和新视频被误配。
    settings = root / 'config.json'
    if settings.exists() and read_json(settings) != config and not force:
        raise ValueError('Output config changed. Use a new output.root or --force to replace generated results.')
    write_json(settings, config)  # 仅保存一份参数，不扫描文件哈希或依赖版本。
    print(f'Processing with {workers} worker(s)', flush=True)
    entries, results = [], []
    # 只有主进程更新总进度、导出release，避免并行覆盖CSV或重复写Parquet。
    for result, entry in iter_processed(groups, config, root, force, workers):
        results.append(result)
        if entry is not None:
            entries.append(entry)
        write_results(root / 'results.csv', sorted(results, key=lambda row: (row['source'], row['sequence_id'])))
        print(f'Progress: {len(results)}/{len(groups)} sequences', flush=True)
    # 5. 整段处理完成后才切 episode；重叠 episode 共享同一份裁剪视频。
    # 有部分序列失败时，成功部分仍可导出；不要把这种 release 误认为全量成功。
    if until == 'export' and entries:
        export_release(entries, config, root / 'release')
    if any(row['status'] == 'failed' for row in results):
        raise RuntimeError(f'Some sequences failed; successful outputs preserved. See {root / "results.csv"}')
    if until == 'export' and not entries:
        raise RuntimeError(f'No accepted sequences to export. See {root / "results.csv"}')
    return root
