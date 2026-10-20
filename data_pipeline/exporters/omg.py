"""导出层：将完整执行序列切成episode，写OMG兼容动作Parquet和视频帧索引。

这里不再重定向或编码视频。多个episode引用同一视频的不同帧区间，不复制MP4。
当前OMG加载器能读取动作；视频条件训练读取分支仍需后续接入。
"""
from pathlib import Path
import hashlib

import numpy as np

from data_pipeline.core.io import read_json, write_json
from data_pipeline.stages.motion import validate_qpos


SPLITS = ('train', 'val', 'test')


def assign_split(group: str, ratios: list[float], seed: str) -> str:
    """按split_group确定train/val/test，同组所有片段与机位始终同属一组。"""
    if len(ratios) != 3 or not np.isfinite(ratios).all() or min(ratios) < 0 or not np.isclose(sum(ratios), 1):
        raise ValueError('Split ratios must be three nonnegative values summing to one')
    # 此处只把分组名稳定映射到[0,1)，不是扫描文件哈希或做数据审计。
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
    return SPLITS[int(np.searchsorted(np.cumsum(ratios), value, side='right'))]


def windows(length: int, size: int, stride: int) -> list[tuple[int, int]]:
    """生成[start,end)帧区间；尾部补一个对齐末帧的窗口，起点去重。"""
    if size < 2 or stride < 1 or stride > size:
        raise ValueError('Require episode_frames >= 2 and 1 <= stride <= episode_frames')
    if length < 2:
        return []
    if length <= size:
        return [(0, length)]
    starts = sorted(set([*range(0, length - size + 1, stride), length - size]))
    return [(start, start + size) for start in starts]


def export_release(entries: list[dict], config: dict, output: Path) -> dict:
    """读取每条序列的paired_motion/video_index，导出动作、episode和视频映射。

    一个episode对应Parquet中的连续若干行；observation.state与action都是36维。
    125维模型特征仍由现有G1MotionFeatureCodec生成，不在这里另写一套。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    policy = config.get('export', {})
    repo_id = policy.get('repo_id', 'local/video-motion-prototype')
    if repo_id == 'THU-MARS/OMG-Data':
        raise ValueError('Custom releases must not impersonate the official dataset')
    ratios = config.get('split', {}).get('ratios', [0.8, 0.1, 0.1])
    seed = str(config.get('split', {}).get('seed', 0))
    episodes, rows, video_records = [], [], []
    # 先按集合排序，保证info.json中每个split对应连续的episode编号区间。
    ordered = sorted(entries, key=lambda entry: (SPLITS.index(assign_split(entry['item']['split_group'], ratios, seed)),
                                               entry['item']['source'], entry['item']['sequence_id']))
    for entry in ordered:
        item, directory = entry['item'], Path(entry['directory'])
        if not read_json(directory / 'quality.json')['accepted']:
            continue
        with np.load(directory / 'paired_motion.npz', allow_pickle=False) as data:
            qpos, times = data['qpos_36'].copy(), data['timestamps'].copy()
            valid = data['executed_state_mask'].copy()
        # t=0是仿真初始化而非执行结果，因此用有效性掩码排除。
        keep = np.flatnonzero(valid)
        if not len(keep):
            continue
        if not np.array_equal(keep, np.arange(keep[0], len(qpos))):
            raise ValueError('Noncontiguous executed validity cannot be silently concatenated')
        # 这是整段30Hz参考/裁剪视频中的索引，不是原片59.94FPS的帧号。
        # 原片帧号保存在裁剪metadata.npz，不能在此把两种索引混用。
        source_indices = keep
        qpos, times = qpos[keep], times[keep]
        validate_qpos(qpos, times)
        mapping = read_json(Path(entry['video_index']))
        split = assign_split(item['split_group'], ratios, seed)
        for start, end in windows(len(qpos), int(policy.get('episode_frames', 240)), int(policy.get('stride_frames', 120))):
            episode_id, row_start = len(episodes), len(rows)
            states = qpos[start:end]
            # action定义为下一帧状态，不是机器人电机控制量。
            # 末帧无下一帧：重复末状态并置action_valid=False，不能跨episode取值。
            actions = np.concatenate((states[1:], states[-1:]))
            for frame, (state, action) in enumerate(zip(states, actions)):
                rows.append({'index': len(rows), 'episode_index': episode_id, 'frame_index': frame,
                             'timestamp': frame / 30, 'task_index': 0,
                             'observation.state': state.tolist(), 'action': action.tolist(),
                             'pipeline.action_valid': frame < len(states) - 1,
                             'pipeline.source_timestamp': float(times[start + frame])})
            episodes.append({'episode_index': episode_id, 'length': len(states), 'tasks': [],
                             'dataset_from_index': row_start, 'dataset_to_index': len(rows),
                             'omg/source_id': item['sequence_id'], 'omg/dataset': item['source'],
                             'omg/split': split, 'omg/source_start_frame': int(source_indices[start]),
                             'omg/source_end_frame': int(source_indices[end - 1]) + 1,
                             'pipeline.split_group': item['split_group']})
            # 视频与动作使用完全相同的窗口索引；不另行按秒数重新切一次视频。
            video_records.append({'episode_index': episode_id, 'source': item['source'],
                                  'sequence_id': item['sequence_id'],
                                  'sequence_timestamps': times[start:end].tolist(),
                                  'views': [{**view,
                                             'frame_indices': np.asarray(view['frame_indices'])[source_indices[start:end]].tolist(),
                                             'frame_pts_s': np.asarray(view['frame_pts_s'])[source_indices[start:end]].tolist()}
                                            for view in mapping['views']]})
    if not episodes:
        raise ValueError('No quality-accepted episodes to export; inspect sequence reports')
    output.mkdir(parents=True, exist_ok=True)
    data_path, episode_path = output / 'data/chunk-000/file-000.parquet', output / 'meta/episodes/chunk-000/file-000.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    # 明确写为float32固定36维数组，避免由Python列表推断成不必要的float64。
    table = pa.Table.from_pylist(rows)
    for name in ('observation.state', 'action'):
        table = table.set_column(table.schema.get_field_index(name), name,
                                 pa.array([row[name] for row in rows], type=pa.list_(pa.float32(), 36)))
    pq.write_table(table, data_path)
    pq.write_table(pa.Table.from_pylist(episodes), episode_path)
    split_ranges = {}
    cursor = 0
    for split in SPLITS:
        count = sum(episode['omg/split'] == split for episode in episodes)
        if count:
            split_ranges[split] = f'{cursor}:{cursor + count}'
        cursor += count
    # meta是训练格式所需的帧率、特征和episode信息，不是历史下载审计。
    write_json(output / 'meta/info.json', {'codebase_version': 'v3.0', 'fps': 30,
               'total_episodes': len(episodes), 'total_frames': len(rows), 'splits': split_ranges,
               'features': {name: {'dtype': 'float32', 'shape': [36]} for name in ('observation.state', 'action')}})
    write_json(output / 'meta/video_index.json', video_records)
    write_json(output / 'meta/omg_manifest.json', {
        'repo_id': repo_id, 'pipeline_schema': 'video-motion-prototype/v1',
        'production_ready': False,
        'video_storage': 'external_video_references_with_decoded_frame_indices',
        'video_preprocess': config.get('video_preprocess', {'enabled': False}),
        'action_terminal_policy': 'repeat_last_state; pipeline.action_valid=false',
        'motion_label': 'tracker_executed; initialization frame omitted',
        'limitations': ['Not an official OMG-Data release', 'Video conditioning loader not implemented',
                        'External video paths are not portable', 'Semantic synchronization not automatically verified'],
    })
    return validate_release(output)


def resolve_asset(root: Path, uri: str) -> Path:
    """旧release可引用绝对路径；独立包的路径相对数据集根目录。"""
    path = Path(uri)
    return path if path.is_absolute() else root / path


def validate_release(root: Path) -> dict:
    """检查导出内部的一致性：行区间、动作前后帧、split和视频索引。"""
    import pyarrow.parquet as pq
    manifest = read_json(root / 'meta/omg_manifest.json')
    bundled = manifest.get('video_storage', '').startswith('bundled_')

    def asset(uri):
        path = resolve_asset(root, uri)
        if bundled and (Path(uri).is_absolute() or not path.resolve().is_relative_to(root.resolve())):
            raise ValueError(f'Bundled assets must stay inside the dataset: {uri}')
        if not path.is_file():
            raise ValueError(f'Video asset missing: {path}')
        return path
    rows = pq.read_table(root / 'data/chunk-000/file-000.parquet').to_pylist()
    episodes = pq.read_table(root / 'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    info = read_json(root / 'meta/info.json')
    videos = read_json(root / 'meta/video_index.json')
    if info['total_frames'] != len(rows) or info['total_episodes'] != len(episodes):
        raise ValueError('Release counts disagree')
    groups = {}
    if len(videos) != len(episodes):
        raise ValueError('Video/episode count mismatch')
    # episode的全局行区间应首尾衔接，不允许有遗漏或错误跨段引用。
    expected_row = 0
    checked_components = set()
    for episode in episodes:
        if episode['dataset_from_index'] != expected_row:
            raise ValueError('Noncontiguous episode rows')
        expected_row = episode['dataset_to_index']
        split, group = episode['omg/split'], episode['pipeline.split_group']
        if group in groups and groups[group] != split:
            raise ValueError('Split leakage detected')
        groups[group] = split
        chunk = rows[episode['dataset_from_index']:episode['dataset_to_index']]
        if len(chunk) != episode['length'] or any(row['episode_index'] != episode['episode_index'] for row in chunk):
            raise ValueError('Episode row identity mismatch')
        qpos = np.asarray([row['observation.state'] for row in chunk])
        validate_qpos(qpos, np.asarray([row['timestamp'] for row in chunk]))
        actions = np.asarray([row['action'] for row in chunk])
        if not np.array_equal(actions[:-1], qpos[1:]) or not np.array_equal(actions[-1], qpos[-1]):
            raise ValueError('Invalid within-episode action alignment')
        if [row['pipeline.action_valid'] for row in chunk] != [True] * (len(chunk) - 1) + [False]:
            raise ValueError('Invalid terminal action mask')
        video = videos[episode['episode_index']]
        if video['episode_index'] != episode['episode_index'] or len(video['sequence_timestamps']) != len(chunk):
            raise ValueError('Video/episode identity mismatch')
        if not np.allclose(video['sequence_timestamps'], [row['pipeline.source_timestamp'] for row in chunk]):
            raise ValueError('Video/motion timestamp mismatch')
        if video.get('execution_video_uri'):
            asset(video['execution_video_uri'])
        for view in video['views']:
            indices, pts = np.asarray(view['frame_indices']), np.asarray(view['frame_pts_s'])
            if len(indices) != len(chunk) or pts.shape != (len(chunk),) or (indices < 0).any() or (np.diff(indices) < 0).any() or not np.isfinite(pts).all():
                raise ValueError('Invalid per-frame video mapping')
            video_path = asset(view['uri'])
            if view.get('crop_metadata_uri'):
                from data_pipeline.components.person_video import validate_person_video
                metadata_path = asset(view['crop_metadata_uri'])
                component_path = metadata_path.parent
                if video_path.resolve() != (component_path / 'person.mp4').resolve():
                    raise ValueError('Crop metadata and person video must share their directory')
                with np.load(metadata_path, allow_pickle=False) as crop:
                    if (indices >= len(crop['output_timestamps'])).any():
                        raise ValueError('Video frame index exceeds packaged video coverage')
                    if not np.allclose(pts, crop['output_timestamps'][indices]):
                        raise ValueError('Video frame PTS disagree with crop metadata')
                # 重叠窗口可能引用同一个视频，媒体检查只做一次。
                if str(component_path) not in checked_components:
                    validate_person_video(component_path)
                    checked_components.add(str(component_path))
                if view['metadata']['width'] != 512 or view['metadata']['height'] != 512:
                    raise ValueError('Preprocessed training video must be 512x512')
    if expected_row != len(rows):
        raise ValueError('Unassigned release rows')
    return {'valid': True, 'episodes': len(episodes), 'frames': len(rows),
            # 仅现有OMG加载器要求这个小型格式文件的身份值；不校验原始文件清单。
            'manifest_sha256': hashlib.sha256((root / 'meta/omg_manifest.json').read_bytes()).hexdigest(),
            'repo_id': manifest['repo_id']}
