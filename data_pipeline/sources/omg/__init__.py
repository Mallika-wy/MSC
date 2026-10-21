"""Read the locally cleaned OMG LeRobot v3 release without training dependencies."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from data_pipeline.io import sha256_file
from data_pipeline.sources.base import SourceMotion, SourceSequence, SourceText


@lru_cache(maxsize=2)
def _metadata(root_string):
    root = Path(root_string)
    info = json.loads((root / 'meta/info.json').read_text())
    if info.get('fps') != 30 or info.get('robot_type') != 'unitree_g1':
        raise ValueError('OMG requires unitree_g1 at 30 FPS')
    if info['features']['observation.state']['shape'] != [36]:
        raise ValueError('OMG observation.state must be qpos36')
    provenance = {'local_cleaned_release': True, 'source_root': str(root)}
    for name in ('info.json', 'omg_manifest.json', 'text_only_cleaning.json', 'quality_cleaning_v1.json'):
        path = root / 'meta' / name
        if path.is_file():
            provenance[name] = sha256_file(path)
    return info, provenance


def discover(root, *, audio=False, text=True, video=False, sequence=None):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info, provenance = _metadata(str(root))
    files = sorted((root / 'meta/episodes').rglob('*.parquet'))
    if not files:
        raise ValueError('OMG episode metadata not found')
    episode_filter = int(sequence.removeprefix('episode-')) if sequence and sequence.startswith('episode-') and sequence.removeprefix('episode-').isdigit() else None
    for path in files:
        batches = (pq.read_table(path, filters=[('episode_index', '=', episode_filter)]).to_batches(max_chunksize=2048)
                   if episode_filter is not None else pq.ParquetFile(path).iter_batches(batch_size=2048))
        for batch in batches:
            for row in batch.to_pylist():
                episode = int(row['episode_index'])
                split = row.get('omg/split')
                if split is None:
                    split = next((name for name, interval in info['splits'].items()
                                  if int(interval.split(':')[0]) <= episode < int(interval.split(':')[1])), None)
                if split not in ('train', 'val', 'test'):
                    raise ValueError(f'Unknown OMG split for episode {episode}')
                relative = info['data_path'].format(chunk_index=int(row['data/chunk_index']), file_index=int(row['data/file_index']))
                motion_path = (root / relative).resolve()
                if not motion_path.is_relative_to(root) or not motion_path.is_file():
                    raise ValueError('Missing/unsafe OMG data shard')
                length = int(row['length'])
                if length < 2 or int(row['dataset_to_index']) - int(row['dataset_from_index']) != length:
                    raise ValueError('Invalid OMG episode range')
                source_id = str(row.get('omg/source_id', episode))
                upstream = str(row.get('omg/dataset', 'unknown'))
                mid = f'episode-{episode:08d}'
                texts = []
                if text and row.get('omg/has_text', True):
                    texts = [SourceText(caption.strip(), 0., length / 30, source='OMG episode tasks')
                             for caption in dict.fromkeys(row.get('tasks') or []) if caption.strip()]
                yield SourceSequence(mid, 'omg', f'{upstream}:{source_id}',
                    [SourceMotion(mid, motion_path, representation='g1_qpos36', metadata=row)],
                    texts=texts, split=split,
                    provenance={**provenance, 'episode': row, 'upstream_dataset': upstream,
                                'upstream_sequence': source_id})


@lru_cache(maxsize=3)
def _row_group(path, mtime, size, group):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).read_row_group(group, columns=[
        'observation.state', 'episode_index', 'index', 'frame_index', 'timestamp'])


def read_motion(motion):
    import pyarrow.parquet as pq
    from data_pipeline.motion.validation import validate_qpos
    row = motion.metadata
    start, end = int(row['dataset_from_index']), int(row['dataset_to_index'])
    file = pq.ParquetFile(motion.path)
    pieces = []
    # Global row indices survive file boundaries. Row-group statistics avoid scans.
    index_column = file.schema_arrow.get_field_index('index')
    for group in range(file.num_row_groups):
        stats = file.metadata.row_group(group).column(index_column).statistics
        if stats is not None and stats.has_min_max and (stats.max < start or stats.min >= end):
            continue
        st = motion.path.stat()
        table = _row_group(str(motion.path), st.st_mtime_ns, st.st_size, group)
        indices = table['index'].to_numpy()
        selected = np.flatnonzero((indices >= start) & (indices < end))
        if len(selected):
            pieces.append(table.take(selected))
    if not pieces:
        raise ValueError('OMG episode not present in declared data shard')
    import pyarrow as pa
    table = pa.concat_tables(pieces)
    order = np.argsort(table['index'].to_numpy())
    table = table.take(order)
    if not np.array_equal(table['index'].to_numpy(), np.arange(start, end)):
        raise ValueError('OMG episode rows are incomplete or duplicated')
    if not np.all(table['episode_index'].to_numpy() == row['episode_index']):
        raise ValueError('OMG episode identity mismatch')
    if not np.array_equal(table['frame_index'].to_numpy(), np.arange(end-start)):
        raise ValueError('OMG frame_index mismatch')
    qpos = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    times = table['timestamp'].to_numpy().astype(np.float64)
    if not np.allclose(times, np.arange(len(qpos))/30, atol=1e-4, rtol=1e-6):
        raise ValueError('OMG timestamp mismatch')
    validate_qpos(qpos, times)
    return qpos
