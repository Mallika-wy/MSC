"""Metadata validation and bounded deep checks. Never loads media encoders."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import numpy as np

from data_pipeline.schema import Inventory, TimeMap
from data_pipeline.io import resolve_uri
from data_pipeline.motion.validation import validate_qpos


def audit_release_roots(roots):
    """One-off cross-release split audit using only identity columns."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sequences, assets = [], []
    for root in map(Path, roots):
        seq = pq.read_table(root / 'meta/sequences.parquet', columns=[
            'sequence_id','source_dataset','source_sequence_id','split_group_id','split'])
        sequences.append(seq)
        links = pq.read_table(root / 'meta/links.parquet', columns=['motion_id','asset_id'])
        if not len(links):
            continue
        motions = pq.read_table(root / 'meta/motions.parquet', columns=['motion_id','sequence_id'])
        motions = motions.filter(pc.is_in(motions['motion_id'], value_set=links['motion_id']))
        motion_splits = pc.take(seq['split'], pc.index_in(motions['sequence_id'], value_set=seq['sequence_id']))
        link_splits = pc.take(motion_splits, pc.index_in(links['motion_id'], value_set=motions['motion_id']))
        table = pq.read_table(root / 'meta/assets.parquet', columns=['asset_id','modality','checksum','source_asset_id'])
        table = table.take(pc.index_in(links['asset_id'], value_set=table['asset_id']))
        assets.append(table.append_column('split', link_splits))
    def check(table, keys):
        counts = table.group_by(keys).aggregate([('split','count_distinct')])
        if pc.any(pc.greater(counts['split_count_distinct'], 1)).as_py():
            raise ValueError(f'Split leakage across Release roots: {keys}')
    seq = pa.concat_tables(sequences)
    check(seq, ['source_dataset','source_sequence_id'])
    check(seq, ['source_dataset','split_group_id'])
    if assets:
        table = pa.concat_tables(assets)
        check(table, ['modality','checksum'])
        check(table, ['modality','source_asset_id'])


def audit_splits(inventories):
    """Audit all splits/roots, including shared audio (not enforced by v1 schema)."""
    seen = {}
    def check(key, split):
        previous = seen.setdefault(key, split)
        if previous != split:
            raise ValueError(f'Split leakage: {key}: {previous}/{split}')
    for inv in inventories:
        for s in inv.sequences:
            check(('sequence', s.source_dataset, s.source_sequence_id), s.split)
            check(('group', s.source_dataset, s.split_group_id), s.split)
        if not inv.links:
            continue
        seq = {s.sequence_id: s for s in inv.sequences}
        motions = {m.motion_id: m for m in inv.motions}
        assets = {a.asset_id: a for a in inv.assets}
        for link in inv.links:
            m, a = motions[link.motion_id], assets[link.asset_id]
            if link.actor_id is not None and link.actor_id != m.actor_id:
                raise ValueError('Link Actor mismatch')
            split = seq[m.sequence_id].split
            check(('asset_checksum', a.modality, a.checksum), split)
            check(('asset_source', a.modality, a.source_asset_id), split)


def _validate_motion_shard(task):
    """Read each needed row group once, including motions crossing its boundary."""
    import pyarrow.parquet as pq
    path, motions = task
    file = pq.ParquetFile(path)
    offsets = np.r_[0, np.cumsum([file.metadata.row_group(i).num_rows for i in range(file.num_row_groups)])]
    current = -1
    try:
        for start, frames, fps in sorted(motions):
            end = start + frames
            first = np.searchsorted(offsets, start, side='right') - 1
            last = np.searchsorted(offsets, end, side='left') - 1
            positions, timestamps = [], []
            for group in range(first, last+1):
                if group != current:
                    table = file.read_row_group(group, use_threads=False)
                    values = table['qpos36'].combine_chunks().values.to_numpy().reshape(-1, 36)
                    times = table['timestamp_sec'].to_numpy()
                    current = group
                lo = max(start, offsets[group]) - offsets[group]
                hi = min(end, offsets[group+1]) - offsets[group]
                positions.append(values[lo:hi])
                timestamps.append(times[lo:hi])
            qpos = positions[0] if len(positions) == 1 else np.concatenate(positions)
            times_for_motion = timestamps[0] if len(timestamps) == 1 else np.concatenate(timestamps)
            validate_qpos(qpos, times_for_motion)
            if not np.allclose(times_for_motion, np.arange(frames) / fps, atol=1e-8, rtol=0):
                raise ValueError('Motion timeline must be local, uniform 30 Hz')
            if (np.sum(qpos[:-1, 3:7] * qpos[1:, 3:7], axis=1) < 0).any():
                raise ValueError('Quaternion sign discontinuity')
    finally:
        file.close()


def validate_release(root, deep_sample=0, *, workers=8):
    root = Path(root)
    if deep_sample < 0:
        raise ValueError('deep_sample must be nonnegative')
    if workers < 1:
        raise ValueError('workers must be positive')
    if json.loads((root / 'dataset.json').read_text()).get('kind') != 'release':
        raise ValueError('Expected release manifest; source inventory is not a release')
    inv = Inventory.read(root, validate=False).validate(root, release=True, check_files=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    storage, quality_groups, ranges = {}, {}, {}
    expected = pa.schema([('qpos36', pa.list_(pa.float32(), 36)), ('timestamp_sec', pa.float64())])
    for m in inv.motions:
        if m.storage_uri not in storage:
            storage[m.storage_uri] = pq.ParquetFile(resolve_uri(root, m.storage_uri, inv.external_roots))
            if storage[m.storage_uri].schema_arrow != expected:
                raise ValueError('Motion storage schema mismatch')
        file = storage[m.storage_uri]
        if m.row_start + m.num_frames > file.metadata.num_rows:
            raise ValueError('Motion row range mismatch')
        ranges.setdefault(m.storage_uri, []).append((m.row_start, m.row_start+m.num_frames))
        quality_groups.setdefault(m.quality_ref, set()).add(m.motion_id)
    for intervals in ranges.values():
        intervals.sort()
        if any(end > next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:])):
            raise ValueError('Overlapping motion row ranges')
    from .quality import read_quality
    for reference, expected_ids in quality_groups.items():
        path = resolve_uri(root, reference, inv.external_roots)
        if path.suffix != '.parquet':
            for mid in expected_ids:
                quality = read_quality(root, reference, mid, inv.external_roots)
                if quality.get('motion_id') != mid or quality.get('accepted') is not True:
                    raise ValueError('Quality record mismatch/rejected')
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                mid = row['motion_id']
                report = json.loads(row['report_json'])
                if mid not in expected_ids or row['accepted'] is not True or report.get('accepted') is not True or report.get('motion_id') != mid:
                    raise ValueError('Quality record mismatch/rejected/duplicate')
                expected_ids.remove(mid)
        if expected_ids:
            raise ValueError('Missing quality records')
    selected = inv.motions[:deep_sample]
    shards = {}
    for m in selected:
        path = str(resolve_uri(root, m.storage_uri, inv.external_roots))
        shards.setdefault(path, []).append((m.row_start, m.num_frames, m.fps))
    if len(selected) >= 256 and len(shards) > 1 and workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(shards)), mp_context=get_context('spawn')) as pool:
            for _ in pool.map(_validate_motion_shard, shards.items()):
                pass
    else:
        for task in shards.items():
            _validate_motion_shard(task)
    for a in inv.assets[:deep_sample]:
        if a.uri is None:
            continue
        source = resolve_uri(root, a.uri, inv.external_roots)
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if source.stat().st_size != a.size_bytes or digest.hexdigest() != a.checksum:
            raise ValueError('Asset checksum/size mismatch')
    for f in inv.features[:deep_sample]:
        from .features import read_feature
        data, times = read_feature(root, f, inv.external_roots)
        if list(data.shape) != f.shape or str(data.dtype) != f.dtype or not np.isfinite(data).all() or len(times) != len(data):
            raise ValueError('Feature payload shape/dtype/values mismatch')
        if times.dtype != np.float64:
            raise ValueError('Token timestamps must be float64 seconds')
        TimeMap().select(times, 0, 1, semantics=f.token_time_semantics)
        asset = next(a for a in inv.assets if a.asset_id == f.asset_id)
        if (times < 0).any() or (asset.duration_sec is not None and (times > asset.duration_sec + 1e-8).any()):
            raise ValueError('Feature timestamps outside asset')
    modalities = {m.motion_id: set() for m in inv.motions}
    sequence_motions = {}
    for m in inv.motions:
        sequence_motions.setdefault(m.sequence_id, []).append(m.motion_id)
    for link in inv.links:
        modalities[link.motion_id].add(link.modality)
    for annotation in inv.annotations:
        for mid in ([annotation.motion_id] if annotation.motion_id else sequence_motions.get(annotation.sequence_id, [])):
            modalities[mid].add('text')
    modalities = {mid: '+'.join(sorted(values)) or 'motion_only' for mid, values in modalities.items()}
    audit_splits([inv])
    from .training_index import validate_training_index
    validate_training_index(root)
    return dict(valid=True, processing_summary=inv.provenance.get('processing_summary', {}),
                rejection_reasons=inv.provenance.get('rejection_reasons', {}), sequences=len(inv.sequences), motions=len(inv.motions), assets=len(inv.assets),
                features=len(inv.features), duration_sec=sum(m.num_frames/m.fps for m in inv.motions),
                sources=dict(Counter(s.source_dataset for s in inv.sequences)),
                modality_combinations=dict(Counter(modalities.values())),
                deep_checked_motions=min(deep_sample, len(inv.motions)),
                deep_checked_features=min(deep_sample, len(inv.features)))
