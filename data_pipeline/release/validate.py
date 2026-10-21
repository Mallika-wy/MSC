"""Metadata validation and bounded deep checks. Never loads media encoders."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from data_pipeline.schema import Inventory, TimeMap
from data_pipeline.io import resolve_uri
from data_pipeline.motion.validation import validate_qpos


def validate_release(root, deep_sample=0):
    from .export import read_motion
    root = Path(root)
    if deep_sample < 0:
        raise ValueError('deep_sample must be nonnegative')
    if json.loads((root / 'dataset.json').read_text()).get('kind') != 'release':
        raise ValueError('Expected release manifest; source inventory is not a release')
    inv = Inventory.read(root, validate=False).validate(root, release=True, check_files=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    storage, quality_groups, ranges = {}, {}, {}
    for m in inv.motions:
        if m.storage_uri not in storage:
            storage[m.storage_uri] = pq.ParquetFile(resolve_uri(root, m.storage_uri, inv.external_roots))
        file = storage[m.storage_uri]
        expected = pa.schema([('qpos36', pa.list_(pa.float32(), 36)), ('timestamp_sec', pa.float64())])
        if file.schema_arrow != expected or m.row_start + m.num_frames > file.metadata.num_rows:
            raise ValueError('Motion storage schema/row range mismatch')
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
    for m in inv.motions[:deep_sample]:
        qpos, times = read_motion(root, m)
        validate_qpos(qpos, times)
        if not np.allclose(times, np.arange(m.num_frames) / m.fps, atol=1e-8, rtol=0):
            raise ValueError('Motion timeline must be local, uniform 30 Hz')
        if (np.sum(qpos[:-1, 3:7] * qpos[1:, 3:7], axis=1) < 0).any():
            raise ValueError('Quaternion sign discontinuity')
    for a in inv.assets[:deep_sample]:
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
    return dict(valid=True, processing_summary=inv.provenance.get('processing_summary', {}),
                rejection_reasons=inv.provenance.get('rejection_reasons', {}), sequences=len(inv.sequences), motions=len(inv.motions), assets=len(inv.assets),
                features=len(inv.features), duration_sec=sum(m.num_frames/m.fps for m in inv.motions),
                sources=dict(Counter(s.source_dataset for s in inv.sequences)),
                modality_combinations=dict(Counter(modalities.values())),
                deep_checked_motions=min(deep_sample, len(inv.motions)),
                deep_checked_features=min(deep_sample, len(inv.features)))
