"""Metadata validation and bounded deep checks. Never loads media encoders."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

from data_pipeline.schemas import Inventory, TimeMap
from data_pipeline.schemas.manifest import resolve_uri
from data_pipeline.stages.motion import validate_qpos


def validate_release(root, deep_sample=0):
    from .export import read_motion
    root = Path(root)
    if deep_sample < 0:
        raise ValueError('deep_sample must be nonnegative')
    if json.loads((root / 'dataset.json').read_text()).get('kind') != 'release':
        raise ValueError('Expected release manifest; source inventory is not a release')
    inv = Inventory.read(root).validate(root, release=True, check_files=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    storage = {}
    for m in inv.motions:
        if m.storage_uri not in storage:
            storage[m.storage_uri] = pq.ParquetFile(resolve_uri(root, m.storage_uri, inv.external_roots))
        file = storage[m.storage_uri]
        expected = pa.schema([('qpos36', pa.list_(pa.float32(), 36)), ('timestamp_sec', pa.float64())])
        if file.schema_arrow != expected or m.row_start + m.num_frames > file.metadata.num_rows:
            raise ValueError('Motion storage schema/row range mismatch')
        quality = json.loads(resolve_uri(root, m.quality_ref, inv.external_roots).read_text())
        if quality.get('motion_id') != m.motion_id or quality.get('accepted') is not True:
            raise ValueError('Quality record mismatch/rejected')
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
        data = np.load(resolve_uri(root, f.feature_uri, inv.external_roots), mmap_mode='r', allow_pickle=False)
        times = np.load(resolve_uri(root, f.timestamp_uri, inv.external_roots), allow_pickle=False)
        if list(data.shape) != f.shape or str(data.dtype) != f.dtype or not np.isfinite(data).all() or len(times) != len(data):
            raise ValueError('Feature payload shape/dtype/values mismatch')
        if times.dtype != np.float64:
            raise ValueError('Token timestamps must be float64 seconds')
        TimeMap().select(times, 0, 1, semantics=f.token_time_semantics)
        asset = next(a for a in inv.assets if a.asset_id == f.asset_id)
        if (times < 0).any() or (asset.duration_sec is not None and (times > asset.duration_sec + 1e-8).any()):
            raise ValueError('Feature timestamps outside asset')
    modalities = {}
    for m in inv.motions:
        values = {l.modality for l in inv.links if l.motion_id == m.motion_id}
        if any(a.motion_id == m.motion_id or (a.motion_id is None and a.sequence_id == m.sequence_id) for a in inv.annotations):
            values.add('text')
        modalities[m.motion_id] = '+'.join(sorted(values)) or 'motion_only'
    return dict(valid=True, sequences=len(inv.sequences), motions=len(inv.motions), assets=len(inv.assets),
                features=len(inv.features), duration_sec=sum(m.num_frames/m.fps for m in inv.motions),
                sources=dict(Counter(s.source_dataset for s in inv.sequences)),
                modality_combinations=dict(Counter(modalities.values())),
                deep_checked_motions=min(deep_sample, len(inv.motions)),
                deep_checked_features=min(deep_sample, len(inv.features)))
