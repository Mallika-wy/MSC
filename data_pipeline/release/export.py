"""Native exporter: each track is stored once, no action/window duplication."""
from dataclasses import replace
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np

from data_pipeline.schema import Inventory
from data_pipeline.io import resolve_uri
from data_pipeline.motion.tracking import validate_qpos


def canonical_qpos(qpos):
    qpos = np.array(qpos, dtype=np.float32, copy=True)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError('Expected finite [T,36] qpos')
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if (norms < 1e-8).any():
        raise ValueError('Invalid root quaternion')
    qpos[:, 3:7] /= norms[:, None]
    for i in range(1, len(qpos)):
        if np.dot(qpos[i-1, 3:7], qpos[i, 3:7]) < 0:
            qpos[i, 3:7] *= -1
    return qpos


def export_release(inventory: Inventory, tracks: dict, output):
    """tracks maps motion_id to an array or NPZ path; quality reports are explicit.

    Publish atomically into a new directory. Inputs and previous releases are untouched.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if set(tracks) != {m.motion_id for m in inventory.motions}:
        raise ValueError('Track payloads must match motion records exactly')
    inventory.validate(output, release=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{output.name}-', dir=output.parent))
    try:
        (temp / 'motion').mkdir()
        (temp / 'reports').mkdir()
        motions = []
        for i, m in enumerate(inventory.motions):
            payload = tracks[m.motion_id]
            if isinstance(payload, (str, Path)):
                with np.load(payload, allow_pickle=False) as data:
                    payload = data['qpos_36'].copy()
            qpos = canonical_qpos(payload)
            times = np.arange(len(qpos), dtype=np.float64) / 30
            validate_qpos(qpos, times)
            if len(qpos) != m.num_frames:
                raise ValueError('Motion frame count mismatch')
            relative = f'motion/shard-{i:05d}.parquet'
            table = pa.table({'qpos36': pa.array(qpos.tolist(), type=pa.list_(pa.float32(), 36)),
                              'timestamp_sec': pa.array(times, type=pa.float64())})
            pq.write_table(table, temp / relative, compression='zstd', row_group_size=16384)
            quality = resolve_uri(output, m.quality_ref, inventory.external_roots)
            report = json.loads(quality.read_text())
            if report.get('accepted') is not True or report.get('motion_id') != m.motion_id:
                raise ValueError('Quality must explicitly accept matching motion_id')
            report_uri = f'reports/quality-{i:05d}.json'
            (temp / report_uri).write_text(json.dumps(report, indent=2))
            motions.append(replace(m, storage_uri=relative, row_start=0, quality_ref=report_uri))
        inv = replace(inventory, motions=motions, external_roots={k: v for k, v in inventory.external_roots.items() if k != "build_quality"})
        inv.write(temp, release=True)
        from .validate import validate_release
        report = validate_release(temp, deep_sample=len(motions))
        (temp / 'reports/dataset_summary.json').write_text(json.dumps(report, indent=2))
        os.rename(temp, output)
        return report
    except BaseException:
        shutil.rmtree(temp)
        raise


def read_motion(root, record):
    import pyarrow.parquet as pq
    # Read only intersecting row groups; do not decode every long sequence per sample.
    path = resolve_uri(root, record.storage_uri)
    file = pq.ParquetFile(path)
    start, end = record.row_start, record.row_start + record.num_frames
    if end > file.metadata.num_rows:
        raise ValueError('Motion row range exceeds shard')
    groups, offset, first_offset = [], 0, None
    for i in range(file.num_row_groups):
        n = file.metadata.row_group(i).num_rows
        if offset < end and offset + n > start:
            groups.append(i)
            if first_offset is None:
                first_offset = offset
        offset += n
    table = file.read_row_groups(groups).slice(start - first_offset, record.num_frames)
    return np.asarray(table['qpos36'].to_pylist(), dtype=np.float32), table['timestamp_sec'].to_numpy()
