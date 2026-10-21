"""Atomic streaming release writer with shared split shards and bounded buffers."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import os
import shutil
import tempfile

import numpy as np

from data_pipeline.schema import Inventory
from data_pipeline.schema.manifest import arrow_schema
from data_pipeline.schema.records import TABLES
from data_pipeline.io import resolve_uri
from data_pipeline.motion.validation import validate_qpos


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


class ReleaseWriter:
    """Incrementally write metadata/arrays; publication happens only after validation."""
    def __init__(self, output, *, max_shard_frames=1_000_000, external_roots=None, provenance=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if type(max_shard_frames) is not int or max_shard_frames < 2:
            raise ValueError('max_shard_frames must be an integer >= 2')
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temp = Path(tempfile.mkdtemp(prefix=f'.{self.output.name}-', dir=self.output.parent))
        self.inventory = Inventory(external_roots=external_roots if external_roots is not None else {},
                                   provenance=provenance if provenance is not None else {})
        self.limit = max_shard_frames
        self.motion_schema = pa.schema([('qpos36', pa.list_(pa.float32(), 36)), ('timestamp_sec', pa.float64())])
        self.quality_schema = pa.schema([('motion_id', pa.string()), ('accepted', pa.bool_()), ('report_json', pa.string())])
        (self.temp / 'meta').mkdir(); (self.temp / 'reports').mkdir(); (self.temp / 'motion').mkdir()
        self.metadata_writers = {name: pq.ParquetWriter(self.temp / 'meta' / f'{name}.parquet', arrow_schema(cls), compression='zstd') for name,cls in TABLES.items()}
        self.metadata_buffers = {name: [] for name in TABLES}
        self.quality_writer = pq.ParquetWriter(self.temp / 'reports/quality-00000.parquet', self.quality_schema, compression='zstd')
        self.qualities = []
        self.writers, self.offsets, self.indices, self.buffers, self.buffered = {}, {}, {}, {}, {}
        self.count = 0
        self.closed = False

    def _flush_motion(self, split):
        import pyarrow as pa
        if self.buffers.get(split):
            self.writers[split].write_table(pa.concat_tables(self.buffers[split]), row_group_size=16384)
            self.buffers[split] = []; self.buffered[split] = 0

    def _flush_metadata(self, name):
        import pyarrow as pa
        if self.metadata_buffers[name]:
            self.metadata_writers[name].write_table(pa.Table.from_pylist(self.metadata_buffers[name], schema=arrow_schema(TABLES[name])))
            self.metadata_buffers[name].clear()

    def _quality(self, row):
        import pyarrow as pa
        if row is not None:
            self.qualities.append(row)
        if len(self.qualities) >= 2048 or row is None:
            if self.qualities:
                self.quality_writer.write_table(pa.Table.from_pylist(self.qualities, schema=self.quality_schema))
                self.qualities.clear()

    def add(self, inventory, tracks, *, qualities=None):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from .quality import read_quality
        if set(tracks) != {m.motion_id for m in inventory.motions}:
            raise ValueError('Track payloads must match motion records exactly')
        if any(f.feature_uri.endswith('.parquet') for f in inventory.features):
            self.inventory.provenance['feature_storage'] = 'parquet_by_feature_id_v1'
        splits = {s.sequence_id: s.split for s in inventory.sequences}
        quality_cache = {}
        motions = []
        for m in inventory.motions:
            split = splits[m.sequence_id]
            if split not in ('train','val','test'):
                raise ValueError('Invalid/unassigned split')
            payload = tracks[m.motion_id]
            if callable(payload):
                payload = payload()
            if isinstance(payload, (str,Path)):
                with np.load(payload, allow_pickle=False) as data:
                    payload = data['qpos_36'].copy()
            qpos = canonical_qpos(payload)
            times = np.arange(len(qpos), dtype=np.float64)/30
            validate_qpos(qpos,times)
            if len(qpos) != m.num_frames:
                raise ValueError('Motion frame count mismatch')
            if split not in self.writers or (self.offsets[split] and self.offsets[split]+len(qpos)>self.limit):
                if split in self.writers:
                    self._flush_motion(split); self.writers.pop(split).close()
                self.indices[split] = self.indices.get(split,-1)+1
                (self.temp / 'motion' / split).mkdir(exist_ok=True)
                relative = f'motion/{split}/shard-{self.indices[split]:05d}.parquet'
                self.writers[split] = pq.ParquetWriter(self.temp / relative, self.motion_schema, compression='zstd')
                self.offsets[split] = 0
            relative = f'motion/{split}/shard-{self.indices[split]:05d}.parquet'
            table = pa.Table.from_arrays([pa.FixedSizeListArray.from_arrays(pa.array(qpos.reshape(-1)),36),pa.array(times)],schema=self.motion_schema)
            self.buffers.setdefault(split,[]).append(table)
            self.buffered[split] = self.buffered.get(split,0)+len(qpos)
            if self.buffered[split] >= 16384:
                self._flush_motion(split)
            quality = qualities[m.motion_id] if qualities is not None else read_quality(
                self.output,m.quality_ref,m.motion_id,inventory.external_roots,quality_cache)
            if quality.get('accepted') is not True or quality.get('motion_id') != m.motion_id:
                raise ValueError('Quality must explicitly accept matching motion_id')
            self._quality(dict(motion_id=m.motion_id,accepted=True,report_json=json.dumps(quality,ensure_ascii=False)))
            motions.append(replace(m,storage_uri=relative,row_start=self.offsets[split],quality_ref='reports/quality-00000.parquet'))
            self.offsets[split] += len(qpos); self.count += 1
        for name in TABLES:
            for record in (motions if name=='motions' else getattr(inventory,name)):
                row = asdict(record)
                if name=='sequences':
                    row['source_provenance'] = json.dumps(row['source_provenance'],sort_keys=True)
                self.metadata_buffers[name].append(row)
                if len(self.metadata_buffers[name])>=2048:
                    self._flush_metadata(name)

    def close(self):
        if self.closed:
            return
        try:
            for split,writer in self.writers.items():
                self._flush_motion(split); writer.close()
            for name,writer in self.metadata_writers.items():
                self._flush_metadata(name); writer.close()
            self._quality(None); self.quality_writer.close()
        finally:
            self.closed = True

    def abort(self):
        try:
            self.close()
        finally:
            shutil.rmtree(self.temp,ignore_errors=True)

    def publish(self):
        try:
            self.close()
            self.inventory.write_manifest(self.temp,release=True)
            from .validate import validate_release
            report = validate_release(self.temp,deep_sample=min(32,self.count))
            report['processing_summary'] = self.inventory.provenance.get('processing_summary', {})
            report['rejection_reasons'] = self.inventory.provenance.get('rejection_reasons', {})
            (self.temp / 'reports/dataset_summary.json').write_text(json.dumps(report,indent=2))
            if self.output.exists():
                raise FileExistsError(self.output)
            os.rename(self.temp,self.output)
            return report
        except BaseException:
            self.abort()
            raise


def export_release(inventory: Inventory, tracks: dict, output, *, max_shard_frames=1_000_000):
    inventory.validate(output,release=True)
    writer = ReleaseWriter(output,max_shard_frames=max_shard_frames,
                           external_roots=dict(inventory.external_roots),provenance=dict(inventory.provenance))
    try:
        writer.add(inventory,tracks)
        return writer.publish()
    except BaseException:
        writer.abort()
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
