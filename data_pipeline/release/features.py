"""Shared feature shards: one row per asset feature, with explicit shape metadata."""
import numpy as np
from data_pipeline.io import resolve_uri


def read_feature(root, feature, external_roots=None):
    import pyarrow.parquet as pq
    path = resolve_uri(root, feature.feature_uri, external_roots)
    if path.suffix != '.parquet':
        return (np.load(path, mmap_mode='r', allow_pickle=False),
                np.load(resolve_uri(root, feature.timestamp_uri, external_roots), allow_pickle=False))
    table = pq.read_table(path, filters=[('feature_id', '=', feature.feature_id)])
    if len(table) != 1:
        raise ValueError('Expected exactly one feature row')
    if table['shape'][0].as_py() != feature.shape or table['dtype'][0].as_py() != feature.dtype:
        raise ValueError('Packed feature metadata mismatch')
    values = table['values'][0].values.to_numpy(zero_copy_only=False).reshape(feature.shape)
    times = table['timestamps'][0].values.to_numpy(zero_copy_only=False).reshape(table['timestamp_shape'][0].as_py())
    return values, times


def feature_schema():
    import pyarrow as pa
    return pa.schema([('feature_id', pa.string()), ('shape', pa.list_(pa.int64())),
        ('dtype', pa.string()), ('values', pa.large_list(pa.float16())),
        ('timestamps', pa.list_(pa.float64())), ('timestamp_shape', pa.list_(pa.int64()))])


def feature_row(identity, values, times):
    return dict(feature_id=identity, shape=list(values.shape), dtype=str(values.dtype),
                values=values.reshape(-1), timestamps=times.reshape(-1),
                timestamp_shape=list(times.shape))


class FeatureShardWriter:
    def __init__(self, root, max_bytes=256*1024*1024):
        from pathlib import Path
        self.root = Path(root)
        self.limit = max_bytes
        self.writers, self.sizes, self.indices = {}, {}, {}

    def add(self, modality, identity, values, times):
        import pyarrow as pa
        import pyarrow.parquet as pq
        size = values.nbytes + times.nbytes
        if modality not in self.writers or self.sizes[modality]+size > self.limit:
            if modality in self.writers:
                self.writers.pop(modality).close()
            self.indices[modality] = self.indices.get(modality,-1)+1
            folder = self.root / modality
            folder.mkdir(parents=True,exist_ok=True)
            path = folder / f'shard-{self.indices[modality]:05d}.parquet'
            self.writers[modality] = pq.ParquetWriter(path,feature_schema(),compression='zstd')
            self.sizes[modality] = 0
        path = self.root / modality / f'shard-{self.indices[modality]:05d}.parquet'
        self.writers[modality].write_table(pa.Table.from_pylist([feature_row(identity,values,times)],schema=feature_schema()))
        self.sizes[modality] += size
        return path

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
