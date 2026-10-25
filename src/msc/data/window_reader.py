"""Read a batch of 35-frame windows, decoding each Parquet row group once."""
from collections import OrderedDict
import numpy as np
import pyarrow.parquet as pq
from data_pipeline.io import resolve_uri


class MotionWindowReader:
    def __init__(self, roots, cache):
        self.roots = roots
        self.cache = cache
        self.files = OrderedDict()

    def open(self, release_index, uri):
        key = (release_index, uri)
        if key in self.files:
            self.files.move_to_end(key)
            return self.files[key]
        file = pq.ParquetFile(resolve_uri(self.roots[release_index], uri))
        offsets = np.r_[0, np.cumsum([file.metadata.row_group(i).num_rows
                                     for i in range(file.num_row_groups)])]
        self.files[key] = (file, offsets)
        # A small fixed motion subset can still span many release shards.
        if len(self.files) > 128:
            _, (old, _) = self.files.popitem(last=False)
            old.close()
        return file, offsets

    @staticmethod
    def decode(file, group):
        column = file.read_row_group(group, columns=['qpos36'], use_threads=False)['qpos36'].combine_chunks()
        return (column.values.to_numpy().reshape(-1, 36),)

    def read(self, windows):
        result = np.empty((len(windows), 35, 36), dtype=np.float32)
        for i, (release_index, motion, _, frame, _) in enumerate(windows):
            file, offsets = self.open(release_index, motion.storage_uri)
            start = motion.row_start + frame - 5
            end = start + 35
            first = np.searchsorted(offsets, start, side='right') - 1
            last = np.searchsorted(offsets, end, side='left') - 1
            for group in range(first, last + 1):
                key = ('row_group', release_index, motion.storage_uri, group)
                values, = self.cache.get(key, lambda: self.decode(file, group))
                lo, hi = max(start, offsets[group]), min(end, offsets[group+1])
                result[i, lo-start:hi-start] = values[lo-offsets[group]:hi-offsets[group]]
        return result
