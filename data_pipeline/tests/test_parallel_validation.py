"""Shard reuse, cross-row-group contracts and spawned validation failures."""
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.release.validate import _validate_motion_shard, validate_release


def write_rows(path, qpos, times, row_group_size=7):
    table = pa.table({
        'qpos36': pa.FixedSizeListArray.from_arrays(pa.array(qpos.reshape(-1)), 36),
        'timestamp_sec': pa.array(times),
    })
    pq.write_table(table, path, row_group_size=row_group_size)


def test_each_needed_group_read_once_with_gaps_and_crossing_motions(tmp_path, monkeypatch):
    ranges = [(4, 8, 30.), (26, 12, 30.), (38, 8, 30.)]
    qpos = np.zeros((50, 36), dtype=np.float32)
    qpos[:, 3] = 1
    times = np.zeros(50, dtype=np.float64)
    for i, (start, frames, fps) in enumerate(ranges):
        qpos[start:start+frames, 3] = (-1)**i  # Sign changes between motions are valid.
        times[start:start+frames] = np.arange(frames)/fps
    path = tmp_path/'shard.parquet'
    write_rows(path, qpos, times)
    original = pq.ParquetFile
    reads = []
    class CountingFile:
        def __init__(self, path):
            self.file = original(path)
            self.metadata = self.file.metadata
            self.num_row_groups = self.file.num_row_groups
        def read_row_group(self, group, **kwargs):
            reads.append(group)
            return self.file.read_row_group(group, **kwargs)
        def close(self):
            self.file.close()
    monkeypatch.setattr(pq, 'ParquetFile', CountingFile)
    _validate_motion_shard((path, ranges[::-1]))
    assert Counter(reads) == Counter([0, 1, 3, 4, 5, 6])


@pytest.mark.parametrize('corruption,match', [
    ('nan', 'Invalid motion values'),
    ('quaternion_norm', 'unit wxyz'),
    ('duplicate_time', 'Invalid motion values'),
    ('time_origin', 'local, uniform'),
    ('quaternion_sign', 'sign discontinuity'),
])
def test_corruption_across_row_group_boundary(tmp_path, corruption, match):
    qpos = np.zeros((15, 36), dtype=np.float32)
    qpos[:, 3] = 1
    times = np.arange(15, dtype=np.float64)/30
    if corruption == 'nan':
        qpos[7, 0] = np.nan
    elif corruption == 'quaternion_norm':
        qpos[7, 3] = 2
    elif corruption == 'duplicate_time':
        times[7] = times[6]
    elif corruption == 'time_origin':
        times += .1
    else:
        qpos[7:, 3] = -1
    path = tmp_path/'shard.parquet'
    write_rows(path, qpos, times)
    with pytest.raises(ValueError, match=match):
        _validate_motion_shard((path, [(0, 15, 30.)]))


def test_parallel_report_matches_serial_and_propagates_corruption(tmp_path):
    from data_pipeline.schema import Inventory, SequenceRecord, MotionRecord
    from data_pipeline.release.export import ReleaseWriter
    root = tmp_path/'release'
    motions = [MotionRecord(f'm{i}', 's', 'person', 'g', 'pending.parquet', 0, 3, 30., 0.,
                           'source_g1', 'not_applicable', 'not_applicable', 'pending.json') for i in range(260)]
    inv = Inventory(sequences=[SequenceRecord('s', 'fixture', 's', 'g', 'train', {'fixture': True})], motions=motions)
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 3] = 1
    writer = ReleaseWriter(root, max_shard_frames=400)
    writer.add(inv, {m.motion_id: qpos for m in motions},
               qualities={m.motion_id: {'motion_id': m.motion_id, 'accepted': True} for m in motions})
    writer.publish()
    assert validate_release(root, 260, workers=2) == validate_release(root, 260, workers=1)
    second = sorted((root/'motion/train').glob('*.parquet'))[1]
    table = pq.read_table(second)
    times = table['timestamp_sec'].to_numpy().copy()
    times[0] = -.1
    pq.write_table(table.set_column(1, 'timestamp_sec', pa.array(times)), second)
    with pytest.raises(ValueError, match='local, uniform'):
        validate_release(root, 260, workers=2)
