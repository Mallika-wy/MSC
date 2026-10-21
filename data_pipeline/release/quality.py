"""Read legacy JSON or shared Parquet quality reports by motion identity."""
import json
from pathlib import Path
from data_pipeline.io import resolve_uri


def read_quality(root, reference, motion_id, external_roots=None, cache=None):
    path = resolve_uri(root, reference, external_roots)
    if path.suffix != '.parquet':
        return json.loads(path.read_text())
    import pyarrow.parquet as pq
    if cache is not None:
        if path not in cache:
            rows = pq.read_table(path).to_pylist()
            mapping = {row['motion_id']: row for row in rows}
            if len(mapping) != len(rows):
                raise ValueError('Duplicate quality motion_id')
            cache[path] = mapping
        row = cache[path].get(motion_id)
    else:
        rows = pq.read_table(path, filters=[('motion_id', '=', motion_id)]).to_pylist()
        if len(rows) != 1:
            raise ValueError('Expected exactly one quality record')
        row = rows[0]
    if row is None:
        raise ValueError('Missing quality record')
    report = json.loads(row['report_json'])
    if row['accepted'] != report.get('accepted') or row['motion_id'] != report.get('motion_id'):
        raise ValueError('Quality record identity/status mismatch')
    return report
