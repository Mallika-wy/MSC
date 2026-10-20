"""Read local official LeRobot metadata without downloading or windowing motion."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from data_pipeline.schemas import Inventory, SequenceRecord
from data_pipeline.schemas.records import SourceMotion


def discover(root, out, *, limit=None, dry_run=False, **_):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    if Path(out).resolve().is_relative_to(root):
        raise ValueError('Inventory output must be outside raw source root')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    info_bytes = (root / 'meta/info.json').read_bytes()
    info = json.loads(info_bytes)
    if info['fps'] != 30 or info['features']['observation.state']['shape'] != [36]:
        raise ValueError('Expected official 30 Hz G1 qpos36')
    inv = Inventory(external_roots={'omg': str(root)}, provenance={'adapter': 'omg_lerobot_v1',
        'info_sha256': hashlib.sha256(info_bytes).hexdigest(), 'audio_policy': 'no raw audio inferred from legacy features'})
    raw = []
    for file in sorted((root / 'meta/episodes').glob('chunk-*/*.parquet')):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                episode = int(row['episode_index'])
                split = row.get('omg/split')
                if not split:
                    for name, value in info['splits'].items():
                        start, end = map(int, value.split(':'))
                        if start <= episode < end:
                            split = name
                            break
                if not split:
                    raise ValueError(f'No official split for episode {episode}')
                if int(row['dataset_to_index']) - int(row['dataset_from_index']) != int(row['length']):
                    raise ValueError('OMG episode row range/length mismatch')
                sid = f'omg:episode:{episode}'
                source = str(row.get('omg/source_id', episode))
                dataset = row.get('omg/dataset', 'unknown')
                group_dataset = dataset.removesuffix('_train').removesuffix('_val').removesuffix('_test')
                # Official segment suffixes are subdivisions, not independent split groups.
                group = source.split('__seg')[0].split('__part')[0]
                inv.sequences.append(SequenceRecord(sid, 'omg', source, f'omg:{group_dataset}:{group}', split,
                    source_provenance={'adapter': 'omg_lerobot_v1', 'episode_index': episode,
                        'info_sha256': inv.provenance['info_sha256'], 'license': 'official OMG upstream source terms',
                        'official_split': split, 'source_dataset': dataset}))
                raw.append(SourceMotion(f'{sid}:g1', sid, 'root://omg/data', 'omg_lerobot_g1', 30.,
                    metadata={'row_start': int(row['dataset_from_index']), 'num_frames': int(row['length']),
                              'captions': row.get('tasks') or [], 'source_start_frame': row.get('omg/source_start_frame', 0)}))
                if limit is not None and len(raw) >= limit:
                    break
            if limit is not None and len(raw) >= limit:
                break
        if limit is not None and len(raw) >= limit:
            break
    if not raw:
        raise ValueError('No local OMG episodes found')
    inv.validate()
    if not dry_run:
        inv.write(out)
        (Path(out) / 'meta/source_motions.json').write_text(json.dumps([asdict(r) for r in raw], indent=2))
    return {'sequences': len(inv.sequences), 'source_motions': len(raw), 'output': str(out)}
