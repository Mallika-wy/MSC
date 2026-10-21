"""Explicit supplemental assets; filenames never imply calibrated synchronization."""
from dataclasses import replace
from pathlib import Path
from .base import SourceAsset, SourceAlignment


def attach_assets(sequences, manifest):
    import pyarrow.parquet as pq
    manifest = Path(manifest).resolve()
    rows = {}
    for row in pq.read_table(manifest).to_pylist():
        rows.setdefault(row['sequence_id'], []).append(row)
    for sequence in sequences:
        assets = {a.asset_id: a for a in sequence.assets}
        alignments = list(sequence.alignments)
        motion_ids = {m.motion_id for m in sequence.motions}
        for row in rows.get(sequence.sequence_id, []):
            if row['motion_id'] not in motion_ids:
                raise ValueError('Supplemental asset refers to an unknown motion')
            if row['modality'] not in ('audio', 'video'):
                raise ValueError('Supplemental assets must be audio or video')
            path = (manifest.parent / row['path']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            asset = SourceAsset(row['asset_id'], path, row['modality'],
                                {k:row[k] for k in ('camera_id','track_id') if row.get(k) is not None})
            if asset.asset_id in assets and assets[asset.asset_id] != asset:
                raise ValueError('Conflicting supplemental asset identity')
            assets[asset.asset_id] = asset
            alignments.append(SourceAlignment(row['motion_id'], asset.asset_id,
                float(row['time_scale']), float(row['time_offset_sec']), row['alignment_status'], row['alignment_source']))
        yield replace(sequence, assets=list(assets.values()), alignments=alignments)
