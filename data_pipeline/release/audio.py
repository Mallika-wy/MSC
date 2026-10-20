"""Turn accepted sample results into the existing MSC native release format."""
from pathlib import Path
import wave

import numpy as np

from data_pipeline.io import write_json
from data_pipeline.schema import (AssetRecord, FeatureRecord, Inventory, ModalityLink,
                                  MotionRecord, SequenceRecord)
from .export import export_release


def external_uri(inventory, path):
    """Give a local file an explicit root alias, shared by files in the same folder."""
    path = Path(path).resolve()
    roots = inventory.external_roots
    alias = next((key for key, value in roots.items() if value == str(path.parent)), None)
    if alias is None:
        alias = f'local_{len(roots)}'
        roots[alias] = str(path.parent)
    return f'root://{alias}/{path.name}'


def publish(dataset, results, output):
    """Publish reference/execution pairs and shared MERT features; omit rejected samples."""
    inventory = Inventory(provenance={'pipeline': 'audio_motion_v2', 'dataset': dataset})
    tracks, sequences, assets = {}, set(), set()
    for result in results:
        sequence_id = f"{dataset}:{result['sequence']}"
        sample_id = f"{dataset}:{result['name']}"
        if sequence_id not in sequences:
            inventory.sequences.append(SequenceRecord(
                sequence_id, dataset, result['sequence'], sequence_id, result['split'],
                {'source_motion': result['motion'], 'source_audio': result['audio']}))
            sequences.add(sequence_id)

        audio = result['stages']['audio']
        asset_id = f"audio:{audio['input_checksum']}"
        if asset_id not in assets:
            wav = Path(result['audio'])
            with wave.open(str(wav)) as stream:
                duration = stream.getnframes() / stream.getframerate()
                sample_rate = stream.getframerate()
            inventory.assets.append(AssetRecord(
                asset_id, 'audio', external_uri(inventory, wav), asset_id,
                audio['input_checksum'], wav.stat().st_size,
                f'{dataset} upstream terms; source media not redistributed', duration, sample_rate))
            inventory.features.append(FeatureRecord(
                f'{asset_id}:mert', asset_id, audio['encoder'], audio['checkpoint_hash'],
                audio['preprocess_hash'], external_uri(inventory, audio['values']),
                external_uri(inventory, audio['timestamps']), 'float16', audio['shape'], 'point',
                'last_transformer_layer', 'mert_chunked_v1', audio['input_checksum'], 75.))
            assets.add(asset_id)
        else:
            duration = next(a.duration_sec for a in inventory.assets if a.asset_id == asset_id)

        reference = result['stages']['reference']
        execution = result['stages']['tracking']
        with np.load(reference['path'], allow_pickle=False) as data:
            reference_frames = len(data['qpos_36'])
        with np.load(execution['path'], allow_pickle=False) as data:
            valid = data['executed_state_mask']
            if not np.array_equal(np.flatnonzero(valid), np.arange(1, len(valid))):
                raise ValueError('Executed labels must exclude exactly the initialization frame')
            label_path = Path(execution['path']).parent / 'labels.npz'
            np.savez_compressed(label_path, qpos_36=data['qpos_36'][valid])
            executed_frames = int(valid.sum())
        parent_id = f'{sample_id}:retarget_reference'
        specifications = [
            ('retarget_reference', reference['path'], reference_frames, 0., reference['quality'], None),
            ('tracker_executed', label_path, executed_frames, 1/30, execution, parent_id),
        ]
        for label, path, frames, origin, quality, parent in specifications:
            motion_id = f'{sample_id}:{label}'
            report_path = Path(path).parent / f'{label}_quality.json'
            write_json(report_path, {**quality, 'motion_id': motion_id, 'production_ready': False})
            inventory.motions.append(MotionRecord(
                motion_id, sequence_id, result['actor'], sequence_id, 'motion/pending.parquet',
                0, frames, 30., origin, label, reference['version'],
                execution['version'] if parent else 'not_applicable',
                external_uri(inventory, report_path), parent_motion_id=parent))
            tracks[motion_id] = Path(path)
            covered = min(frames / 30, duration - origin)
            if covered <= 0:
                raise ValueError('No audio coverage for motion')
            inventory.links.append(ModalityLink(
                f'{motion_id}:audio', motion_id, asset_id, 'audio', 0., covered, 1., origin,
                'estimated', 'matched source excerpt; offset not independently calibrated',
                actor_id=result['actor']))
    return export_release(inventory, tracks, Path(output) / 'release')
