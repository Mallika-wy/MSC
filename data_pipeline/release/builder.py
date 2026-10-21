"""Assemble processor artifacts into the single MSC six-table release."""
from pathlib import Path

from data_pipeline.schema import (Inventory, SequenceRecord, MotionRecord, AssetRecord,
                                  ModalityLink, TextAnnotation, FeatureRecord)
from data_pipeline.storage.store import load_arrays, fingerprint
from data_pipeline.io import external_uri
from .export import ReleaseWriter
from .features import FeatureShardWriter


class ReleaseBuilder:
    def __init__(self, output, config=None):
        self.output = Path(output)
        self.config = config or {}

    def publish(self, results):
        if self.output.exists():
            raise FileExistsError(self.output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Prepared reports/features are consolidated and survive publication as
        # external feature references. Re-publishing uses the work ledger instead.
        prepared = self.output.parent / 'cache/prepared'
        prepared.mkdir(parents=True, exist_ok=True)
        feature_root = prepared / ('features-' + fingerprint(str(self.output.resolve()))[:16])
        inv = Inventory(provenance={'pipeline': 'dataset_centric_v3', 'release_policy': self.config,
                                    'train_default': self.config.get('train_default'),
                                    'quality_storage': 'parquet_by_motion_id_v1',
                                    'feature_storage': 'parquet_by_feature_id_v1'})
        tracks, sequences, assets, features = {}, set(), {}, set()
        from collections import Counter
        processing = Counter()
        reasons = Counter()
        audit_groups, audit = {}, []
        feature_writer = FeatureShardWriter(feature_root)
        roots, provenance = inv.external_roots, inv.provenance
        provenance['split_audit'] = audit
        writer = ReleaseWriter(self.output, max_shard_frames=self.config.get('max_shard_frames', 1_000_000),
                               external_roots=roots, provenance=provenance)
        if self.config.get('split_policy') == 'song_disjoint':
            from .splits import song_disjoint
            results = song_disjoint(results, self.config)
        try:
            for result in results:
                processing[result['state']] += 1
                for motion in result.get('motions', []):
                    processing['motion_' + motion['state']] += 1
                    for artifact in motion.get('artifacts', []):
                        reasons.update(artifact['quality'].get('rejection_reasons', []))
                if result['state'] not in ('accepted', 'partial'):
                    continue
                inv = Inventory(external_roots=roots, provenance=provenance)
                tracks, qualities = {}, {}
                source = result['sequence']
                dataset = source['source_dataset']
                sid = f"{dataset}:{source['sequence_id']}"
                split = source['split']
                song_ids = sorted({a['checksum'] for a in result.get('assets', []) if a['modality'] == 'audio'})
                for group in [source['split_group_id'], *song_ids]:
                    old = audit_groups.setdefault(group, split)
                    if old != split:
                        audit.append({'group': group, 'splits': sorted({old, split})})
                if sid in sequences:
                    raise ValueError(f'Duplicate source sequence: {sid}')
                sequences.add(sid)
                inv.sequences.append(SequenceRecord(sid, dataset, source['sequence_id'],
                    source['split_group_id'], split, source.get('provenance') or {'source_root': dataset}))
                asset_map = {}
                for asset in result.get('assets', []):
                    aid = f"{asset['modality']}:{asset['checksum']}"
                    if asset['modality'] == 'video':
                        aid += ':' + fingerprint([dataset, asset['source_id'], asset.get('camera_id'), asset.get('person_track_id')])[:16]
                    asset_map[asset['source_id']] = aid
                    if aid not in assets:
                        path = Path(asset['path'])
                        record = AssetRecord(aid, asset['modality'], external_uri(inv, path), f"{dataset}:{asset['source_id']}" if asset['modality'] == 'video' else aid,
                            asset['checksum'], path.stat().st_size, f'{dataset} upstream terms; source media not redistributed',
                            asset.get('duration_sec'), asset.get('sample_rate_hz'), asset.get('frame_rate'),
                            camera_id=asset.get('camera_id'), person_track_id=asset.get('person_track_id'))
                        inv.assets.append(record); assets[aid] = record
                    feature = asset.get('feature')
                    if feature:
                        meta = feature['metadata']
                        if meta.get('input_checksum') != asset['checksum']:
                            raise ValueError('Feature input checksum does not match asset')
                        fid = f"{aid}:{fingerprint([meta['encoder'], meta['checkpoint_hash'], meta['preprocess_hash']])}"
                        if fid not in features:
                            arrays = load_arrays(feature['arrays'])
                            values, times = arrays['values'], arrays['timestamps']
                            feature_path = feature_writer.add(asset['modality'], fid, values, times)
                            uri = external_uri(inv, feature_path)
                            inv.features.append(FeatureRecord(fid, aid, meta['encoder'], meta['checkpoint_hash'],
                                meta['preprocess_hash'], uri, uri, str(values.dtype), list(values.shape), 'point',
                                'last_transformer_layer', 'mert_chunked_v1', asset['checksum'], 75.))
                            features.add(fid)
                for motion_result in result['motions']:
                    if motion_result['state'] != 'accepted':
                        continue
                    source_motion = motion_result['source']
                    published = self.config.get('publish_labels')
                    for artifact in motion_result['artifacts']:
                        if not artifact['quality'].get('accepted') or (published is not None and artifact['label_type'] not in published):
                            continue
                        label = artifact['label_type']
                        mid = f"{dataset}:{source_motion['motion_id']}:{label}"
                        parent = f"{dataset}:{source_motion['motion_id']}:{artifact['parent_label']}" if artifact.get('parent_label') else None
                        quality = {**artifact['quality'], 'motion_id': mid, 'production_ready': False}
                        qualities[mid] = quality
                        inv.motions.append(MotionRecord(mid, sid, source_motion['actor_id'], source['split_group_id'],
                            'motion/pending.parquet', 0, artifact['num_frames'], 30., artifact['time_origin_sec'], label,
                            next((a['version'] for a in motion_result['artifacts'] if a['label_type'] in ('retarget_reference', 'source_g1')), 'not_applicable'),
                            artifact['version'] if label == 'tracker_executed' else 'not_applicable',
                            'reports/pending.parquet', parent_motion_id=parent))
                        reference = artifact['arrays']
                        tracks[mid] = lambda reference=reference: load_arrays(reference)['qpos_36']
                        origin, duration = artifact['time_origin_sec'], artifact['num_frames']/30
                        for i, text in enumerate(source.get('texts', [])):
                            if text.get('actor_id') not in (None, source_motion['actor_id']):
                                continue
                            lo, hi = max(0., text['start_sec']-origin), min(duration, text['end_sec']-origin)
                            if hi > lo:
                                inv.annotations.append(TextAnnotation(f'{mid}:text:{i}', sid, text['text'], text['language'],
                                    lo, hi, text['source'], 1., mid))
                        for alignment in source.get('alignments', []):
                            if alignment['motion_id'] != source_motion['motion_id']:
                                continue
                            aid = asset_map.get(alignment['asset_id'])
                            status = alignment['status']
                            if aid is None or status == 'unknown' or (self.config.get('alignment_policy') == 'verified_only' and status != 'verified'):
                                continue
                            scale = alignment['time_scale']
                            if scale <= 0:
                                raise ValueError('Alignment requires positive scale')
                            offset = alignment['time_offset_sec'] + scale*origin
                            asset = assets[aid]
                            lo = max(0., -offset/scale)
                            hi = min(duration, (asset.duration_sec-offset)/scale) if asset.duration_sec is not None else duration
                            if hi > lo:
                                inv.links.append(ModalityLink(f'{mid}:{fingerprint(alignment)[:16]}', mid, aid, asset.modality,
                                    lo, hi, scale, offset, status, alignment['source'], view_id=asset.camera_id, actor_id=source_motion['actor_id']))
                writer.add(inv, tracks, qualities=qualities)
        except BaseException:
            writer.abort()
            raise
        finally:
            feature_writer.close()
        if not writer.count:
            writer.abort()
            return None
        provenance['processing_summary'] = dict(processing)
        provenance['rejection_reasons'] = dict(reasons)
        return writer.publish()
