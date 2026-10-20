"""Deterministic synthetic multimodal fixtures; never claims real encoder output."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from data_pipeline.schemas import (Inventory, SequenceRecord, MotionRecord, AssetRecord,
                                  ModalityLink, TextAnnotation, FeatureRecord)
from data_pipeline.release.export import export_release
from data_pipeline.release.package import package_release


def fixture_records(root):
    """Four modality combinations, plus two actors and two cameras sharing assets."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    inv = Inventory(external_roots={'fixture': str(root.resolve())}, provenance={'synthetic': True, 'fixture_version': '1'})
    tracks = {}
    audio_times = np.arange(1500, dtype=np.float64) / 75
    video_times = np.stack((np.arange(100) / 10, (np.arange(100) + 1) / 10), axis=1).astype(np.float64)
    assets = [('audio', 'song', audio_times, [1500, 8], 'point'),
              ('video', 'front', video_times, [100, 2, 6], 'interval'),
              ('video', 'side', video_times, [100, 1, 6], 'interval')]
    for modality, name, times, shape, semantics in assets:
        media = root / f'{name}.txt'
        media.write_text(f'Synthetic {modality} identity {name}; not decodable raw media.')
        checksum = hashlib.sha256(media.read_bytes()).hexdigest()
        aid = f'fixture:{name}'
        inv.assets.append(AssetRecord(aid, modality, f'root://fixture/{media.name}', aid, checksum,
            media.stat().st_size, 'synthetic fixture; no upstream media', 20. if modality == 'audio' else 10.,
            camera_id=name if modality == 'video' else None))
        np.save(root / f'{name}-tokens.npy', np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(np.float16) / 100)
        np.save(root / f'{name}-times.npy', times)
        inv.features.append(FeatureRecord(f'{aid}:mock', aid, f'synthetic_{modality}', 'synthetic-no-checkpoint',
            'synthetic-preprocess-v1', f'root://fixture/{name}-tokens.npy', f'root://fixture/{name}-times.npy',
            'float16', shape, semantics, 'synthetic', 'fixture-v1', checksum, 75. if modality == 'audio' else 10.))
    for task in ('text', 'audio', 'video', 'combined'):
        sid = f'fixture:{task}'
        inv.sequences.append(SequenceRecord(sid, 'fixture', task, 'fixture:shared-song-and-cameras', 'train', {'synthetic': True}))
        for actor in range(2 if task == 'combined' else 1):
            mid = f'{sid}:actor{actor}'
            qpos = np.zeros((90, 36), dtype=np.float32)
            qpos[:, 0] = np.arange(90) / 100 + actor
            qpos[:, 2:4] = [.8, 1.]
            tracks[mid] = qpos
            quality = root / f'{task}-{actor}-quality.json'
            quality.write_text(json.dumps({'motion_id': mid, 'accepted': True, 'synthetic': True, 'production_ready': False}))
            inv.motions.append(MotionRecord(mid, sid, f'actor{actor}', 'fixture:group', 'motion/pending.parquet',
                0, len(qpos), 30., 0., 'source_g1', 'not_applicable', 'not_applicable', f'root://fixture/{quality.name}'))
            if task in ('text', 'combined'):
                inv.annotations.append(TextAnnotation(f'{mid}:caption', sid, 'Synthetic: walk forward', 'en', 0., 2., 'synthetic', 1., mid))
                inv.annotations.append(TextAnnotation(f'{mid}:event', sid, 'Synthetic: lift arm', 'en', 2., 3., 'synthetic', .8, mid))
            for modality, names in [('audio', ['song']), ('video', ['front', 'side'] if task == 'combined' else ['front'])]:
                if task not in (modality, 'combined'):
                    continue
                for name in names:
                    inv.links.append(ModalityLink(f'{mid}:{name}', mid, f'fixture:{name}', modality,
                        0., 3., 1., 13.4 if modality == 'audio' else .2, 'verified', 'synthetic-affine-v1',
                        view_id=name if modality == 'video' else None, actor_id=f'actor{actor}'))
    return inv, tracks


def make_fixture(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix='msc-fixture-') as temp:
        root = Path(temp)
        inv, tracks = fixture_records(root / 'source')
        export_release(inv, tracks, root / 'reference')
        package_release(root / 'reference', output, mode='bundle')
    return output
