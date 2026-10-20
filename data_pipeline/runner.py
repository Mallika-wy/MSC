"""Independent source/motion/text stages with atomic native release publication."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np

from data_pipeline.schemas import Inventory, MotionRecord, TextAnnotation
from data_pipeline.io import write_json
from data_pipeline.schemas.records import SourceMotion
from data_pipeline.stages.motion import decode
from data_pipeline.stages.motion import validate_qpos
from data_pipeline.release.export import export_release


def build(config, *, stages=('motion', 'text'), limit=None, dry_run=False):
    import yaml
    config_path = Path(config).resolve()
    settings = yaml.safe_load(config_path.read_text())
    def path(value):
        return (config_path.parent / value).resolve()
    source_root, output = path(settings['inventory']), path(settings['output'])
    work = path(settings.get('work', str(output.parent / f'.{output.name}-stages')))
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError('Release output must be outside inventory')
    if output.exists() and not dry_run:
        raise FileExistsError(output)
    inventory = Inventory.read(source_root)
    for raw_root in inventory.external_roots.values():
        if output.is_relative_to(Path(raw_root).resolve()) or work.is_relative_to(Path(raw_root).resolve()):
            raise ValueError('Output/work must be outside raw source roots')
    raw = [SourceMotion(**r) for r in json.loads((source_root / 'meta/source_motions.json').read_text())]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        raw = raw[:limit]
    if not raw:
        raise ValueError('No source motions')
    stages = set(stages)
    if not stages <= {'motion', 'text'} or 'motion' not in stages:
        raise ValueError('Current build supports motion and optional text; audio/video feature stages are pending')
    tracker = settings.get('tracker', {})
    if tracker.get('enabled') and not tracker.get('version'):
        raise ValueError('Enabled tracker requires explicit controller version/revision')
    if tracker.get('enabled') and any(r.format in ('omg_lerobot_g1', 'g1_qpos_wxyz') for r in raw):
        raise ValueError('Tracker applies to retargeted human motion; native G1 sources bypass this stage')
    if dry_run:
        return {'source_motions': len(raw), 'stages': sorted(stages), 'output': str(output),
                'tracker_enabled': bool(tracker.get('enabled'))}
    sequences = {s.sequence_id: s for s in inventory.sequences}
    selected = {r.sequence_id for r in raw}
    assignments = settings.get('split_assignments', {})
    def assigned_split(sequence):
        return assignments.get(sequence.sequence_id, assignments.get(sequence.split_group_id, sequence.split))
    for s in inventory.sequences:
        if s.source_dataset == 'omg' and assigned_split(s) != s.split:
            raise ValueError('Official OMG splits cannot be reassigned')
    inv = Inventory(sequences=[replace(s, split=assigned_split(s))
        for s in inventory.sequences if s.sequence_id in selected], external_roots=dict(inventory.external_roots),
        provenance={**inventory.provenance, 'build_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    'tracker': tracker, 'stages': sorted(stages), 'processing': {}})
    tracks = {}
    with tempfile.TemporaryDirectory(prefix='msc-quality-') as temp:
        inv.external_roots['build_quality'] = temp
        def add_track(source, qpos, origin, label, retarget_version, tracker_version, report, parent=None):
            mid = f'{source.source_motion_id}:{label}'
            report = {**report, 'motion_id': mid, 'source_motion_id': source.source_motion_id, 'production_ready': False}
            if report.get('accepted') is not True:
                raise ValueError(f'Quality rejected {mid}: {report.get("rejection_reasons")}')
            name = hashlib.sha256(mid.encode()).hexdigest()
            (Path(temp) / f'{name}.json').write_text(json.dumps(report, indent=2))
            inv.motions.append(MotionRecord(mid, source.sequence_id, source.actor_id,
                sequences[source.sequence_id].split_group_id, f'motion/pending-{name}.parquet', 0, len(qpos),
                30., origin, label, retarget_version, tracker_version, f'root://build_quality/{name}.json', parent_motion_id=parent))
            tracks[mid] = qpos
            if 'text' in stages:
                for j, text in enumerate(source.metadata.get('captions', [])):
                    if str(text).strip():
                        inv.annotations.append(TextAnnotation(f'{mid}:text:{j}', source.sequence_id, str(text), 'en',
                            0., len(qpos)/30, 'official_OMG_episode_tasks', 1., mid))
            return mid
        for source in raw:
            if source.format in ('omg_lerobot_g1', 'g1_qpos_wxyz'):
                qpos, times = decode(source, inventory, source_root)
                add_track(source, qpos, source.time_origin_sec + float(times[0]), 'source_g1',
                    'not_applicable', 'not_applicable', {'accepted': True, 'scope': 'representation_and_timeline_only'})
                continue
            from data_pipeline.stages.retarget import retarget
            from data_pipeline.stages.arm_twist import screen_reference
            retarget_settings = dict(settings['retarget'])
            for key in ('smpl_model', 'gmr_root'):
                if retarget_settings.get(key):
                    retarget_settings[key] = str(path(retarget_settings[key]))
            reference, version, fingerprint = retarget(source, inventory, source_root, retarget_settings, work / 'retarget')
            with np.load(reference, allow_pickle=False) as data:
                qpos, times = data['qpos_36'].copy(), data['source_timestamps'].copy()
            validate_qpos(qpos, times)
            if not np.allclose(times, np.arange(len(times))/30, atol=1e-8, rtol=0):
                raise ValueError('Reference timeline must start at zero and be uniform 30 Hz')
            qc = screen_reference(reference, settings.get('quality', {}))
            qc['scope'] = 'retarget_reference_arm_twist_only'
            write_json(reference.parent / 'reference_quality.json', {**qc, 'motion_id': f'{source.source_motion_id}:retarget_reference'})
            inv.provenance['processing'][source.source_motion_id] = fingerprint
            mid = add_track(source, qpos, source.time_origin_sec, 'retarget_reference', version, 'not_applicable', qc)
            if tracker.get('enabled'):
                from data_pipeline.stages.motion import execute_and_screen
                from data_pipeline.stages.retarget import sha256_file
                execution = dict(tracker)
                execution['onnx'] = str(path(execution['onnx']))
                identity = {'retarget_version': version, 'tracker': execution,
                            'stage_sha256': sha256_file(Path(__file__).parent / 'stages/motion.py'),
                            'quality_sha256': sha256_file(Path(__file__).parent / 'stages/motion_quality.py'),
                            'onnx_sha256': sha256_file(execution['onnx']), 'quality': settings.get('quality', {})}
                cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                directory = work / 'tracking' / cache_key
                directory.mkdir(parents=True, exist_ok=True)
                quality = execute_and_screen(reference, {'execution': execution, 'quality': settings.get('quality', {})},
                                             directory)
                (directory / 'fingerprint.json').write_text(json.dumps(identity, indent=2))
                with np.load(directory / 'paired_motion.npz', allow_pickle=False) as data:
                    valid = np.flatnonzero(data['executed_state_mask'])
                    if len(valid) < 2 or not np.array_equal(valid, np.arange(1, len(data['qpos_36']))):
                        raise ValueError('Expected contiguous executed states excluding initialization at t=0')
                    executed = data['qpos_36'][valid].copy()
                    origin = source.time_origin_sec + float(data['timestamps'][valid[0]])
                quality['scope'] = 'tracker_execution_and_tracking_error'
                add_track(source, executed, origin, 'tracker_executed', version, f'holomotion:{cache_key}', quality, mid)
        return export_release(inv, tracks, output)
