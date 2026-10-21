"""Motion processing and the single owner of reference/execution caches."""
from functools import lru_cache
from pathlib import Path
import tempfile
import numpy as np

from data_pipeline.storage.artifacts import MotionArtifact
from data_pipeline.storage.store import Store, fingerprint, digest
from data_pipeline.motion.validation import validate_qpos


@lru_cache(maxsize=8)
def environment_key(dataset, settings_json):
    import json
    settings = json.loads(settings_json)
    from data_pipeline.sources import load_dataset
    import inspect
    paths = [Path(__file__), Path(__file__).resolve().parents[1] / 'motion/retarget.py',
             Path(__file__).resolve().parents[1] / 'motion/decoding.py', Path(inspect.getfile(load_dataset(dataset).decode_motion))]
    root = Path(settings['gmr_root'])
    paths.extend(p for p in (root / 'general_motion_retargeting').rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.yml', '.toml', '.xml', '.urdf') and '.git' not in p.parts)
    if settings.get('smpl_model'):
        paths.append(Path(settings['smpl_model']))
    import subprocess
    revision = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    diff = subprocess.check_output(['git','-C',str(root),'diff','--binary','HEAD'])
    import hashlib
    from data_pipeline.robot import robot_contract
    return fingerprint({'files': {str(p): digest(p) for p in paths}, 'revision': revision, 'worktree_diff': hashlib.sha256(diff).hexdigest(), 'robot': robot_contract()})


def process_motion(source, dataset, settings, *, diagnostic=False, tracker_version='not_applicable'):
    import json
    from data_pipeline.motion.arm_twist import screen_reference
    from data_pipeline.motion.tracking import track
    from data_pipeline.motion.retarget import retarget
    native = source.representation == 'g1_qpos36'
    if native and not settings.tracker['enabled']:
        from data_pipeline.sources.omg import read_motion
        qpos = read_motion(source)
        reference = dict(kind='source_g1', path=str(source.path), metadata=source.metadata, checksum=digest(source.path))
        quality = {'accepted': True, 'stage': 'source_g1', 'checks': ['finite_qpos36', 'unit_quaternion', '30fps']}
        return [MotionArtifact('source_g1', reference, len(qpos), quality)]
    store = Store(settings.output / 'cache/motion.sqlite')
    if not native and settings.retarget['enabled'] is False:
        raise ValueError('Human motion requires retargeting')
    identity = {'source': digest(source.path), 'metadata': source.metadata, 'actor': source.actor_id,
                'dataset': dataset, 'implementation': digest(__file__)}
    if not native:
        identity['environment'] = environment_key(dataset, json.dumps(settings.retarget, sort_keys=True))
        identity['settings'] = settings.retarget
    key = 'reference:' + fingerprint(identity)
    with store.lock(key):
        cached = store.get(key)
        if cached is None:
            if native:
                from data_pipeline.sources.omg import read_motion
                qpos = read_motion(source)
                report = {}
            else:
                qpos, report = retarget(source, dataset, settings.retarget)
            validate_qpos(qpos, np.arange(len(qpos))/30)
            version = 'not_applicable' if native else f'gmr:{key}'
            reference = store.put(key, {'version': version, 'report': report}, qpos_36=qpos)
        else:
            metadata, arrays = cached
            version, qpos = metadata['version'], arrays['qpos_36']
            report = metadata['report']
            reference = dict(store=str(store.path), key=key)
    validate_qpos(qpos, np.arange(len(qpos))/30)
    label = 'source_g1' if native else 'retarget_reference'
    quality = {'accepted': True, 'stage': 'source_g1'} if native else screen_reference(qpos, settings.quality)
    quality['processing'] = report
    artifacts = [MotionArtifact(label, reference, len(qpos), quality, version)]
    if settings.tracker['enabled'] and (quality['accepted'] or diagnostic):
        tracking_key = 'tracking:' + fingerprint({'reference': key, 'version': tracker_version,
            'tracker': settings.tracker, 'diagnostic': diagnostic, 'quality': settings.quality,
            'implementation': digest(Path(__file__).resolve().parents[1] / 'motion/tracking.py'),
            'screening': digest(Path(__file__).resolve().parents[1] / 'motion/quality.py')})
        target = settings.output / 'work/diagnostics' / fingerprint(tracking_key)
        with store.lock(tracking_key):
            cached = store.get(tracking_key)
            if diagnostic and not (target / 'quality.json').is_file():
                cached = None
            if cached is None:
                with tempfile.TemporaryDirectory(dir=store.path.parent, prefix='.tracking-') as work:
                    executed, execution = track(qpos, settings.tracker, settings.quality, Path(work), diagnostic=diagnostic)
                    validate_qpos(executed, np.arange(len(executed))/30)
                    if len(executed) != len(qpos) - 1:
                        raise ValueError('Executed labels must exclude exactly the initialization frame')
                    if diagnostic:
                        import shutil
                        from data_pipeline.io import write_json
                        shutil.copytree(work, target, dirs_exist_ok=True)
                        write_json(target / 'quality.json', execution)
                tracking_ref = store.put(tracking_key, execution, qpos_36=executed)
            else:
                execution, arrays = cached
                executed = arrays['qpos_36']
                tracking_ref = dict(store=str(store.path), key=tracking_key)
        execution = {**execution, 'accepted': bool(execution['accepted'] and quality['accepted']),
                     'diagnostic_only': not quality['accepted']}
        artifacts.append(MotionArtifact('tracker_executed', tracking_ref, len(executed), execution,
                                       tracker_version, 1/30, label))
    return artifacts
