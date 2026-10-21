"""Motion-only processing with separate reference and execution caches."""
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
    paths.extend(p for p in (root / 'general_motion_retargeting').rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.xml', '.urdf') and '.git' not in p.parts)
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
                version = 'not_applicable'
            else:
                from data_pipeline.sources.sample import Sample
                with tempfile.TemporaryDirectory(dir=store.path.parent, prefix='.motion-') as work:
                    path, version, _ = retarget(Sample(source.motion_id, source.motion_id, source.path, None,
                        actor=source.actor_id), dataset, settings.retarget, Path(work))
                    with np.load(path, allow_pickle=False) as data:
                        qpos = data['qpos_36'].copy()
            validate_qpos(qpos, np.arange(len(qpos))/30)
            reference = store.put(key, {'version': version}, qpos_36=qpos)
        else:
            metadata, arrays = cached
            version, qpos = metadata['version'], arrays['qpos_36']
            reference = dict(store=str(store.path), key=key)
    validate_qpos(qpos, np.arange(len(qpos))/30)
    label = 'source_g1' if native else 'retarget_reference'
    # Native OMG has already undergone the documented quality cleaning. Validate
    # its numerical contract, without silently running the historical deletion policy.
    with tempfile.TemporaryDirectory(dir=store.path.parent, prefix='.motion-') as work:
        work = Path(work)
        path = work / 'reference.npz'
        np.savez(path, qpos_36=qpos, fps=np.float32(30), source_timestamps=np.arange(len(qpos))/30)
        quality = {'accepted': True, 'stage': 'source_g1', 'checks': ['finite_qpos36', 'unit_quaternion', '30fps']} if native else screen_reference(path, settings.quality)
        artifacts = [MotionArtifact(label, reference, len(qpos), quality, version)]
        if settings.tracker['enabled'] and (quality['accepted'] or diagnostic):
            tracking_key = 'tracking:' + fingerprint({'reference': key, 'version': tracker_version,
                'tracker': settings.tracker, 'diagnostic': diagnostic, 'quality': settings.quality,
                'implementation': digest(Path(__file__).resolve().parents[1] / 'motion/tracking.py'),
                'screening': digest(Path(__file__).resolve().parents[1] / 'motion/quality.py')})
            target = settings.output / 'work/diagnostics' / fingerprint(tracking_key)
            with store.lock(tracking_key):
                cached = store.get(tracking_key)
                if diagnostic and not target.is_dir():
                    cached = None
                if cached is None:
                    execution = track(path, version, {**settings.tracker, 'save_video': diagnostic}, settings.quality,
                                      tracker_version, work / 'tracking')
                    with np.load(execution['path'], allow_pickle=False) as data:
                        valid = data['executed_state_mask']
                        if not np.array_equal(np.flatnonzero(valid), np.arange(1, len(valid))):
                            raise ValueError('Executed labels must exclude exactly the initialization frame')
                        executed = data['qpos_36'][valid].copy()
                    execution = {k:v for k,v in execution.items() if k not in ('path', 'execution_video')}
                    tracking_ref = store.put(tracking_key, execution, qpos_36=executed)
                else:
                    execution, arrays = cached
                    executed = arrays['qpos_36']
                    tracking_ref = dict(store=str(store.path), key=tracking_key)
            execution = {**execution, 'accepted': bool(execution['accepted'] and quality['accepted']),
                         'diagnostic_only': not quality['accepted']}
            artifacts.append(MotionArtifact('tracker_executed', tracking_ref, len(executed), execution,
                                             tracker_version, 1/30, label))
            if diagnostic:
                # Keep only explicitly requested diagnostics; normal runs leave no sample directories.
                import shutil
                if (work / 'tracking').exists():
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(work, target)
    return artifacts
