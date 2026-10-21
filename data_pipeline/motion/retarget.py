"""Shared GMR→G1 stage; dataset adapters only decode human motion."""
import contextlib
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from data_pipeline.sources import load_dataset

from data_pipeline.io import sha256_file, write_json
from data_pipeline.robot import G1_JOINT_NAMES
from .decoding import HumanMotion


def load_gmr(gmr_root):
    """Load the requested local GMR checkout without copying or modifying it."""
    root = Path(gmr_root).resolve()
    if not (root / 'general_motion_retargeting/motion_retarget.py').is_file():
        raise FileNotFoundError(f'GMR checkout not found: {root}')
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('general_motion_retargeting')
    except ModuleNotFoundError as exc:
        raise ImportError(f'GMR dependency {exc.name!r} is missing in this Python; install the offline GMR dependencies here') from exc
    finally:
        sys.path.remove(str(root))
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise RuntimeError('A different GMR checkout is already imported; start a fresh process')
    return module.GeneralMotionRetargeting


def run_gmr(motion: HumanMotion, settings):
    """Shared by SMPL, SMPL-X, BVH or other decoders producing HumanMotion."""
    from .tracking import validate_qpos
    t, j = len(motion.timestamps), len(motion.joint_names)
    if (t < 2 or j == 0 or len(set(motion.joint_names)) != j
            or motion.positions.shape != (t, j, 3) or motion.orientations.shape != (t, j, 4)
            or not np.isfinite(motion.positions).all() or not np.isfinite(motion.orientations).all()
            or not np.isfinite(motion.timestamps).all() or not np.isfinite(motion.height_m) or motion.height_m <= 0
            or not np.allclose(motion.timestamps, np.arange(t)/30, atol=1e-8, rtol=0)
            or not np.allclose(np.linalg.norm(motion.orientations, axis=-1), 1, atol=1e-3)):
        raise ValueError('Invalid human motion: expected finite Z-up world joints and 30 Hz local time')
    solver_class = load_gmr(settings['gmr_root'])
    import mujoco
    import torch
    from msc._omg.kinematics import G1Kinematics
    solver = solver_class(src_human=motion.gmr_source, tgt_robot='unitree_g1',
        actual_human_height=motion.height_m, verbose=False, use_velocity_limit=False)
    names = [mujoco.mj_id2name(solver.model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(solver.model.njnt) if solver.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if names != list(G1_JOINT_NAMES):
        raise ValueError('GMR joint order does not match the MSC G1 contract')
    qpos = []
    for i in range(t):
        frame = {name: (motion.positions[i, k].copy(), motion.orientations[i, k].copy())
                 for k, name in enumerate(motion.joint_names)}
        if i == 0:
            for _ in range(8):
                solver.retarget(frame)
        qpos.append(solver.retarget(frame).copy())
    qpos = np.asarray(qpos, dtype=np.float32)
    for i in range(1, len(qpos)):
        if np.dot(qpos[i-1, 3:7], qpos[i, 3:7]) < 0:
            qpos[i, 3:7] *= -1
    validate_qpos(qpos, motion.timestamps)
    # Preserve the existing sequence-wide sole correction; do not flatten jumps.
    kinematics = G1Kinematics()
    with torch.no_grad():
        body = kinematics.forward_kinematics(torch.from_numpy(qpos))
        points, radii = kinematics.get_sole_proxy_points(body['body_pos_w'], body['body_quat_w'])
    z_shift = .005 - float((points[..., 2] - radii).min())
    qpos[:, 2] += z_shift
    violation = np.maximum(np.maximum(kinematics.joint_lower_limits.numpy() - qpos[:, 7:],
                                      qpos[:, 7:] - kinematics.joint_upper_limits.numpy()), 0)
    return qpos, {'ground_constant_shift_m': z_shift, 'joint_limit_max_violation_rad': float(violation.max())}


def retarget(sample, dataset, settings, work):
    """Decode one person's motion, then fit the common G1 robot with GMR."""
    decoder = load_dataset(dataset).decode_motion
    path = sample.motion
    model = Path(settings['smpl_model']).resolve() if settings.get('smpl_model') else None
    gmr_root = Path(settings['gmr_root']).resolve()
    revision = subprocess.check_output(['git', '-C', str(gmr_root), 'rev-parse', 'HEAD'], text=True).strip()
    # Fingerprint local work as well as HEAD; never require changing the user's GMR checkout.
    diff = subprocess.check_output(['git', '-C', str(gmr_root), 'diff', '--binary', 'HEAD'])
    untracked = subprocess.check_output(['git', '-C', str(gmr_root), 'ls-files', '--others', '--exclude-standard', '-z'])
    extra_code = {}
    for raw_name in untracked.decode().split('\0'):
        name = Path(raw_name)
        if raw_name and name.suffix in ('.py', '.json', '.yaml', '.yml', '.toml'):
            extra_code[raw_name] = sha256_file(gmr_root / name)
    from data_pipeline.robot import robot_contract
    fingerprint = {'robot': robot_contract(), 'source_sha256': sha256_file(path), 'sample': sample.name, 'actor': sample.actor,
                   'smpl_sha256': sha256_file(model) if model else None, 'gmr_revision': revision,
                   'gmr_worktree_diff_sha256': hashlib.sha256(diff).hexdigest(), 'gmr_untracked_code': extra_code,
                   'stage_sha256': sha256_file(__file__), 'decoder_sha256': sha256_file(inspect.getfile(decoder)),
                   'human_fk_sha256': sha256_file(Path(__file__).parent / 'decoding.py'),
                   'settings': settings,
                   'dataset': dataset}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    output = Path(work) / key
    reference = output / 'motion.npz'
    if not reference.is_file() or not (output / 'fingerprint.json').is_file():
        output.mkdir(parents=True, exist_ok=True)
        motion = decoder(path, settings)
        with (output / 'gmr.log').open('w') as log, contextlib.redirect_stdout(log):
            qpos, quality = run_gmr(motion, settings)
        np.savez_compressed(reference, qpos_36=qpos, fps=np.float32(30), source_timestamps=motion.timestamps,
                            joint_names=np.asarray(G1_JOINT_NAMES))
        write_json(output / 'report.json', {**motion.provenance, **quality,
                   'source_motion_id': sample.name, 'production_ready': False})
        write_json(output / 'fingerprint.json', fingerprint)
    return reference, f'gmr:{key}', fingerprint
