"""Shared GMR→G1 stage; dataset adapters only decode human motion."""
import importlib
from pathlib import Path
import sys

import numpy as np

from data_pipeline.sources import load_dataset

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
    from .validation import validate_qpos
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


def retarget(source, dataset, settings):
    """Decode one person's motion, then fit the common G1 robot with GMR."""
    motion = load_dataset(dataset).decode_motion(source.path, settings)
    qpos, quality = run_gmr(motion, settings)
    return qpos, {**motion.provenance, **quality}
