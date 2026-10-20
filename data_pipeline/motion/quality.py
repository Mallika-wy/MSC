"""动作执行质检与按显式时间戳重采样。

Fall thresholds follow OMG Appendix B.5; frame counting uses the supplied
executed-state sampling rate. Joint-limit tolerance is a configurable local
policy, not a threshold published by the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class MotionScreeningConfig:
    min_root_height_m: float = 0.20
    max_root_tilt_deg: float = 85.0
    low_root_height_m: float = 0.35
    low_root_tilt_deg: float = 60.0
    consecutive_fall_frames: int = 10
    strict_joint_limits: bool = True
    joint_limit_tolerance_rad: float = 1e-4


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(ends - starts)) if starts.size else 0


def screen_executed_motion(
    qpos_36: np.ndarray,
    *,
    fps: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    config: MotionScreeningConfig = MotionScreeningConfig(),
) -> dict:
    qpos = np.asarray(qpos_36, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not len(qpos):
        raise ValueError('Expected a nonempty (T,36) trajectory')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    lower, upper = np.asarray(joint_lower), np.asarray(joint_upper)
    if lower.shape != (29,) or upper.shape != (29,) or not np.isfinite([lower, upper]).all() or (lower > upper).any():
        raise ValueError('Expected finite ordered 29D joint bounds')
    if config.consecutive_fall_frames < 1 or config.joint_limit_tolerance_rad < 0:
        raise ValueError('Invalid screening configuration')
    result = {'frames': len(qpos), 'screening_fps': float(fps), 'thresholds': asdict(config), 'accepted': False, 'rejection_reasons': []}
    if not np.isfinite(qpos).all():
        result['rejection_reasons'].append('nonfinite_state')
        return result
    norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
    if (norms < 1e-8).any():
        result['rejection_reasons'].append('invalid_root_quaternion')
        return result
    quat = qpos[:, 3:7] / norms[:, None]
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
    height = qpos[:, 2]
    fallen = (height < config.min_root_height_m) | (tilt > config.max_root_tilt_deg) | ((height < config.low_root_height_m) & (tilt > config.low_root_tilt_deg))
    longest = longest_true_run(fallen)
    violation = np.maximum(np.maximum(lower - qpos[:, 7:], qpos[:, 7:] - upper), 0)
    joint_bad = violation > config.joint_limit_tolerance_rad
    if longest >= config.consecutive_fall_frames:
        result['rejection_reasons'].append('persistent_fall')
    if config.strict_joint_limits and joint_bad.any():
        result['rejection_reasons'].append('executed_joint_limit_violation')
    velocity = np.diff(qpos[:, 7:], axis=0) * fps
    result.update({
        'accepted': not result['rejection_reasons'],
        'root_height_min_m': float(height.min()),
        'root_height_max_m': float(height.max()),
        'root_tilt_max_deg': float(tilt.max()),
        'fall_frames': int(fallen.sum()),
        'longest_fall_run_frames': longest,
        'longest_fall_run_s': longest / fps,
        'joint_limit_violation_values': int(joint_bad.sum()),
        'joint_limit_violation_frames': int(joint_bad.any(axis=1).sum()),
        'joint_limit_max_violation_rad': float(violation.max()),
        'quaternion_max_norm_error': float(np.abs(norms - 1).max()),
        'max_joint_speed_rad_s': float(np.abs(velocity).max()) if len(velocity) else 0.0,
    })
    return result


def resample_motion_at_times(qpos_36: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Interpolate explicit timestamps with quaternion SLERP; forbid extrapolation."""
    qpos = np.asarray(qpos_36, dtype=np.float64)
    times, target = np.asarray(source_times, dtype=np.float64), np.asarray(target_times, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or times.shape != (len(qpos),) or not len(qpos):
        raise ValueError('Expected matching nonempty qpos (T,36) and timestamps (T,)')
    if target.ndim != 1 or not np.isfinite(target).all() or not np.isfinite(times).all() or not np.isfinite(qpos).all():
        raise ValueError('Motion and timestamps must be finite arrays')
    if (np.diff(times) <= 0).any() or (np.diff(target) < 0).any():
        raise ValueError('Source times must strictly increase; target times must not decrease')
    if target.size and (target[0] < times[0] - 1e-8 or target[-1] > times[-1] + 1e-8):
        raise ValueError('Extrapolation is forbidden; target timestamps exceed executed coverage')
    target = np.clip(target, times[0], times[-1])
    norms = np.linalg.norm(qpos[:, 3:7], axis=-1)
    if (norms < 1e-8).any():
        raise ValueError('Invalid root quaternion')
    quat = qpos[:, 3:7] / norms[:, None]
    out = np.empty((len(target), 36), dtype=np.float32)
    for column in [0, 1, 2, *range(7, 36)]:
        out[:, column] = np.interp(target, times, qpos[:, column])
    if len(target):
        if len(qpos) == 1:
            out[:, 3:7] = quat[0]
        else:
            out[:, 3:7] = Slerp(times, Rotation.from_quat(quat[:, [1, 2, 3, 0]]))(target).as_quat()[:, [3, 0, 1, 2]]
        for frame in range(1, len(out)):
            if np.dot(out[frame - 1, 3:7], out[frame, 3:7]) < 0:
                out[frame, 3:7] *= -1
    return out
