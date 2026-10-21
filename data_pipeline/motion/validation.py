"""Shared G1 array contract without tracker dependencies."""
import numpy as np


def validate_qpos(qpos, times):
    """检查公共动作契约：qpos[T,36]、时间戳[T]、有限值及单位四元数。

    36维 = 根位置xyz[3] + 根四元数wxyz[4] + G1关节角[29]。
    """
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or times.shape != (len(qpos),):
        raise ValueError('Expected at least two G1 qpos frames (T,36) and matching timestamps')
    if not np.isfinite(qpos).all() or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Invalid motion values or timestamps')
    if np.max(abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-3:
        raise ValueError('Root quaternion must be unit wxyz')

