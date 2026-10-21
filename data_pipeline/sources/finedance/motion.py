"""FineDance 315D -> local axis angles -> world-space human joints."""
import numpy as np
from scipy.spatial.transform import Rotation

from data_pipeline.motion.decoding import human_motion


def rotation_6d_to_matrix(six):
    """FineDance/PyTorch3D stores the first two ROWS of the rotation matrix."""
    first, second = six[..., :3], six[..., 3:]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - (first * second).sum(axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def decode_motion(path, settings):
    data = np.load(path, allow_pickle=False)
    if data.ndim != 2 or data.shape[1] != 315:
        raise ValueError('Expected FineDance shape (frames, 315)')
    matrices = rotation_6d_to_matrix(data[:, 3:].reshape(-1, 52, 6))
    pose = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 52, 3)
    return human_motion(pose, data[:, :3], np.zeros(10, dtype=np.float32),
                        pose[:, 22:37].reshape(-1, 45), pose[:, 37:52].reshape(-1, 45), settings)
