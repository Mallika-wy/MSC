"""Shared SMPL-X forward kinematics; dataset decoding lives in sources/."""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class HumanMotion:
    """GMR input: joint positions in metres, world orientations in wxyz, 30Hz seconds."""
    joint_names: tuple[str, ...]
    positions: np.ndarray
    orientations: np.ndarray
    timestamps: np.ndarray
    height_m: float
    gmr_source: str
    provenance: dict = field(default_factory=dict)


def human_motion(pose, trans, betas, left, right, settings):
    """Convert local 30 Hz axis angles to Z-up world joints for GMR."""
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES

    if (len(pose) < 2 or trans.shape != (len(pose), 3)
            or not all(np.isfinite(value).all() for value in (pose, trans, betas, left, right))):
        raise ValueError('Invalid source motion')
    model = smplx.SMPLX(model_path=str(settings['smpl_model']), gender='neutral',
                        use_pca=False, flat_hand_mean=True)
    betas = np.pad(betas, (0, max(0, model.num_betas - len(betas))))[:model.num_betas]
    positions, rotations = [], []
    with torch.no_grad():
        neutral = model(return_verts=True)
        height = float(neutral.vertices[0, :, 1].max() - neutral.vertices[0, :, 1].min())
        for start in range(0, len(pose), 64):
            sl = slice(start, start + 64)
            zeros = torch.zeros((len(pose[sl]), 3), dtype=torch.float32)
            result = model(betas=torch.from_numpy(betas).float().reshape(1, -1),
                global_orient=torch.from_numpy(pose[sl, 0]).float(),
                body_pose=torch.from_numpy(pose[sl, 1:22].reshape(-1, 63)).float(),
                left_hand_pose=torch.from_numpy(left[sl]).float(),
                right_hand_pose=torch.from_numpy(right[sl]).float(),
                jaw_pose=zeros, leye_pose=zeros, reye_pose=zeros,
                transl=torch.from_numpy(trans[sl]).float(), return_full_pose=True)
            positions.append(result.joints[:, :55].numpy())
            rotations.append(result.full_pose.numpy().reshape(-1, 55, 3))
    positions = np.concatenate(positions)
    local = Rotation.from_rotvec(np.concatenate(rotations).reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
    parents = model.parents.numpy()
    world = np.empty_like(local)
    for j in range(55):
        world[:, j] = local[:, j] if parents[j] < 0 else world[:, parents[j]] @ local[:, j]
    correction = Rotation.from_euler('x', 90, degrees=True)
    positions = correction.apply(positions.reshape(-1, 3)).reshape(positions.shape)
    origin = positions[0, 0].copy()
    origin[2] = 0
    positions -= origin
    quats = Rotation.from_matrix((correction.as_matrix() @ world).reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 55, 4)
    return HumanMotion(tuple(JOINT_NAMES[:55]), positions, quats, np.arange(len(pose))/30, height, 'smplx',
        {'proxy': 'neutral SMPL-X', 'source_normalized_horizontal_origin': origin.tolist()})
