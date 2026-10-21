"""Shared SMPL-X forward kinematics; dataset decoding lives in sources/."""
from dataclasses import dataclass, field
from functools import lru_cache

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


@lru_cache(maxsize=2)
def _body_model(path):
    import smplx
    import torch
    model = smplx.SMPLX(model_path=path, gender='neutral', use_pca=False, flat_hand_mean=True).eval()
    with torch.no_grad():
        vertices = model(return_verts=True).vertices
        height = float(vertices[0, :, 1].max() - vertices[0, :, 1].min())
    return model, height


def human_motion(pose, trans, betas, left, right, settings):
    """Evaluate the 55-joint skeleton without per-frame mesh skinning."""
    import torch
    from smplx.joint_names import JOINT_NAMES
    from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform

    if (len(pose) < 2 or trans.shape != (len(pose), 3)
            or not all(np.isfinite(value).all() for value in (pose, trans, betas, left, right))):
        raise ValueError('Invalid source motion')
    model, height = _body_model(str(settings['smpl_model']))
    betas = np.pad(betas, (0, max(0, model.num_betas - len(betas))))[:model.num_betas]
    positions, rotations = [], []
    with torch.no_grad():
        shaped = model.v_template + blend_shapes(torch.as_tensor(betas).float()[None], model.shapedirs)
        joints = vertices2joints(model.J_regressor, shaped)
        for start in range(0, len(pose), 64):
            sl = slice(start, start + 64)
            count = len(pose[sl])
            full_pose = torch.cat((torch.from_numpy(pose[sl, :22].reshape(count, 66)).float(),
                                   torch.zeros((count, 9), dtype=torch.float32), torch.from_numpy(left[sl]).float(),
                                   torch.from_numpy(right[sl]).float()), dim=1) + model.pose_mean
            matrices = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(count, 55, 3, 3)
            posed_joints, _ = batch_rigid_transform(matrices, joints.expand(count, -1, -1), model.parents)
            positions.append((posed_joints + torch.from_numpy(trans[sl]).float()[:, None]).numpy())
            rotations.append(full_pose.numpy().reshape(-1, 55, 3))
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
