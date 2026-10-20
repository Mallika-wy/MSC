"""Decode FineDance and single-person GDance motions to shared GMR input."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_6d_to_matrix(six):
    """FineDance follows PyTorch3D: the 6D vectors are matrix rows."""
    a, b = six[..., :3], six[..., 3:]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - (a*b).sum(axis=-1, keepdims=True)*a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.stack((a, b, np.cross(a, b)), axis=-2)


def decode_motion(source, motion_path, settings):
    import smplx
    import torch
    from smplx.joint_names import JOINT_NAMES
    from data_pipeline.stages.retarget import HumanMotion

    if source.fps != 30:
        raise ValueError('Expected 30 Hz motion')
    if source.format == 'finedance_smplx':
        data = np.load(motion_path, allow_pickle=False)
        if data.ndim != 2 or data.shape[1] != 315:
            raise ValueError('Expected FineDance shape (frames, 315)')
        trans = data[:, :3]
        six = data[:, 3:].reshape(-1, 52, 6)
        matrix = rotation_6d_to_matrix(six)
        pose = Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 52, 3)
        left, right = pose[:, 22:37].reshape(-1, 45), pose[:, 37:52].reshape(-1, 45)
        betas = np.zeros(10, dtype=np.float32)
    elif source.format == 'gdance_smpl':
        with np.load(motion_path, allow_pickle=False) as data:
            pose = data['smpl_poses'].reshape(-1, 24, 3)
            trans = data['root_trans'].copy()
            betas = data['smpl_betas'].mean(axis=0)[:10].astype(np.float32)
        left = right = np.zeros((len(pose), 45), dtype=np.float32)
    else:
        raise ValueError(f'Unsupported dance format {source.format}')
    if len(pose) < 2 or not np.isfinite(pose).all() or not np.isfinite(trans).all():
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
    origin = positions[0, 0].copy(); origin[2] = 0
    positions -= origin
    quats = Rotation.from_matrix((correction.as_matrix() @ world).reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 55, 4)
    return HumanMotion(tuple(JOINT_NAMES[:55]), positions, quats, np.arange(len(pose))/30, height, 'smplx',
        {'source_format': source.format, 'proxy': 'neutral SMPL-X', 'source_normalized_horizontal_origin': origin.tolist()})
