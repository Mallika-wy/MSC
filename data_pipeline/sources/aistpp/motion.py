"""Decode 60 Hz AIST++ SMPL annotations into 30 Hz world-space human joints."""
from pathlib import Path

def decode_motion(motion_path, settings):
    """AIST SMPL60 → world-space joints30; no dataset-specific IK solver."""
    import inspect
    import pickle
    import numpy as np
    import torch
    import smplx
    from scipy.spatial.transform import Rotation
    from data_pipeline.motion.decoding import HumanMotion

    gender = settings['gender']
    if gender != 'male' and not settings.get('allow_nonofficial_model', False):
        raise ValueError('AIST uses male SMPL; other models require allow_nonofficial_model')
    payload = pickle.loads(Path(motion_path).read_bytes())
    # SMPL每帧24个局部轴角旋转；AIST标注60Hz，取0,2,4...生成30Hz参考。
    pose = np.asarray(payload['smpl_poses']).reshape(-1, 24, 3)
    trans = np.asarray(payload['smpl_trans'])
    scale = float(np.asarray(payload['smpl_scaling']).reshape(-1)[0])
    indices = np.arange(0, len(pose), 2)
    if not np.isfinite(pose).all() or not np.isfinite(trans).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid original SMPL values')
    # 2. 用SMPL前向运动学恢复人体位置；不能把局部轴角直接当成全局关节朝向。
    # 官方AIST使用male；neutral只有显式允许时才作为当前原型近似。
    aliases = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
               'object': object, 'unicode': str, 'str': str}
    added = [name for name in aliases if name not in np.__dict__]
    old_getargspec = getattr(inspect, 'getargspec', None)
    try:
        for name in added:
            setattr(np, name, aliases[name])
        if old_getargspec is None:
            inspect.getargspec = inspect.getfullargspec
        model = smplx.SMPL(model_path=str(Path(settings['smpl_model']).resolve()), gender=gender)
    finally:
        for name in added:
            delattr(np, name)
        if old_getargspec is None:
            delattr(inspect, 'getargspec')
    with torch.no_grad():
        neutral = model(return_verts=True)
        height = float(neutral.vertices[0, :, 1].max() - neutral.vertices[0, :, 1].min())
        joints = []
        for start in range(0, len(indices), 64):
            selected = indices[start:start + 64]
            # AIST拟合的平移带全局scale；标准SMPL米制空间要用trans/scale。
            result = model(global_orient=torch.from_numpy(pose[selected, 0]).float(), body_pose=torch.from_numpy(pose[selected, 1:].reshape(-1, 69)).float(), transl=torch.from_numpy(trans[selected] / scale).float(), return_verts=False)
            joints.append(result.joints[:, :24].numpy())
    positions = np.concatenate(joints)
    # 使用原生SMPL关节名；SMPL-X在腕部之后的索引不同，不能直接照搬完整列表。
    names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
    # 3. AIST世界坐标Y向上，机器人Z向上；Rx(+90°)把(x,y,z)变成(x,-z,y)。
    correction = Rotation.from_euler('x', 90, degrees=True)
    positions = correction.apply(positions.reshape(-1, 3)).reshape(positions.shape)
    # 只把首帧骨盆水平位置设为原点，不删帧、不改时间；逆平移信息写入report。
    horizontal_origin = positions[0, 0].copy()
    horizontal_origin[2] = 0
    positions -= horizontal_origin
    parents = model.parents.numpy()
    local_rot = Rotation.from_rotvec(pose[indices].reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    global_rot = np.empty_like(local_rot)
    # 沿父子链累乘：R_world(child)=R_world(parent)@R_local(child)。
    # 坐标系修正只左乘全局旋转，不对每个局部关节重复施加世界旋转。
    for joint in range(24):
        global_rot[:, joint] = local_rot[:, joint] if parents[joint] < 0 else global_rot[:, parents[joint]] @ local_rot[:, joint]
    global_rot = correction.as_matrix() @ global_rot
    orientations = Rotation.from_matrix(global_rot.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 24, 4)
    return HumanMotion(tuple(names), positions, orientations, indices / 60, height, 'smplx',
        {'source_frames': len(pose), 'source_scaling': scale, 'smpl_gender': gender,
         'world_rotation': correction.as_matrix().tolist(),
         'source_normalized_horizontal_origin': horizontal_origin.tolist()})
