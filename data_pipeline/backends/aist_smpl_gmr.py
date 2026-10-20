#!/usr/bin/env python3
"""AIST动作解码后端：原始SMPL → 人体全局关节 → GMR → G1参考动作。

由data_pipeline/stages/motion.py以独立模块子进程调用。这里仅生成reference，
不是最终tracker执行标签；--skip-video用于流水线，避免重复渲染预览。
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import pickle
import subprocess
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import torch


def main():
    """保持整段时间连续，输出30Hz qpos及源帧对应关系。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', default='gBR_sBM_cAll_d04_mBR0_ch01')
    parser.add_argument('--source-root', type=Path, help='AIST source directory; defaults to data/AIST++')
    parser.add_argument('--camera', default='c01')
    parser.add_argument('--motion-path', type=Path, help='CSV motion_path: PKL file or ZIP archive')
    parser.add_argument('--motion-member', default='', help='PKL member inside motion ZIP')
    parser.add_argument('--video-path', type=Path, help='CSV video_path')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gmr-root', type=Path, default=Path('../GMR'))
    parser.add_argument('--dependency-site', type=Path, default=Path('/home1/gmh/anaconda3/envs/gmr/lib/python3.10/site-packages'))
    parser.add_argument('--smpl-model', type=Path, required=True)
    parser.add_argument('--gender', choices=['male', 'female', 'neutral'], required=True)
    parser.add_argument('--allow-nonofficial-model', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-video', action='store_true', help='Generate motion and QA without rendering')
    args = parser.parse_args()
    if args.gender != 'male' and not args.allow_nonofficial_model:
        parser.error('AIST uses male SMPL; a nonofficial model requires explicit smoke-test opt-in')
    root = Path(__file__).resolve().parents[2]
    source = args.source_root.resolve() if args.source_root else root / 'data/AIST++'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.overwrite)
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(source / 'runtime_deps'))
    sys.path.insert(0, str(args.gmr_root.resolve()))
    sys.path.append(str(args.dependency_site.resolve()))
    # 旧SMPL模型依赖旧版numpy/chumpy接口；兼容补丁只影响当前子进程。
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex, 'object': object, 'unicode': str, 'str': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    import smplx
    import mujoco
    from scipy.spatial.transform import Rotation
    from general_motion_retargeting import GeneralMotionRetargeting
    from omg.robots.g1.constants import G1_JOINT_NAMES
    from omg.robots.g1.kinematics import G1Kinematics
    from omg.render.mujoco import render_qpos_video

    torch.set_num_threads(4)
    # 1. CSV可以指定ZIP+member，也可以直接指定PKL；不需要解压整个动作包。
    motion_path = args.motion_path or source / 'archives/motions_complete.zip'
    if motion_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(motion_path) as archive:
            raw = archive.read(args.motion_member or f'motions/{args.sequence}.pkl')
    else:
        raw = motion_path.read_bytes()
    payload = pickle.loads(raw)
    if args.sequence in (source / 'archives/ignore_list.txt').read_text().split():
        raise ValueError('Sequence is in the official ignore list')
    # SMPL每帧24个局部轴角旋转；AIST标注60Hz，取0,2,4...生成30Hz参考。
    pose = np.asarray(payload['smpl_poses']).reshape(-1, 24, 3)
    trans = np.asarray(payload['smpl_trans'])
    scale = float(np.asarray(payload['smpl_scaling']).reshape(-1)[0])
    indices = np.arange(0, len(pose), 2)
    if not np.isfinite(pose).all() or not np.isfinite(trans).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid original SMPL values')
    # 2. 用SMPL前向运动学恢复人体位置；不能把局部轴角直接当成全局关节朝向。
    # 官方AIST使用male；neutral只有显式允许时才作为当前原型近似。
    model = smplx.SMPL(model_path=str(args.smpl_model.resolve()), gender=args.gender)
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
    start_time = time.perf_counter()
    # 4. GMR接收每个身体节点的全局位置与四元数，逐帧求机器人IK。
    with (output / 'gmr.log').open('w') as log, contextlib.redirect_stdout(log):
        retarget = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1', actual_human_height=height, verbose=False, use_velocity_limit=False)
        actual_names = [mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(retarget.model.njnt) if retarget.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        assert actual_names == list(G1_JOINT_NAMES)
        qpos = []
        for frame in range(len(indices)):
            human = {name: (positions[frame, j], orientations[frame, j]) for j, name in enumerate(names)}
            # 多次求解首帧用于初始化，但不添加时间步，也不丢弃原始首帧。
            if frame == 0:
                for _ in range(8):
                    retarget.retarget(human)
            qpos.append(retarget.retarget(human).copy())
    qpos = np.asarray(qpos, dtype=np.float32)
    # q与-q表示同一旋转；统一相邻帧符号，避免后续插值出现不必要的跳变。
    for frame in range(1, len(qpos)):
        if np.dot(qpos[frame - 1, 3:7], qpos[frame, 3:7]) < 0:
            qpos[frame, 3:7] *= -1
    elapsed = time.perf_counter() - start_time
    kinematics = G1Kinematics()

    def soles(q):
        state = kinematics.forward_kinematics(torch.from_numpy(q))
        points, radii = kinematics.get_sole_proxy_points(state['body_pos_w'], state['body_quat_w'])
        return points.numpy(), radii.numpy()

    points, radii = soles(qpos)
    minimum_before = float((points[..., 2] - radii).min())
    # 5. 整段只应用一个高度平移；不能逐帧强制最低脚贴地，否则会抹掉跳跃。
    z_shift = 0.005 - minimum_before
    qpos[:, 2] += z_shift
    lower = kinematics.joint_lower_limits.numpy()
    upper = kinematics.joint_upper_limits.numpy()
    violation = np.maximum(np.maximum(lower - qpos[:, 7:], qpos[:, 7:] - upper), 0)
    video_name = args.sequence.replace('_cAll_', f'_{args.camera}_') + '.mp4'
    video_path = args.video_path or source / 'videos' / video_name
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video_path)]))
    duration = float(probe['format']['duration'])
    np.savez_compressed(output / 'motion.npz', qpos_36=qpos, fps=np.float32(30), source_frame_indices=indices, source_timestamps=indices / 60, joint_names=np.array(G1_JOINT_NAMES))
    report = {'sequence': args.sequence, 'smpl_gender': args.gender, 'production_ready': False,
              'source_frames': len(pose), 'output_frames': len(qpos), 'fps': 30,
              'source_scaling': scale, 'template_height_m': height,
              'world_rotation': correction.as_matrix().tolist(),
              'source_normalized_horizontal_origin': horizontal_origin.tolist(),
              'ground_constant_shift_m': z_shift, 'video_duration_s': duration,
              'joint_limit_max_violation_rad': float(violation.max()),
              'retarget_elapsed_s': elapsed}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.skip_video:
        return
    render_qpos_video(qpos, output / 'robot.mp4', fps=30, width=640, height=480, title='AIST -> GMR -> G1 (neutral SMPL smoke)', overlay_lines=['30 FPS | original first frame preserved', 'OMG G1 rendering | GMR G1 IK'])
    rendered = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=height', '-of', 'json', str(output / 'robot.mp4')]))
    rendered_height = int(rendered['streams'][0]['height'])
    # The renderer appends a header above the requested scene height.
    source_top = (rendered_height - 360) // 2
    video_filter = f'[0:v]fps=30,scale=640:360,pad=640:{rendered_height}:0:{source_top}:black,setsar=1[a];[1:v]setsar=1[b];[a][b]hstack=inputs=2[v]'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(video_path), '-i', str(output / 'robot.mp4'), '-filter_complex', video_filter, '-map', '[v]', '-map', '0:a?', '-t', str(len(qpos) / 30), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', str(output / 'comparison.mp4')], check=True)
    print(f'Preview: {output / "comparison.mp4"}', flush=True)


if __name__ == '__main__':
    main()
