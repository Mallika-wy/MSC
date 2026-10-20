"""动作分支：原生格式 → G1参考动作 → 仿真实际执行动作 → 30Hz训练标签。

reference 是重定向器希望机器人达到的姿态；executed 才是控制器实际执行后的姿态。
两者不能混为一谈，最终标签保留 executed，并记录其有效时间范围。
"""
from pathlib import Path
import subprocess
import sys

import numpy as np

from data_pipeline.core.io import read_json, write_json
from data_pipeline.core.pairs import Pair


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


def decode_aist(pair, config, output, project):
    """读取 CSV 指定的 AIST SMPL，交给已有 GMR 后端生成30Hz参考动作。"""
    backend = config['retarget']
    if pair.motion_fps != 60:
        raise ValueError('aist_smpl requires 60 Hz annotations')
    # 后端在子进程中运行，隔离旧SMPL/chumpy依赖；调试时可启用 subProcess。
    # CSV 同时提供真实视频路径、ZIP路径和member，不在调度器里猜文件位置。
    args = [sys.executable, '-m', 'data_pipeline.backends.aist_smpl_gmr',
            '--sequence', pair.sequence_id, '--source-root', pair.source_root or str(Path(pair.motion_path).parent.parent),
            '--motion-path', pair.motion_path, '--motion-member', pair.motion_member,
            '--video-path', pair.video_path, '--gmr-root', backend['gmr_root'],
            '--smpl-model', backend['smpl_model'], '--gender', backend['gender'],
            '--output', str(output), '--skip-video', '--overwrite']
    if backend.get('dependency_site'):
        args += ['--dependency-site', backend['dependency_site']]
    if backend.get('allow_nonofficial_model'):
        args.append('--allow-nonofficial-model')
    with (output.parent / 'retarget.log').open('w') as log:
        subprocess.run(args, cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
    if pair.motion_time_offset_s:
        # 只移动时间轴，不删首帧、不改变动作速度，用于与公共序列时间对齐。
        with np.load(output / 'motion.npz', allow_pickle=False) as data:
            payload = dict(data)
        payload['source_timestamps'] += pair.motion_time_offset_s
        np.savez_compressed(output / 'motion.npz', **payload)


def decode_g1(pair, config, output, project):
    """已是规范G1的NPZ无需人体重定向，但仍需检查格式和后续执行质量。"""
    # 此格式名明确承诺：官方29关节顺序、米制、Z向上、根四元数wxyz。
    if pair.motion_fps != 30 or pair.motion_member:
        raise ValueError('g1_qpos_wxyz requires a normal 30 Hz NPZ file')
    with np.load(pair.motion_path, allow_pickle=False) as data:
        qpos, times = data['qpos_36'].astype(np.float32), data['timestamps'].astype(np.float64)
    times += pair.motion_time_offset_s
    validate_qpos(qpos, times)
    np.savez_compressed(output / 'motion.npz', qpos_36=qpos, source_timestamps=times,
                        source_frame_indices=np.arange(len(qpos)), fps=np.float32(30))


# 新动作格式只在这里注册对应解码器；不同数据源若格式相同，可以共用解码器。
DECODERS = {'aist_smpl': decode_aist, 'g1_qpos_wxyz': decode_g1}


def prepare_reference(pair: Pair, config, directory, project, force=False):
    """选择解码器并获得统一参考动作；已有文件可复用，force=True 时重算。"""
    decoder = DECODERS.get(pair.motion_format)
    if decoder is None:
        raise ValueError(f'Unsupported motion_format: {pair.motion_format}; add a decoder')
    output = directory / 'reference'
    output.mkdir(exist_ok=True)
    path = output / 'motion.npz'
    if force or not path.is_file():
        decoder(pair, config, output, project)
    with np.load(path, allow_pickle=False) as data:
        validate_qpos(data['qpos_36'], data['source_timestamps'])
        if float(data['fps']) != 30 or not np.allclose(np.diff(data['source_timestamps']), 1/30, atol=1e-7):
            raise ValueError('Reference must be uniformly sampled at 30 Hz')
    return path


def execute_and_screen(reference, config, output, project, force=False):
    """执行参考轨迹、检查原始控制频率下的质量，再按标签时间重采样。

    输出 paired_motion.npz：qpos_36、公共时间戳、30Hz帧率、执行有效性掩码。
    输出 quality.json：跌倒/限位检查和执行与参考之间的跟踪误差。
    """
    motion_path = output / 'paired_motion.npz'
    quality_path = output / 'quality.json'

    if not force and motion_path.is_file() and quality_path.is_file():
        return read_json(quality_path)

    return _execute_motion(reference, config, output, force=force)


def _execute_motion(reference, config, output, force=False):
    """实际执行与质检；仅在缓存未命中时加载仿真依赖。"""
    import mujoco
    import torch

    from data_pipeline.stages.motion_quality import (
        MotionScreeningConfig,
        resample_motion_at_times,
        screen_executed_motion,
    )
    from omg.robots.g1.kinematics import G1Kinematics
    from omg.tracking.holomotion.reference import resample_qpos
    from omg.tracking.holomotion.runner import (
        TrackerRunConfig,
        run_holomotion_tracker,
    )

    torch.set_num_threads(4)

    with np.load(reference, allow_pickle=False) as data:
        ref = data['qpos_36'].copy()
        source_times = data['source_timestamps'].copy()

    rollout = output / 'tracker_rollout.npz'
    if force or not rollout.is_file():
        # 控制器50Hz执行；每次控制包含10个物理子步，这不是最终标签帧率。
        tracker_config = TrackerRunConfig(
            reference=reference,
            holomotion_onnx=config['execution']['onnx'],
            providers=config['execution'].get('providers', 'CPUExecutionProvider'),
            target_fps=50.0,
            control_substeps=10,
            action_clip=10.0,
            output=rollout,
        )
        run_holomotion_tracker(tracker_config)

    with np.load(rollout, allow_pickle=False) as data:
        executed = data['executed_qpos_36'].copy()
        tracked_ref = data['reference_qpos_36'].copy()
        fps = float(data['fps'])
        model = mujoco.MjModel.from_xml_path(str(data['robot_xml'].item()))

    dt = float(model.opt.timestep) * 10
    # 不用文件名判断旧rollout是否可用，而是核对保存的参考采样和控制时间步。
    if (
        not np.isclose(dt, 1 / fps)
        or executed.shape != tracked_ref.shape
        or not np.array_equal(tracked_ref, resample_qpos(ref, 30, fps))
    ):
        raise ValueError('Rollout does not match reference/timing; rerun with --force')

    kinematics = G1Kinematics()
    policy = config.get('quality', {})
    # 跌倒的“连续帧数”必须在原控制频率上统计，不能先降到30Hz再数。
    screening = screen_executed_motion(
        executed,
        fps=fps,
        joint_lower=kinematics.joint_lower_limits.numpy(),
        joint_upper=kinematics.joint_upper_limits.numpy(),
        config=MotionScreeningConfig(
            strict_joint_limits=policy.get('strict_joint_limits', False),
        ),
    )
    if (
        not np.isfinite(executed).all()
        or 'invalid_root_quaternion' in screening['rejection_reasons']
    ):
        raise ValueError('Invalid executed motion')

    # 第i个保存的执行状态发生在(i+1)*dt，而不是i*dt；否则会错位20ms。
    # 显式补上t=0初始化状态方便插值，但最后的掩码会排除它，不当作执行标签。
    aligned = resample_motion_at_times(
        np.concatenate((ref[:1], executed)),
        np.arange(len(executed) + 1) * dt,
        np.arange(len(ref)) / 30,
    )
    with torch.no_grad():
        # 通过同一套G1正向运动学比较全局身体位置，而不只比较关节角。
        body = kinematics.forward_kinematics(torch.from_numpy(aligned))['body_pos_w'].numpy()
        target = kinematics.forward_kinematics(torch.from_numpy(ref))['body_pos_w'].numpy()
    metrics = {
        'joint_rmse_rad': float(np.sqrt(np.mean((aligned[1:, 7:] - ref[1:, 7:]) ** 2))),
        'global_body_mpjpe_mm': float(
            np.linalg.norm(body[1:] - target[1:], axis=-1).mean() * 1000
        ),
    }
    reasons = list(screening['rejection_reasons'])
    # 跟踪误差阈值是配置里的本地策略，不是声称来自论文的官方阈值。
    for metric, value in metrics.items():
        threshold = policy.get('max_' + metric)
        if threshold is not None and value > threshold:
            reasons.append('local_threshold_exceeded:' + metric)

    quality = {
        'accepted': not reasons,
        'rejection_reasons': reasons,
        'screening': screening,
        'tracking': metrics,
        'production_ready': False,
    }
    np.savez_compressed(
        output / 'paired_motion.npz',
        qpos_36=aligned,
        timestamps=source_times,
        executed_state_mask=np.arange(len(ref)) > 0,
        fps=np.float32(30),
    )
    write_json(output / 'quality.json', quality)
    return quality
