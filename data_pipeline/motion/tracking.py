"""Execute a G1 reference at 50 Hz, save video, and produce 30 Hz labels."""
from pathlib import Path
import hashlib
import json

import numpy as np

from data_pipeline.io import read_json, sha256_file, write_json


def track(reference, reference_version, execution, quality, tracker_version, output):
    """Choose a cache directory tied to this reference, controller and screening policy."""
    identity = dict(reference=reference_version, tracker=tracker_version, execution=execution,
                    quality=quality, implementation=sha256_file(__file__),
                    screening=sha256_file(Path(__file__).with_name('quality.py')))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(output) / key
    directory.mkdir(parents=True, exist_ok=True)
    result = execute_and_screen(reference, {'execution': execution, 'quality': quality}, directory)
    write_json(directory / 'fingerprint.json', identity)
    return {**result, 'path': str((directory / 'paired_motion.npz').resolve()), 'version': tracker_version}


from .validation import validate_qpos


def resample_executed(reference, executed, *, control_fps=50., motion_fps=30.):
    from .quality import resample_motion_at_times
    if control_fps != 50 or motion_fps != 30:
        raise ValueError('Controller contract is 50 Hz execution and 30 Hz labels')
    query = np.arange(len(reference), dtype=np.float64) / motion_fps
    source_times = np.arange(len(executed) + 1, dtype=np.float64) / control_fps
    aligned = resample_motion_at_times(np.concatenate((reference[:1], executed)), source_times, query)
    return aligned, query > 0


def execute_and_screen(reference, config, output):
    """执行参考轨迹、检查原始控制频率下的质量，再按标签时间重采样。

    输出 paired_motion.npz：qpos_36、公共时间戳、30Hz帧率、执行有效性掩码。
    输出 quality.json：跌倒/限位检查和执行与参考之间的跟踪误差。
    """
    motion_path = output / 'paired_motion.npz'
    quality_path = output / 'quality.json'
    rollout = output / 'tracker_rollout.npz'
    from data_pipeline.video.execution import save_execution_video

    if motion_path.is_file() and quality_path.is_file():
        quality = read_json(quality_path)
        if config.get('execution', {}).get('save_video', True):
            video = save_execution_video(rollout)
            quality['execution_video'] = str(video.resolve())
        return quality

    import mujoco
    import torch

    from data_pipeline.motion.quality import (
        MotionScreeningConfig,
        screen_executed_motion,
    )
    from msc._omg.kinematics import G1Kinematics
    from omg.tracking.holomotion.reference import resample_qpos
    from omg.tracking.holomotion.runner import (
        TrackerRunConfig,
        run_holomotion_tracker,
    )

    torch.set_num_threads(4)

    with np.load(reference, allow_pickle=False) as data:
        ref = data['qpos_36'].copy()
        source_times = data['source_timestamps'].copy()

    if not rollout.is_file():
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

    # Save every physical rollout, including those later rejected by screening.
    if config.get('execution', {}).get('save_video', True):
        save_execution_video(rollout)

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
        raise ValueError('Rollout does not match reference/timing; use a new work directory')

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
    aligned, executed_mask = resample_executed(ref, executed, control_fps=fps)
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
        'execution_video': str((output / 'execution.mp4').resolve()),
        'production_ready': False,
    }
    np.savez_compressed(
        output / 'paired_motion.npz',
        qpos_36=aligned,
        timestamps=source_times,
        executed_state_mask=executed_mask,
        fps=np.float32(30),
    )
    write_json(output / 'quality.json', quality)
    return quality
