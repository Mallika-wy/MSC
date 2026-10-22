"""Execute a G1 reference at 50 Hz and return screened 30 Hz label arrays."""

import numpy as np

from .validation import validate_qpos


def resample_executed(reference, executed, *, control_fps=50., motion_fps=30.):
    from .quality import resample_motion_at_times
    if control_fps != 50 or motion_fps != 30:
        raise ValueError('Controller contract is 50 Hz execution and 30 Hz labels')
    query = np.arange(len(reference), dtype=np.float64) / motion_fps
    source_times = np.arange(len(executed) + 1, dtype=np.float64) / control_fps
    aligned = resample_motion_at_times(np.concatenate((reference[:1], executed)), source_times, query)
    return aligned, query > 0


def track(reference, execution, policy, output, *, diagnostic=False):
    """只运行计算；output 是外部 Tracker 所需的临时文件桥接目录。"""
    output.mkdir(parents=True, exist_ok=True)
    rollout = output / 'tracker_rollout.npz'

    import torch

    from data_pipeline.motion.quality import (
        MotionScreeningConfig,
        screen_executed_motion,
    )
    from msc.motion.kinematics import G1Kinematics
    from .sonic import run

    torch.set_num_threads(1 if execution.get('inference_batch_size', 1) > 1 else 4)

    ref = np.asarray(reference, dtype=np.float32)
    times = np.arange(len(ref), dtype=np.float64) / 30
    validate_qpos(ref, times)
    reference_path = output / 'reference.npz'
    np.savez_compressed(reference_path, qpos_36=ref, source_timestamps=times, fps=np.float32(30))
    run(ref, execution, rollout)
    if diagnostic:
        from .tracking_video import save_execution_video
        save_execution_video(rollout)

    with np.load(rollout, allow_pickle=False) as data:
        executed = data['executed_qpos_36'].copy()
        fps = float(data['fps'])

    kinematics = G1Kinematics()
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
        'production_ready': False,
    }
    return aligned[executed_mask], quality
