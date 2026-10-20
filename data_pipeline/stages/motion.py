"""G1 decoding, numerical validation and optional tracker execution."""
from pathlib import Path

import numpy as np

from data_pipeline.io import read_json, write_json
from data_pipeline.schemas.manifest import resolve_uri


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


def read_global_rows(directory, start, length):
    import pyarrow.parquet as pq
    if type(start) is not int or type(length) is not int or start < 0 or length < 2:
        raise ValueError('Invalid source row range')
    remaining, offset, pieces, timestamps = length, 0, [], []
    for path in sorted(Path(directory).glob('chunk-*/*.parquet')):
        file = pq.ParquetFile(path)
        for group in range(file.num_row_groups):
            count = file.metadata.row_group(group).num_rows
            if remaining and offset + count > start:
                lo = max(0, start - offset)
                take = min(count - lo, remaining)
                data = file.read_row_group(group, columns=['observation.state', 'timestamp']).slice(lo, take)
                pieces.append(np.asarray(data['observation.state'].to_pylist(), dtype=np.float32))
                timestamps.append(data['timestamp'].to_numpy().astype(np.float64))
                remaining -= take
                start += take
            offset += count
            if not remaining:
                qpos = np.concatenate(pieces)
                return qpos, np.concatenate(timestamps)
    raise ValueError('Source row range exceeds local parquet files')


def decode(source, inventory, inventory_root):
    path = resolve_uri(inventory_root, source.uri, inventory.external_roots)
    if source.fps != 30:
        raise ValueError('Native G1 requires 30 Hz; use an explicit retarget/resampling stage')
    if source.format == 'omg_lerobot_g1':
        qpos, times = read_global_rows(path, source.metadata['row_start'], source.metadata['num_frames'])
    elif source.format == 'g1_qpos_wxyz':
        with np.load(path, allow_pickle=False) as data:
            qpos = data['qpos_36'].astype(np.float32)
            times = data['timestamps'].astype(np.float64)

    else:
        raise ValueError(f'No native decoder for {source.format}; retarget is required')
    validate_qpos(qpos, times)
    # Legacy LeRobot timestamp columns may be float32; tolerate quantization, not a missing frame.
    if not np.allclose(times, times[0] + np.arange(len(times))/30, atol=1e-5, rtol=0):
        raise ValueError('Nonuniform native G1 timeline')
    return qpos, times


def resample_executed(reference, executed, *, control_fps=50., motion_fps=30.):
    from .motion_quality import resample_motion_at_times
    if control_fps != 50 or motion_fps != 30:
        raise ValueError('Current AIST controller contract is 50 Hz execution and 30 Hz labels')
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
    from data_pipeline.stages.execution_video import save_execution_video

    if motion_path.is_file() and quality_path.is_file():
        video = save_execution_video(rollout)
        quality = read_json(quality_path)
        quality['execution_video'] = str(video.resolve())
        return quality

    import mujoco
    import torch

    from data_pipeline.stages.motion_quality import (
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
