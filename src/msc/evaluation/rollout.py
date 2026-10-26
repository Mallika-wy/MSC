"""Full-motion selection and autoregressive generation, at the canonical 30 FPS."""
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
import torch

from data_pipeline.release.export import read_motion
from data_pipeline.release.training_index import TrainingMotion
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics
from msc.motion.rotations import quaternion_apply


@dataclass
class MotionCase:
    release_index: int
    motion: TrainingMotion
    task: str
    spans: list

    def condition_at(self, frame):
        # Prefer a valid full-horizon condition; at the final partial horizon,
        # its annotation/link remains valid for another 29 target frames.
        full = [s for s in self.spans if s[0] <= frame < s[1]]
        covered = full or [s for s in self.spans if s[0] <= frame < s[1] + 29]
        return max(covered, key=lambda s: s[0])[2]

    def record(self):
        return dict(release_index=self.release_index, motion=asdict(self.motion), task=self.task,
                    frames=self.motion.num_frames, duration_s=self.motion.num_frames / 30,
                    conditions=[dict(start_frame=a, end_frame_exclusive=b + 29,
                                     condition=list(c)) for a, b, c in self.spans])


def covers_full_motion(spans, frames):
    """Window-start interval [a,b) covers target frames [a,b+29)."""
    cursor = 5
    for start, end, _ in sorted(spans, key=lambda s: s[0]):
        if start > cursor:
            return False
        cursor = max(cursor, end + 29)
        if cursor >= frames:
            return True
    return False


def select_motions(dataset, count, seed):
    """Uniformly sample distinct motions, not windows weighted by clip length."""
    import pyarrow as pa
    import pyarrow.compute as pc
    table = dataset.rollout_spans.table.unify_dictionaries()
    grouped = table.group_by(['release_index', 'motion_id', 'task']).aggregate(
        [('start', 'min'), ('end', 'max'), ('num_frames', 'max')])
    coarse = pc.and_(pc.less_equal(grouped['start_min'], 5),
                     pc.greater_equal(pc.add(grouped['end_max'], 29), grouped['num_frames_max']))
    grouped = grouped.filter(coarse)
    for name in ('motion_id', 'task'):
        grouped = grouped.set_column(grouped.schema.get_field_index(name), name,
                                     pc.cast(grouped[name], pa.string()))
    grouped = grouped.sort_by([
        ('release_index', 'ascending'), ('motion_id', 'ascending'), ('task', 'ascending')])
    cases = []
    for index in np.random.default_rng(seed).permutation(len(grouped)):
        row = grouped.slice(int(index), 1).to_pylist()[0]
        records = list(dataset.motion_spans(row['release_index'], row['motion_id'], row['task']))
        spans = [(lo, hi, condition) for _, _, _, lo, hi, condition in records]
        motion = records[0][1]
        if covers_full_motion(spans, motion.num_frames):
            cases.append(MotionCase(row['release_index'], motion, row['task'], spans))
            if len(cases) == count:
                return cases
    raise ValueError(f'Only {len(cases)} motions have conditions covering every generated frame; requested {count}')


def motion_seed(seed, case):
    key = f'{seed}:{case.release_index}:{case.motion.motion_id}:{case.task}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little') % (2**63)


def reanchor_previous(codec, previous, executed_frames, old_pos, old_quat, new_pos, new_quat):
    """Shift the unexecuted plan; change coordinates while preserving all 125D."""
    tail = previous[:, executed_frames:]
    shifted = torch.cat((tail, tail[:, -1:].expand(-1, executed_frames, -1)), 1)
    components = codec.split_features(shifted)
    world = codec.decode_to_world_qpos36(components, old_pos, old_quat)
    _, heading, _ = codec._canonical_anchor(old_pos, old_quat)
    links = quaternion_apply(heading.unsqueeze(-2).expand(1, 30, 29, 4),
                             components.body_link_pos_local) + old_pos.unsqueeze(-2)
    bodies = torch.cat((world[..., :3].unsqueeze(-2), links), -2)
    return codec.assemble_features(codec.canonicalize(world, bodies, None, new_pos, new_quat, 30.))


def condition_arguments(dataset, case, frame, model, device):
    # Read external conditions only. Tail queries do not read a nonexistent
    # 35-frame GT window, and no later motion history is taken from the dataset.
    sample = {}
    condition = case.condition_at(frame)
    dataset._init_codec()
    dataset._condition(sample, case.release_index, case.task, frame, condition)
    if case.task == 'text':
        return dict(text=sample['text']), condition
    names = ('audio_tokens', 'audio_mask', 'audio_times') if case.task == 'audio' else (
        'video_tokens', 'video_mask', 'video_intervals')
    arguments = {name: sample[name][None].to(device) for name in names}
    if case.task == 'audio':
        arguments['audio_metadata'] = model.audio_metadata
    return arguments, condition


@torch.no_grad()
def generate_full_motion(model, dataset, case, *, seed, device, rtc=False, num_steps=20,
                         cfg_scale=1., replan_frames=15, guidance_alpha=.9,
                         mask_schedule='exponential', precision='bf16', controller_factory=None):
    codec = G1MotionFeatureCodec(G1Kinematics(), num_prev_states=5,
                                canonical_frame_idx=4, rotation_representation='rot6d')
    target, source_times = read_motion(dataset.roots[case.release_index], case.motion)
    target = target.astype(np.float32)
    history = target[:5].copy()
    controller = controller_factory(history) if controller_factory else None
    committed = [history.copy()]
    previous = None
    chunk_records = []
    generator = torch.Generator(device=device).manual_seed(motion_seed(seed, case))
    old_executed = replan_frames
    frame = 5
    while frame < len(target):
        take = min(replan_frames, len(target) - frame)
        h = torch.from_numpy(history.copy())[None]
        physical, pos, quat = codec.prev_state_features_from_history(
            h, codec.kinematics.forward_body_positions(h), None, 30.)
        prior = None
        if rtc and previous is not None:
            prior = reanchor_previous(codec, previous, old_executed,
                                      old_pos, old_quat, pos, quat).to(device)
        arguments, condition = condition_arguments(dataset, case, frame, model, device)
        noise = torch.randn((1, 30, 125), device=device, generator=generator)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=str(device).startswith('cuda') and precision == 'bf16'):
            prediction = model.sample(physical.to(device), initial_noise=noise, num_steps=num_steps,
                cfg_scale=cfg_scale, previous_motion=prior,
                rtc_mode='test_time' if prior is not None else 'off', inference_delay=0,
                execution_horizon=replan_frames, rtc_mask_schedule=mask_schedule,
                rtc_guidance_alpha=guidance_alpha, **arguments).float().cpu()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        latency = time.monotonic() - started
        world = codec.decode_to_world_qpos36(codec.split_features(prediction), pos, quat)[0].numpy()
        if not np.isfinite(world).all():
            raise FloatingPointError(f'Nonfinite motion at {case.motion.motion_id}, frame {frame}')
        committed.append(world[:take].copy())
        chunk_records.append(dict(start_frame=frame, generated_frames=take,
            rtc=prior is not None, inference_seconds=latency, condition=list(condition),
            history_source='GT initialization' if previous is None else (
                'SONIC measured state' if controller else 'generated motion')))
        if controller:
            controller.execute(world, take)
            if frame + take < len(target):
                history = controller.measured_history()
        else:
            history = np.concatenate((history, world[:take]))[-5:].copy()
        previous, old_pos, old_quat, old_executed = prediction, pos, quat, take
        frame += take
    predicted = np.concatenate(committed)
    arrays = dict(target=target, predicted=predicted, source_timestamps=source_times,
                  timestamps=np.arange(len(target)) / 30, fps=np.float32(30))
    if controller:
        arrays['executed'] = np.concatenate((target[:5], controller.trajectory(len(target) - 5)))
        arrays['executed_50hz'] = np.asarray(controller.states, dtype=np.float32)
    assert len(predicted) == len(target)
    return arrays, chunk_records


def trajectory_metrics(predicted, target):
    """Exclude the five shared initialization frames from all reported errors."""
    kin = G1Kinematics()
    pred, gt = predicted[5:], target[5:]
    pred_fk = kin.forward_body_positions(torch.from_numpy(pred)[None])
    gt_fk = kin.forward_body_positions(torch.from_numpy(gt)[None])
    amplitude = float(pred[:, 7:].std(0).mean())
    target_amplitude = float(gt[:, 7:].std(0).mean())
    pv = np.diff(predicted[4:, 7:], axis=0) * 30
    tv = np.diff(target[4:, 7:], axis=0) * 30
    return dict(evaluated_frames=len(pred), joint_rmse_rad=float(np.sqrt(np.mean((pred[:, 7:] - gt[:, 7:])**2))),
        root_position_rmse_m=float(np.sqrt(np.mean((pred[:, :3] - gt[:, :3])**2))),
        fk_mpjpe_m=float((pred_fk - gt_fk).norm(dim=-1).mean()),
        joint_amplitude_rad=amplitude, target_joint_amplitude_rad=target_amplitude,
        amplitude_ratio=amplitude / target_amplitude if target_amplitude > 1e-8 else None,
        velocity_residual_rms=float(np.sqrt(np.mean((pv - tv)**2))),
        root_endpoint_error_m=float(np.linalg.norm(pred[-1, :3] - gt[-1, :3])),
        root_z_min=float(pred[:, 2].min()), root_z_max=float(pred[:, 2].max()),
        min_upright_z=float((1 - 2 * (pred[:, 4]**2 + pred[:, 5]**2)).min()))
