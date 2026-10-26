"""Exact Train-window 125D moments without expanding overlapping futures.

FK is evaluated once per stored frame. Prefix sums integrate each 30-frame
future, then its positions and rotations are expressed in the anchor heading.
Overlapping annotation/task spans retain their sampling multiplicity.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from msc.config import load_config
from msc.factory import build_dataset
from msc.motion.kinematics import G1Kinematics
from msc.motion.rotations import quaternion_to_matrix


@torch.no_grad()
def canonical_moments(qpos, starts, ends, kinematics, chunk_frames=65536):
    """Return count, sum and squared sum of the dataset's future features.

    starts/ends are absolute future-start intervals in this shard, not motion
    frame intervals. Each start uses the preceding frame as its anchor.
    """
    events = np.zeros(len(qpos) + 1, dtype=np.int64)
    np.add.at(events, np.asarray(starts) - 1, 1)
    np.add.at(events, np.asarray(ends) - 1, -1)
    multiplicity = np.cumsum(events[:-1])
    if multiplicity.min() < 0:
        raise ValueError('Invalid window intervals')
    selected = np.flatnonzero(multiplicity)
    if not len(selected) or selected[0] < 4 or selected[-1] + 30 >= len(qpos):
        raise ValueError('Invalid 5-history/30-future window bounds')
    indices = torch.as_tensor(selected, device=qpos.device)
    weights = torch.as_tensor(multiplicity[selected], device=qpos.device, dtype=torch.float64)
    count = int(multiplicity.sum()) * 30

    def future_sum(values):
        prefix = values.double().cumsum(0)
        return prefix[indices + 30] - prefix[indices]

    def weighted(values):
        shape = (-1,) + (1,) * (values.ndim - 1)
        return (values * weights.reshape(shape)).sum(0)

    rotations = quaternion_to_matrix(F.normalize(qpos[:, 3:7], dim=-1))
    yaw = torch.atan2(rotations[indices, 1, 0], rotations[indices, 0, 0]).double()
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]

    # Canonical root rotation: first two rows of Rz(-anchor_yaw) @ R_world.
    rx, ry = rotations[:, 0].double(), rotations[:, 1].double()
    sx, sy = future_sum(rx), future_sum(ry)
    xx, yy, xy = future_sum(rx.square()), future_sum(ry.square()), future_sum(rx * ry)
    rot_sum = torch.stack((weighted(c * sx + s * sy), weighted(-s * sx + c * sy))).flatten()
    rot_square = torch.stack((weighted(c*c*xx + s*s*yy + 2*c*s*xy),
                              weighted(s*s*xx + c*c*yy - 2*c*s*xy))).flatten()
    del rotations, rx, ry, sx, sy, xx, yy, xy

    # The root and all 29 link positions share the same anchor translation.
    positions = torch.empty((len(qpos), 30, 3), device=qpos.device, dtype=torch.float64)
    positions[:, 0] = qpos[:, :3]
    for first in range(0, len(qpos), chunk_frames):
        last = min(first + chunk_frames, len(qpos))
        positions[first:last, 1:] = kinematics.forward_body_positions(qpos[first:last])[:, 1:]
    # Center the shard to reduce cancellation for trajectories far from origin.
    origin = qpos[0, :3].double()
    positions -= origin
    anchor = qpos[indices, :3].double() - origin
    sums = []
    centered_squares = []
    for axis in range(3):
        values = positions[:, :, axis]
        total = future_sum(values)
        offset = anchor[:, axis, None]
        sums.append(total - 30 * offset)
        centered_squares.append(future_sum(values.square()) - 2 * offset * total + 30 * offset.square())
    px, py = positions[:, :, 0], positions[:, :, 1]
    ax, ay = anchor[:, 0, None], anchor[:, 1, None]
    cross = future_sum(px * py) - ax * future_sum(py) - ay * future_sum(px) + 30 * ax * ay
    xx, yy, zz = centered_squares
    position_sum = torch.stack((weighted(c * sums[0] + s * sums[1]),
                                weighted(-s * sums[0] + c * sums[1]), weighted(sums[2])), -1)
    position_square = torch.stack((weighted(c*c*xx + s*s*yy + 2*c*s*cross),
                                   weighted(s*s*xx + c*c*yy - 2*c*s*cross), weighted(zz)), -1)
    joints = qpos[:, 7:].double()
    joint_sum, joint_square = weighted(future_sum(joints)), weighted(future_sum(joints.square()))
    total = torch.cat((position_sum[0], rot_sum, joint_sum, position_sum[1:].flatten()))
    square = torch.cat((position_square[0], rot_square, joint_square, position_square[1:].flatten()))
    if total.shape != (125,) or not torch.isfinite(total).all() or not torch.isfinite(square).all():
        raise ValueError('Nonfinite normalization moments')
    return count, total, square


def main():
    import pyarrow.parquet as pq
    from data_pipeline.io import resolve_uri
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg.data.split != 'train':
        parser.error('Normalization statistics require Train Split')
    output = Path(args.output).resolve()
    if any(output.is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        parser.error('Statistics output must be outside Release roots')
    rank, world_size = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = args.device
    if device.startswith('cuda'):
        if world_size > 1:
            device = f"cuda:{os.environ['LOCAL_RANK']}"
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group('gloo')
    try:
        torch.set_num_threads(cfg.train.num_threads)
        dataset = build_dataset(cfg)
        table = dataset.spans.table.select(['release_index', 'storage_uri', 'row_start', 'start', 'end']).unify_dictionaries()
        groups = table.group_by(['release_index', 'storage_uri'], use_threads=False).aggregate([
            ('row_start', 'list'), ('start', 'list'), ('end', 'list')]).to_pylist()
        kin = G1Kinematics().to(device)
        totals = torch.zeros(251, dtype=torch.float64)
        started = time.monotonic()
        for number in range(rank, len(groups), world_size):
            group = groups[number]
            path = resolve_uri(dataset.roots[group['release_index']], group['storage_uri'])
            column = pq.read_table(path, columns=['qpos36'], use_threads=False)['qpos36'].combine_chunks()
            qpos = torch.from_numpy(column.values.to_numpy().reshape(-1, 36).copy()).to(device)
            offsets = np.asarray(group['row_start_list'])
            count, total, square = canonical_moments(qpos, offsets + group['start_list'], offsets + group['end_list'], kin)
            totals[0] += count
            totals[1:126] += total.cpu()
            totals[126:] += square.cpu()
            print(json.dumps(dict(rank=rank, shard=number + 1, shards=len(groups), count=int(totals[0]),
                                  elapsed_seconds=time.monotonic()-started)), flush=True)
            del qpos, column
        if world_size > 1:
            dist.all_reduce(totals)
        if int(totals[0]) != len(dataset) * 30:
            raise ValueError('Normalization count does not match the entire Train window pool')
        if rank == 0:
            mean = totals[1:126] / totals[0]
            variance = totals[126:] / totals[0] - mean.square()
            if variance.min() < -1e-8:
                raise ValueError('Negative feature variance')
            result = dict(version=1, representation='g1_canonical_125d',
                feature='root_pos_local+root_rot_local_rot6d_6+joint_dof_29+body_link_pos_local_29x3',
                rotation_representation='rot6d', split='train', dataset_identity=dataset.identity,
                population='all_future_frames_of_all_training_windows_with_sampling_multiplicity',
                windows=len(dataset), count=int(totals[0]), num_prev_states=5, canonical_frame_idx=4,
                sequence_length=30, fps=30, mean=mean.tolist(),
                std=variance.clamp_min(0).sqrt().clamp_min(1e-6).tolist(), std_floor=1e-6,
                algorithm='shard_fk_heading_prefix_moments_float64_v1')
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(output)
            print(f'[MSC] Normalization statistics saved to {output}', flush=True)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
