"""Streaming train-only dynamics scales for the exact selected task/window pools."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from msc.config import load_config
from msc.factory import build_dataset
from msc.models.losses import joint_dynamics


def compute_stats(dataset):
    count = [0,0]
    sums = [torch.zeros(29,dtype=torch.float64) for _ in range(2)]
    squares = [x.clone() for x in sums]
    for index in range(len(dataset)):
        item = dataset[index]
        v,a,vm,am = joint_dynamics(item['motion_future'][None,...,9:38],
                                   item['motion_history'][None,...,9:38],item['motion_mask'][None])
        for j,(value,mask) in enumerate(((v,vm),(a,am))):
            values = value[mask].double()
            count[j] += len(values)
            sums[j] += values.sum(0)
            squares[j] += values.square().sum(0)
    if min(count)==0:
        raise ValueError('No valid training dynamics')
    scales = [(squares[j]/count[j]-(sums[j]/count[j]).square()).clamp_min(0).sqrt().clamp_min(1e-4).tolist() for j in range(2)]
    return dict(version=1,split='train',dataset_identity=dataset.identity,
                velocity_scale=scales[0],acceleration_scale=scales[1],count=count,scale_floor=1e-4)


def shard_moments(task):
    """Each joint frame is weighted by its exact number of training windows."""
    import pyarrow.parquet as pq
    path, starts, ends = task
    column = pq.read_table(path, columns=['qpos36'])['qpos36'].combine_chunks()
    joints = column.values.to_numpy().reshape(-1, 36)[:, 7:]
    events = np.zeros(len(joints) + 1, dtype=np.int64)
    np.add.at(events, starts, 1)
    np.add.at(events, ends, -1)
    cumulative = np.cumsum(np.cumsum(events[:-1]))
    weights = cumulative.copy()
    weights[30:] -= cumulative[:-30]
    weights = weights[2:]
    values = (30 * (joints[2:] - joints[1:-1]),
              900 * (joints[2:] - 2 * joints[1:-1] + joints[:-2]))
    sums, squares = [], []
    for value in values:
        value = value.astype(np.float64)
        sums.append((value * weights[:, None]).sum(0))
        squares.append((value ** 2 * weights[:, None]).sum(0))
    return int(weights.sum()), np.array(sums), np.array(squares)


def compute_release_stats(dataset, workers=4):
    """Equivalent to compute_stats, without decoding overlapping windows repeatedly.

    Canonical joint channels 9:38 are unchanged qpos joint angles 7:36. Every
    published window has 30 valid future frames and at least two history frames.
    Group by shard to read each joint trajectory once, preserving window weights.
    """
    table = dataset.spans.table.select(['release_index', 'storage_uri', 'row_start', 'start', 'end']).unify_dictionaries()
    groups = table.group_by(['release_index', 'storage_uri']).aggregate([
        ('row_start', 'list'), ('start', 'list'), ('end', 'list')]).to_pylist()
    tasks = [(str(dataset.roots[g['release_index']] / g['storage_uri']),
              np.array(g['row_start_list']) + np.array(g['start_list']),
              np.array(g['row_start_list']) + np.array(g['end_list'])) for g in groups]
    count = 0
    sums, squares = np.zeros((2, 29)), np.zeros((2, 29))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for n, total, square in pool.map(shard_moments, tasks):
            count += n
            sums += total
            squares += square
    scales = np.sqrt(np.maximum(0, squares / count - (sums / count) ** 2)).clip(1e-4)
    return dict(version=1, split='train', dataset_identity=dataset.identity,
                velocity_scale=scales[0].tolist(), acceleration_scale=scales[1].tolist(),
                count=[count, count], scale_floor=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('overrides',nargs='*')
    args = parser.parse_args()
    cfg = load_config(args.config,args.overrides)
    if cfg.data.split != 'train' or cfg.data.backend != 'msc_release':
        parser.error('Statistics require msc_release Train Split')
    output = Path(args.output).resolve()
    if any(output.is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        parser.error('Statistics output must be outside Release roots')
    result = compute_release_stats(build_dataset(cfg), args.workers)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(output)


if __name__ == '__main__':
    main()
