"""Streaming train-only dynamics scales for the exact selected task/window pools."""
import argparse
import json
from pathlib import Path
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('overrides',nargs='*')
    args = parser.parse_args()
    cfg = load_config(args.config,args.overrides)
    if cfg.data.split != 'train' or cfg.data.backend != 'msc_release':
        parser.error('Statistics require msc_release Train Split')
    output = Path(args.output).resolve()
    if any(output.is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        parser.error('Statistics output must be outside Release roots')
    result = compute_stats(build_dataset(cfg))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(output)


if __name__ == '__main__':
    main()
