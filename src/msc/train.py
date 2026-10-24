import argparse
import os
import torch
import torch.distributed as dist
from msc.config import load_config
from msc.factory import seed_everything, build_dataset, build_model
from msc.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train independent MSC flow matching")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        if cfg.train.device.startswith('cuda'):
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            cfg.train.device = f'cuda:{local_rank}'
        dist.init_process_group(backend='nccl' if cfg.train.device.startswith('cuda') else 'gloo')
    seed_everything(cfg.train.seed)
    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.audio_metadata, getattr(dataset,"feature_metadata",None))
    try:
        Trainer(model, dataset, cfg).fit()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
