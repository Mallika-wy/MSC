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
    lead = int(os.environ.get('RANK', '0')) == 0
    if lead:
        print(f'[MSC] PyTorch {torch.__version__} loaded; reading configuration.', flush=True)
    cfg = load_config(args.config, args.overrides)
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        if cfg.train.device.startswith('cuda'):
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            cfg.train.device = f'cuda:{local_rank}'
        dist.init_process_group(backend='nccl' if cfg.train.device.startswith('cuda') else 'gloo')
    seed_everything(cfg.train.seed)
    if lead:
        print('[MSC] Reading published training window indices...', flush=True)
    dataset = build_dataset(cfg)
    if lead:
        print(f'[MSC] Dataset ready: {len(dataset):,} windows; loading frozen T5 and motion model...', flush=True)
    model = build_model(cfg, dataset.audio_metadata, getattr(dataset,"feature_metadata",None))
    try:
        if lead:
            print('[MSC] Model loaded; initializing optimizer, device and validation data...', flush=True)
        trainer = Trainer(model, dataset, cfg)
        if lead:
            print(f'[MSC] Starting training on {cfg.train.device}; first metrics after '
                  f'{cfg.train.log_every} optimizer steps. Expert compilation: {cfg.train.compile_expert}.', flush=True)
        trainer.fit()
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
