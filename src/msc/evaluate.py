import argparse
import json
import torch
from torch.utils.data import DataLoader
from msc.sample import load_model
from msc.factory import build_dataset, to_device, seed_everything
from msc.data.collator import collate_motion


def main():
    p = argparse.ArgumentParser(description="Evaluate masked flow loss; not a motion quality benchmark")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root")
    p.add_argument("--split", default="val")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--t5-path")
    args = p.parse_args()
    if args.batches < 1:
        p.error("--batches must be positive")
    torch.set_num_threads(2)
    seed_everything(42)
    model, cfg, _ = load_model(args.checkpoint, args.device, args.t5_path)
    cfg.data.split = args.split
    if args.data_root:
        cfg.data.root = args.data_root
    dataset = build_dataset(cfg)
    if dataset.audio_metadata != model.audio_metadata:
        raise ValueError("Evaluation audio provenance differs from checkpoint")
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, collate_fn=collate_motion)
    total = count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            loss = model.compute_loss(to_device(batch, args.device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            size = len(batch["motion_future"])
            total += loss.item() * size
            count += size
    print(json.dumps(dict(masked_flow_loss=total / count, samples=count)))


if __name__ == "__main__":
    main()
