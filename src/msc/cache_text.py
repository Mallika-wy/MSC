import argparse
from pathlib import Path
import torch
from msc.models.conditioning.text import T5TextEncoder


def main():
    p = argparse.ArgumentParser(description="Precompute local frozen T5 tokens from one caption per line")
    p.add_argument("--t5-path", required=True)
    p.add_argument("--captions", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    encoder = T5TextEncoder(args.t5_path, max_length=args.max_length, cache_dir=args.cache_dir).to(args.device)
    captions = Path(args.captions).read_text().splitlines()
    encoder.write_cache(captions, torch.device(args.device))
    print(f"Cached {len(set(captions))} captions")


if __name__ == "__main__":
    main()
