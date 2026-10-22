import argparse
from pathlib import Path
import torch
import json
from data_pipeline.io import sha256_file
from msc.data.release_dataset import fingerprint
from msc.models.conditioning.text import T5TextEncoder


def main():
    p = argparse.ArgumentParser(description="Precompute local frozen T5 tokens from one caption per line")
    p.add_argument("--t5-path", required=True)
    p.add_argument("--captions")
    p.add_argument('--release-root',action='append',default=[])
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-length", type=int, default=50)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    encoder = T5TextEncoder(args.t5_path, max_length=args.max_length, cache_dir=args.cache_dir).to(args.device)
    if bool(args.captions) == bool(args.release_root):
        p.error('Choose --captions or one/more --release-root')
    if args.release_root:
        from data_pipeline.schema.manifest import Inventory
        captions = sorted({a.text for root in args.release_root for a in Inventory.read(root).annotations if a.text.strip()})
    else:
        captions = Path(args.captions).read_text().splitlines()
    root = Path(args.cache_dir)
    if any(root.resolve().is_relative_to(Path(release).resolve()) for release in args.release_root):
        p.error('T5 cache output must be outside read-only Release roots')
    root.mkdir(parents=True,exist_ok=True)
    identity = dict(checkpoint_hash=encoder.fingerprint,output_dim=encoder.output_dim,
                    max_length=args.max_length,preprocess='T5Tokenizer-v1',dtype='float32')
    manifest = dict(version=2,encoder=identity,entries={},checksums={})
    for text in dict.fromkeys(captions):
        name = fingerprint(text)+'.pt'
        features,mask = encoder._encode([text],torch.device(args.device))
        temporary = root / (name+'.tmp')
        torch.save(dict(encoder=identity,text=text,features=features[0].cpu(),mask=mask[0].cpu()),temporary)
        temporary.replace(root/name)
        manifest['entries'][fingerprint(text)] = name
        manifest['checksums'][name] = sha256_file(root/name)
    temporary = root/'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest,sort_keys=True,indent=2))
    temporary.replace(root/'manifest.json')
    print(f"Cached {len(set(captions))} captions")


if __name__ == "__main__":
    main()
