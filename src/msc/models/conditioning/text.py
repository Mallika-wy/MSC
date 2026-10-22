"""Local-only T5 encoding and versioned frozen-token cache."""
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn


def model_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Local T5 directory required: {path}")
    digest = hashlib.sha256()
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix in {".json", ".model", ".safetensors", ".bin"}:
            digest.update(item.name.encode())
            with item.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


class TextAdapter(nn.Sequential):
    def __init__(self,input_dim,output_dim=512):
        super().__init__(nn.LayerNorm(input_dim),nn.Linear(input_dim,output_dim),
                         nn.SiLU(),nn.Linear(output_dim,output_dim))


class T5TextEncoder(nn.Module):
    def __init__(self, model_path, *, frozen=True, max_length=50, cache_dir=None):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        if not frozen and cache_dir:
            raise ValueError("Static text cache cannot be used with trainable T5")
        self.frozen = frozen
        self.max_length = max_length
        self.fingerprint = model_fingerprint(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.encoder = T5EncoderModel.from_pretrained(str(model_path), local_files_only=True)
        self.encoder.requires_grad_(not frozen)
        self.output_dim = self.encoder.config.d_model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def cache_identity(self, text):
        return dict(version=1, model_sha256=self.fingerprint, text=text,
                    max_length=self.max_length, dtype="float32")

    def cache_path(self, text):
        key = hashlib.sha256(json.dumps(self.cache_identity(text), sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.pt"

    def _encode(self, captions, device):
        tokens = self.tokenizer(captions, max_length=self.max_length, padding="max_length",
                                truncation=True, return_tensors="pt").to(device)
        # Cache entries are FP32; frozen live encoding uses the same precision
        # even when the surrounding motion model runs under autocast.
        with torch.no_grad() if self.frozen else nullcontext():
            with torch.autocast(device_type=torch.device(device).type, enabled=False) if self.frozen else nullcontext():
                features = self.encoder(**tokens).last_hidden_state
        return features.float(), tokens.attention_mask.bool()

    def forward(self, captions, device):
        if not captions:
            raise ValueError("Empty text batch")
        if self.cache_dir is None:
            return self._encode(captions, device)
        entries = []
        for caption in captions:
            path = self.cache_path(caption)
            if path.exists():
                obj = torch.load(path, map_location="cpu", weights_only=True)
                if obj["identity"] != self.cache_identity(caption):
                    raise ValueError(f"Text cache identity mismatch: {path}")
                entries.append((obj["features"].to(device), obj["mask"].to(device)))
            else:
                f, m = self._encode([caption], device)
                entries.append((f[0], m[0]))
        return torch.stack([f for f, _ in entries]), torch.stack([m for _, m in entries])

    @torch.no_grad()
    def write_cache(self, captions, device):
        if not self.frozen or self.cache_dir is None:
            raise ValueError("Cache writing requires frozen T5 and cache_dir")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for caption in dict.fromkeys(captions):
            if self.cache_path(caption).exists():
                continue
            features, mask = self._encode([caption], device)
            path = self.cache_path(caption)
            temp = path.with_suffix(".tmp")
            torch.save(dict(identity=self.cache_identity(caption), features=features[0].cpu(), mask=mask[0].cpu()), temp)
            temp.replace(path)
