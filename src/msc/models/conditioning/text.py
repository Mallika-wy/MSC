"""Local-only frozen T5 encoding."""
import hashlib
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
    def __init__(self, model_path, *, frozen=True, max_length=50):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        self.frozen = frozen
        self.max_length = max_length
        self.fingerprint = model_fingerprint(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.encoder = T5EncoderModel.from_pretrained(str(model_path), local_files_only=True)
        self.encoder.requires_grad_(not frozen)
        self.output_dim = self.encoder.config.d_model
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def _encode(self, captions, device):
        tokens = self.tokenizer(captions, max_length=self.max_length, padding=True,
                                truncation=True, return_tensors="pt").to(device)
        # Follow the training autocast context, as in the reference OMG encoder.
        with torch.no_grad():
            features = self.encoder(**tokens).last_hidden_state
        return features.float(), tokens.attention_mask.bool()

    def forward(self, captions, device):
        if not captions:
            raise ValueError("Empty text batch")
        # Windows from the same annotation often repeat a caption in one batch.
        # Encode each distinct string once, online, then restore sample order.
        unique = list(dict.fromkeys(captions))
        indices = {caption: i for i, caption in enumerate(unique)}
        inverse = torch.tensor([indices[caption] for caption in captions], device=device)
        features, mask = self._encode(unique, device)
        return features[inverse], mask[inverse]
