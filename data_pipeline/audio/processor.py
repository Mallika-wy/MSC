"""Independent audio processing with a single consolidated MERT cache."""
import hashlib
import json
from pathlib import Path

import numpy as np

from data_pipeline.io import sha256_file
from data_pipeline.storage.store import Store, digest
from data_pipeline.storage.artifacts import FeatureArtifact
from .mert import MertEncoder


class AudioCache:
    """Own feature paths and fingerprints; the encoder only computes arrays."""

    def __init__(self, model_path, *, chunk_seconds=5, device='cpu'):
        self.encoder = MertEncoder(model_path, chunk_seconds=chunk_seconds, device=device)
        model = Path(model_path)
        self.chunk_seconds = chunk_seconds
        self.checkpoint_hash = sha256_file(model / 'pytorch_model.bin')
        self.preprocess_hash = hashlib.sha256(json.dumps({
            'preprocessor': sha256_file(model / 'preprocessor_config.json'),
            'chunk_seconds': chunk_seconds, 'implementation': sha256_file(Path(__file__).with_name('mert.py')),
            'model_code': {p.name: sha256_file(p) for p in model.glob('*.py')},
            'model_config': sha256_file(model / 'config.json'),
        }, sort_keys=True).encode()).hexdigest()

    def extract(self, wav, output):
        checksum = digest(wav)
        key = hashlib.sha256((checksum + self.checkpoint_hash + self.preprocess_hash).encode()).hexdigest()
        store = Store(Path(output) / 'features.sqlite')
        with store.lock(key):
            cached = store.get(key)
            if cached is not None:
                report, _ = cached
                return {**report, 'arrays': dict(store=str(store.path), key=key)}
            values, times, duration = self.encoder.encode(wav)
            if (times.shape != (len(values),) or times.dtype != np.float64 or
                not np.isfinite(values).all() or not np.isfinite(times).all() or
                (np.diff(times) <= 0).any()):
                raise ValueError('Invalid MERT features or timestamps')
            report = dict(encoder='m-a-p/MERT-v1-95M', checkpoint_hash=self.checkpoint_hash,
                preprocess_hash=self.preprocess_hash, input_checksum=checksum,
                chunk_seconds=self.chunk_seconds, shape=list(values.shape), duration_sec=duration)
            reference = store.put(key, report, values=values, timestamps=times)
            return {**report, 'arrays': reference}


def process_audio(source, encoder, output):
    result = encoder.extract(source.path, output)
    return FeatureArtifact(source.asset_id, result['arrays'], result)
