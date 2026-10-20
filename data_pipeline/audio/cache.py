"""Cache MERT values and timestamps by audio checksum and encoder settings."""
import hashlib
import json
from pathlib import Path

import numpy as np

from data_pipeline.io import read_json, sha256_file, write_json
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
        checksum = sha256_file(wav)
        key = hashlib.sha256((checksum + self.checkpoint_hash + self.preprocess_hash).encode()).hexdigest()
        directory = Path(output) / key
        values_path = directory / 'values.npy'
        times_path = directory / 'times.npy'
        report_path = directory / 'features.json'
        if report_path.is_file() and values_path.is_file() and times_path.is_file():
            report = read_json(report_path)
            # Catch incomplete writes before advertising cached features.
            values = np.load(values_path, mmap_mode='r', allow_pickle=False)
            times = np.load(times_path, mmap_mode='r', allow_pickle=False)
            if list(values.shape) == report['shape'] and times.shape == (len(values),):
                return report

        values, times, duration = self.encoder.encode(wav)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(values_path, values)
        np.save(times_path, times)
        report = dict(encoder='m-a-p/MERT-v1-95M', checkpoint_hash=self.checkpoint_hash,
                      preprocess_hash=self.preprocess_hash, input_checksum=checksum,
                      chunk_seconds=self.chunk_seconds, shape=list(values.shape),
                      duration_sec=duration, values=str(values_path.resolve()),
                      timestamps=str(times_path.resolve()))
        write_json(report_path, report)
        return report
