"""Read WAV audio and extract timestamped features with local MERT weights."""
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly


def read_audio(path, target_rate=24000):
    """Decode PCM, mix channels to mono, then resample to MERT's sample rate."""
    with wave.open(str(path)) as stream:
        rate = stream.getframerate()
        channels = stream.getnchannels()
        width = stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width == 1:
        values = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        values = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        signed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = ((signed ^ (1 << 23)) - (1 << 23)).astype(np.float32) / (1 << 23)
    elif width == 4:
        values = np.frombuffer(raw, dtype='<i4').astype(np.float32) / (1 << 31)
    else:
        raise ValueError(f'Unsupported PCM width {width}: {path}')
    mono = values.reshape(-1, channels).mean(axis=1)
    return resample_poly(mono, target_rate, rate).astype(np.float32)


class MertEncoder:
    """Load the local encoder once and return arrays without managing cache files."""

    def __init__(self, model_path, *, chunk_seconds=5, device='cpu'):
        self.path = Path(model_path)
        self.chunk_seconds = chunk_seconds
        self.device = device
        self.model = None

    def load(self):
        if self.model is None:
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.path, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                self.path, trust_remote_code=True, local_files_only=True).eval().to(self.device)

    def encode(self, wav):
        """Return features, token centre times, and audio duration; do not write files."""
        import torch
        self.load()
        samples = read_audio(wav, self.processor.sampling_rate)
        rate = int(self.processor.sampling_rate)
        step = int(self.chunk_seconds * rate)
        if step < 400:
            raise ValueError('MERT chunks must contain at least 400 samples')
        features, times = [], []
        with torch.inference_mode():
            for start in range(0, len(samples), step):
                piece = samples[start:start + step]
                if len(piece) < 400:
                    break
                inputs = self.processor(piece, sampling_rate=rate, return_tensors='pt')
                inputs = {name: value.to(self.device) for name, value in inputs.items()}
                hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1][0]
                features.append(hidden.cpu().numpy().astype(np.float16))
                times.append((start + np.arange(len(hidden)) * 320 + 199.5) / rate)
        if not features:
            raise ValueError(f'Audio is too short for MERT: {wav}')
        values = np.concatenate(features)
        return values, np.concatenate(times), len(samples) / rate
