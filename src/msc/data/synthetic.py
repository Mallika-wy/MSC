"""Deterministic synthetic fixtures in the external canonical feature space."""
import torch
from torch.utils.data import Dataset
from msc.models.representation import MotionRepresentation

SYNTHETIC_AUDIO = dict(backend="mert_precomputed", feature_type="synthetic_test_only", dimension=768,
                       token_rate_hz=75, preprocessing_version="msc-synthetic-v2")


class SyntheticDataset(Dataset):
    def __init__(self, size=32, text_dim=768, seed=17, short_sequences=True):
        self.size, self.text_dim, self.seed = size, text_dim, seed
        self.short_sequences = short_sequences
        self.representation = MotionRepresentation()
        self.audio_metadata = SYNTHETIC_AUDIO.copy()
        self.identity = f'synthetic-v2:{size}:{text_dim}:{seed}:{short_sequences}'
        self.feature_metadata = {'synthetic':True,'features':[['video','synthetic','fixture','fixture',[4,16]]]}
        self.task_ranges = {}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(self.seed + index)
        phase = index % 4 * 0.3
        timeline = torch.arange(-5, 30).float()
        values = 0.1 * torch.sin(timeline[:, None] / 30 + phase).expand(-1, 125).clone()
        values += torch.linspace(-0.03, 0.03, 125)[None]
        motion = self.representation.denormalize(values)
        length = 23 if self.short_sequences and index % 5 == 0 else 30
        mask = torch.arange(30) < length
        has_text, has_audio = index % 4 in (0, 2), index % 4 in (1, 2)
        audio_length = 68 + index % 3 * 4
        audio_times = torch.arange(audio_length)/75
        return dict(motion_history=motion[:5], motion_future=motion[5:], motion_mask=mask,
                    text="", has_text=torch.tensor(has_text),
                    text_features=torch.randn(5, self.text_dim, generator=gen) * 0.1,
                    text_mask=torch.full((5,), has_text),
                    audio_features=torch.randn(audio_length, 768, generator=gen) * 0.1,
                    audio_mask=(audio_times < length/30) & has_audio, audio_times=audio_times)
