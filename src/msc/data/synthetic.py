"""Deterministic synthetic fixtures in the external canonical feature space."""
import torch
from torch.utils.data import Dataset
from msc.models.representation import MotionRepresentation

SYNTHETIC_AUDIO = dict(backend="omg_precomputed", feature_type="synthetic_test_only", dimension=35,
                       fps=30, preprocessing_version="msc-synthetic-v1")


class SyntheticDataset(Dataset):
    def __init__(self, size=32, text_dim=768, seed=17, short_sequences=True):
        self.size, self.text_dim, self.seed = size, text_dim, seed
        self.short_sequences = short_sequences
        self.representation = MotionRepresentation()
        self.audio_metadata = SYNTHETIC_AUDIO.copy()

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
        return dict(motion_history=motion[:5], motion_future=motion[5:], motion_mask=mask,
                    text="", has_text=torch.tensor(has_text),
                    text_features=torch.randn(5, self.text_dim, generator=gen) * 0.1,
                    text_mask=torch.full((5,), has_text),
                    audio_features=torch.randn(30, 35, generator=gen) * 0.1,
                    audio_mask=mask & has_audio)
