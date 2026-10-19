import random
import numpy as np
import torch
from msc.models.msc_model import MSCModel
from msc.models.conditioning.text import T5TextEncoder
from msc.data.synthetic import SyntheticDataset


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config):
    if config.data.backend == "synthetic":
        return SyntheticDataset(config.data.size, config.model.text_dim, seed=config.train.seed)
    from msc.data.omg_dataset import OMGDataset
    return OMGDataset(config.data.root, split=config.data.split, limit_size=config.data.limit_size,
                      use_text=config.model.use_text, use_audio=config.model.use_audio)


def build_model(config, audio_metadata=None):
    m = config.model
    encoder = None
    if m.use_text and m.text_backend == "t5":
        encoder = T5TextEncoder(m.t5_path, frozen=m.freeze_text, max_length=m.text_max_length,
                                cache_dir=m.text_cache_dir)
        if encoder.output_dim != m.text_dim:
            raise ValueError("text_dim does not match local T5 output dimension")
    return MSCModel(hidden_dim=m.hidden_dim, num_layers=m.num_layers, num_heads=m.num_heads,
        dropout=m.dropout, text_dim=m.text_dim, text_encoder=encoder, use_text=m.use_text,
        use_audio=m.use_audio, text_dropout=m.text_dropout, audio_dropout=m.audio_dropout,
        history_dropout=m.history_dropout, stats_path=m.stats_path, audio_metadata=audio_metadata,
        rtc=m.rtc, max_delay=m.max_delay)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
