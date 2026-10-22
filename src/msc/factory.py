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
    if config.data.backend == 'msc_release':
        from msc.data.release_dataset import ReleaseDataset
        from dataclasses import asdict
        d = asdict(config.data)
        for key in ('backend','root','size','num_workers'):
            d.pop(key)
        return ReleaseDataset(**d,text_cache_dir=config.model.text_cache_dir)
    raise ValueError(f'Unsupported dataset backend: {config.data.backend}')


def build_model(config, audio_metadata=None, feature_metadata=None):
    m = config.model
    condition = m.condition
    text = condition.get('text',{})
    audio = condition.get('audio',{})
    video = condition.get('video',{})
    use_text, use_audio = text.get('enabled',m.use_text),audio.get('enabled',m.use_audio)
    encoder = None
    text_dim = text.get('dim',m.text_dim)
    metadata = feature_metadata or {}
    if config.data.backend == 'msc_release' and use_text:
        if not metadata.get('text'):
            # Audio-only release training should explicitly disable text.
            raise ValueError('Enabled text requires an offline T5 cache manifest')
        actual_dim = metadata['text']['output_dim']
        if text_dim not in ('auto',actual_dim):
            raise ValueError('Configured text dimension differs from T5 cache metadata')
        text_dim = actual_dim
    elif use_text and text.get('backend',m.text_backend) == 't5':
        encoder = T5TextEncoder(m.t5_path,frozen=True,max_length=m.text_max_length,cache_dir=m.text_cache_dir)
        text_dim = encoder.output_dim
    if use_text and text_dim == 'auto':
        raise ValueError('text dim auto requires real T5/cache metadata')
    video_dim, spatial = None,None
    if video.get('enabled',False):
        identities = {tuple(x[-1]) for x in metadata.get('features',[]) if x[0]=='video'}
        if len(identities) != 1:
            raise ValueError('Enabled video requires one verified K/D feature layout')
        spatial,video_dim = next(iter(identities))
        if video.get('dim','auto') not in ('auto',video_dim) or video.get('spatial_tokens',spatial) != spatial:
            raise ValueError('Video config differs from actual features')
    return MSCModel(hidden_dim=m.hidden_dim,num_layers=m.num_layers,num_heads=m.num_heads,
        dropout=m.dropout,text_dim=text_dim,text_encoder=encoder,use_text=use_text,use_audio=use_audio,
        text_dropout=text.get('dropout',m.text_dropout),audio_dropout=audio.get('dropout',m.audio_dropout),
        history_dropout=condition.get('history_dropout',m.history_dropout),stats_path=m.stats_path,
        audio_metadata=audio_metadata,rtc=m.rtc,max_delay=m.max_delay,video_dim=video_dim,
        spatial_tokens=spatial,video_dropout=video.get('dropout',.1),ffn_dim=m.ffn_dim,
        temporal_bias=condition.get('temporal_bias','relative_distance'),losses=m.losses,
        dynamic_stats_path=m.dynamic_stats_path,feature_metadata=feature_metadata)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
