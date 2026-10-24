"""Independent modality adapters assemble one read-only condition memory."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .text import TextAdapter
from .audio import AudioAdapter
from .video import VideoAdapter


@dataclass
class ConditionMemory:
    tokens: torch.Tensor
    mask: torch.Tensor
    relative_times: torch.Tensor
    temporal_support: torch.Tensor
    modality_ids: torch.Tensor

    def null(self):
        mask = self.mask & (self.modality_ids == 0)
        return ConditionMemory(self.tokens.masked_fill(~mask[..., None], 0), mask,
                               self.relative_times, self.temporal_support, self.modality_ids)


def projection(input_dim, hidden):
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden),
                         nn.SiLU(), nn.Linear(hidden, hidden))


def position(times, dim):
    frequencies = torch.exp(torch.arange(dim // 2, device=times.device, dtype=torch.float32)
                            * (-math.log(10000) / max(1, dim // 2 - 1)))
    phase = times.float()[..., None] * frequencies * 2 * math.pi
    return torch.cat((phase.sin(), phase.cos()), -1)


class ConditionBridge(nn.Module):
    def __init__(self, hidden, text_dim=None, audio_dim=768, video_dim=None, spatial_tokens=None,
                 text_dropout=.1, audio_dropout=.1, video_dropout=.1):
        super().__init__()
        self.history = projection(125, hidden)
        if video_dim is not None and (not isinstance(spatial_tokens, int) or spatial_tokens <= 0):
            raise ValueError('Video requires a verified positive spatial token count')
        self.adapters = nn.ModuleDict()
        for name, adapter, dim in [('text', TextAdapter, text_dim),
                                   ('audio', AudioAdapter, audio_dim),
                                   ('video', VideoAdapter, video_dim)]:
            if dim is not None:
                self.adapters[name] = adapter(dim, hidden)
        self.types = nn.Embedding(4, hidden)
        self.spatial = nn.Embedding(spatial_tokens, hidden) if video_dim is not None else None
        self.dropouts = dict(text=text_dropout, audio=audio_dropout, video=video_dropout)
        self.hidden = hidden

    def temporal_values(self, modality, batch, mask):
        """Return relative center times and support for each flattened token."""
        if modality == 'text':
            times = torch.zeros_like(mask, dtype=torch.float32)
            return times, torch.stack((times, times), -1)
        if modality == 'audio':
            times = batch['audio_times'].to(mask.device).float()
            if times.shape != mask.shape:
                raise ValueError('Audio timestamps must match token mask')
            times = times.masked_fill(~mask, 0)
            support = torch.stack((times, times), -1)
        else:
            intervals = batch['video_intervals'].to(mask.device).float()
            if intervals.shape != (*mask.shape[:2], 2):
                raise ValueError('Video interval shape mismatch')
            clip_mask = mask.any(-1)
            intervals = intervals.masked_fill(~clip_mask[..., None], 0)
            if ((intervals[..., 1] <= intervals[..., 0]) & clip_mask).any():
                raise ValueError('Invalid video time interval')
            support = intervals[:, :, None].expand(-1, -1, mask.shape[2], -1).flatten(1, 2)
            support = support.masked_fill(~mask.flatten(1, 2)[..., None], 0)
            times = support.mean(-1)
        if not torch.isfinite(support).all():
            raise ValueError(f'Valid {modality} times contain NaN/Inf')
        return times, support

    def forward(self, batch, history, *, require_external=False):
        batch_size = len(history)
        history_times = torch.arange(-5, 0, device=history.device)[None].expand(batch_size, -1) / 30
        tokens = [self.history(history) + self.types.weight[0] + position(history_times, self.hidden)]
        masks = [torch.ones((batch_size, 5), device=history.device, dtype=torch.bool)]
        times = [history_times]
        supports = [torch.stack((history_times, history_times), -1)]
        ids = [torch.zeros_like(history_times, dtype=torch.long)]
        active = torch.zeros(batch_size, device=history.device, dtype=torch.long)

        for modality_id, modality in enumerate(('text', 'audio', 'video'), 1):
            features = batch.get(modality + '_tokens')
            if features is None:
                continue
            if modality not in self.adapters:
                raise ValueError(f'{modality} conditioning is disabled')
            mask = batch.get(modality + '_mask')
            if mask is None or mask.shape != features.shape[:-1] or features.shape[0] != batch_size:
                raise ValueError(f'{modality} tokens require aligned mask')
            expected_dim = self.adapters[modality][0].normalized_shape[0]
            expected_rank = 4 if modality == 'video' else 3
            if features.ndim != expected_rank or features.shape[-1] != expected_dim:
                raise ValueError(f'{modality} feature layout differs from adapter metadata')
            if modality == 'video' and features.shape[2] != self.spatial.num_embeddings:
                raise ValueError('Video spatial layout differs from verified metadata')
            mask = mask.to(history.device).bool()
            active += mask.reshape(batch_size, -1).any(-1).long()
            features = features.detach().to(history.device).masked_fill(~mask[..., None], 0)
            if not torch.isfinite(features).all():
                raise ValueError(f'Valid {modality} tokens contain NaN/Inf')
            token_times, support = self.temporal_values(modality, batch, mask)
            projected = self.adapters[modality](features)
            if modality == 'video':
                projected = (projected + self.spatial.weight[None, None]).flatten(1, 2)
                mask = mask.flatten(1, 2)
            if self.training and self.dropouts[modality]:
                keep = torch.rand(batch_size, 1, device=mask.device) >= self.dropouts[modality]
                mask = mask & keep
            projected = projected + self.types.weight[modality_id]
            if modality != 'text':
                projected = projected + position(token_times, self.hidden)
            tokens.append(projected.masked_fill(~mask[..., None], 0))
            masks.append(mask)
            times.append(token_times)
            supports.append(support)
            ids.append(torch.full_like(token_times, modality_id, dtype=torch.long))

        if (active > 1).any():
            raise ValueError('Each sample may use only one external modality')
        if require_external and (active == 0).any():
            raise ValueError('Training samples require one external modality before dropout')
        if len(tokens) > 2:
            # Each row has only one external modality. Share its padded slots
            # across modalities instead of allocating text + audio + video slots.
            length = max(value.shape[1] for value in tokens[1:])
            def padded(value):
                padding = (0, length - value.shape[1])
                if value.ndim == 3:
                    padding = (0, 0, *padding)
                return F.pad(value, padding)
            tokens = [tokens[0], sum(padded(value) for value in tokens[1:])]
            times = [times[0], sum(padded(value.masked_fill(~mask, 0))
                                   for value, mask in zip(times[1:], masks[1:]))]
            supports = [supports[0], sum(padded(value.masked_fill(~mask[..., None], 0))
                                         for value, mask in zip(supports[1:], masks[1:]))]
            ids = [ids[0], sum(padded(value.masked_fill(~mask, 0))
                               for value, mask in zip(ids[1:], masks[1:]))]
            masks = [masks[0], torch.stack([padded(mask) for mask in masks[1:]]).any(0)]
        return ConditionMemory(torch.cat(tokens, 1), torch.cat(masks, 1), torch.cat(times, 1),
                               torch.cat(supports, 1), torch.cat(ids, 1))
