"""One structured memory for history and independent frozen feature modalities."""
from dataclasses import dataclass
import math
import torch
from torch import nn
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
        return ConditionMemory(self.tokens.masked_fill(~mask[...,None],0), mask,
                               self.relative_times,self.temporal_support,self.modality_ids)


def projection(input_dim, hidden):
    return nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden))


def position(times, dim):
    freq = torch.exp(torch.arange(dim//2,device=times.device,dtype=torch.float32)*(-math.log(10000)/max(1,dim//2-1)))
    phase = times.float()[...,None]*freq*2*math.pi
    return torch.cat((phase.sin(),phase.cos()),-1)


class ConditionBridge(nn.Module):
    def __init__(self, hidden, text_dim=None, audio_dim=768, video_dim=None, spatial_tokens=None,
                 text_dropout=.1,audio_dropout=.1,video_dropout=.1):
        super().__init__()
        self.history = projection(125,hidden)
        if video_dim is not None and (not isinstance(spatial_tokens,int) or spatial_tokens<=0):
            raise ValueError('Video requires a verified positive spatial token count')
        adapters = dict(text=TextAdapter,audio=AudioAdapter,video=VideoAdapter)
        self.adapters = nn.ModuleDict({m:adapters[m](d,hidden) for m,d in
                                      [('text',text_dim),('audio',audio_dim),('video',video_dim)] if d is not None})
        self.types = nn.Embedding(4,hidden)
        self.spatial = nn.Embedding(spatial_tokens,hidden) if video_dim is not None else None
        self.dropouts = dict(text=text_dropout,audio=audio_dropout,video=video_dropout)
        self.hidden = hidden

    def forward(self,batch,history):
        b = len(history)
        ht = torch.arange(-5,0,device=history.device,dtype=torch.float32)[None].expand(b,-1)/30
        tokens = [self.history(history)+self.types.weight[0]+position(ht,self.hidden)]
        masks = [torch.ones((b,5),device=history.device,dtype=torch.bool)]
        times, supports = [ht],[torch.stack((ht,ht),-1)]
        ids = [torch.zeros_like(ht,dtype=torch.long)]
        for mid,modality in enumerate(('text','audio','video'),1):
            features = batch.get(modality+'_tokens',batch.get(modality+'_features'))
            if features is None:
                continue
            if modality not in self.adapters:
                raise ValueError(f'{modality} conditioning is disabled')
            mask = batch.get(modality+'_mask')
            if mask is None or mask.shape != features.shape[:-1] or features.shape[0] != b:
                raise ValueError(f'{modality} tokens require aligned mask')
            mask = mask.to(history.device).bool()
            features = features.detach().to(history.device).masked_fill(~mask[...,None],0)
            if not torch.isfinite(features).all():
                raise ValueError(f'Valid {modality} tokens contain NaN/Inf')
            if modality == 'text':
                t = torch.zeros(features.shape[:2],device=history.device)
                support = torch.stack((t,t),-1)
            elif modality == 'audio':
                t = batch['audio_times'].to(history.device).float()
                if t.shape != mask.shape:
                    raise ValueError('Audio timestamps must match token mask')
                t = t.masked_fill(~mask,0)
                support = torch.stack((t,t),-1)
            else:
                if features.ndim != 4 or features.shape[2] != self.spatial.num_embeddings:
                    raise ValueError('Video spatial layout differs from verified metadata')
                support = batch['video_intervals'].to(history.device).float()
                if support.shape != (*features.shape[:2],2):
                    raise ValueError('Video interval shape mismatch')
                support = support.masked_fill(~mask.any(-1)[...,None],0)
                if ((support[...,1] <= support[...,0]) & mask.any(-1)).any():
                    raise ValueError('Invalid video time interval')
                support = support[:,:,None,:].expand(-1,-1,features.shape[2],-1).reshape(b,-1,2)
                t = support.mean(-1)
            if not torch.isfinite(support).all():
                raise ValueError(f'Valid {modality} times contain NaN/Inf')
            projected = self.adapters[modality](features)
            if modality == 'video':
                projected = projected+self.spatial.weight[None,None]
                projected = projected.flatten(1,2)
                mask = mask.flatten(1,2)
            if self.training and self.dropouts[modality]:
                mask = mask & (torch.rand(b,1,device=mask.device)>=self.dropouts[modality])
            projected = projected+self.types.weight[mid]
            if modality != 'text':
                projected = projected+position(t,self.hidden)
            tokens.append(projected.masked_fill(~mask[...,None],0))
            masks.append(mask); times.append(t); supports.append(support)
            ids.append(torch.full_like(t,mid,dtype=torch.long))
        return ConditionMemory(torch.cat(tokens,1),torch.cat(masks,1),torch.cat(times,1),
                               torch.cat(supports,1),torch.cat(ids,1))
