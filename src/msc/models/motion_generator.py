"""Future-only OMG Motion DiT: history/text cross attention and audio FiLM.

Adapted from OMG transformer.py (MIT), commit 510bc0d; see THIRD_PARTY.md.
"""
import torch
from torch import nn
from msc._omg.attention import TimestepEmbedding, RotarySelfAttention, _qk_normalized_cross_attention


class MotionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = RotarySelfAttention(dim, heads, dropout=dropout, qk_norm=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.audio_film = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.audio_film[-1].weight)
        nn.init.zeros_(self.audio_film[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_mask, valid_mask, audio=None, audio_mask=None):
        cross = _qk_normalized_cross_attention(self.cross_attn, query=self.norm_cross(x),
                    context=context, context_key_padding_mask=~context_mask)
        x = x + self.dropout(cross)
        x = x + self.dropout(self.self_attn(self.norm_self(x), key_padding_mask=~valid_mask))
        h = self.norm_ff(x)
        if audio is not None:
            scale, shift = self.audio_film(audio).chunk(2, dim=-1)
            scale = scale.masked_fill(~audio_mask[..., None], 0)
            shift = shift.masked_fill(~audio_mask[..., None], 0)
            h = h * (1 + scale.tanh()) + shift
        return (x + self.dropout(self.ff(h))).masked_fill(~valid_mask[..., None], 0)


class MotionGenerator(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=12, num_heads=8, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads or (hidden_dim // num_heads) % 2:
            raise ValueError("hidden_dim / num_heads must be an even integer for RoPE")
        self.time_embed = TimestepEmbedding(hidden_dim)
        self.input_proj = nn.Linear(125 + hidden_dim, hidden_dim)
        self.history_projector = nn.Sequential(nn.LayerNorm(125), nn.Linear(125, hidden_dim),
                                               nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.extra_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList([MotionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 125)

    def forward(self, x, timesteps, conditions, valid_mask=None):
        b, t, d = x.shape
        if (t, d) != (30, 125):
            raise ValueError("Motion input must have shape [B,30,125]")
        if valid_mask is None:
            valid_mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
        if valid_mask.shape != (b, t) or not valid_mask.any(dim=1).all():
            raise ValueError("Each sample needs at least one valid future frame")
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None].expand(b, t)
        if timesteps.shape != (b, t):
            raise ValueError("timesteps must have shape [B] or [B,30]")
        time = self.time_embed(timesteps)
        h = self.input_proj(torch.cat([x.masked_fill(~valid_mask[..., None], 0).to(time.dtype), time], dim=-1))
        for layer in self.layers:
            h = layer(h, conditions["context"], conditions["context_mask"], valid_mask,
                      conditions.get("audio"), conditions.get("audio_mask"))
        return self.output(self.norm(h)).masked_fill(~valid_mask[..., None], 0)
