"""MSC v3 future-only flow expert with gated AdaLN-Zero blocks."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention import TimestepEmbedding, RotarySelfAttention, normalize_qk
from .conditioning.fusion import position


class MotionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ffn_dim=None, temporal_bias='relative_distance'):
        super().__init__()
        self.heads = heads
        self.temporal_bias = temporal_bias
        self.norm_self, self.norm_cross, self.norm_ff = (
            nn.LayerNorm(dim, elementwise_affine=False) for _ in range(3)
        )
        self.self_attn = RotarySelfAttention(dim, heads, dropout=dropout)
        self.query, self.key, self.value, self.cross_out = (nn.Linear(dim, dim) for _ in range(4))
        ffn_dim = ffn_dim or 3 * dim
        self.ff = nn.Sequential(nn.Linear(dim, ffn_dim), nn.SiLU(), nn.Dropout(dropout),
                                nn.Linear(ffn_dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.alpha = nn.Parameter(torch.full((heads,), -2.))
        self.dropout = nn.Dropout(dropout)

    def attention_bias(self, memory, horizon):
        query_times = torch.arange(horizon, device=memory.tokens.device)[None, :, None] / 30
        support = memory.temporal_support.masked_fill(~memory.mask[..., None], 0)
        lo, hi = support.unbind(-1)
        distance = (lo[:, None] - query_times).clamp_min(0) + (query_times - hi[:, None]).clamp_min(0)
        distance = distance.masked_fill(memory.modality_ids[:, None] == 1, 0)
        bias = -F.softplus(self.alpha)[None, :, None, None] * distance[:, None]
        if self.temporal_bias == 'none':
            bias = torch.zeros_like(bias)
        return bias.masked_fill(~memory.mask[:, None, None], float('-inf'))

    def cross_attention(self, h, memory):
        batch, horizon, dim = h.shape

        def split(x):
            return x.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)

        q = normalize_qk(split(self.query(h)))
        k = normalize_qk(split(self.key(memory.tokens)))
        v = split(self.value(memory.tokens))
        cross = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.attention_bias(memory, horizon).to(q.dtype),
            dropout_p=self.dropout.p if self.training else 0.,
        )
        return self.cross_out(cross.transpose(1, 2).reshape(batch, horizon, dim))

    def forward(self, x, time, memory, mask):
        self_shift, self_scale, self_gate, cross_shift, cross_scale, cross_gate, ff_shift, ff_scale, ff_gate = (
            self.modulation(time).chunk(9, -1)
        )
        h = self.norm_self(x) * (1 + self_scale) + self_shift
        x = x + self_gate * self.dropout(self.self_attn(h, key_padding_mask=~mask))
        h = self.norm_cross(x) * (1 + cross_scale) + cross_shift
        x = x + cross_gate * self.dropout(self.cross_attention(h, memory))
        h = self.norm_ff(x) * (1 + ff_scale) + ff_shift
        return (x + ff_gate * self.dropout(self.ff(h))).masked_fill(~mask[..., None], 0)


class MotionGenerator(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=12, num_heads=16, dropout=.1,
                 ffn_dim=None, temporal_bias='relative_distance'):
        super().__init__()
        if hidden_dim % num_heads or (hidden_dim // num_heads) % 2:
            raise ValueError('hidden_dim / num_heads must be even for RoPE')
        self.time_embed = TimestepEmbedding(hidden_dim)
        self.input_proj = nn.Linear(125, hidden_dim)
        self.layers = nn.ModuleList([
            MotionBlock(hidden_dim, num_heads, dropout, ffn_dim, temporal_bias) for _ in range(num_layers)
        ])
        self.norm, self.output = nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 125)

    def forward(self, x, timesteps, conditions, valid_mask=None):
        batch, horizon, dim = x.shape
        if (horizon, dim) != (30, 125):
            raise ValueError('Motion input must be [B,30,125]')
        if valid_mask is None:
            valid_mask = torch.ones((batch, horizon), dtype=torch.bool, device=x.device)
        if valid_mask.shape != (batch, horizon) or not valid_mask.any(-1).all():
            raise ValueError('Each sample needs at least one valid future frame')
        if timesteps.shape not in {(batch,), (batch, horizon)}:
            raise ValueError('timesteps must be [B] or [B,30]')
        # Standard FM uses one time per sample. Compute modulation once and
        # broadcast across frames; RTC keeps its per-frame time contract.
        time = self.time_embed(timesteps[:, None] if timesteps.ndim == 1 else timesteps)
        h = self.input_proj(x.masked_fill(~valid_mask[..., None], 0).to(time.dtype))
        h = h + position(torch.arange(horizon, device=x.device)[None] / 30, h.shape[-1]).to(h.dtype)
        for layer in self.layers:
            h = layer(h, time, conditions, valid_mask)
        return self.output(self.norm(h)).masked_fill(~valid_mask[..., None], 0)
