# Adapted from OMG (MIT), commit 510bc0d. See THIRD_PARTY.md.
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = timesteps.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("t,d->td", positions, self.inv_freq.to(device=device))
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
        return cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0), sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.forward(q.shape[-2], q.device, q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even attention head dim, got {self.head_dim}")
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = float(dropout)
        self.qk_norm = bool(qk_norm)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self.rope.apply(q, k)

        if self.qk_norm:
            head_scale = math.sqrt(self.head_dim)
            q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype) * head_scale
            k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype) * head_scale

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(
                batch_size,
                1,
                1,
                seq_len,
                device=x.device,
                dtype=x.dtype,
            )
            attn_mask = attn_mask.masked_fill(
                key_padding_mask[:, None, None, :].bool(),
                torch.finfo(x.dtype).min,
            )

        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        h = self.out_proj(h)
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1).bool(), 0.0)
        return h

def _qk_normalized_cross_attention(
    attention: nn.Module,
    query: torch.Tensor,
    context: torch.Tensor,
    context_key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Cross-attend with per-head query/key directions instead of magnitudes.

    QK normalization removes the otherwise unidentifiable radial degree of
    freedom in query and key projections.  Scaling either projection therefore
    cannot sharpen the logits or amplify the input Jacobian.  Multiplying unit
    vectors by ``sqrt(head_dim)`` preserves the variance of standard scaled
    dot-product attention at initialization.
    """
    if hasattr(attention, "_qkv_same_embed_dim") and not attention._qkv_same_embed_dim:
        raise ValueError("QK-normalized cross-attention requires equal query/key/value dimensions")

    embed_dim = attention.embed_dim
    num_heads = attention.num_heads
    head_dim = embed_dim // num_heads
    if hasattr(attention, "q_proj"):
        q = attention.q_proj(query)
        k = attention.k_proj(context)
        v = attention.v_proj(context)
    else:
        if attention.in_proj_weight is None:
            raise ValueError("QK-normalized cross-attention requires packed projection weights")
        weight_q, weight_k, weight_v = attention.in_proj_weight.chunk(3, dim=0)
        if attention.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = attention.in_proj_bias.chunk(3, dim=0)
        q = F.linear(query, weight_q, bias_q)
        k = F.linear(context, weight_k, bias_k)
        v = F.linear(context, weight_v, bias_v)
    batch_size, query_len, _ = q.shape
    context_len = k.shape[1]
    q = q.view(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, context_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, context_len, num_heads, head_dim).transpose(1, 2)

    head_scale = math.sqrt(head_dim)
    q = F.normalize(q.float(), dim=-1).to(dtype=q.dtype) * head_scale
    k = F.normalize(k.float(), dim=-1).to(dtype=k.dtype) * head_scale

    attn_mask = None
    if context_key_padding_mask is not None:
        attn_mask = torch.zeros(
            batch_size,
            1,
            1,
            context_len,
            device=q.device,
            dtype=q.dtype,
        )
        attn_mask = attn_mask.masked_fill(
            context_key_padding_mask[:, None, None, :].bool(),
            torch.finfo(q.dtype).min,
        )
    h = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=float(getattr(attention, "dropout", 0.0)) if attention.training else 0.0,
        is_causal=False,
    )
    h = h.transpose(1, 2).reshape(batch_size, query_len, embed_dim)
    return attention.out_proj(h)

def _expand_timesteps(timesteps: torch.Tensor, seq_len: int) -> torch.Tensor:
    if timesteps.ndim == 1:
        return timesteps[:, None].expand(-1, seq_len)
    if timesteps.ndim == 2:
        return timesteps
    raise ValueError(f"Expected timestep shape (B,) or (B,T), got {tuple(timesteps.shape)}")
