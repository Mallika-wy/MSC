"""Shared future-only action expert with structured cross attention."""
import torch
from torch import nn
from torch.nn import functional as F
from msc.models.attention import TimestepEmbedding, RotarySelfAttention
from .conditioning.fusion import position


class MotionBlock(nn.Module):
    def __init__(self, dim, heads, dropout, ffn_dim=None, temporal_bias='relative_distance'):
        super().__init__()
        self.heads, self.dim, self.temporal_bias = heads, dim, temporal_bias
        self.norm_self, self.norm_cross, self.norm_ff = (nn.LayerNorm(dim) for _ in range(3))
        self.self_attn = RotarySelfAttention(dim,heads,dropout=dropout,qk_norm=True)
        self.query, self.key, self.value, self.cross_out = (nn.Linear(dim,dim) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(dim,ffn_dim or 4*dim),nn.SiLU(),nn.Dropout(dropout),nn.Linear(ffn_dim or 4*dim,dim))
        self.modulation = nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        self.alpha = nn.Parameter(torch.full((heads,),-2.))
        self.dropout = nn.Dropout(dropout)

    def attention_bias(self, memory, horizon):
        query_times = torch.arange(horizon,device=memory.tokens.device,dtype=torch.float32)[None,:,None]/30
        lo,hi = memory.temporal_support.unbind(-1)
        distance = (lo[:,None]-query_times).clamp_min(0)+(query_times-hi[:,None]).clamp_min(0)
        distance = distance.masked_fill(memory.modality_ids[:,None,:]==1,0)
        bias = -F.softplus(self.alpha)[None,:,None,None]*distance[:,None]
        if self.temporal_bias == 'none':
            bias = bias*0
        return bias.masked_fill(~memory.mask[:,None,None,:],float('-inf'))

    def forward(self,x,time,memory,mask):
        ss,bs,sc,bc,sf,bf = self.modulation(time).chunk(6,-1)
        h = self.norm_self(x)*(1+ss.tanh())+bs
        x = x+self.dropout(self.self_attn(h,key_padding_mask=~mask))
        h = self.norm_cross(x)*(1+sc.tanh())+bc
        b,t,d = h.shape
        split = lambda v: v.reshape(b,-1,self.heads,d//self.heads).transpose(1,2)
        q,k,v = split(self.query(h)),split(self.key(memory.tokens)),split(self.value(memory.tokens))
        cross = F.scaled_dot_product_attention(q,k,v,attn_mask=self.attention_bias(memory,t).to(q.dtype),
                    dropout_p=self.dropout.p if self.training else 0.)
        x = x+self.dropout(self.cross_out(cross.transpose(1,2).reshape(b,t,d)))
        h = self.norm_ff(x)*(1+sf.tanh())+bf
        return (x+self.dropout(self.ff(h))).masked_fill(~mask[...,None],0)


class MotionGenerator(nn.Module):
    def __init__(self,hidden_dim=512,num_layers=6,num_heads=8,dropout=.1,ffn_dim=None,temporal_bias='relative_distance'):
        super().__init__()
        if hidden_dim%num_heads or (hidden_dim//num_heads)%2:
            raise ValueError('hidden_dim / num_heads must be even for RoPE')
        self.time_embed = TimestepEmbedding(hidden_dim)
        self.input_proj = nn.Linear(125,hidden_dim)
        self.layers = nn.ModuleList([MotionBlock(hidden_dim,num_heads,dropout,ffn_dim,temporal_bias) for _ in range(num_layers)])
        self.norm, self.output = nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,125)

    def forward(self,x,timesteps,conditions,valid_mask=None):
        b,t,d = x.shape
        if (t,d)!=(30,125):
            raise ValueError('Motion input must be [B,30,125]')
        if valid_mask is None:
            valid_mask = torch.ones((b,t),dtype=torch.bool,device=x.device)
        if valid_mask.shape!=(b,t) or not valid_mask.any(-1).all():
            raise ValueError('Each sample needs at least one valid future frame')
        if timesteps.ndim==1:
            timesteps = timesteps[:,None].expand(b,t)
        if timesteps.shape!=(b,t):
            raise ValueError('timesteps must be [B] or [B,30]')
        time = self.time_embed(timesteps)
        h = self.input_proj(x.masked_fill(~valid_mask[...,None],0).to(time.dtype))
        h = h+position(torch.arange(t,device=x.device)[None]/30,h.shape[-1]).to(h.dtype)
        for layer in self.layers:
            h = layer(h,time,conditions,valid_mask)
        return self.output(self.norm(h)).masked_fill(~valid_mask[...,None],0)
