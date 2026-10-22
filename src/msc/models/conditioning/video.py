"""Frozen video features are supplied by the Release, never encoded in workers.

The bridge uses the same independent projection as the other modalities and
requires the verified K/D layout; no V-JEPA checkpoint geometry is assumed.
"""
from torch import nn


class VideoAdapter(nn.Sequential):
    def __init__(self,input_dim,output_dim=512):
        super().__init__(nn.LayerNorm(input_dim),nn.Linear(input_dim,output_dim),
                         nn.SiLU(),nn.Linear(output_dim,output_dim))
