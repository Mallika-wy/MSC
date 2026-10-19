from torch import nn


class AudioAdapter(nn.Sequential):
    def __init__(self, input_dim=35, output_dim=512):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, 2 * output_dim),
                         nn.SiLU(), nn.Linear(2 * output_dim, output_dim))
