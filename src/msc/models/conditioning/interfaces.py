from typing import Protocol
import torch


class TokenEncoder(Protocol):
    """Future modalities return tokens [B,L,D] and valid mask [B,L]."""
    def __call__(self, inputs, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]: ...
