"""Thin DDP loss wrapper and rank slices of the existing global batch order."""
from torch import nn
from torch.utils.data import Sampler


class TrainingLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch, noise, times, step):
        return self.model.compute_loss(batch, noise=noise, times=times,
                                       return_metrics=True, step=step)


class RankBatchSampler(Sampler):
    def __init__(self, sampler, rank, world_size, batch_size):
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size

    def __iter__(self):
        global_size = self.batch_size * self.world_size
        for indices in self.sampler:
            # Match DistributedSampler's padding of the last global batch.
            padding = global_size - len(indices)
            indices = indices + (indices * ((padding + len(indices) - 1) // len(indices)))[:padding]
            start = self.rank * self.batch_size
            yield indices[start:start + self.batch_size]

    def __len__(self):
        return len(self.sampler)
