"""Global window permutations adapted from OMG's DistributedPermutationSampler."""
from bisect import bisect_right
import math
from torch.utils.data import Sampler


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


class _GlobalPermutation:
    """Map positions to unique window indices without storing an index array."""

    def __init__(self, size, seed, epoch):
        self.size = size
        bits = max(2, (size - 1).bit_length())
        self.half_bits = (bits + bits % 2) // 2
        self.half_mask = (1 << self.half_bits) - 1
        key = _splitmix64(seed + epoch * 0xD1B54A32D192ED03)
        self.round_keys = [key ^ (i * 0x9E3779B97F4A7C15) for i in range(4)]

    def __getitem__(self, position):
        # Four Feistel rounds are bijective on the enclosing power-of-four domain.
        # Cycle-walking skips values outside the actual dataset, retaining bijectivity.
        while True:
            left = position >> self.half_bits
            right = position & self.half_mask
            for key in self.round_keys:
                mixed = _splitmix64(right ^ key) & self.half_mask
                left, right = right, left ^ mixed
            position = (left << self.half_bits) | right
            if position < self.size:
                return position


class EpochBatchSampler(Sampler):
    """Globally permute all windows each epoch; resume directly at a batch offset."""

    def __init__(self, size, batch_size, start, stop, seed, ranges=None):
        self.ranges = ranges
        self.ends = []
        for a, b in ranges or []:
            self.ends.append((self.ends[-1] if self.ends else 0) + b - a)
        self.size = self.ends[-1] if ranges else size
        self.batch_size, self.start, self.stop, self.seed = batch_size, start, stop, seed
        self.batches_per_epoch = math.ceil(self.size / batch_size)

    def __iter__(self):
        step = self.start
        while step < self.stop:
            epoch, batch_offset = divmod(step, self.batches_per_epoch)
            permutation = _GlobalPermutation(self.size, self.seed, epoch)
            for first in range(batch_offset * self.batch_size, self.size, self.batch_size):
                indices = [permutation[i] for i in range(first, min(first + self.batch_size, self.size))]
                if self.ranges:
                    for i, index in enumerate(indices):
                        span = bisect_right(self.ends, index)
                        indices[i] = self.ranges[span][0] + index - (self.ends[span - 1] if span else 0)
                yield indices
                step += 1
                if step >= self.stop:
                    return

    def __len__(self):
        return self.stop - self.start
