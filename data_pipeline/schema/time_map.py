"""The single affine timeline convention: asset = scale * motion + offset."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TimeMap:
    scale: float = 1.0
    offset_sec: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.scale) or self.scale <= 0 or not math.isfinite(self.offset_sec):
            raise ValueError('Time map requires finite positive scale and finite offset')

    def to_asset(self, times):
        return np.asarray(times) * self.scale + self.offset_sec

    def to_motion(self, times):
        return (np.asarray(times) - self.offset_sec) / self.scale

    @classmethod
    def from_aist_offset(cls, old_offset_sec, motion_origin_sec=0):
        # sequence = asset + old_offset; sequence = local_motion + origin.
        return cls(1., motion_origin_sec - old_offset_sec)

    def select(self, timestamps, start, end, *, margin_sec=0., semantics='point'):
        if not np.isfinite([start, end, margin_sec]).all() or end <= start or margin_sec < 0:
            raise ValueError('Invalid query interval')
        times = np.asarray(timestamps, dtype=np.float64)
        lo, hi = self.to_asset([start - margin_sec, end + margin_sec])
        if not np.isfinite(times).all():
            raise ValueError('Nonfinite timestamps')
        if semantics == 'point':
            if times.ndim != 1 or (np.diff(times) <= 0).any():
                raise ValueError('Point timestamps must strictly increase')
            return (times >= lo) & (times < hi)
        if semantics != 'interval' or times.ndim != 2 or times.shape[1] != 2:
            raise ValueError('Expected interval timestamps [T,2]')
        if (times[:, 1] <= times[:, 0]).any() or (np.diff(times[:, 0]) <= 0).any():
            raise ValueError('Invalid interval timestamps')
        return (times[:, 1] > lo) & (times[:, 0] < hi)
