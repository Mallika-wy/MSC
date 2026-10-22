"""MSC canonical G1 125D contract. No alternate motion representation."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn

ASSETS = Path(__file__).resolve().parents[1] / "assets"


class MotionRepresentation(nn.Module):
    name = "g1_canonical_125d"
    feature_dim = 125
    fps = 30
    # Four preceding states plus the current state predict the next second.
    history_frames = 5
    future_frames = 30
    canonical_frame_idx = 4
    slices = {"root_position": slice(0, 3), "root_rot6d": slice(3, 9),
              "joint_angles": slice(9, 38), "link_positions": slice(38, 125)}

    def __init__(self, stats_path=None):
        super().__init__()
        path = Path(stats_path) if stats_path else ASSETS / "stats/g1_125d_stats.json"
        content = path.read_bytes()
        stats = json.loads(content)
        mean, std = [torch.tensor(stats[k], dtype=torch.float32) for k in ("mean", "std")]
        if mean.shape != (125,) or std.shape != (125,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Expected finite canonical G1 125D normalization statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        self.stats_sha256 = hashlib.sha256(content).hexdigest()

    def normalize(self, x):
        return (x - self.mean.to(x)) / self.std.to(x)

    def denormalize(self, x):
        return x * self.std.to(x) + self.mean.to(x)

    def metadata(self):
        return dict(type=self.name, rotation="rot6d", fps=30, history_frames=5,
                    future_frames=30, canonical_frame_idx=4, stats_sha256=self.stats_sha256)
