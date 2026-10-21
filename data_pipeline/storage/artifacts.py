"""Small typed processor outputs, serialized in the work ledger."""
from dataclasses import dataclass, field


@dataclass
class MotionArtifact:
    label_type: str
    arrays: dict
    num_frames: int
    quality: dict
    version: str = 'not_applicable'
    time_origin_sec: float = 0.
    parent_label: str | None = None


@dataclass
class FeatureArtifact:
    asset_id: str
    arrays: dict
    metadata: dict = field(default_factory=dict)
