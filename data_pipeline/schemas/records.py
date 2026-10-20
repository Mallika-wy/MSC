"""MSC v1 typed entities. Times are seconds; intervals are half open."""
from dataclasses import dataclass, field
import math

REPRESENTATION = 'g1_qpos36_wxyz_zup_radians'


@dataclass
class SequenceRecord:
    sequence_id: str
    source_dataset: str
    source_sequence_id: str
    split_group_id: str
    split: str = 'unassigned'
    source_provenance: dict = field(default_factory=dict)


@dataclass
class MotionRecord:
    motion_id: str
    sequence_id: str
    actor_id: str
    group_id: str
    storage_uri: str
    row_start: int
    num_frames: int
    fps: float
    time_origin_sec: float
    label_type: str
    retarget_version: str
    tracker_version: str
    quality_ref: str
    representation: str = REPRESENTATION
    parent_motion_id: str | None = None


@dataclass
class AssetRecord:
    asset_id: str
    modality: str
    uri: str
    source_asset_id: str
    checksum: str
    size_bytes: int
    license_ref: str
    duration_sec: float | None = None
    sample_rate_hz: float | None = None
    frame_rate: float | None = None
    camera_id: str | None = None
    person_track_id: str | None = None


@dataclass
class ModalityLink:
    link_id: str
    motion_id: str
    asset_id: str
    modality: str
    motion_valid_start_sec: float
    motion_valid_end_sec: float
    time_scale: float
    time_offset_sec: float
    alignment_status: str
    alignment_source: str
    view_id: str | None = None
    actor_id: str | None = None


@dataclass
class TextAnnotation:
    annotation_id: str
    sequence_id: str
    text: str
    language: str
    start_sec: float
    end_sec: float
    source: str
    confidence: float
    motion_id: str | None = None


@dataclass
class FeatureRecord:
    feature_id: str
    asset_id: str
    encoder_name: str
    encoder_checkpoint_hash: str
    preprocess_hash: str
    feature_uri: str
    timestamp_uri: str
    dtype: str
    shape: list[int]
    token_time_semantics: str
    aggregation_method: str
    extraction_version: str
    input_asset_checksum: str
    token_rate_hz: float | None = None


TABLES = {'sequences': SequenceRecord, 'motions': MotionRecord, 'assets': AssetRecord,
          'links': ModalityLink, 'annotations': TextAnnotation, 'features': FeatureRecord}


@dataclass
class SourceMotion:
    source_motion_id: str
    sequence_id: str
    uri: str
    format: str
    fps: float
    actor_id: str = 'actor0'
    member: str = ''
    time_origin_sec: float = 0.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip() for v in (self.source_motion_id, self.sequence_id, self.uri, self.format, self.actor_id)):
            raise ValueError('Source motion requires explicit identity, URI, format and actor')
        if not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(self.time_origin_sec):
            raise ValueError('Invalid source motion timeline')
        if not isinstance(self.metadata, dict):
            raise ValueError('Source motion metadata must be a dict')
