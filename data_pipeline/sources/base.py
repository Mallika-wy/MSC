"""Dataset discovery descriptions, independent of processors and release storage."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceMotion:
    motion_id: str
    path: Path
    actor_id: str = 'actor0'
    representation: str = 'human'
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceAsset:
    asset_id: str
    path: Path
    modality: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceText:
    text: str
    start_sec: float
    end_sec: float
    language: str = 'en'
    source: str = 'dataset'
    actor_id: str | None = None


@dataclass(frozen=True)
class SourceAlignment:
    motion_id: str
    asset_id: str
    time_scale: float = 1.
    time_offset_sec: float = 0.
    status: str = 'estimated'
    source: str = 'matched source excerpt; offset not independently calibrated'


@dataclass(frozen=True)
class SourceSequence:
    sequence_id: str
    source_dataset: str
    split_group_id: str
    motions: list[SourceMotion]
    assets: list[SourceAsset] = field(default_factory=list)
    texts: list[SourceText] = field(default_factory=list)
    alignments: list[SourceAlignment] = field(default_factory=list)
    split: str = 'train'
    provenance: dict = field(default_factory=dict)


def paired_sequence(dataset, sequence_id, pairs, split='train'):
    """Build a sequence from per-actor motions and optional shared WAV files."""
    motions, assets, alignments = [], {}, []
    for motion, audio in pairs:
        if not motion.path.is_file():
            raise FileNotFoundError(motion.path)
        motions.append(motion)
        if audio is not None:
            if not audio.is_file():
                raise FileNotFoundError(audio)
            identity = str(audio.resolve())
            assets.setdefault(identity, SourceAsset(identity, audio, 'audio'))
            alignments.append(SourceAlignment(motion.motion_id, identity))
    return SourceSequence(sequence_id, dataset, sequence_id, motions,
                          list(assets.values()), alignments=alignments, split=split,
                          provenance={'source_sequence': sequence_id})
