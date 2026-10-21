"""One person performing one motion, paired with a raw audio file."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    sequence: str
    motion: Path
    audio: Path | None
    split: str = 'train'
    actor: str = 'actor0'

    def validate(self, *, require_audio=True):
        for path in ((self.motion, self.audio) if require_audio else (self.motion,)):
            if path is None or not path.is_file():
                raise FileNotFoundError(path)
        return self
