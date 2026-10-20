"""One person performing one motion, paired with a raw audio file."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    sequence: str
    motion: Path
    audio: Path
    split: str = 'train'
    actor: str = 'actor0'

    def validate(self):
        for path in (self.motion, self.audio):
            if not path.is_file():
                raise FileNotFoundError(path)
        return self
