"""Read GDance's per-person index produced by split.py."""
import csv
from pathlib import Path

from data_pipeline.sources.sample import Sample


def samples(root, *, require_audio=True):
    with (root / 'index.csv').open() as stream:
        for row in csv.DictReader(stream):
            audio = Path(row['audio_path'])
            if not audio.is_absolute():
                audio = root / audio
            yield Sample(row['motion_id'], row['source_sequence'], root / row['motion_path'], audio,
                         row['split'], f"person_{int(row['actor_index']):02d}").validate(require_audio=require_audio)
