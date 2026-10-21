"""Video-only crop processor; no motion, audio or visual encoder dependency."""
from pathlib import Path
from data_pipeline.video import PersonVideoConfig, preprocess_person_video
from data_pipeline.storage.store import fingerprint, digest


def process_video(source, output, config):
    settings = {k:v for k,v in config.items() if k != 'enabled'}
    if source.metadata.get('track_id') is not None:
        settings['track_id'] = int(source.metadata['track_id'])
    key = fingerprint({'source': digest(source.path), 'settings': settings,
                       'weights': digest(settings['weights']),
                       'implementation': {p.name: digest(p) for p in (Path(__file__).resolve().parents[1] / 'video').glob('*.py')}})
    result = preprocess_person_video(source.path, Path(output) / key, PersonVideoConfig(**settings))
    import numpy as np
    with np.load(result['metadata'], allow_pickle=False) as data:
        result['source_time_origin_sec'] = float(data['source_requested_pts'][0])
        result['track_id'] = int(data['track_id'])
    return result
