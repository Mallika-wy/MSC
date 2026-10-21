"""Audio processing, independent of motion and source adapters."""
from pathlib import Path
import numpy as np
from data_pipeline.storage.artifacts import FeatureArtifact


def process_audio(source, encoder, output):
    result = encoder.extract(source.path, output)
    arrays = result.get('arrays')
    if arrays is None:
        from data_pipeline.storage.store import Store, fingerprint
        store = Store(Path(output) / 'features.sqlite')
        key = fingerprint({k:v for k,v in result.items() if k not in ('values','timestamps')})
        arrays = store.put(key, result,
            values=np.load(result['values'], allow_pickle=False),
            timestamps=np.load(result['timestamps'], allow_pickle=False))
    return FeatureArtifact(source.asset_id, arrays, result)
