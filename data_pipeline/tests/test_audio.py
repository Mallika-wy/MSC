import numpy as np

from data_pipeline.audio.cache import AudioCache
from data_pipeline.tests.test_pipeline import make_wav


def test_shared_audio_is_encoded_once_and_preprocessing_changes_invalidate_cache(tmp_path, monkeypatch):
    from data_pipeline.audio.mert import MertEncoder
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('pytorch_model.bin', 'config.json', 'preprocessor_config.json'):
        (model / name).write_bytes(b'synthetic')
    wav = tmp_path / 'group.wav'
    make_wav(wav)
    calls = []
    def encode(encoder, path):
        calls.append(path)
        return np.zeros((3, 768), dtype=np.float16), np.arange(3, dtype=np.float64)/75, 1.
    monkeypatch.setattr(MertEncoder, 'encode', encode)
    first = AudioCache(model).extract(wav, tmp_path / 'features')
    again = AudioCache(model).extract(wav, tmp_path / 'features')
    assert first == again and len(calls) == 1
    changed = AudioCache(model, chunk_seconds=10).extract(wav, tmp_path / 'features')
    assert changed['values'] != first['values'] and len(calls) == 2


def test_mert_chunk_timestamps_keep_the_second_chunks_audio_offset(monkeypatch):
    import torch
    from types import SimpleNamespace
    from data_pipeline.audio import mert

    class Processor:
        sampling_rate = 24000
        def __call__(self, samples, **kwargs):
            return {'input_values': torch.from_numpy(samples)[None]}

    class Model:
        def __call__(self, input_values, **kwargs):
            count = (input_values.shape[1] - 400) // 320 + 1
            return SimpleNamespace(hidden_states=[torch.zeros((1, count, 768))])

    monkeypatch.setattr(mert, 'read_audio', lambda *args: np.zeros(5 * 24000 + 800, dtype=np.float32))
    encoder = mert.MertEncoder('/not-loaded')
    encoder.processor, encoder.model = Processor(), Model()
    values, times, duration = encoder.encode('/mock.wav')
    assert values.dtype == np.float16 and times.dtype == np.float64
    assert duration == (5 * 24000 + 800) / 24000
    np.testing.assert_allclose(times[[0, 374]], [199.5/24000, 5 + 199.5/24000])
    assert (np.diff(times) > 0).all()
