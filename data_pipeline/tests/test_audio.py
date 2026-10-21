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


def _extract_in_process(model, wav, output, counter):
    import time
    from data_pipeline.audio.mert import MertEncoder
    def encode(self, path):
        with open(counter, 'a') as stream:
            stream.write('encoded\n')
        time.sleep(.2)
        return np.zeros((3, 768), dtype=np.float16), np.arange(3)/75, 1.
    MertEncoder.encode = encode
    return AudioCache(model).extract(wav, output)


def test_concurrent_shared_audio_is_encoded_once(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('pytorch_model.bin', 'config.json', 'preprocessor_config.json'):
        (model / name).write_bytes(b'synthetic')
    wav = tmp_path / 'group.wav'
    make_wav(wav)
    counter = tmp_path / 'encodes.txt'
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(_extract_in_process, model, wav, tmp_path / 'features', counter)
                   for _ in range(2)]
        reports = [f.result() for f in futures]
    assert reports[0] == reports[1]
    assert counter.read_text().splitlines() == ['encoded']


def test_consolidated_audio_cache_invalidation_and_corruption(tmp_path, monkeypatch):
    from data_pipeline.audio.mert import MertEncoder
    from data_pipeline.storage.store import Store, load_arrays
    model = tmp_path / 'model'; model.mkdir()
    for name in ('pytorch_model.bin','config.json','preprocessor_config.json'):
        (model/name).write_bytes(b'synthetic')
    wav = tmp_path/'song.wav'; make_wav(wav)
    calls = []
    def encode(*args):
        calls.append(1)
        return np.ones((3,8),dtype=np.float16),np.arange(3,dtype=np.float64)/75,1.
    monkeypatch.setattr(MertEncoder,'encode',encode)
    output = tmp_path/'cache'
    first = AudioCache(model,consolidated=True).extract(wav,output)
    second = AudioCache(model,consolidated=True).extract(wav,output)
    assert first==second and len(calls)==1
    changed = AudioCache(model,consolidated=True,chunk_seconds=10).extract(wav,output)
    assert changed['arrays']['key'] != first['arrays']['key'] and len(calls)==2
    assert load_arrays(first['arrays'])['values'].shape==(3,8)
    with Store(first['arrays']['store']).connect() as db:
        db.execute('UPDATE artifacts SET payload=? WHERE key=?',(b'incomplete',first['arrays']['key']))
    again = AudioCache(model,consolidated=True).extract(wav,output)
    assert len(calls)==3 and again==first
    assert len(list(output.glob('*.sqlite')))==1
    assert not list(output.rglob('*.npy'))
