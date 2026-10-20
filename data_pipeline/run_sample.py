"""Run one real audio/motion pair through GMR, tracker and MERT."""
import argparse
import json
from pathlib import Path
import sys
import wave

import numpy as np

from data_pipeline.io import write_json
from data_pipeline.schemas import Inventory
from data_pipeline.schemas.records import SourceMotion
from data_pipeline.stages.arm_twist import screen_reference
from data_pipeline.stages.motion import execute_and_screen
from data_pipeline.stages.retarget import retarget

ROOT = Path(__file__).resolve().parent.parent
OMG = ROOT.parent / 'OMG'


def audio_features(wav, output):
    import torch
    from scipy.signal import resample_poly
    sys.path.append(str(OMG / '.venv/lib/python3.10/site-packages'))
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    model_path = ROOT / 'model/MERT-v1-95M'
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with wave.open(str(wav), 'rb') as stream:
        rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
        raw = stream.readframes(stream.getnframes())
    if width != 2:
        raise ValueError(f'Expected 16-bit PCM WAV: {wav} has width {width}')
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768
    samples = resample_poly(samples, 24000, rate).astype(np.float32)
    features, times = [], []
    # Limit attention memory for long recordings; token context resets at every 5 s.
    step = 5 * 24000
    with torch.inference_mode():
        for start in range(0, len(samples), step):
            piece = samples[start:start+step]
            if len(piece) < 400:
                break
            inputs = processor(piece, sampling_rate=24000, return_tensors='pt')
            result = model(**{k: v.to(device) for k, v in inputs.items()}, output_hidden_states=True)
            values = result.hidden_states[-1][0].cpu().numpy().astype(np.float16)
            features.append(values)
            times.append((start + np.arange(len(values))*320 + 199.5) / 24000)
    np.save(output / 'audio_mert.npy', np.concatenate(features))
    np.save(output / 'audio_mert_times.npy', np.concatenate(times))
    return {'encoder': 'm-a-p/MERT-v1-95M', 'chunk_seconds': 5, 'shape': list(np.concatenate(features).shape),
            'duration_sec': len(samples)/24000, 'token_time': 'convolution receptive-field centre'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['aistpp', 'finedance', 'gdance'])
    parser.add_argument('--output', type=Path, help='Output directory for this sample')
    args = parser.parse_args()
    if args.dataset == 'aistpp':
        root = ROOT / 'data/raw_audio/AISTpp/edge_aistpp'
        path = sorted((root / 'motions').glob('*.pkl'))[0]
        wav, fmt, fps = root / 'wavs' / (path.stem + '.wav'), 'aist_smpl', 60
        smpl = ROOT / 'model/smpl/SMPL_MALE.pkl'
    elif args.dataset == 'finedance':
        root = ROOT / 'data/raw_audio/FineDance/finedance'
        path = sorted((root / 'motion').glob('*.npy'))[0]
        wav, fmt, fps = root / 'music_wav' / (path.stem + '.wav'), 'finedance_smplx', 30
        smpl = ROOT / 'model/smplx/SMPLX_NEUTRAL.npz'
    else:
        root = ROOT / 'data/processed/AIOZ-GDance-single-person'
        path = sorted((root / 'motions_smpl').glob('*.npz'))[0]
        wav = ROOT / 'data/raw_audio/AIOZ-GDance/musics' / (path.stem.rsplit('__person_', 1)[0] + '.wav')
        fmt, fps, smpl = 'gdance_smpl', 30, ROOT / 'model/smplx/SMPLX_NEUTRAL.npz'
    output = args.output or ROOT / 'data/processed' / args.dataset / path.stem
    output.mkdir(parents=True, exist_ok=True)
    source = SourceMotion(path.stem, path.stem, 'root://source/' + path.relative_to(root).as_posix(), fmt, fps)
    inventory = Inventory(external_roots={'source': str(root)})
    status = {'dataset': args.dataset, 'motion': str(path), 'audio': str(wav), 'stages': {}}
    def save(stage, result):
        status['stages'][stage] = result
        write_json(output / 'status.json', status)
        print(json.dumps({stage: result}, ensure_ascii=False), flush=True)
    settings = {'gmr_root': str(ROOT.parent / 'GMR'), 'smpl_model': str(smpl), 'gender': 'male'}
    reference, version, _ = retarget(source, inventory, root, settings, output / 'retarget')
    quality = screen_reference(reference, {'arm_twist': {'enabled': True}})
    save('reference', {'path': str(reference), 'version': version, 'quality': quality})
    save('audio', audio_features(wav, output))
    # Run rejected references too for diagnostic video; do not treat them as accepted.
    track = output / 'tracking'; track.mkdir(exist_ok=True)
    if (track / 'tracker_rollout.npz').is_file():
        from omg.tracking.holomotion.reference import resample_qpos
        with np.load(reference, allow_pickle=False) as source, np.load(track / 'tracker_rollout.npz', allow_pickle=False) as saved:
            if not np.array_equal(saved['reference_qpos_36'], resample_qpos(source['qpos_36'], 30, float(saved['fps']))):
                raise ValueError('Tracker cache belongs to a different reference; choose a new --output directory')
    result = execute_and_screen(reference, {'execution': {'onnx': str(ROOT / 'model/holomotion/model.onnx'),
                                'providers': 'CPUExecutionProvider'}, 'quality': {'strict_joint_limits': False}}, track)
    result['diagnostic_only'] = not quality['accepted']
    save('tracking', result)


if __name__ == '__main__':
    main()
