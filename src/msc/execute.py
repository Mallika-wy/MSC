"""Execute sampled canonical 125D motion through the pipeline's SONIC tracker."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_pipeline.config import Settings
from data_pipeline.motion.tracking import track
from msc._omg.feature_codec import G1MotionFeatureCodec
from msc._omg.kinematics import G1Kinematics


def main():
    parser = argparse.ArgumentParser(description='Execute sampled 30 Hz motion with SONIC')
    parser.add_argument('--sample', required=True, help='msc-sample output .pt with motion [B,T,125]')
    parser.add_argument('--initial-state', required=True, help='NPZ qpos_36 [36] or [B,36] at the history anchor')
    parser.add_argument('--config', default='configs/data_pipeline/omg_text.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()

    sampled = torch.load(args.sample, map_location='cpu', weights_only=True)
    features = torch.as_tensor(sampled['motion'] if isinstance(sampled, dict) else sampled, dtype=torch.float32)
    if features.ndim == 2:
        features = features[None]
    with np.load(args.initial_state, allow_pickle=False) as data:
        initial = np.asarray(data['qpos_36'], dtype=np.float32)
    if initial.ndim == 1:
        initial = initial[None]
    if features.ndim != 3 or features.shape[-1] != 125 or initial.shape != (len(features), 36):
        raise ValueError('Expected sampled [B,T,125] and matching initial qpos_36 [B,36]')

    codec = G1MotionFeatureCodec(G1Kinematics(), num_prev_states=5,
                                 canonical_frame_idx=4, rotation_representation='rot6d')
    world = codec.decode_to_world_qpos36(codec.split_features(features),
                                          torch.from_numpy(initial[:, None, :3]),
                                          torch.from_numpy(initial[:, None, 3:7])).numpy()
    settings = Settings.read(args.config)
    output = Path(args.output)
    for index, (start, future) in enumerate(zip(initial, world)):
        target = output / f'sample-{index:05d}'
        target.mkdir(parents=True, exist_ok=True)
        reference = np.concatenate([start[None], future]).astype(np.float32)
        executed, quality = track(reference, settings.tracker, settings.quality,
                                  target, diagnostic=args.diagnostic)
        np.savez_compressed(target / 'executed.npz', qpos_36=executed, fps=np.float32(30),
                            time_origin_sec=np.float64(1 / 30))
        (target / 'quality.json').write_text(json.dumps(quality, indent=2))
        if not args.diagnostic:
            (target / 'tracker_rollout.npz').unlink()
            (target / 'reference.npz').unlink()
        print(f'{target}: {len(executed)} frames, accepted={quality["accepted"]}', flush=True)


if __name__ == '__main__':
    main()
