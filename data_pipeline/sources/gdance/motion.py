"""Decode a single GDance actor using neutral SMPL-X as a SMPL proxy."""
import numpy as np

from data_pipeline.motion.decoding import human_motion


def decode_motion(path, settings):
    with np.load(path, allow_pickle=False) as data:
        pose = data['smpl_poses'].reshape(-1, 24, 3)
        translation = data['root_trans'].copy()
        betas = data['smpl_betas'].mean(axis=0)[:10].astype(np.float32)
    hands = np.zeros((len(pose), 45), dtype=np.float32)
    return human_motion(pose, translation, betas, hands, hands, settings)
