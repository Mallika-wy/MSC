import numpy as np
import json
import pytest
import torch

from msc.data.normalization_stats import canonical_moments
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics


@pytest.mark.parametrize('translation', [0., 10000.])
def test_prefix_moments_match_encoded_overlapping_training_windows(translation):
    generator = torch.Generator().manual_seed(74)
    qpos = torch.randn(100, 36, generator=generator) * .15
    qpos[:, :3] += translation
    qpos[:, 3] += 1
    kin = G1Kinematics()
    # Include overlapping annotations and duplicate task spans: their windows
    # must be counted multiple times, just as in the real training sampler.
    starts, ends = np.array([5, 12, 12, 49]), np.array([20, 26, 26, 70])
    count, total, square = canonical_moments(qpos, starts, ends, kin, chunk_frames=17)
    indices = torch.tensor([k for lo, hi in zip(starts, ends) for k in range(lo, hi)])
    frames = indices[:, None] + torch.arange(30)
    body_positions = kin.forward_body_positions(qpos)
    codec = G1MotionFeatureCodec(kin, num_prev_states=5, canonical_frame_idx=4,
                                 rotation_representation='rot6d')
    features = codec.assemble_features(codec.canonicalize(
        qpos[frames], body_positions[frames], None,
        qpos[indices-1, :3], qpos[indices-1, 3:7], 30.)).double().flatten(0, 1)
    assert count == len(features) == sum(ends-starts)*30
    torch.testing.assert_close(total/count, features.mean(0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close((square/count-(total/count).square()).clamp_min(0).sqrt(),
                               features.std(0, correction=0), rtol=1e-5, atol=1e-6)


def test_normalization_rejects_out_of_bounds_windows():
    with pytest.raises(ValueError, match='window bounds'):
        canonical_moments(torch.zeros(40, 36), [4], [5], G1Kinematics())


def test_normalization_rejects_wrong_temporal_contract(tmp_path):
    from msc.models.representation import MotionRepresentation
    path = tmp_path / 'stats.json'
    path.write_text(json.dumps(dict(dataset_identity='train-pool', split='train',
        representation='g1_canonical_125d', fps=30, num_prev_states=10,
        canonical_frame_idx=9, sequence_length=60, mean=[0.]*125, std=[1.]*125)))
    with pytest.raises(ValueError, match='5/30-frame Train contract'):
        MotionRepresentation(path)


def test_training_rejects_normalization_from_another_pool(cfg, toy_dataset):
    from msc.factory import build_model
    from msc.training.trainer import Trainer
    model = build_model(cfg, toy_dataset.audio_metadata)
    model.representation.stats_dataset_identity = 'different-train-pool'
    with pytest.raises(ValueError, match='Normalization statistics do not match'):
        Trainer(model, toy_dataset, cfg)
