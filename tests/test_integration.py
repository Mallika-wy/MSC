"""Opt-in local tests. No weights or datasets are downloaded."""
import os
from pathlib import Path
import pytest
import torch
from msc.data.collator import collate_motion
from msc.factory import build_model
from msc.models.conditioning.text import T5TextEncoder

pytestmark = pytest.mark.integration


def local_path(name):
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        pytest.skip(f"Set {name} to enable local integration")
    return value


@pytest.fixture(scope="module")
def real_data():
    from msc.data.omg_dataset import OMGDataset
    # The pinned official release: episode 0=no audio, 153=short, 8160=audio.
    return OMGDataset(local_path("MSC_TEST_DATA_ROOT"), limit_size=8161)


def real_item(data, episode_id):
    # Select a known episode's first window without constructing millions of indices.
    ep = next(e for e in data.reader.episodes if e["episode_index"] == episode_id)
    original = data.reader.samples
    try:
        data.reader.samples = [dict(ep, fixed_window_start=0)]
        return data[0], data.reader[0]
    finally:
        data.reader.samples = original


def test_real_data_masks_and_audio_values(real_data):
    for ep_id, valid, audio in [(0, 30, 0), (153, 30, 0), (8160, 30, 30)]:
        batch, raw = real_item(real_data, ep_id)
        assert batch["motion_future"].shape == (30, 125)
        assert batch["motion_history"].shape == (5, 125)
        assert batch["motion_mask"].sum() == valid
        assert batch["audio_mask"].sum() == audio
        if audio:
            ep = next(e for e in real_data.reader.episodes if e["episode_index"] == ep_id)
            stored = real_data.reader._read_episode(ep)["omg.audio.feature"][:30]
            torch.testing.assert_close(batch["audio_features"], torch.as_tensor(stored).float(), atol=0, rtol=0)
        assert torch.isfinite(batch["motion_future"]).all()
        assert not batch["audio_mask"][~batch["motion_mask"]].any()


def test_original_omg_numerical_parity(real_data, monkeypatch):
    source = local_path("MSC_TEST_OMG_SOURCE")
    monkeypatch.syspath_prepend(str(Path(source) / "src"))
    from omg.motion.feature_codec import G1MotionFeatureCodec
    from omg.robots.g1.kinematics import G1Kinematics
    from omg.motion.representation import G1MotionRepresentation
    from msc.models.representation import MotionRepresentation
    from omg.core.tensor import get_valid_mask
    kin = G1Kinematics(str(Path(source) / "assets/robots/g1/g1_kinematics.json"))
    codec = G1MotionFeatureCodec(kin, num_prev_states=5, canonical_frame_idx=4, rotation_representation="rot6d")
    rep = MotionRepresentation()
    oldrep = G1MotionRepresentation(stats_path=str(Path(source) / "assets/stats/g1_125d_stats.json"),
        kinematics_path=str(Path(source) / "assets/robots/g1/g1_kinematics.json"), num_prev_states=5,
        canonical_frame_idx=4, feat_dim=125, sequence_length=30, rotation_representation="rot6d")
    for ep_id in (0, 153, 8160):
        current, raw = real_item(real_data, ep_id)
        ep = next(e for e in real_data.reader.episodes if e["episode_index"] == ep_id)
        episode, qpos, _, _ = real_data.reader._prepare_episode_kinematics(ep)
        fk = kin.forward_kinematics(qpos)
        from omg.utils.rotation_conversions import standardize_quaternion
        body_quat = standardize_quaternion(torch.nn.functional.normalize(fk["body_quat_w"], dim=-1))
        prev = real_data.reader._get_prev_indices(0, len(qpos))
        history, pos, quat = codec.prev_state_features_from_history(qpos[prev][None],
            fk["body_pos_w"][prev][None], body_quat[prev][None], fps=torch.tensor([30.0]))
        length = min(30, len(qpos))
        comps = codec.canonicalize(qpos[:length][None], fk["body_pos_w"][:length][None],
            body_quat[:length][None], anchor_root_pos=pos, anchor_root_quat=quat,
            fps=torch.tensor([30.0]), valid_mask=get_valid_mask(length, length)[None])
        expected = codec.assemble_features(comps)[0]
        torch.testing.assert_close(current["motion_history"], history[0], atol=0, rtol=0)
        torch.testing.assert_close(current["motion_future"][:length], expected, atol=0, rtol=0)
        torch.testing.assert_close(rep.normalize(expected), oldrep.normalize_features(expected), atol=0, rtol=0)


def test_t5_cache_and_real_train_sample(real_data, cfg, tmp_path):
    cfg.model.text_backend = "t5"
    cfg.model.text_dim = 768
    cfg.model.t5_path = local_path("MSC_TEST_T5_PATH")
    cfg.model.text_cache_dir = str(tmp_path / "text_cache")
    model = build_model(cfg, real_data.audio_metadata)
    items = [real_item(real_data, ep)[0] for ep in (0, 153, 8160)]
    batch = collate_motion(items)
    encoder = model.text_encoder
    model.train()
    assert not encoder.encoder.training
    direct, mask = encoder(batch["text"], torch.device("cpu"))
    encoder.write_cache(batch["text"], torch.device("cpu"))
    cached, cached_mask = encoder(batch["text"], torch.device("cpu"))
    torch.testing.assert_close(direct, cached, atol=0, rtol=0)
    assert torch.equal(mask, cached_mask)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    loss = model.compute_loss(batch)
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in encoder.encoder.parameters())
    out = model.sample(batch["motion_history"], text=batch["text"], audio=batch["audio_features"],
        audio_mask=batch["audio_mask"], audio_metadata=real_data.audio_metadata)
    assert out.shape == (3, 30, 125) and torch.isfinite(out).all()
    print("REAL_DATA_LOSS", loss.item())


def test_t5_unfreeze_gradients(tmp_path):
    # A tiny local T5 model verifies trainability without downloading another model.
    from transformers import T5Config, T5EncoderModel
    import shutil
    source = Path(local_path("MSC_TEST_T5_PATH"))
    local = tmp_path / "tiny-t5"
    local.mkdir()
    shutil.copyfile(source / "spiece.model", local / "spiece.model")
    tiny = T5EncoderModel(T5Config(vocab_size=32128, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2))
    tiny.save_pretrained(local)
    encoder = T5TextEncoder(local, frozen=False, max_length=8)
    encoder.train()
    features, mask = encoder(["walk forward"], torch.device("cpu"))
    features.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.encoder.parameters())
    with pytest.raises(ValueError, match="Static text cache"):
        T5TextEncoder(local, frozen=False, cache_dir=tmp_path / "cache")
