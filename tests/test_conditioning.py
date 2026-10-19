import pytest
import torch


@pytest.mark.parametrize("text,audio", [(True, False), (False, True), (True, True), (False, False)])
def test_modalities(model, batch, text, audio):
    if not text:
        batch.pop("text_features")
    if not audio:
        batch.pop("audio_features")
    out = model(torch.randn(4, 30, 125), torch.rand(4), batch)
    assert torch.isfinite(out).all()
    assert out.shape == (4, 30, 125)


def test_masked_condition_values_irrelevant(model, batch):
    model.eval()
    # Nonzero FiLM makes this test sensitive to actual audio leakage.
    with torch.no_grad():
        for layer in model.motion_generator.layers:
            layer.audio_film[-1].weight.normal_(std=0.05)
    x, t = torch.randn(4, 30, 125), torch.rand(4)
    expected = model(x, t, batch)
    changed = dict(batch)
    changed["audio_features"] = batch["audio_features"].masked_fill(~batch["audio_mask"][..., None], float("nan"))
    changed["text_features"] = batch["text_features"].masked_fill(~batch["text_mask"][..., None], float("nan"))
    torch.testing.assert_close(model(x, t, changed), expected)


def test_zero_audio_is_not_missing(model, batch):
    model.eval()
    with torch.no_grad():
        for layer in model.motion_generator.layers:
            layer.audio_film[-1].bias.fill_(0.5)
    batch["audio_features"].zero_()
    batch["audio_mask"].fill_(True)
    x, t = torch.randn(4, 30, 125), torch.rand(4)
    present = model(x, t, batch)
    batch["audio_mask"].fill_(False)
    missing = model(x, t, batch)
    assert not torch.allclose(present, missing)


def test_audio_provenance_rejected(model, batch):
    with pytest.raises(ValueError, match="provenance"):
        model.sample(batch["motion_history"], audio=batch["audio_features"],
                     audio_mask=batch["audio_mask"], audio_metadata={"feature_type": "current35"})


def test_all_condition_dropout(model, batch):
    model.text_dropout = model.audio_dropout = model.history_dropout = 1.0
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
