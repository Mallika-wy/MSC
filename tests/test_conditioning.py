import pytest
import torch


@pytest.mark.parametrize("audio", [True, False])
def test_modalities(model, batch, audio):
    if not audio:
        batch.pop("audio_tokens")
    out = model(torch.randn(4, 30, 125), torch.rand(4), batch)
    assert torch.isfinite(out).all()
    assert out.shape == (4, 30, 125)


def test_masked_condition_values_irrelevant(model, batch):
    model.eval()
    x, t = torch.randn(4, 30, 125), torch.rand(4)
    expected = model(x, t, batch)
    changed = dict(batch)
    changed["audio_tokens"] = batch["audio_tokens"].masked_fill(~batch["audio_mask"][..., None], float("nan"))
    torch.testing.assert_close(model(x, t, changed), expected)


def test_zero_audio_is_not_missing(model, batch):
    model.eval()
    batch["audio_tokens"].zero_()
    batch["audio_mask"].fill_(True)
    x, t = torch.randn(4, 30, 125), torch.rand(4)
    present = model(x, t, batch)
    batch["audio_mask"].fill_(False)
    missing = model(x, t, batch)
    assert not torch.allclose(present, missing)


def test_audio_provenance_rejected(model, batch):
    with pytest.raises(ValueError, match="provenance"):
        model.sample(batch["motion_history"], audio_tokens=batch["audio_tokens"],
                     audio_mask=batch["audio_mask"], audio_metadata={"feature_type": "current35"})


def test_all_condition_dropout(model, batch):
    model.condition_bridge.dropouts.update(text=1.,audio=1.)
    loss = model.compute_loss(batch)
    assert torch.isfinite(loss)
