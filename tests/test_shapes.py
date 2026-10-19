import pytest
import torch
from msc.models.representation import MotionRepresentation


def test_forward(model, batch):
    velocity = model(torch.randn(4, 30, 125), torch.rand(4), batch)
    assert velocity.shape == (4, 30, 125)
    assert torch.isfinite(velocity).all()


def test_representation_roundtrip():
    rep = MotionRepresentation()
    x = torch.randn(2, 30, 125)
    torch.testing.assert_close(rep.normalize(rep.denormalize(x)), x, atol=3e-5, rtol=1e-4)


def test_empty_future_rejected(model, batch):
    batch["motion_mask"][0] = False
    with pytest.raises(ValueError, match="valid future"):
        model.compute_loss(batch)


def test_sample(model, batch):
    result = model.sample(batch["motion_history"][:1], num_steps=10)
    assert result.shape == (1, 30, 125)
    assert torch.isfinite(result).all()
    assert model.training
