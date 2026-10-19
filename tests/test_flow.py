import pytest
import torch
from msc.models.flow_matching import FlowMatching


@pytest.mark.parametrize("steps", [1, 10, 37])
def test_euler_direction_and_endpoint(steps):
    data, noise = torch.randn(2, 30, 125), torch.randn(2, 30, 125)
    seen = []
    def oracle(x, t, conditions, valid_mask=None):
        seen.append(t[0].item())
        return noise - data
    out = FlowMatching().sample(oracle, {"context": torch.zeros(2, 1, 8)},
                               initial_noise=noise, num_steps=steps)
    torch.testing.assert_close(out, data, atol=1e-5, rtol=1e-5)
    assert len(seen) == steps
    assert seen[0] == 1000
    assert all(a > b for a, b in zip(seen, seen[1:]))


def test_masked_padding_does_not_change_loss(model, batch):
    model.eval()
    noise = torch.randn_like(batch["motion_future"])
    times = torch.rand(4)
    loss = model.compute_loss(batch, noise=noise, times=times)
    altered = dict(batch)
    altered["motion_future"] = batch["motion_future"].masked_fill(~batch["motion_mask"][..., None], float("nan"))
    other = model.compute_loss(altered, noise=noise, times=times)
    torch.testing.assert_close(loss, other)


def test_loss_and_gradient(model, batch):
    loss = model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.motion_generator.output.weight.grad.abs().sum() > 0
    assert model.text_adapter.weight.grad.abs().sum() > 0
    # FiLM is deliberately zero-initialized: adapter gradients begin after FiLM updates.
    assert model.motion_generator.layers[0].audio_film[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-0.001)
    model.zero_grad(set_to_none=True)
    model.compute_loss(batch).backward()
    assert model.audio_adapter[1].weight.grad.abs().sum() > 0


def test_cfg_identity(model, batch):
    model.eval()
    cond = model.conditions(batch)
    noise = torch.randn(4, 30, 125)
    a = model.flow.sample(model.motion_generator, cond, initial_noise=noise)
    b = model.flow.sample(model.motion_generator, cond, initial_noise=noise, null_conditions=cond, cfg_scale=2.5)
    torch.testing.assert_close(a, b)
