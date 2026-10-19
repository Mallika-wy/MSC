import pytest
import torch

from msc.models.flow_matching import FlowMatching


def _conditions(batch=2):
    return {"context": torch.zeros(batch, 1, 8)}


def test_rtc_training_loss_freezes_prefix_and_excludes_it():
    flow = FlowMatching()
    data = torch.randn(2, 30, 125)
    noise = torch.randn_like(data)
    mask = torch.ones(2, 30, dtype=torch.bool)
    seen = {}

    def generator(x, time, conditions, valid_mask=None):
        seen["x"] = x.detach()
        seen["time"] = time.detach()
        return noise - data

    loss = flow.loss(generator, data, _conditions(), mask, noise=noise, times=torch.full((2,), 0.7),
                     rtc_training=True, max_delay=8, delays=torch.tensor([3, 0]))
    assert torch.isclose(loss, torch.tensor(0.0))
    torch.testing.assert_close(seen["x"][0, :3], data[0, :3])
    torch.testing.assert_close(seen["time"][0, :3], torch.zeros(3))
    torch.testing.assert_close(seen["time"][1], torch.full((30,), 700.0))


def test_training_rtc_sampling_hard_freezes_prefix():
    flow = FlowMatching()
    previous = torch.randn(1, 30, 125)
    initial = torch.randn_like(previous)

    def zero_velocity(x, time, conditions, valid_mask=None):
        return torch.zeros_like(x)

    out = flow.sample_training_rtc(zero_velocity, _conditions(1), previous_actions=previous,
                                   inference_delay=5, max_delay=8, num_steps=4, initial_noise=initial)
    torch.testing.assert_close(out[:, :5], previous[:, :5])
    torch.testing.assert_close(out[:, 5:], initial[:, 5:])


def test_test_time_rtc_handles_guidance_and_preserves_grad_mode():
    flow = FlowMatching()
    previous = torch.randn(1, 30, 125)

    def generator(x, time, conditions, valid_mask=None):
        return x * 0.25

    out = flow.sample_rtc(generator, _conditions(1), previous_actions=previous,
                          inference_delay=4, execution_horizon=12, num_steps=3)
    assert out.shape == previous.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("schedule", ["hard", "linear", "exponential", "simple"])
def test_rtc_masks_have_expected_regions(schedule):
    mask = FlowMatching.rtc_mask(30, 4, 12, schedule, "cpu")
    torch.testing.assert_close(mask[:4], torch.ones(4))
    if schedule != "simple":
        torch.testing.assert_close(mask[-12:], torch.zeros(12))
    assert (mask >= 0).all() and (mask <= 1).all()
