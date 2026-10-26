import copy

import pytest
import torch

from experiments.benchmark_motion_block import CandidateBlock, inputs
from msc.models.conditioning.fusion import ConditionMemory
from msc.models.motion_generator import MotionBlock


@pytest.mark.parametrize('length', [50,113])
@pytest.mark.parametrize('partial,time_frames', [(False,1),(False,30),(True,1)])
@pytest.mark.parametrize('packed,unmasked', [(True,False),(False,True),(True,True)])
def test_compute_candidates_preserve_outputs_inputs_and_parameter_gradients(length,partial,time_frames,packed,unmasked):
    base = MotionBlock(64,4,0.,192).double()
    with torch.no_grad():
        # Open the initially zero AdaLN gates so all attention paths contribute.
        base.modulation[-1].weight.normal_(std=.02)
        base.modulation[-1].bias.normal_(std=.02)
    candidate = CandidateBlock(64,4,0.,192,packed_kv=packed,
                               all_frames_valid=unmasked and not partial).double()
    candidate.load_state_dict(base.state_dict(), strict=True)
    assert list(base.state_dict()) == list(candidate.state_dict())
    data = inputs(2,64,length,dtype=torch.float64,partial=partial,time_frames=time_frames)
    cloned = [value.detach().clone().requires_grad_() for value in (data[0],data[1],data[2].tokens)]
    memory = ConditionMemory(cloned[2],data[2].mask,data[2].relative_times,
                             data[2].temporal_support,data[2].modality_ids)
    expected = base(*data)
    actual = candidate(cloned[0],cloned[1],memory,data[3])
    torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
    probe = torch.randn_like(expected)
    (expected*probe).mean().backward()
    (actual*probe).mean().backward()
    for left,right in zip((data[0],data[1],data[2].tokens),cloned):
        torch.testing.assert_close(left.grad,right.grad,atol=1e-9,rtol=1e-6)
    for (name,left),(other,right) in zip(base.named_parameters(),candidate.named_parameters()):
        assert name == other
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad,right.grad,atol=1e-9,rtol=1e-6)
    assert base.key.weight.grad.abs().sum() > 0
    assert base.value.weight.grad.abs().sum() > 0
