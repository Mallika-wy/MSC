"""MSC v3 architecture and feature contracts; video layouts here are synthetic."""
import pytest
import torch
from msc.models.msc_model import MSCModel
from msc.models.attention import normalize_qk
from msc.models.motion_generator import MotionBlock, MotionGenerator
from msc.models.conditioning.fusion import ConditionMemory, ConditionBridge
from msc.models.audit import parameter_report


def small_model(**kwargs):
    return MSCModel(hidden_dim=128, num_layers=2, num_heads=4, ffn_dim=384,
                    dropout=0., text_dropout=0., audio_dropout=0., video_dropout=0., **kwargs)


def open_gates(model):
    with torch.no_grad():
        for layer in model.motion_generator.layers:
            layer.modulation[-1].bias.view(9, -1)[[2, 5, 8]] = .1


def test_zero_gate_identity_and_learning(batch):
    model = small_model(use_text=False)
    block = model.motion_generator.layers[0]
    memory = model.encode_conditions(batch)
    x, time = torch.randn(4,30,128), torch.randn(4,30,128)
    mask = torch.ones(4,30,dtype=torch.bool)
    torch.testing.assert_close(block(x,time,memory,mask), x, atol=0, rtol=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    model.compute_loss(batch).backward()
    assert block.modulation[-1].weight.grad.abs().sum() > 0
    assert block.self_attn.qkv.weight.grad.abs().sum() == 0
    assert model.audio_adapter[-1].weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    model.compute_loss(batch).backward()
    for module in [block.self_attn.qkv, block.query, block.key, block.value,
                   block.ff[0], model.audio_adapter[-1], model.motion_generator.output]:
        assert module.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('spatial,dim', [(2,6), (3,10)])
def test_video_synthetic_forward_backward_and_mask(batch, spatial, dim):
    model = small_model(use_text=False, use_audio=False, video_dim=dim, spatial_tokens=spatial)
    open_gates(model)
    data = {k:v for k,v in batch.items() if not k.startswith('audio_')}
    mask = torch.ones(4,3,spatial,dtype=torch.bool)
    mask[:,-1,-1] = False
    data.update(video_tokens=torch.randn(4,3,spatial,dim), video_mask=mask,
                video_intervals=torch.tensor([[-.2,.2],[.2,.6],[.6,1.1]]).expand(4,-1,-1).clone())
    memory = model.encode_conditions(data)
    assert (memory.temporal_support[~memory.mask] == 0).all()
    loss = model.compute_loss(data)
    loss.backward()
    assert model.condition_bridge.adapters['video'][-1].weight.grad.abs().sum() > 0
    assert model.condition_bridge.spatial.weight.grad.abs().sum() > 0
    noise, time = torch.randn(4,30,125), torch.rand(4)
    expected = model(noise,time,data)
    data['video_tokens'][~mask] = float('nan')
    torch.testing.assert_close(model(noise,time,data), expected)
    generated = model.sample(data['motion_history'], video_tokens=data['video_tokens'],
                             video_mask=mask, video_intervals=data['video_intervals'], num_steps=2)
    assert generated.shape == (4,30,125) and torch.isfinite(generated).all()
    data['video_intervals'][0,0] = torch.tensor([1.,0.])
    with pytest.raises(ValueError,match='interval'):
        model.encode_conditions(data)
    data['video_intervals'][0,0] = torch.tensor([0.,.2])
    data['video_tokens'] = torch.randn(4,3,spatial,dim+1)
    with pytest.raises(ValueError,match='layout'):
        model.encode_conditions(data)


def test_mixed_modalities_and_input_contract(batch):
    model = small_model()
    open_gates(model)
    data = dict(batch)
    data['audio_mask'] = batch['audio_mask'].clone()
    data['audio_mask'][:2] = False
    data.update(text_tokens=torch.randn(4,7,768), text_mask=torch.zeros(4,7,dtype=torch.bool))
    data['text_mask'][:2] = True
    model.compute_loss(data).backward()
    assert model.text_adapter[-1].weight.grad.abs().sum() > 0
    assert model.audio_adapter[-1].weight.grad.abs().sum() > 0
    model.eval()
    x,t = torch.randn(4,30,125),torch.rand(4)
    original = model(x,t,data)
    data['text_tokens'][0] += 100
    changed = model(x,t,data)
    assert not torch.allclose(original[0],changed[0])
    torch.testing.assert_close(original[1:],changed[1:],atol=0,rtol=0)
    data['audio_mask'][0] = True
    with pytest.raises(ValueError,match='only one'):
        model.compute_loss(data)
    data['audio_mask'].zero_()
    data['text_mask'].zero_()
    with pytest.raises(ValueError,match='require one'):
        model.compute_loss(data)


def test_qk_norm_and_temporal_mask():
    x = torch.randn(2,4,30,32)
    torch.testing.assert_close(normalize_qk(x), normalize_qk(7*x))
    memory = ConditionMemory(torch.randn(2,6,128),torch.tensor([[True]*5+[False]]*2),
        torch.zeros(2,6),torch.zeros(2,6,2),torch.zeros(2,6,dtype=torch.long))
    memory.temporal_support[:,-1] = float('nan')
    layer = MotionBlock(128,4,0.)
    bias = layer.attention_bias(memory,30)
    assert torch.isneginf(bias[...,-1]).all()
    assert torch.isfinite(bias[...,:5]).all()


def test_mixed_memory_shared_padding_matches_concatenation_and_gradients(batch):
    model = small_model(video_dim=6, spatial_tokens=2).eval()
    open_gates(model)
    data = dict(batch)
    data['audio_mask'] = batch['audio_mask'].clone()
    data['audio_mask'][[0, 2, 3]] = False
    data.update(text_tokens=torch.randn(4, 7, 768), text_mask=torch.zeros(4, 7, dtype=torch.bool),
                video_tokens=torch.randn(4, 3, 2, 6), video_mask=torch.zeros(4, 3, 2, dtype=torch.bool),
                video_intervals=torch.tensor([[-.2, .2], [.2, .6], [.6, 1.1]]).expand(4, -1, -1))
    data['text_mask'][0] = True
    data['video_mask'][2:] = True
    packed = model.encode_conditions(data, require_external=True)
    memories = [model.encode_conditions({key: value for key, value in data.items()
                if not key.startswith(('text_', 'audio_', 'video_')) or key.startswith(modality + '_')})
                for modality in ('text', 'audio', 'video')]
    reference = ConditionMemory(**{name: torch.cat([getattr(memories[0], name)[:, :5],
                *[getattr(memory, name)[:, 5:] for memory in memories]], 1)
                for name in ('tokens', 'mask', 'relative_times', 'temporal_support', 'modality_ids')})
    assert packed.tokens.shape[1] == 5 + max(7, data['audio_tokens'].shape[1], 6)
    assert packed.tokens.shape[1] < reference.tokens.shape[1]
    for name in ('tokens', 'relative_times', 'temporal_support', 'modality_ids'):
        torch.testing.assert_close(getattr(packed, name)[packed.mask], getattr(reference, name)[reference.mask])
    x, times = torch.randn(4, 30, 125), torch.rand(4)
    actual = model.motion_generator(x, times, packed)
    expected = model.motion_generator(x, times, reference)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    parameters = [p for p in model.parameters() if p.requires_grad]
    probe = torch.randn_like(actual)
    left = torch.autograd.grad((actual * probe).sum(), parameters, allow_unused=True)
    right = torch.autograd.grad((expected * probe).sum(), parameters, allow_unused=True)
    for a, b in zip(left, right):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('mode', ['train','test_time'])
def test_v3_rtc_with_cfg(batch, mode):
    model = small_model(use_text=False,rtc=True)
    open_gates(model)
    model.compute_loss(batch,rtc_delays=torch.tensor([0,1,7,29])).backward()
    kwargs = dict(audio_tokens=batch['audio_tokens'],audio_mask=batch['audio_mask'],
        audio_times=batch['audio_times'],audio_metadata=None,previous_motion=batch['motion_future'],
        rtc_mode=mode,inference_delay=3,num_steps=3,cfg_scale=2.,initial_noise=torch.randn(4,30,125))
    model.audio_metadata = {'synthetic':True}
    kwargs['audio_metadata'] = model.audio_metadata
    first = model.sample(batch['motion_history'],**kwargs)
    second = model.sample(batch['motion_history'],**kwargs)
    assert first.shape == (4,30,125) and torch.isfinite(first).all()
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    if mode == 'train':
        torch.testing.assert_close(first[:,:3],batch['motion_future'][:,:3])


def test_full_parameter_budget():
    # Meta tensors exercise the actual architecture without allocating 1.2 GB.
    with torch.device('meta'):
        model = torch.nn.Module()
        model.motion_generator = MotionGenerator(ffn_dim=3072)
        model.condition_bridge = ConditionBridge(1024,text_dim=768)
    report = parameter_report(model)
    assert report['trainable'] == 296_883_767


def test_scalar_time_broadcast_matches_frame_time(batch):
    import copy
    model = small_model(use_text=False)
    open_gates(model)
    other = copy.deepcopy(model)
    times = torch.rand(4)
    noise = torch.randn_like(batch['motion_future'])
    scalar = model.compute_loss(batch,noise=noise,times=times)
    expanded = other.compute_loss(batch,noise=noise,times=times[:,None].expand(-1,30))
    torch.testing.assert_close(scalar,expanded)
    scalar.backward()
    expanded.backward()
    for a,b in zip(model.parameters(),other.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-5)


@pytest.mark.integration
def test_compiled_expert_matches_eager_forward_and_gradients(batch):
    import copy
    if not torch.cuda.is_available():
        pytest.skip('Compiled CUDA parity requires CUDA and a working Triton installation')
    pytest.importorskip('triton')
    model = small_model(use_text=False).cuda().eval()
    open_gates(model)
    compiled = copy.deepcopy(model)
    compiled.motion_generator.compile(dynamic=True)
    data = {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    noise, times = torch.randn_like(data['motion_future']), torch.rand(4,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        eager_loss = model.compute_loss(data,noise=noise,times=times)
        compiled_loss = compiled.compute_loss(data,noise=noise,times=times)
    torch.testing.assert_close(compiled_loss,eager_loss,atol=.002,rtol=.01)
    eager_loss.backward()
    compiled_loss.backward()
    for a,b in zip(model.parameters(),compiled.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=2e-4,rtol=.1)
