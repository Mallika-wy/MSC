"""Correctness tests use explicitly synthetic releases/features, not real results."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
import torch
import pyarrow.parquet as pq
from data_pipeline.tests.fixtures import fixture_records
from data_pipeline.release.export import export_release
from data_pipeline.schema.manifest import Inventory
from msc.data.release_dataset import ReleaseDataset
from data_pipeline.release.validate import audit_splits
from msc.data.collator import collate_motion
from msc.data.temporal import select_tokens
from msc.models.conditioning.fusion import ConditionMemory
from msc.models.losses import dynamics_loss, joint_dynamics
from msc.models.msc_model import MSCModel
from msc.training.trainer import EpochBatchSampler, Trainer
from msc.factory import build_model, build_dataset, seed_everything
from msc.config import Config, load_config
from conftest import ToyDataset


@pytest.fixture
def release(tmp_path):
    source = tmp_path/'source'
    inv,tracks = fixture_records(source)
    inv.provenance['train_default'] = 'source_g1'
    for i,f in enumerate(inv.features):
        if 'audio' in f.encoder_name:
            values = np.broadcast_to(np.linspace(-1,1,768,dtype=np.float16),(1500,768)).copy()
            np.save(source/'song-tokens.npy',values)
            inv.features[i] = replace(f,shape=list(values.shape),encoder_name='synthetic_mert')
    root = tmp_path/'release'
    export_release(inv,tracks,root)
    # Small groups exercise windows intersecting several groups.
    shard = root/'motion/train/shard-00000.parquet'
    table = pq.read_table(shard)
    pq.write_table(table,shard,row_group_size=12)
    return root


def test_window_io_parity_and_determinism(release,monkeypatch):
    import msc.data.release_dataset as module
    reads = []
    groups = []
    parquet_file = pq.ParquetFile
    class File:
        def __init__(self,*args,**kwargs):
            self.file=parquet_file(*args,**kwargs)
            self.metadata=self.file.metadata
            self.num_row_groups=self.file.num_row_groups
        def read_row_groups(self,indices):
            groups.append((len(indices),self.num_row_groups))
            return self.file.read_row_groups(indices)
    monkeypatch.setattr(pq,'ParquetFile',File)
    original = module.read_motion
    def reader(root,record):
        reads.append(record)
        return original(root,record)
    monkeypatch.setattr(module,'read_motion',reader)
    data = ReleaseDataset([release],tasks=['audio'],cache_bytes=0)
    item = data[5]
    other = data[5]
    assert all(x.num_frames == 35 for x in reads)
    _,motion,_,k,_ = data.window_identity(5)
    assert reads[0].row_start == motion.row_start+k-5
    assert all(1 < n < total for n,total in groups)
    assert item['window_id'] == other['window_id']
    torch.testing.assert_close(item['motion_future'],other['motion_future'],rtol=0,atol=0)
    q,_ = original(release,motion)
    q = torch.tensor(q[k-5:k+30])[None]
    fk = data.codec.kinematics.forward_kinematics(q)
    h,p,r = data.codec.prev_state_features_from_history(q[:,:5],fk['body_pos_w'][:,:5],None,30.)
    comp = data.codec.canonicalize(q[:,5:],fk['body_pos_w'][:,5:],None,p,r,30.)
    torch.testing.assert_close(item['motion_history'],h[0],rtol=0,atol=0)
    torch.testing.assert_close(item['motion_future'],data.codec.assemble_features(comp)[0],rtol=0,atol=0)
    decoded = data.codec.decode_to_world_qpos36(comp,p,r)
    torch.testing.assert_close(decoded,q[:,5:],atol=1e-5,rtol=1e-5)
    assert data.cache.size==0


def test_tasks_masks_and_real_feature_shapes(release):
    data = ReleaseDataset([release],tasks=['audio', 'video'])
    samples = [data[data.task_ranges[t][0][0]] for t in ('audio','video')]
    batch = collate_motion(samples)
    assert batch['audio_tokens'].shape[-1]==768
    assert batch['video_tokens'].shape[-2:]==(2,6)
    assert batch['audio_mask'][0].any()
    assert not batch['audio_mask'][1].any()
    model = MSCModel(hidden_dim=32,num_layers=1,num_heads=4,use_text=False,video_dim=6,spatial_tokens=2,
                     text_dropout=0,audio_dropout=0,video_dropout=0)
    memory = model.encode_conditions(batch)
    assert isinstance(memory,ConditionMemory)
    assert memory.mask[:,:5].all()
    null = memory.null()
    assert null.mask[:,:5].all() and not null.mask[:,5:].any()
    video = (memory.modality_ids==3) & memory.mask
    actual = memory.temporal_support[video]
    expected = batch['video_intervals'][:, :, None].expand(-1, -1, 2, -1)[batch['video_mask']]
    torch.testing.assert_close(actual, expected)
    loss = model.compute_loss(batch)
    loss.backward()
    for adapter in model.condition_bridge.adapters.values():
        assert adapter[1].weight.grad is not None
        assert torch.isfinite(adapter[1].weight.grad).all()


def test_missing_video_rejected(release):
    from data_pipeline.release.training_index import build_training_index
    table = pq.read_table(release/'meta/features.parquet')
    pq.write_table(table.slice(0,1),release/'meta/features.parquet')
    build_training_index(release)
    with pytest.raises(ValueError,match='No valid video'):
        ReleaseDataset([release],tasks=['video'])


def test_split_song_and_actor_audit(release):
    inv = Inventory.read(release)
    bad = copy.deepcopy(inv)
    bad.sequences[0].split='val'
    with pytest.raises(ValueError,match='leakage'):
        audit_splits([inv,bad])
    bad = copy.deepcopy(inv)
    bad.links[0].actor_id='wrong-actor'
    with pytest.raises(ValueError,match='Actor'):
        audit_splits([bad])
    bad = copy.deepcopy(inv)
    # Different source IDs/groups, still sharing the same song checksum.
    bad.sequences = [replace(s,source_sequence_id='other:'+s.source_sequence_id,split_group_id='other',split='val') for s in bad.sequences]
    bad.motions = [replace(m,motion_id='other:'+m.motion_id,sequence_id=m.sequence_id) for m in bad.motions]
    for link in bad.links:
        link.motion_id='other:'+link.motion_id
    with pytest.raises(ValueError,match='asset_.*leakage|leakage:.*asset_'):
        audit_splits([inv,bad])


def test_temporal_affine_and_intervals():
    from types import SimpleNamespace
    link = SimpleNamespace(time_scale=2.,time_offset_sec=.3)
    times = np.arange(100)/10
    indices,local = select_tokens(times,link,1.,2.)
    np.testing.assert_allclose(local,(times[indices]-.3)/2-1)
    assert local.min()>=-1e-12 and local.max()<1
    intervals = np.stack((times,times+.2),-1)
    indices,local = select_tokens(intervals,link,1.,2.,semantics='interval')
    assert (local[:,1]>0).all() and (local[:,0]<1).all()


def test_valid_nan_and_invalid_nan(model,batch):
    model.eval()
    reference = model.encode_conditions(batch)
    altered = dict(batch)
    altered['audio_tokens'] = batch['audio_tokens'].masked_fill(~batch['audio_mask'][...,None],float('nan'))
    altered['audio_times'] = batch['audio_times'].masked_fill(~batch['audio_mask'],float('nan'))
    other = model.encode_conditions(altered)
    torch.testing.assert_close(reference.tokens,other.tokens)
    altered['audio_tokens'] = batch['audio_tokens'].clone()
    altered['audio_tokens'][batch['audio_mask']]=float('inf')
    with pytest.raises(ValueError,match='NaN/Inf'):
        model.encode_conditions(altered)


def test_precomputed_audio_tokens_stay_frozen(model,batch):
    batch['audio_tokens']=batch['audio_tokens'].clone().requires_grad_()
    model.compute_loss(batch).backward()
    assert batch['audio_tokens'].grad is None
    assert model.audio_adapter[-1].weight.grad.abs().sum()>0


@pytest.mark.parametrize('length',[1,2,30])
def test_dynamics_boundaries_masks_and_zero_variance(length):
    full = torch.arange(-2,length).float()[None,:,None].expand(1,-1,29)/30
    history,joints = full[:,:2],full[:,2:]
    mask = torch.ones(1,length,dtype=torch.bool)
    v,a,vm,am = joint_dynamics(joints,history,mask)
    torch.testing.assert_close(v,torch.ones_like(v),atol=2e-6,rtol=1e-5)
    torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-4,rtol=0)
    vel,acc = dynamics_loss(joints,joints,history,mask,torch.zeros(29),torch.zeros(29))
    assert vel==acc==0
    pred = joints.clone().requires_grad_()
    errors = torch.where(torch.arange(length)%2==0,.1,-.1)[None,:,None]
    vel,acc = dynamics_loss(pred+errors,joints,history,mask,torch.ones(29),torch.ones(29))
    assert acc>vel
    (vel+acc).backward()
    assert pred.grad.abs().sum()>0
    if length>2:
        mask[:,1]=False
        _,_,vm,am = joint_dynamics(joints,history,mask)
        assert not vm[:,1:3].any() and not am[:,1:4].any()


@pytest.mark.parametrize('tau',[0.,.5,1.])
def test_auxiliary_gradient_through_clean_estimate(model,batch,tau):
    model.eval()
    future = model.representation.normalize(batch['motion_future'])
    noise = torch.randn_like(future)
    x = (1-tau)*future+tau*noise
    velocity = model.motion_generator(x,torch.full((4,),tau*1000),model.encode_conditions(batch),batch['motion_mask'])
    clean = model.representation.denormalize(x-tau*velocity)
    vel,acc = dynamics_loss(clean[...,9:38],batch['motion_future'][...,9:38],batch['motion_history'][...,9:38],batch['motion_mask'],torch.ones(29),torch.ones(29))
    (vel+acc).backward()
    magnitude = model.motion_generator.output.weight.grad.abs().sum()
    assert magnitude==0 if tau==0 else magnitude>0


def test_zero_weights_exact_fm_and_frame_times(model,batch):
    model.eval()
    noise = torch.randn_like(batch['motion_future'])
    times = torch.rand(4,30)
    model.loss_config = dict(joint_velocity=0.,joint_acceleration=0.,dynamic_warmup_steps=0)
    loss = model.compute_loss(batch,noise=noise,times=times)
    expected = model.flow.loss(model.motion_generator,model.representation.normalize(batch['motion_future']),model.encode_conditions(batch),batch['motion_mask'],noise=noise,times=times)
    torch.testing.assert_close(loss,expected,atol=0,rtol=0)


@pytest.mark.parametrize('size,batch_size',[(1,8),(10,3),(65543,257)])
def test_epoch_sampler_coverage_and_resume(size,batch_size):
    steps=(size+batch_size-1)//batch_size
    args=dict(size=size,batch_size=batch_size,stop=steps*2,seed=17)
    whole=list(EpochBatchSampler(start=0,**args))
    for epoch in range(2):
        indices=[i for batch in whole[epoch*steps:(epoch+1)*steps] for i in batch]
        assert len(indices)==size and sorted(indices)==list(range(size))
    offset=max(1,steps//2)
    assert whole[offset:]==list(EpochBatchSampler(start=offset,**args))
    if size>2:
        assert whole[:steps]!=whole[steps:]


def test_epoch_sampler_validation_ranges():
    sampler=EpochBatchSampler(100,3,0,2,17,ranges=[(3,5),(8,11)])
    indices=[i for batch in sampler for i in batch]
    assert sorted(indices)==[3,4,8,9,10]


def test_sampling_encodes_once_with_cfg(model,batch,monkeypatch):
    calls = []
    original = model.encode_conditions
    def wrapped(batch):
        calls.append(1)
        return original(batch)
    monkeypatch.setattr(model,'encode_conditions',wrapped)
    noise = torch.randn(4,30,125)
    one = model.sample(batch['motion_history'],initial_noise=noise,cfg_scale=2.,num_steps=3)
    assert len(calls)==1
    two = model.sample(batch['motion_history'],initial_noise=noise,cfg_scale=2.,num_steps=3)
    torch.testing.assert_close(one,two,atol=0,rtol=0)


def test_nested_configuration():
    cfg = load_config('configs/msc_v3_2xa6000.yaml',['model.condition.text.enabled=false','data.tasks=[audio]','model.losses.joint_velocity=0.0'])
    assert cfg.model.condition['text']['enabled'] is False
    assert cfg.model.losses['joint_velocity']==0


def test_temporal_bias_point_interval_text_and_mask():
    from msc.models.motion_generator import MotionBlock
    memory = ConditionMemory(torch.zeros(1,4,8),torch.tensor([[True,True,True,False]]),
        torch.tensor([[0.,0.,0.,float('nan')]]),
        torch.tensor([[[.5,.5],[.2,.8],[100.,100.],[0.,0.]]]),torch.tensor([[2,3,1,2]]))
    layer=MotionBlock(8,2,0.)
    bias=layer.attention_bias(memory,30)
    scale=torch.nn.functional.softplus(layer.alpha)
    torch.testing.assert_close(bias[0,:,0,0],-.5*scale)
    torch.testing.assert_close(bias[0,:,15,1],torch.zeros(2))
    torch.testing.assert_close(bias[0,:,:,2],torch.zeros(2,30))
    assert torch.isneginf(bias[0,:,:,3]).all()
    layer.temporal_bias='none'
    assert (layer.attention_bias(memory,30)[...,:3]==0).all()


def test_cache_budget_and_short_subset(release):
    from msc.data.release_dataset import ByteCache
    cache=ByteCache(32)
    for k in range(10):
        cache.get(k,lambda:(np.ones(4,dtype=np.float32),))
        assert cache.size<=32
    assert len(cache.items)==2
    data=ReleaseDataset([release],tasks=['audio'],max_motions_per_task=1,max_windows_per_motion=3)
    assert len(data)==3
    assert len({data[i]['window_id'] for i in range(3)})==3
    assert len({data.window_identity(i)[1].motion_id for i in range(3)})==1


def test_motion_manifest_selects_exact_caption_interval_and_survives_pickle(release, tmp_path):
    import pickle
    full = ReleaseDataset([release], tasks=['text'])
    ri, motion, _, lo, hi, condition = full.spans[0]
    end = min(hi, lo + 3)
    manifest = tmp_path / 'selection.json'
    manifest.write_text(json.dumps({'rows': [dict(release_index=ri, motion_id=motion.motion_id,
        text=condition[2].strip(), start_frame=lo, end_frame_exclusive=end)]}))
    selected = ReleaseDataset([release], tasks=['text'], motion_manifest=str(manifest))
    assert len(selected) == end - lo
    assert selected.identity != full.identity
    restored = pickle.loads(pickle.dumps(selected))
    assert restored.identity == selected.identity
    for i in range(len(selected)):
        assert restored.window_identity(i)[1].motion_id == motion.motion_id
        assert restored.window_identity(i)[3] == lo + i
        torch.testing.assert_close(restored[i]['motion_future'], selected[i]['motion_future'])
        assert restored[i]['text'].strip() == condition[2].strip()


def test_training_start_reads_only_prebuilt_index(release, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Training startup must not read Inventory or Parquet metadata')
    monkeypatch.setattr(Inventory, 'read', forbidden)
    monkeypatch.setattr(pq, 'read_table', forbidden)
    data = ReleaseDataset([release], tasks=['text', 'audio'],
                          max_motions_per_task=1, max_windows_per_motion=2)
    assert len(data) == 4
    assert data[data.task_ranges['text'][0][0]]['text'].strip()


def test_training_index_publication_detects_stale_and_corrupt_files(release):
    from data_pipeline.release.training_index import validate_training_index
    summary = validate_training_index(release)
    provenance = release / 'meta/provenance.json'
    data = json.loads(provenance.read_text())
    data['test_revision'] = 'changed'
    provenance.write_text(json.dumps(data))
    rebuilt = validate_training_index(release)
    assert rebuilt['identity'] != summary['identity']
    path = release / 'training' / rebuilt['splits']['train']['allow_estimated']['file']
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='index checksum'):
        validate_training_index(release)


@pytest.mark.parametrize('scale,context', [(1., 0.), (1., .25), (2., .25)])
def test_media_index_matches_individual_window_queries(scale, context):
    from types import SimpleNamespace
    from data_pipeline.release.training_index import _media_runs
    from msc.data.temporal import covers
    times = np.arange(1500, dtype=np.float64) / 75
    # Include MERT block-boundary gaps and a genuinely irregular region.
    times = np.delete(times, [374, 375, 800, 801, 802, 803])
    times[1100:1110] += np.arange(10) * .00003
    link = SimpleNamespace(time_scale=scale, time_offset_sec=.2, link_id='link',
                           motion_valid_start_sec=.1, motion_valid_end_sec=5.)
    feature = SimpleNamespace(token_time_semantics='point', token_rate_hz=75., feature_id='feature')
    motion = dict(num_frames=150)
    runs = _media_runs(motion, [(link, feature, times)], context)
    indexed = {k for lo, hi, _ in runs for k in range(lo, hi)}
    expected = set()
    for k in range(5, 121):
        start, end = k/30, (k+30)/30
        if link.motion_valid_start_sec > start+1e-8 or link.motion_valid_end_sec < end-1e-8:
            continue
        _, local = select_tokens(times, link, start, end, context)
        if covers(local, 0., 1., 'point', 75*scale):
            expected.add(k)
    assert indexed == expected


def test_text_index_matches_annotation_coverage():
    from data_pipeline.release.training_index import _text_runs
    annotations = [dict(annotation_id='long', text='walk', start_sec=0., end_sec=5.),
                   dict(annotation_id='short', text='turn', start_sec=.333333333, end_sec=2.2),
                   dict(annotation_id='tail', text='stop', start_sec=2.4, end_sec=5.)]
    runs = _text_runs(dict(num_frames=150), annotations)
    for k in range(5, 121):
        valid = [a for a in annotations if a['start_sec'] <= k/30+1e-8 and a['end_sec'] >= (k+30)/30-1e-8]
        expected = min(valid, key=lambda a: (a['end_sec']-a['start_sec'], a['annotation_id']))['text']
        actual = next(cond[0] for start,end,cond in runs if start <= k < end)
        assert actual == expected


def test_statistics_identity_and_zero_scales(release,cfg,tmp_path):
    from msc.data.stats import compute_stats
    data=ReleaseDataset([release],tasks=['audio'],max_motions_per_task=1,max_windows_per_motion=2)
    stats=compute_stats(data)
    assert stats['split']=='train' and stats['dataset_identity']==data.identity
    assert min(stats['velocity_scale'])>=1e-4
    path=tmp_path/'stats.json'
    path.write_text(json.dumps(stats))
    cfg.data.backend='msc_release';cfg.data.roots=[str(release)];cfg.data.tasks=['audio']
    cfg.model.use_text=False;cfg.model.losses=dict(joint_velocity=.05,joint_acceleration=.01);cfg.model.dynamic_stats_path=str(path)
    cfg.train.validation_every=0
    model=build_model(cfg,data.audio_metadata,data.feature_metadata)
    assert model.dynamic_stats_identity==data.identity
    model.dynamic_stats_identity='wrong'
    with pytest.raises(ValueError,match='statistics'):
        Trainer(model,data,cfg)


def test_evaluation_measures_amplitude_and_rollout(release,cfg,tmp_path):
    from msc.evaluate import evaluate_dataset
    data=ReleaseDataset([release],tasks=['audio'],max_motions_per_task=1)
    cfg.model.use_text=False
    model=build_model(cfg,data.audio_metadata,data.feature_metadata)
    # Evaluate index zero to guarantee two contiguous chunks exist.
    from msc.evaluate import rollout
    from msc.motion.codec import G1MotionFeatureCodec
    from msc.motion.kinematics import G1Kinematics
    codec=G1MotionFeatureCodec(G1Kinematics(),num_prev_states=5,canonical_frame_idx=4,rotation_representation='rot6d')
    directory=tmp_path/'rollout';directory.mkdir()
    result=rollout(model.eval(),data,0,codec,2,'cpu',directory)
    assert result['chunks']==2 and len(result['boundary_joint_jumps'])==1
    result=evaluate_dataset(model,data,batches=1,batch_size=1,output_dir=tmp_path/'eval')
    metrics=result['tasks']['audio'][0]
    assert 'joint_amplitude' in metrics and 'target_joint_amplitude' in metrics
    assert 'condition_controls' in metrics and 'fk_mpjpe' in metrics


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_bf16_dynamics_finite(cfg,toy_dataset,batch):
    cfg.model.losses=dict(joint_velocity=.05,joint_acceleration=.01,dynamic_warmup_steps=0)
    data=toy_dataset
    model=build_model(cfg,data.audio_metadata,data.feature_metadata).to('cuda:0')
    batch={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_validation_does_not_change_training_rng(release,cfg,tmp_path):
    import hashlib
    inv=Inventory.read(release)
    inv.sequences=[replace(s,source_dataset='validation_fixture',split='val') for s in inv.sequences]
    changed={}
    for asset in inv.assets:
        asset.uri=None
        asset.source_asset_id='val:'+asset.source_asset_id
        asset.checksum=hashlib.sha256(('val:'+asset.checksum).encode()).hexdigest()
        changed[asset.asset_id]=asset.checksum
    for feature in inv.features:
        feature.input_asset_checksum=changed[feature.asset_id]
    inv.external_roots['quality_fixture']=str(release)
    inv.motions=[replace(m,quality_ref='root://quality_fixture/'+m.quality_ref) for m in inv.motions]
    from data_pipeline.release.export import read_motion
    tracks={m.motion_id:read_motion(release,m)[0] for m in inv.motions}
    validation=tmp_path/'validation'
    export_release(inv,tracks,validation)
    cfg.data.backend='msc_release';cfg.data.roots=[str(release),str(validation)]
    cfg.data.tasks=['audio'];cfg.data.max_motions_per_task=1;cfg.data.max_windows_per_motion=2
    cfg.model.use_text=False;cfg.model.losses=dict(joint_velocity=0.,joint_acceleration=0.)
    cfg.train.validation_every=1;cfg.train.validation_batches=1;cfg.train.steps=2
    cfg.train.output_dir=str(tmp_path/'validated')
    data=build_dataset(cfg)
    seed_everything(cfg.train.seed)
    trainer=Trainer(build_model(cfg,data.audio_metadata,data.feature_metadata),data,cfg)
    before=torch.get_rng_state().clone()
    values=trainer.validate()
    assert 'audio' in values
    torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
    trainer.fit()
    assert (tmp_path/'validated/validation.jsonl').exists()


def test_origin_is_not_added_twice(release,monkeypatch):
    inv=Inventory.read(release)
    inv.motions=[replace(m,time_origin_sec=1/30) for m in inv.motions]
    with monkeypatch.context() as patch:
        patch.setattr(Inventory,'read',lambda *a,**kw:inv)
        data=ReleaseDataset([release],tasks=['audio'],max_motions_per_task=1)
        sample=data[0]
        ri,m,task,k,condition=data.window_identity(0)
        link=next(l for l in inv.links if l.link_id==condition[1])
        feature=next(f for f in inv.features if f.feature_id==condition[2])
        from data_pipeline.release.features import read_feature
        _,times=read_feature(release,feature,inv.external_roots)
        _,expected=select_tokens(times,link,k/30,(k+30)/30,.25)
        torch.testing.assert_close(sample['audio_times'],torch.tensor(expected,dtype=torch.float32),atol=0,rtol=0)


def test_release_raw_text_and_mixed_batch(release):
    data=ReleaseDataset([release],tasks=['text', 'audio'],max_windows_per_motion=1)
    text=data[data.task_ranges['text'][0][0]]
    audio=data[data.task_ranges['audio'][0][0]]
    assert text['text'].strip() and 'text_tokens' not in text
    assert 'text' not in audio
    batch=collate_motion([text,audio])
    assert batch['text']==[text['text'],'']
    assert batch['audio_mask'][1].any() and not batch['audio_mask'][0].any()


@pytest.mark.integration
def test_release_online_t5_training_and_sampling(release):
    import os
    path=os.environ.get('MSC_TEST_T5_PATH')
    if not path:
        pytest.skip('Set MSC_TEST_T5_PATH for real online T5')
    data=ReleaseDataset([release],tasks=['text', 'audio'],max_windows_per_motion=1)
    batch=collate_motion([data[data.task_ranges[t][0][0]] for t in ('text','audio')])
    cfg=Config()
    cfg.model.hidden_dim=64; cfg.model.num_layers=2; cfg.model.num_heads=4
    cfg.data.roots=[str(release)]
    cfg.model.t5_path=path
    cfg.model.condition={'text':{'enabled':True,'dim':'auto','dropout':0.},
                         'audio':{'enabled':True,'dropout':0.}}
    model=build_model(cfg,data.audio_metadata,data.feature_metadata)
    model.train()
    calls=[]
    hook=model.text_encoder.encoder.register_forward_pre_hook(lambda *args:calls.append(1))
    memory=model.encode_conditions(batch)
    assert not memory.mask[1][memory.modality_ids[1]==1].any()
    calls.clear()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.compute_loss(batch).backward()
    optimizer.step()
    optimizer.zero_grad()
    calls.clear()
    loss=model.compute_loss(batch)
    loss.backward()
    assert len(calls)==1
    assert not model.text_encoder.encoder.training
    assert all(not p.requires_grad and p.grad is None for p in model.text_encoder.parameters())
    assert model.text_adapter[-1].weight.grad.abs().sum()>0
    calls.clear()
    generated=model.sample(batch['motion_history'],text=batch['text'],num_steps=2,cfg_scale=2.)
    assert len(calls)==1 and generated.shape==(2,30,125) and generated.isfinite().all()
    hook.remove()


def test_shard_statistics_match_window_statistics(release):
    from msc.data.stats import compute_stats, compute_release_stats
    data = ReleaseDataset([release],tasks=['text','audio'],max_motions_per_task=1,max_windows_per_motion=5)
    expected, actual = compute_stats(data), compute_release_stats(data,workers=1)
    assert actual['count'] == expected['count']
    assert actual['dataset_identity'] == expected['dataset_identity']
    np.testing.assert_allclose(actual['velocity_scale'],expected['velocity_scale'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(actual['acceleration_scale'],expected['acceleration_scale'],rtol=1e-6,atol=1e-7)


def test_batched_window_reader_codec_and_conditions_parity(release, monkeypatch):
    data = ReleaseDataset([release], tasks=['text','audio','video'])
    indices = [data.task_ranges[t][0][0]+i for t in data.tasks for i in (0,3,8)]
    expected = [data[index] for index in indices]
    # A fresh reader must also handle windows crossing the tiny fixture row groups.
    data.cache = None
    actual = data.__getitems__(indices)
    for left, right in zip(expected, actual):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=1e-6)
            else:
                assert left[key] == right[key]
    calls=[]
    original=data.window_reader.decode
    monkeypatch.setattr(data.window_reader,'decode',lambda *args:(calls.append(1) or original(*args)))
    data.__getitems__(indices)
    assert not calls  # Cached row groups are reused across different windows.


def test_row_groups_decode_once_per_batch_even_without_lru_capacity(release, monkeypatch):
    data = ReleaseDataset([release], tasks=['text'], cache_bytes=0)
    data.__getitems__([0])
    calls = []
    original = data.window_reader.decode
    monkeypatch.setattr(data.window_reader, 'decode', lambda file, group: (calls.append(group) or original(file, group)))
    indices = [0, 3, 8, 0, 3, 8]
    actual = data.__getitems__(indices)
    assert len(calls) == len(set(calls))
    for index, sample in zip(indices, actual):
        torch.testing.assert_close(sample['motion_future'], data[index]['motion_future'], rtol=0, atol=0)


def test_mmap_storage_cache_preserves_features_and_avoids_parquet_decoding(release, tmp_path, monkeypatch):
    from pathlib import Path
    from msc.data.storage_cache import build_storage_cache, StorageCache
    from msc.data.window_reader import MotionWindowReader
    import msc.data.release_dataset as module
    cfg = Config()
    cfg.data.roots = [str(release)]
    cfg.data.tasks = ['text', 'audio', 'video']
    directory = tmp_path/'cache'
    build_storage_cache(cfg, directory, workers=2, splits=('train',))
    original = ReleaseDataset([release], tasks=cfg.data.tasks)
    indices = [original.task_ranges[t][0][0] + i for t in cfg.data.tasks for i in (0, 3, 8)]
    expected = original.__getitems__(indices)
    cached = ReleaseDataset([release], tasks=cfg.data.tasks, cache_dir=str(directory))
    def unexpected(*args, **kwargs):
        raise AssertionError('Cached training read a Parquet payload')
    monkeypatch.setattr(MotionWindowReader, 'decode', unexpected)
    read_feature = module.read_feature
    def direct_npy(root, feature, external_roots):
        assert not feature.feature_uri.endswith('.parquet')
        return read_feature(root, feature, external_roots)
    monkeypatch.setattr(module, 'read_feature', direct_npy)
    for left, right in zip(expected, cached.__getitems__(indices)):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            else:
                assert left[key] == right[key]
    assert original.identity == cached.identity
    assert cached.__getstate__()['cache_dir'] == str(directory)
    # Replacing a source invalidates its derivative instead of serving stale data.
    import os
    source = next(p for p in StorageCache(directory).entries if '/motion/' in p)
    stat = Path(source).stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert StorageCache(directory).motion(source) is None


def test_packed_feature_mmap_matches_parquet_payload(release, tmp_path):
    from pathlib import Path
    from msc.data.storage_cache import convert_source, StorageCache
    from data_pipeline.release.features import FeatureShardWriter, read_feature
    data = ReleaseDataset([release], tasks=['audio'])
    feature = next(iter(data._lookup[0][1].values()))
    values, times = read_feature(release, feature, data.external_roots[0])
    writer = FeatureShardWriter(tmp_path/'packed')
    source = writer.add('audio', feature.feature_id, values, times)
    writer.close()
    feature = replace(feature, feature_uri=str(source), timestamp_uri=str(source))
    directory = tmp_path/'cache'
    directory.mkdir()
    key, item = convert_source((str(source), 'feature', {feature.feature_id: feature}, directory))
    (directory/'manifest.json').write_text(json.dumps(dict(version=1, sources={key:item})))
    actual = StorageCache(directory).feature(source, feature)
    assert all(isinstance(value, np.memmap) for value in actual)
    for left, right in zip((values, times), actual):
        np.testing.assert_array_equal(left, right)


@pytest.mark.integration
def test_online_t5_dedup_preserves_tokens_and_sample_order():
    import os
    from msc.models.conditioning.text import T5TextEncoder
    path = os.environ.get('MSC_TEST_T5_PATH')
    if not path:
        pytest.skip('Set MSC_TEST_T5_PATH for real online T5')
    encoder = T5TextEncoder(path)
    captions = ['walk forward', 'turn left', 'walk forward', 'turn left']
    expected, expected_mask = encoder._encode(captions, 'cpu')
    calls=[]
    hook=encoder.encoder.register_forward_pre_hook(lambda _,args,kwargs:calls.append(kwargs['input_ids'].shape[0]),with_kwargs=True)
    actual, mask = encoder(captions,'cpu')
    hook.remove()
    assert calls == [2]
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(mask,expected_mask)
    assert not encoder.encoder.training
