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
from data_pipeline.io import sha256_file
from msc.data.release_dataset import ReleaseDataset, audit_splits, fingerprint
from msc.data.collator import collate_motion
from msc.data.temporal import select_tokens
from msc.models.conditioning.fusion import ConditionMemory
from msc.models.losses import dynamics_loss, joint_dynamics
from msc.models.msc_model import MSCModel
from msc.training.trainer import StepBatchSampler, Trainer
from msc.factory import build_model, build_dataset, seed_everything
from msc.config import load_config


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


def cached_text(root,cache):
    cache.mkdir()
    identity = dict(output_dim=32,checkpoint_hash='synthetic',preprocess='fixture')
    manifest = dict(version=2,encoder=identity,entries={},checksums={})
    for a in Inventory.read(root).annotations:
        name = fingerprint(a.text)+'.pt'
        torch.save(dict(encoder=identity,text=a.text,features=torch.arange(96).reshape(3,32).float()/96,mask=torch.ones(3,dtype=torch.bool)),cache/name)
        manifest['entries'][fingerprint(a.text)] = name
        manifest['checksums'][name] = sha256_file(cache/name)
    (cache/'manifest.json').write_text(json.dumps(manifest))
    return cache


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
    data = ReleaseDataset([release],task_weights={'audio':1},cache_bytes=0)
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


def test_tasks_masks_and_real_feature_shapes(release,tmp_path):
    cache = cached_text(release,tmp_path/'text-cache')
    data = ReleaseDataset([release],task_weights={'text':1,'audio':1,'video':1},text_cache_dir=cache)
    samples = [data[data.task_ranges[t][0][0]] for t in ('text','audio','video')]
    batch = collate_motion(samples)
    assert batch['audio_tokens'].shape[-1]==768
    assert batch['video_tokens'].shape[-2:]==(2,6)
    assert not batch['audio_mask'][0].any()
    assert batch['audio_mask'][1].any()
    model = MSCModel(hidden_dim=32,num_layers=1,num_heads=4,text_dim=32,video_dim=6,spatial_tokens=2,
                     text_dropout=0,audio_dropout=0,video_dropout=0)
    memory = model.encode_conditions(batch)
    assert isinstance(memory,ConditionMemory)
    assert memory.mask[:,:5].all()
    null = memory.null()
    assert null.mask[:,:5].all() and not null.mask[:,5:].any()
    video = memory.modality_ids==3
    actual = memory.temporal_support[video].reshape(3,-1,2,2)
    torch.testing.assert_close(actual[:,:,0],actual[:,:,1])
    loss = model.compute_loss(batch)
    loss.backward()
    for adapter in model.condition_bridge.adapters.values():
        assert adapter[1].weight.grad.abs().sum()>0


def test_missing_video_rejected(release):
    inv = Inventory.read(release)
    inv.features = [f for f in inv.features if 'video' not in f.encoder_name]
    # Mutate only in-memory test inventory, never real Release data.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Inventory,'read',lambda *a,**k: inv)
        with pytest.raises(ValueError,match='No valid video'):
            ReleaseDataset([release],task_weights={'video':1})


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
    altered['audio_features'] = batch['audio_features'].masked_fill(~batch['audio_mask'][...,None],float('nan'))
    altered['audio_times'] = batch['audio_times'].masked_fill(~batch['audio_mask'],float('nan'))
    other = model.encode_conditions(altered)
    torch.testing.assert_close(reference.tokens,other.tokens)
    altered['audio_features'] = batch['audio_features'].clone()
    altered['audio_features'][batch['audio_mask']]=float('inf')
    with pytest.raises(ValueError,match='NaN/Inf'):
        model.encode_conditions(altered)


def test_precomputed_encoder_features_stay_frozen(model,batch):
    batch['text_features']=batch['text_features'].clone().requires_grad_()
    batch['audio_features']=batch['audio_features'].clone().requires_grad_()
    model.compute_loss(batch).backward()
    assert batch['text_features'].grad is None
    assert batch['audio_features'].grad is None
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


def test_task_sampler_resume_and_weighting():
    ranges = {'text':[(0,3),(8,10)],'audio':[(3,8)]}
    args = dict(size=10,batch_size=8,stop=1000,seed=17,task_ranges=ranges,task_weights={'text':.7,'audio':.3})
    whole = list(StepBatchSampler(start=0,**args))
    resumed = list(StepBatchSampler(start=43,**args))
    assert whole[43:]==resumed
    text = 0
    for batch in whole:
        tasks = {'audio' if 3<=i<8 else 'text' for i in batch}
        assert len(tasks)==1
        text += 'text' in tasks
    assert .65<text/len(whole)<.75


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
    cfg = load_config('configs/trimodal_release.yaml',['model.condition.text.enabled=false','data.task_weights.text=0.0','model.losses.joint_velocity=0.0'])
    assert cfg.model.condition['text']['enabled'] is False
    assert cfg.model.losses['joint_velocity']==0


def test_real_release_resume_with_workers(release,cfg,tmp_path):
    cfg.data.backend='msc_release'
    cfg.data.roots=[str(release)]
    cfg.data.task_weights={'audio':1.}
    cfg.data.num_workers=2
    cfg.model.use_text=False
    cfg.model.losses={'joint_velocity':0.,'joint_acceleration':0.}
    cfg.train.validation_every=0
    cfg.train.steps=2
    cfg.train.output_dir=str(tmp_path/'full')
    data=build_dataset(cfg)
    seed_everything(cfg.train.seed)
    full=Trainer(build_model(cfg,data.audio_metadata,data.feature_metadata),data,cfg)
    full.fit()
    part=copy.deepcopy(cfg)
    part.train.steps=1
    part.train.output_dir=str(tmp_path/'part')
    seed_everything(cfg.train.seed)
    Trainer(build_model(part,data.audio_metadata,data.feature_metadata),data,part).fit()
    part.train.steps=2
    part.train.resume=str(tmp_path/'part/last.pt')
    resumed=Trainer(build_model(part,data.audio_metadata,data.feature_metadata),data,part)
    resumed.fit()
    assert resumed.last_loss==full.last_loss
    for key,value in full.model.state_dict().items():
        torch.testing.assert_close(value,resumed.model.state_dict()[key],atol=0,rtol=0)


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
    data=ReleaseDataset([release],task_weights={'audio':1},max_motions_per_task=1,max_windows_per_motion=3)
    assert len(data)==3
    assert len({data[i]['window_id'] for i in range(3)})==3
    assert len({data.window_identity(i)[1].motion_id for i in range(3)})==1


def test_statistics_identity_and_zero_scales(release,cfg,tmp_path):
    from msc.data.stats import compute_stats
    data=ReleaseDataset([release],task_weights={'audio':1},max_motions_per_task=1,max_windows_per_motion=2)
    stats=compute_stats(data)
    assert stats['split']=='train' and stats['dataset_identity']==data.identity
    assert min(stats['velocity_scale'])>=1e-4
    path=tmp_path/'stats.json'
    path.write_text(json.dumps(stats))
    cfg.data.backend='msc_release';cfg.data.roots=[str(release)];cfg.data.task_weights={'audio':1.}
    cfg.model.use_text=False;cfg.model.dynamic_stats_path=str(path)
    cfg.train.validation_every=0
    model=build_model(cfg,data.audio_metadata,data.feature_metadata)
    assert model.dynamic_stats_identity==data.identity
    model.dynamic_stats_identity='wrong'
    with pytest.raises(ValueError,match='statistics'):
        Trainer(model,data,cfg)


def test_checkpoint_rejects_legacy_and_wrong_data(cfg,tmp_path):
    from msc.training.checkpoint import load_checkpoint
    path=tmp_path/'legacy.pt'
    torch.save(dict(version=1),path)
    with pytest.raises(ValueError,match='Unsupported'):
        load_checkpoint(path)
    cfg.train.steps=1;cfg.train.output_dir=str(tmp_path/'run')
    data=build_dataset(cfg)
    Trainer(build_model(cfg,data.audio_metadata,data.feature_metadata),data,cfg).fit()
    changed=copy.deepcopy(cfg)
    changed.train.resume=str(tmp_path/'run/last.pt')
    changed.data.size+=1
    other=build_dataset(changed)
    with pytest.raises(ValueError,match='identity'):
        Trainer(build_model(changed,other.audio_metadata,other.feature_metadata),other,changed)


def test_evaluation_measures_amplitude_and_rollout(release,cfg,tmp_path):
    from msc.evaluate import evaluate_dataset
    data=ReleaseDataset([release],task_weights={'audio':1},max_motions_per_task=1)
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
def test_bf16_dynamics_finite(cfg,batch):
    cfg.model.losses=dict(joint_velocity=.05,joint_acceleration=.01,dynamic_warmup_steps=0)
    data=build_dataset(cfg)
    model=build_model(cfg,data.audio_metadata,data.feature_metadata).to('cuda:1')
    batch={k:v.to('cuda:1') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
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
    cfg.data.task_weights={'audio':1.};cfg.data.max_motions_per_task=1;cfg.data.max_windows_per_motion=2
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
        data=ReleaseDataset([release],task_weights={'audio':1.},max_motions_per_task=1)
        sample=data[0]
        ri,m,task,k,condition=data.window_identity(0)
        link=next(l for l in inv.links if l.link_id==condition[1])
        feature=next(f for f in inv.features if f.feature_id==condition[2])
        from data_pipeline.release.features import read_feature
        _,times=read_feature(release,feature,inv.external_roots)
        _,expected=select_tokens(times,link,k/30,(k+30)/30,.25)
        torch.testing.assert_close(sample['audio_times'],torch.tensor(expected,dtype=torch.float32),atol=0,rtol=0)


def test_checkpoint_rejects_previous_names_and_versions(cfg,tmp_path):
    from msc.training.checkpoint import load_checkpoint
    cfg.train.steps=1;cfg.train.output_dir=str(tmp_path/'native')
    data=build_dataset(cfg)
    Trainer(build_model(cfg,data.audio_metadata,data.feature_metadata),data,cfg).fit()
    path=tmp_path/'native/last.pt'
    saved=load_checkpoint(path)
    assert saved['version']==3
    previous=tmp_path/'previous.pt'
    older=copy.deepcopy(saved)
    older['version']=2
    torch.save(older,previous)
    with pytest.raises(ValueError,match='Unsupported'):
        load_checkpoint(previous)
    saved['metadata']['representation']['type']='previous_canonical_name'
    torch.save(saved,previous)
    with pytest.raises(ValueError,match='g1_canonical_125d'):
        load_checkpoint(previous)
