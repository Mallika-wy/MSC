"""Deterministic per-task losses, motion metrics and condition controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from msc.sample import load_model
from msc.factory import build_dataset, to_device
from msc.data.collator import collate_motion
from msc.models.losses import joint_dynamics
from msc.training.trainer import EpochBatchSampler
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics


def motion_metrics(pred,target,history,mask):
    pv,pa,vm,am = joint_dynamics(pred[...,9:38],history[...,9:38],mask)
    tv,ta,_,_ = joint_dynamics(target[...,9:38],history[...,9:38],mask)
    jerk_mask = am[:,1:] & am[:,:-1]
    jerk = 30*(pa[:,1:]-pa[:,:-1])
    gt_jerk = 30*(ta[:,1:]-ta[:,:-1])
    def rms(x,m):
        return float(x[m].square().mean().sqrt()) if m.any() else None
    def amplitude(x):
        return float(torch.stack([row[valid,9:38].std(0,unbiased=False).mean() for row,valid in zip(x,mask)]).mean())
    def displacement(x):
        return float(torch.stack([(row[valid][-1,:3]-row[valid][0,:3]).norm() for row,valid in zip(x,mask)]).mean())
    metrics = dict(joint_rmse=rms(pred[...,9:38]-target[...,9:38],mask),
        velocity_residual=rms(pv-tv,vm),acceleration_residual=rms(pa-ta,am),
        jerk_rms=rms(jerk,jerk_mask),target_jerk_rms=rms(gt_jerk,jerk_mask),
        root_position_rmse=rms(pred[...,:3]-target[...,:3],mask),
        joint_amplitude=amplitude(pred),target_joint_amplitude=amplitude(target),
        root_displacement=displacement(pred),target_root_displacement=displacement(target),
        chunk_boundary_joint_jump=float((pred[:,0,9:38]-history[:,-1,9:38]).norm(dim=-1).mean()),
        target_chunk_boundary_joint_jump=float((target[:,0,9:38]-history[:,-1,9:38]).norm(dim=-1).mean()))
    return metrics


def sample_batch(model,batch,noise):
    memory = model.encode_conditions(batch)
    return model.representation.denormalize(model.flow.sample(model.motion_generator,memory,initial_noise=noise))


def task_indices(dataset,task,batches,batch_size):
    if getattr(dataset,'task_ranges',None):
        return EpochBatchSampler(len(dataset),batch_size,0,batches,1729,
                                ranges=dataset.task_ranges[task])
    return EpochBatchSampler(len(dataset),batch_size,0,batches,1729)


@torch.no_grad()
def evaluate_dataset(model,dataset,*,batches=4,batch_size=4,device='cpu',output_dir=None,sonic_config=None,rollout_chunks=2):
    model.eval()
    tasks = sorted(getattr(dataset,'task_ranges',{}))
    results = {}
    codec = G1MotionFeatureCodec(G1Kinematics(),num_prev_states=5,canonical_frame_idx=4,rotation_representation='rot6d')
    generator = torch.Generator(device=device).manual_seed(1729)
    for task in tasks:
        records = []
        for bi,indices in enumerate(task_indices(dataset,task,batches,batch_size)):
            batch = to_device(collate_motion([dataset[i] for i in indices]),device)
            noise = torch.randn(batch['motion_future'].shape,generator=generator,device=device)
            times = torch.rand(len(indices),generator=generator,device=device)
            _,losses = model.compute_loss(batch,noise=noise,times=times,step=model.loss_config.get('dynamic_warmup_steps',0),return_metrics=True)
            pred = sample_batch(model,batch,noise)
            shuffled = dict(batch)
            # An independently chosen task example also works when batch_size=1.
            alternatives = list(task_indices(dataset,task,bi+2,len(indices)))[-1]
            control = to_device(collate_motion([dataset[i] for i in alternatives]),device)
            for key in list(shuffled):
                if key == 'text' or key.startswith(('text_','audio_','video_')):
                    shuffled.pop(key)
            shuffled.update({k:v for k,v in control.items() if k == 'text' or k.startswith(('text_','audio_','video_'))})
            shuffled_pred = sample_batch(model,shuffled,noise)
            memory = model.encode_conditions(batch)
            missing_pred = model.representation.denormalize(model.flow.sample(model.motion_generator,memory.null(),initial_noise=noise))
            record = {k:float(v) for k,v in losses.items()}
            record.update(motion_metrics(pred,batch['motion_future'],batch['motion_history'],batch['motion_mask']))
            record['condition_controls'] = dict(
                matched_joint_rmse=float((pred[...,9:38]-batch['motion_future'][...,9:38]).square().mean().sqrt()),
                shuffled_joint_rmse=float((shuffled_pred[...,9:38]-batch['motion_future'][...,9:38]).square().mean().sqrt()),
                missing_joint_rmse=float((missing_pred[...,9:38]-batch['motion_future'][...,9:38]).square().mean().sqrt()),
                matched_vs_shuffled_rms=float((pred-shuffled_pred).square().mean().sqrt()),
                matched_vs_missing_rms=float((pred-missing_pred).square().mean().sqrt()),
                same_control_windows=indices==alternatives,
                semantic_match_status='requires task-specific semantic/human assessment')
            if 'anchor_qpos' in batch:
                anchor = batch['anchor_qpos'].cpu()
                world = codec.decode_to_world_qpos36(codec.split_features(pred.cpu()),anchor[:,None,:3],anchor[:,None,3:7])
                target_world = codec.decode_to_world_qpos36(codec.split_features(batch['motion_future'].cpu()),anchor[:,None,:3],anchor[:,None,3:7])
                pfk = codec.kinematics.forward_kinematics(world)['body_pos_w']
                tfk = codec.kinematics.forward_kinematics(target_world)['body_pos_w']
                record['fk_mpjpe'] = float((pfk-tfk).norm(dim=-1).mean())
                if output_dir:
                    path = Path(output_dir)/task/f'batch-{bi:03d}'
                    path.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(motion=pred.cpu()),path/'sample.pt')
                    np.savez_compressed(path/'initial.npz',qpos_36=anchor.numpy())
                    np.savez_compressed(path/'qpos36.npz',qpos_36=world.numpy(),target_qpos_36=target_world.numpy())
                    record['rollout'] = rollout(model,dataset,indices[0],codec,rollout_chunks,device,path,sonic_config)
            records.append(record)
        results[task] = records
    sonic=[r['rollout']['sonic'] for records in results.values() for r in records
           if 'rollout' in r and 'accepted' in r['rollout'].get('sonic',{})]
    return dict(tasks=results,sonic_summary=dict(evaluated=len(sonic),
        failure_rate=sum(not r['accepted'] for r in sonic)/len(sonic) if sonic else None),
        note='Condition controls measure dependence; they do not establish semantic alignment.')


@torch.no_grad()
def rollout(model,dataset,index,codec,chunks,device,output,sonic_config=None):
    """Use generated history, re-anchor every chunk, and keep the same Motion/task."""
    ri,m,task,k,_ = dataset.window_identity(index)
    # Start at the first eligible window of the sampled motion so short subset
    # selection cannot accidentally truncate the continuity check at its tail.
    k = min(start for _,_,_,start,_,_ in dataset.motion_spans(ri,m.motion_id,task))
    first = dataset.read_window(ri,m.motion_id,task,k)
    history = first['motion_history'][None].to(device)
    anchor = first['anchor_qpos'][None]
    world_chunks,targets = [],[]
    gen = torch.Generator(device=device).manual_seed(713)
    for chunk in range(chunks):
        sample = dataset.read_window(ri,m.motion_id,task,k+chunk*30)
        if sample is None:
            break
        batch = to_device(collate_motion([sample]),device)
        batch['motion_history'] = history
        noise = torch.randn((1,30,125),device=device,generator=gen)
        pred = sample_batch(model,batch,noise).cpu()
        world = codec.decode_to_world_qpos36(codec.split_features(pred),anchor[:,None,:3],anchor[:,None,3:7])
        original_anchor = sample['anchor_qpos'][None]
        target = codec.decode_to_world_qpos36(codec.split_features(sample['motion_future'][None]),original_anchor[:,None,:3],original_anchor[:,None,3:7])
        world_chunks.append(world[0]);targets.append(target[0])
        last = world[:,-5:]
        fk = codec.kinematics.forward_kinematics(last)
        history,_,_ = codec.prev_state_features_from_history(last,fk['body_pos_w'],None,30.)
        history = history.to(device)
        anchor = last[:,-1]
    if not world_chunks:
        return dict(status='Pending: no contiguous next window')
    world,target = torch.cat(world_chunks),torch.cat(targets)
    np.savez_compressed(output/'rollout.npz',qpos_36=world.numpy(),target_qpos_36=target.numpy(),fps=30.)
    result = dict(start_motion_frame=k,chunks=len(world_chunks),root_drift=float((world[-1,:3]-target[-1,:3]).norm()),
                  joint_amplitude=float(world[:,7:].std(0,unbiased=False).mean()),
                  target_joint_amplitude=float(target[:,7:].std(0,unbiased=False).mean()),
                  boundary_joint_jumps=[float((world[i]-world[i-1])[7:].norm()) for i in range(30,len(world),30)])
    if sonic_config:
        from data_pipeline.config import Settings
        from data_pipeline.motion.tracking import track
        settings = Settings.read(sonic_config)
        directory = output/'sonic'
        directory.mkdir(exist_ok=True)
        reference = np.concatenate((first['anchor_qpos'][None].numpy(),world.numpy()))
        executed,quality = track(reference,settings.tracker,settings.quality,directory,diagnostic=False)
        np.savez_compressed(directory/'executed.npz',qpos_36=executed,fps=30.)
        result['sonic'] = quality
    else:
        result['sonic'] = dict(status='Pending: run evaluation with --sonic-config')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',action='append')
    p.add_argument('--split',default='val')
    p.add_argument('--batches',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--t5-path')
    p.add_argument('--output',default='.runs/msc-v3-evaluation')
    p.add_argument('--sonic-config')
    p.add_argument('--rollout-chunks',type=int,default=2)
    args = p.parse_args()
    if args.batches<1 or args.rollout_chunks<1:
        p.error('batches and rollout chunks must be positive')
    torch.set_num_threads(2)
    model,cfg,metadata = load_model(args.checkpoint,args.device,args.t5_path)
    cfg.data.split = args.split
    if args.split != 'train' and cfg.train.validation_motions_per_task:
        cfg.data.max_motions_per_task=cfg.train.validation_motions_per_task
    if args.data_root:
        cfg.data.roots=args.data_root
    dataset = build_dataset(cfg)
    if json.dumps(dataset.audio_metadata,sort_keys=True)!=json.dumps(model.audio_metadata,sort_keys=True):
        raise ValueError('Evaluation audio provenance differs from checkpoint')
    output = Path(args.output)
    if any(output.resolve().is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        p.error('Evaluation output must be outside read-only Release roots')
    output.mkdir(parents=True,exist_ok=True)
    result = evaluate_dataset(model,dataset,batches=args.batches,batch_size=cfg.train.batch_size,device=args.device,
                              output_dir=output,sonic_config=args.sonic_config,rollout_chunks=args.rollout_chunks)
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
