"""GPU-gated 200-step overfit and three 1000-step controlled experiments."""
import argparse
import copy
import json
import hashlib
from pathlib import Path
import torch
from msc.config import load_config
from msc.factory import build_dataset,build_model,seed_everything,to_device
from msc.data.stats import compute_stats
from msc.data.collator import collate_motion
from msc.training.trainer import Trainer
from msc.training.trainer import StepBatchSampler
from msc.evaluate import evaluate_dataset


def prepare(cfg,output):
    cfg.model.dynamic_stats_path=None
    dataset = build_dataset(cfg)
    if any(cfg.model.losses.get(k,0) for k in ('joint_velocity','joint_acceleration')):
        path = output/'dynamics-stats.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(compute_stats(dataset),indent=2))
        cfg.model.dynamic_stats_path=str(path)
    return dataset


def fixed_loss(model,dataset,device):
    batch=to_device(collate_motion([dataset[i] for i in range(min(4,len(dataset)))]),device)
    gen=torch.Generator(device=device).manual_seed(1729)
    noise=torch.randn(batch['motion_future'].shape,device=device,generator=gen)
    times=torch.rand(len(noise),device=device,generator=gen)
    model.eval()
    with torch.no_grad():
        # Overfit diagnostic remains FM-only across the auxiliary warmup boundary.
        loss=model.compute_loss(batch,noise=noise,times=times,step=-1)
    return float(loss)


def initial_identity(model,dataset,cfg):
    parameters=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        parameters.update(name.encode())
        parameters.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    sampling=hashlib.sha256()
    sampler=StepBatchSampler(len(dataset),cfg.train.batch_size,0,
        cfg.train.steps*cfg.train.accumulation_steps,cfg.train.seed,dataset.task_ranges,cfg.data.task_weights)
    for batch in sampler:
        sampling.update(json.dumps(batch).encode())
    return dict(initial_state_sha256=parameters.hexdigest(),sampler_sequence_sha256=sampling.hexdigest(),
                noise_seed_offset=1000003,evaluation_seed=1729)


def run(cfg,output,sonic_config=None):
    output=Path(output).resolve()
    if any(output.is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        raise ValueError('Experiment outputs must be outside Release roots')
    output.mkdir(parents=True,exist_ok=True)
    report=dict(overfit={},comparisons={},protocol=dict(overfit_steps=200,comparison_steps=1000,
        dynamic_warmup_steps=100,train_motions_per_task=8,val_motions_per_task=4,windows_per_motion=16))
    def save():
        (output/'report.json').write_text(json.dumps(report,indent=2))
    if not torch.cuda.is_available() or not cfg.train.device.startswith('cuda'):
        report['status']='Pending: CUDA required for full short experiments'
        save(); return report
    cfg=copy.deepcopy(cfg)
    cfg.data.max_motions_per_task=8
    cfg.data.max_windows_per_motion=16
    cfg.train.validation_motions_per_task=4
    cfg.train.batch_size=8
    cfg.train.accumulation_steps=1
    cfg.train.resume=None
    cfg.train.log_every=10
    cfg.model.losses['dynamic_warmup_steps']=100
    for task,weight in cfg.data.task_weights.items():
        if weight<=0:
            continue
        task_cfg=copy.deepcopy(cfg)
        task_cfg.data.task_weights={task:1.}
        task_cfg.train.steps=200
        task_cfg.train.schedule_steps=1000
        task_cfg.train.warmup_steps=0
        task_cfg.train.validation_every=0
        task_cfg.train.checkpoint_every=200
        task_cfg.train.output_dir=str(output/'overfit'/task)
        directory=Path(task_cfg.train.output_dir)
        for modality in ('text','audio','video'):
            task_cfg.model.condition.setdefault(modality,{})['enabled']=modality==task
        try:
            data=prepare(task_cfg,directory)
        except (FileNotFoundError,ValueError) as error:
            report['overfit'][task]=dict(status='Pending',reason=str(error))
            save();continue
        seed_everything(task_cfg.train.seed)
        model=build_model(task_cfg,data.audio_metadata,data.feature_metadata)
        trainer=Trainer(model,data,task_cfg)
        identity=initial_identity(model,data,task_cfg)
        initial=fixed_loss(model,data,task_cfg.train.device)
        # Evaluation must not alter training dropout RNG.
        seed_everything(task_cfg.train.seed+7919)
        trainer.fit()
        final=fixed_loss(model,data,task_cfg.train.device)
        controls=evaluate_dataset(model,data,batches=1,batch_size=4,device=task_cfg.train.device,
            output_dir=directory/'evaluation',sonic_config=sonic_config)
        report['overfit'][task]=dict(status='Complete',initial_fixed_flow=initial,final_fixed_flow=final,
            ratio=final/initial,improved=final<initial,profile=trainer.profile,evaluation=controls,identity=identity)
        save()
        del trainer,model
        torch.cuda.empty_cache()
    # All three arms use the same selected dataset, validation windows, seeds,
    # architecture, schedule, initial weights and per-microstep noise/tau RNG.
    validation_cfg=copy.deepcopy(cfg)
    validation_cfg.data.split='val'
    validation_cfg.data.max_motions_per_task=4
    try:
        data=prepare(cfg,output)
        validation=build_dataset(validation_cfg)
    except (FileNotFoundError,ValueError) as error:
        report['comparisons']=dict(status='Pending',reason=str(error))
        save();return report
    for name,vel,acc in [('flow_only',0.,0.),('flow_velocity',.05,0.),('flow_velocity_acceleration',.05,.01)]:
        arm=copy.deepcopy(cfg)
        arm.model.losses.update(joint_velocity=vel,joint_acceleration=acc)
        arm.train.steps=1000
        arm.train.schedule_steps=1000
        arm.train.warmup_steps=0
        arm.train.validation_every=100
        arm.train.checkpoint_every=1000
        arm.train.output_dir=str(output/name)
        seed_everything(arm.train.seed)
        model=build_model(arm,data.audio_metadata,data.feature_metadata)
        trainer=Trainer(model,data,arm)
        identity=initial_identity(model,data,arm)
        seed_everything(arm.train.seed+7919)
        trainer.fit()
        evaluation=evaluate_dataset(model,validation,batches=4,batch_size=4,device=arm.train.device,
            output_dir=Path(arm.train.output_dir)/'evaluation',sonic_config=sonic_config)
        report['comparisons'][name]=dict(status='Complete',profile=trainer.profile,evaluation=evaluation,
            dataset_identity=data.identity,validation_identity=validation.identity,identity=identity)
        save()
        del trainer,model
        torch.cuda.empty_cache()
    report['status']='Complete'
    save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',default='.runs/trimodal-short-experiments')
    parser.add_argument('--sonic-config')
    parser.add_argument('overrides',nargs='*')
    args=parser.parse_args()
    report=run(load_config(args.config,args.overrides),args.output,args.sonic_config)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
