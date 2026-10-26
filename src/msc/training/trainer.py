import contextlib
import copy
import json
import math
import time
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from .distributed import RankBatchSampler, TrainingLoss
from .tracking import flatten_metrics, offline_run
from torch.utils.data import DataLoader
from msc.data.collator import collate_motion
from msc.factory import to_device, build_dataset
from .checkpoint import save_checkpoint, load_checkpoint, metadata, restore_rng
from .sampling import EpochBatchSampler, InterleavedEpisodeBatchSampler


def worker_init(worker_id):
    import random
    import numpy as np
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class Trainer:
    def __init__(self, model, dataset, config):
        self.config, self.dataset = config, dataset
        stats_identity = model.representation.stats_dataset_identity
        if stats_identity is not None and stats_identity != dataset.identity:
            raise ValueError('Normalization statistics do not match selected Release dataset')
        cfg = config.train
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(cfg.deterministic)
        torch.backends.cudnn.benchmark = False
        if any(Path(cfg.output_dir).resolve().is_relative_to(Path(root).resolve()) for root in config.data.roots):
            raise ValueError('Training output must be outside read-only Release roots')
        torch.set_num_threads(cfg.num_threads)
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        if cfg.precision != "fp32" and self.device.type != "cuda":
            raise ValueError("AMP requires CUDA; use precision=fp32 for CPU")
        if cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is not supported on this device")
        self.amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[cfg.precision]
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=cfg.learning_rate, weight_decay=cfg.weight_decay, fused=self.device.type=='cuda')
        def schedule(step):
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            progress = min(1.0, (step - cfg.warmup_steps) / (cfg.schedule_steps - cfg.warmup_steps))
            floor = cfg.min_learning_rate / cfg.learning_rate
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
        self.step = 0
        self.last_loss = None
        self.profile = {}
        self.validation_dataset = None
        if self.rank == 0 and config.data.backend == 'msc_release' and cfg.validation_every > 0:
            val_config = copy.deepcopy(config)
            val_config.data.split = 'val'
            val_config.data.max_motions_per_task = cfg.validation_motions_per_task
            self.validation_dataset = build_dataset(val_config)
        if config.data.backend == 'msc_release' and any(model.loss_config.get(k,0) for k in ('joint_velocity','joint_acceleration')):
            if model.dynamic_stats_identity != dataset.identity:
                raise ValueError('Train dynamics statistics do not match selected Release dataset')
        if cfg.resume:
            self.resume(cfg.resume)
        if cfg.compile_expert:
            # Compile in place: parameter names and the saved model stay unchanged.
            self.model.motion_generator.compile(dynamic=True)
        self.training_model = TrainingLoss(self.model)
        if self.world_size > 1:
            # Different tasks can leave different adapters unused on each rank.
            self.training_model = DistributedDataParallel(
                self.training_model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=True)
            if cfg.bf16_gradient_reduce:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
                self.training_model.register_comm_hook(None, bf16_compress_hook)
        torch.manual_seed(cfg.seed + self.rank)

    @torch.no_grad()
    def validate(self):
        if self.validation_dataset is None:
            return {}
        results = {}
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1729)
            self.model.eval()
            for task in sorted(self.validation_dataset.task_ranges):
                sampler = EpochBatchSampler(len(self.validation_dataset),self.config.train.batch_size,0,
                    self.config.train.validation_batches,1729,
                    ranges=self.validation_dataset.task_ranges[task])
                loader = DataLoader(self.validation_dataset,batch_sampler=sampler,collate_fn=collate_motion,num_workers=0)
                totals = {}
                for batch in loader:
                    with self.autocast():
                        _,metrics = self.model.compute_loss(to_device(batch,self.device),return_metrics=True,step=self.step)
                    for key,value in metrics.items():
                        totals[key] = totals.get(key,0.)+float(value)/len(loader)
                results[task] = totals
            self.model.train()
        return results

    def autocast(self):
        return (torch.autocast(self.device.type, dtype=self.amp_dtype)
                if self.amp_dtype is not None else contextlib.nullcontext())

    def resume(self, path):
        obj = load_checkpoint(path)
        if obj["metadata"] != metadata(self.model, self.config, self.dataset):
            raise ValueError("Checkpoint representation, modality or data identity differs")
        old = obj["config"]["train"]
        for key in ("seed", "deterministic", "batch_size", "accumulation_steps", "learning_rate", "weight_decay",
                    "warmup_steps", "schedule_steps", "precision", "max_grad_norm"):
            if old[key] != self.config.to_dict()["train"][key]:
                raise ValueError(f"Resume training setting changed: {key}")
        current_model = self.config.to_dict()["model"]
        for key, value in obj["config"]["model"].items():
            if key not in {"t5_path", "stats_path", "dynamic_stats_path"} and current_model[key] != value:
                raise ValueError(f"Resume model setting changed: {key}")
        self.model.load_state_dict(obj["model"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.scheduler.load_state_dict(obj["scheduler"])
        self.scaler.load_state_dict(obj["scaler"])
        self.step = obj["step"]
        restore_rng(obj["rng"])

    def fit(self):
        tracking = contextlib.nullcontext()
        if self.rank == 0 and self.config.train.wandb_project:
            tracking = offline_run(self.config)
        with tracking as run:
            return self._fit(run)

    def _fit(self, run=None):
        cfg = self.config.train
        out = Path(cfg.output_dir)
        if self.rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
            (out / "metadata.json").write_text(json.dumps(metadata(self.model, self.config, self.dataset), indent=2))
        if cfg.sampling == 'episode_interleaved':
            sampler = InterleavedEpisodeBatchSampler(self.dataset.episode_ranges, cfg.batch_size * self.world_size,
                self.step * cfg.accumulation_steps, cfg.steps * cfg.accumulation_steps, cfg.seed,
                cfg.active_episodes, cfg.windows_per_episode)
        else:
            sampler = EpochBatchSampler(len(self.dataset), cfg.batch_size * self.world_size,
                self.step * cfg.accumulation_steps, cfg.steps * cfg.accumulation_steps, cfg.seed)
        if self.world_size > 1:
            sampler = RankBatchSampler(sampler, self.rank, self.world_size, cfg.batch_size)
        loader = DataLoader(self.dataset, batch_sampler=sampler, collate_fn=collate_motion,
            pin_memory=self.device.type == 'cuda',
            num_workers=self.config.data.num_workers, worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(cfg.seed+self.rank),
            multiprocessing_context='spawn' if self.config.data.num_workers else None)
        self.model.train()
        if self.device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        starting_step = self.step
        last_log_time, last_log_step, last_log_samples = started, starting_step, 0
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        interval_loss, interval_steps = 0.0, 0
        samples_seen = 0
        task_metrics = {}
        task_samples = {}
        # Per-microbatch probes measure each task's contribution before gradients
        # from different accumulation steps are combined in parameter.grad.
        micro_grad_norms = {}
        probes = [('expert_output',self.model.motion_generator.output.weight),
                  ('history_output',self.model.condition_bridge.history[-1].weight)]
        probes += [(name+'_output',adapter[-1].weight) for name,adapter in self.model.condition_bridge.adapters.items()]
        handles = [p.register_hook(lambda grad,name=name: micro_grad_norms.__setitem__(name,grad.detach().float().norm()))
                   for name,p in probes]
        data_wait_seconds = 0.
        last_batch_end = started
        for microstep, batch in enumerate(loader, 1):
            data_wait_seconds += time.monotonic() - last_batch_end
            micro_grad_norms.clear()
            batch = to_device(batch, self.device)
            absolute_microstep = self.step*cfg.accumulation_steps+(microstep-1)%cfg.accumulation_steps
            flow_rng = torch.Generator(device=self.device).manual_seed(cfg.seed+1000003+absolute_microstep*self.world_size+self.rank)
            noise = torch.randn(batch['motion_future'].shape,device=self.device,generator=flow_rng)
            times = torch.rand(len(noise),device=self.device,generator=flow_rng)
            synchronize = microstep % cfg.accumulation_steps == 0
            sync_context = (self.training_model.no_sync() if self.world_size > 1 and not synchronize
                            else contextlib.nullcontext())
            with sync_context:
                with self.autocast():
                    loss, components = self.training_model(batch, noise, times, self.step)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                self.scaler.scale(loss / cfg.accumulation_steps).backward()
            accumulated_loss += loss.detach().item() / cfg.accumulation_steps
            samples_seen += len(batch['motion_future'])
            for name in batch.get('task', []):
                task_samples[name] = task_samples.get(name, 0) + 1
            tasks = set(batch.get('task',[]))
            task = next(iter(tasks)) if len(tasks)==1 else 'mixed'
            current = task_metrics.setdefault(task,dict(count=0))
            current['count'] += 1
            for name,value in micro_grad_norms.items():
                key = 'gradient_probe_'+name
                current[key] = current.get(key,0.)+float(value)*cfg.accumulation_steps/self.scaler.get_scale()
            for key,value in components.items():
                current[key] = current.get(key,0.)+float(value)
            for modality in ('text','audio','video'):
                if modality+'_mask' in batch:
                    key = modality+'_valid_fraction'
                    current[key] = current.get(key,0.)+float(batch[modality+'_mask'].float().mean())
            if microstep % cfg.accumulation_steps:
                last_batch_end = time.monotonic()
                continue
            self.scaler.unscale_(self.optimizer)
            group_norms = {name: float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in module.parameters() if p.grad is not None])))
                           for name,module in [('expert',self.model.motion_generator),*self.model.condition_bridge.adapters.items()]
                           if any(p.grad is not None for p in module.parameters())} if (self.step+1) % cfg.log_every == 0 else {}
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm,
                                                       error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.last_loss = accumulated_loss
            interval_loss += accumulated_loss
            interval_steps += 1
            if self.step % cfg.log_every == 0:
                payload = dict(loss_sum=interval_loss, loss_steps=interval_steps, step_loss=accumulated_loss,
                               tasks=task_metrics, samples=task_samples,
                               samples_seen=samples_seen, data_wait_seconds=data_wait_seconds,
                               peak_memory=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0,
                               reserved_memory=torch.cuda.memory_reserved(self.device) if self.device.type=='cuda' else 0)
                summaries = [payload]
                if self.world_size > 1:
                    summaries = [None] * self.world_size
                    dist.all_gather_object(summaries, payload)
                if self.rank == 0:
                    combined_tasks, combined_samples = {}, {}
                    for summary in summaries:
                        for name, count in summary['samples'].items():
                            combined_samples[name] = combined_samples.get(name, 0) + count
                        for task, values in summary['tasks'].items():
                            target = combined_tasks.setdefault(task, {})
                            for key, value in values.items():
                                target[key] = target.get(key, 0.) + value
                    now = time.monotonic()
                    global_samples = sum(s['samples_seen'] for s in summaries)
                    interval = now - last_log_time
                    seconds_per_step = interval / (self.step - last_log_step)
                    record = dict(step=self.step,
                        loss=sum(s['loss_sum'] for s in summaries)/sum(s['loss_steps'] for s in summaries),
                        step_loss=sum(s['step_loss'] for s in summaries)/self.world_size,
                        learning_rate=self.optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm),
                        gradient_norms=group_norms, task_samples=combined_samples,
                        elapsed_seconds=now-started,
                        data_wait_seconds=[s['data_wait_seconds'] for s in summaries],
                        samples_per_second=global_samples/(now-started),
                        seconds_per_step=seconds_per_step,
                        interval_samples_per_second=(global_samples-last_log_samples)/interval,
                        eta_hours=(cfg.steps-self.step)*seconds_per_step/3600,
                        rank_peak_memory_gib=[s['peak_memory']/2**30 for s in summaries],
                        rank_reserved_memory_gib=[s['reserved_memory']/2**30 for s in summaries],
                        tasks={task:{k:v/values['count'] for k,v in values.items() if k!='count'}
                               for task,values in combined_tasks.items()})
                    print(json.dumps(record), flush=True)
                    with (out/'metrics.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n')
                    if run is not None:
                        run.log({'optimizer_step': self.step, **flatten_metrics(record, 'train')})
                    last_log_time, last_log_step, last_log_samples = now, self.step, global_samples
                task_metrics = {}
                interval_loss, interval_steps = 0.0, 0
            accumulated_loss = 0.0
            if self.rank == 0 and cfg.validation_every > 0 and self.step % cfg.validation_every == 0:
                validation = self.validate()
                if validation:
                    with (out/'validation.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=self.step,tasks=validation))+'\n')
                    if run is not None:
                        run.log({'optimizer_step': self.step, **flatten_metrics(validation, 'val')})
            if self.rank == 0 and (self.step % cfg.checkpoint_every == 0 or self.step == cfg.steps):
                save_checkpoint(out / "checkpoints" / f"step_{self.step}.pt", self)
            if self.world_size > 1 and (self.step % cfg.checkpoint_every == 0 or
                    (cfg.validation_every > 0 and self.step % cfg.validation_every == 0) or self.step == cfg.steps):
                dist.barrier()
            last_batch_end = time.monotonic()
        for handle in handles:
            handle.remove()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic()-started
        self.profile = dict(elapsed_seconds=elapsed,optimizer_steps=self.step,
            executed_steps=self.step-starting_step,
            samples_per_second=samples_seen/max(elapsed,1e-9),
            task_samples=task_samples,
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in self.model.parameters()),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None)
        profiles = [self.profile]
        if self.world_size > 1:
            profiles = [None] * self.world_size
            dist.all_gather_object(profiles, self.profile)
        if self.rank == 0:
            self.profile['world_size'] = self.world_size
            self.profile['global_samples_per_second'] = sum(p['samples_per_second'] for p in profiles)
            self.profile['rank_peak_gpu_memory_bytes'] = [p['peak_gpu_memory_bytes'] for p in profiles]
            (out/'profile.json').write_text(json.dumps(self.profile, indent=2))
            if run is not None:
                run.summary.update(flatten_metrics(self.profile, 'profile'))
        return self.last_loss
