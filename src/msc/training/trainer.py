import contextlib
import copy
import json
import math
import time
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from msc.data.collator import collate_motion
from msc.factory import to_device, build_dataset
from .checkpoint import save_checkpoint, load_checkpoint, metadata, restore_rng
from bisect import bisect_right


class EpochBatchSampler(Sampler):
    """Visit each window once per epoch; bounded block shuffling supports exact resume."""
    block_size = 65536

    def __init__(self, size, batch_size, start, stop, seed, ranges=None):
        self.ranges = ranges
        self.ends = []
        for a,b in ranges or []:
            self.ends.append((self.ends[-1] if self.ends else 0)+b-a)
        self.size = self.ends[-1] if ranges else size
        self.batch_size, self.start, self.stop, self.seed = batch_size,start,stop,seed
        self.batches_per_epoch = math.ceil(self.size / batch_size)

    def __iter__(self):
        step = self.start
        while step < self.stop:
            epoch, batch_offset = divmod(step,self.batches_per_epoch)
            skip = batch_offset*self.batch_size
            blocks = math.ceil(self.size/self.block_size)
            generator = torch.Generator().manual_seed(self.seed+epoch)
            order = torch.randperm(blocks,generator=generator).tolist()
            batch = []
            for block in order:
                first = block*self.block_size
                length = min(self.block_size,self.size-first)
                if skip >= length:
                    skip -= length
                    continue
                generator = torch.Generator().manual_seed(self.seed+epoch*1000003+block*104729)
                indices = torch.randperm(length,generator=generator).tolist()
                for local in indices[skip:]:
                    index = first+local
                    if self.ranges:
                        span = bisect_right(self.ends,index)
                        index = self.ranges[span][0]+index-(self.ends[span-1] if span else 0)
                    batch.append(index)
                    if len(batch)==self.batch_size:
                        yield batch
                        step += 1
                        if step >= self.stop:
                            return
                        batch = []
                skip = 0
            if batch:
                yield batch
                step += 1

    def __len__(self):
        return self.stop-self.start


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
        cfg = config.train
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
                                          lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        def schedule(step):
            if step < cfg.warmup_steps:
                return (step + 1) / max(1, cfg.warmup_steps)
            progress = min(1.0, (step - cfg.warmup_steps) / (cfg.schedule_steps - cfg.warmup_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
        self.step = 0
        self.last_loss = None
        self.profile = {}
        self.validation_dataset = None
        if config.data.backend == 'msc_release' and cfg.validation_every > 0:
            val_config = copy.deepcopy(config)
            val_config.data.split = 'val'
            val_config.data.max_motions_per_task = cfg.validation_motions_per_task
            self.validation_dataset = build_dataset(val_config)
        if config.data.backend == 'msc_release' and any(model.loss_config.get(k,0) for k in ('joint_velocity','joint_acceleration')):
            if model.dynamic_stats_identity != dataset.identity:
                raise ValueError('Train dynamics statistics do not match selected Release dataset')
        if cfg.resume:
            self.resume(cfg.resume)

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
        cfg = self.config.train
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        (out / "metadata.json").write_text(json.dumps(metadata(self.model, self.config, self.dataset), indent=2))
        sampler = EpochBatchSampler(len(self.dataset), cfg.batch_size,
            self.step * cfg.accumulation_steps, cfg.steps * cfg.accumulation_steps, cfg.seed)
        loader = DataLoader(self.dataset, batch_sampler=sampler, collate_fn=collate_motion,
            num_workers=self.config.data.num_workers, worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(cfg.seed),
            multiprocessing_context='spawn' if self.config.data.num_workers else None)
        self.model.train()
        if self.device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.monotonic()
        starting_step = self.step
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        samples_seen = 0
        task_metrics = {}
        # Per-microbatch probes measure each task's contribution before gradients
        # from different accumulation steps are combined in parameter.grad.
        micro_grad_norms = {}
        probes = [('expert_output',self.model.motion_generator.output.weight),
                  ('history_output',self.model.condition_bridge.history[-1].weight)]
        probes += [(name+'_output',adapter[-1].weight) for name,adapter in self.model.condition_bridge.adapters.items()]
        handles = [p.register_hook(lambda grad,name=name: micro_grad_norms.__setitem__(name,grad.detach().float().norm()))
                   for name,p in probes]
        for microstep, batch in enumerate(loader, 1):
            micro_grad_norms.clear()
            batch = to_device(batch, self.device)
            absolute_microstep = self.step*cfg.accumulation_steps+(microstep-1)%cfg.accumulation_steps
            flow_rng = torch.Generator(device=self.device).manual_seed(cfg.seed+1000003+absolute_microstep)
            noise = torch.randn(batch['motion_future'].shape,device=self.device,generator=flow_rng)
            times = torch.rand(len(noise),device=self.device,generator=flow_rng)
            with self.autocast():
                loss, components = self.model.compute_loss(batch,noise=noise,times=times,return_metrics=True,step=self.step)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            self.scaler.scale(loss / cfg.accumulation_steps).backward()
            accumulated_loss += loss.detach().item() / cfg.accumulation_steps
            samples_seen += len(batch['motion_future'])
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
                continue
            self.scaler.unscale_(self.optimizer)
            group_norms = {name: float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in module.parameters() if p.grad is not None])))
                           for name,module in [('expert',self.model.motion_generator),*self.model.condition_bridge.adapters.items()]
                           if any(p.grad is not None for p in module.parameters())}
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm,
                                                       error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.last_loss = accumulated_loss
            if self.step % cfg.log_every == 0:
                record = dict(step=self.step, loss=accumulated_loss,
                    learning_rate=self.optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm))
                record['tasks'] = {task:{k:v/values['count'] for k,v in values.items() if k!='count'} for task,values in task_metrics.items()}
                record['gradient_norms'] = group_norms
                print(json.dumps(record), flush=True)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                task_metrics = {}
            accumulated_loss = 0.0
            if cfg.validation_every > 0 and self.step % cfg.validation_every == 0:
                validation = self.validate()
                if validation:
                    with (out/'validation.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=self.step,tasks=validation))+'\n')
            if self.step % cfg.checkpoint_every == 0 or self.step == cfg.steps:
                save_checkpoint(out / "last.pt", self)
        for handle in handles:
            handle.remove()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
        elapsed = time.monotonic()-started
        self.profile = dict(elapsed_seconds=elapsed,optimizer_steps=self.step,
            executed_steps=self.step-starting_step,
            samples_per_second=samples_seen/max(elapsed,1e-9),
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            total_parameters=sum(p.numel() for p in self.model.parameters()),
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None)
        (out/'profile.json').write_text(json.dumps(self.profile,indent=2))
        return self.last_loss
