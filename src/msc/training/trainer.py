import contextlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler
from msc.data.collator import collate_motion
from msc.factory import to_device
from .checkpoint import save_checkpoint, load_checkpoint, metadata, restore_rng


class StepBatchSampler(Sampler):
    """Uniform windows with replacement, reproducible without a huge randperm.

    Each microbatch has its own RNG seed; resume skips no model RNG draws and
    does not read earlier data. Official training windows themselves are fixed.
    """
    def __init__(self, size, batch_size, start, stop, seed):
        self.size, self.batch_size, self.start, self.stop, self.seed = size, batch_size, start, stop, seed

    def __iter__(self):
        for step in range(self.start, self.stop):
            gen = torch.Generator().manual_seed(self.seed + step)
            yield torch.randint(self.size, (self.batch_size,), generator=gen).tolist()

    def __len__(self):
        return self.stop - self.start


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
        if cfg.resume:
            self.resume(cfg.resume)

    def autocast(self):
        return (torch.autocast(self.device.type, dtype=self.amp_dtype)
                if self.amp_dtype is not None else contextlib.nullcontext())

    def resume(self, path):
        obj = load_checkpoint(path)
        if obj["metadata"] != metadata(self.model, self.config):
            raise ValueError("Checkpoint representation, modality or data identity differs")
        old = obj["config"]["train"]
        for key in ("seed", "batch_size", "accumulation_steps", "learning_rate", "weight_decay",
                    "warmup_steps", "schedule_steps", "precision", "max_grad_norm"):
            if old[key] != self.config.to_dict()["train"][key]:
                raise ValueError(f"Resume training setting changed: {key}")
        current_model = self.config.to_dict()["model"]
        for key, value in obj["config"]["model"].items():
            if key not in {"t5_path", "text_cache_dir", "stats_path"} and current_model[key] != value:
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
        (out / "metadata.json").write_text(json.dumps(metadata(self.model, self.config), indent=2))
        sampler = StepBatchSampler(len(self.dataset), cfg.batch_size,
            self.step * cfg.accumulation_steps, cfg.steps * cfg.accumulation_steps, cfg.seed)
        loader = DataLoader(self.dataset, batch_sampler=sampler, collate_fn=collate_motion,
            num_workers=self.config.data.num_workers, worker_init_fn=worker_init,
            generator=torch.Generator().manual_seed(cfg.seed))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for microstep, batch in enumerate(loader, 1):
            batch = to_device(batch, self.device)
            with self.autocast():
                loss = self.model.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            self.scaler.scale(loss / cfg.accumulation_steps).backward()
            accumulated_loss += loss.detach().item() / cfg.accumulation_steps
            if microstep % cfg.accumulation_steps:
                continue
            self.scaler.unscale_(self.optimizer)
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
                print(json.dumps(record), flush=True)
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
            accumulated_loss = 0.0
            if self.step % cfg.checkpoint_every == 0 or self.step == cfg.steps:
                save_checkpoint(out / "last.pt", self)
        return self.last_loss
