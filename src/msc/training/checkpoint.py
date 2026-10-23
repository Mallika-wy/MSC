"""Atomic trusted-local checkpoints, including RNG and deterministic data position."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from msc.models.representation import MotionRepresentation


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def metadata(model, config, dataset=None):
    return dict(representation=model.representation.metadata(), audio=model.audio_metadata,
                text_model_sha256=model.text_encoder.fingerprint if model.text_encoder else None,
                data=dict(backend=config.data.backend, split=config.data.split),
                architecture=model.architecture_version,features=model.feature_metadata,
                dataset_identity=getattr(dataset,'identity',None),
                dynamic_stats_sha256=model.dynamic_stats_sha256,
                sampling=dict(version=3,mode='epoch_shuffle',tasks=config.data.tasks,seed=config.train.seed,block_size=65536),
                noise=dict(seed_offset=1000003,mode='per_microstep',rng='torch'),
                losses=dict(version=1,config=model.loss_config),
                flow=dict(type="linear_velocity", time_scale=1000.0, time_sampling="uniform", direction="1_to_0"))


def save_checkpoint(path, trainer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = dict(version=3, model=trainer.model.state_dict(), optimizer=trainer.optimizer.state_dict(),
               scheduler=trainer.scheduler.state_dict(), scaler=trainer.scaler.state_dict(),
               step=trainer.step, config=trainer.config.to_dict(), rng=rng_state(),
               metadata=metadata(trainer.model, trainer.config, trainer.dataset))
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, temp)
    temp.replace(path)


def load_checkpoint(path):
    # Contains Python and NumPy RNG state; load only checkpoints you trust.
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != 3:
        raise ValueError("Unsupported MSC checkpoint version; old checkpoints are not supported")
    name = MotionRepresentation.name
    if (obj.get('metadata',{}).get('representation',{}).get('type') != name or
        obj.get('config',{}).get('model',{}).get('representation') != name):
        raise ValueError('Checkpoint representation must be g1_canonical_125d')
    return obj
