"""Optional offline W&B runs and scalar metrics from existing JSON records."""
from pathlib import Path


def flatten_metrics(metrics, prefix):
    result = {}
    items = enumerate(metrics) if isinstance(metrics, (list, tuple)) else metrics.items()
    for name, value in items:
        key = f'{prefix}/{name}'
        if isinstance(value, (dict, list, tuple)):
            result.update(flatten_metrics(value, key))
        elif isinstance(value, (int, float)):
            result[key] = value
    return result


def offline_run(config):
    import wandb

    output = Path(config.train.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=config.train.wandb_project,
                     name=config.train.wandb_name or output.name,
                     mode='offline', dir=str(output), config=config.to_dict(),
                     settings=wandb.Settings(disable_git=True, disable_code=True))
    run.define_metric('optimizer_step')
    # Training and validation can log separately at the same optimizer step.
    run.define_metric('*', step_metric='optimizer_step')
    return run
