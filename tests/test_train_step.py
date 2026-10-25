from msc.factory import build_model
from msc.training.trainer import Trainer
import pytest
import torch
import json


def test_gradient_accumulation(cfg, toy_dataset, tmp_path):
    cfg.train.steps = 2
    cfg.train.accumulation_steps = 2
    cfg.train.output_dir = str(tmp_path)
    data = toy_dataset
    trainer = Trainer(build_model(cfg, data.audio_metadata), data, cfg)
    trainer.fit()
    assert trainer.step == 2
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert record['seconds_per_step'] > 0
        assert record['interval_samples_per_second'] * record['seconds_per_step'] == pytest.approx(8)
        assert record['eta_hours'] == pytest.approx((2-record['step'])*record['seconds_per_step']/3600)


def test_rank_sampler_shards_global_batches():
    from msc.training.trainer import EpochBatchSampler
    from msc.training.distributed import RankBatchSampler
    def rank_batches(rank):
        base = EpochBatchSampler(19, 8, 0, 3, 42)
        return list(RankBatchSampler(base, rank, 2, 4))
    left, right = rank_batches(0), rank_batches(1)
    assert all(len(batch)==4 for batch in left+right)
    for a,b in zip(left[:2],right[:2]):
        assert not set(a) & set(b)
    assert set(i for batch in left+right for i in batch) == set(range(19))


def test_logged_loss_averages_interval_and_keeps_last_step(cfg, toy_dataset, tmp_path):
    cfg.train.steps = 5
    cfg.train.log_every = 2
    cfg.train.accumulation_steps = 2
    cfg.train.output_dir = str(tmp_path)
    trainer = Trainer(build_model(cfg, toy_dataset.audio_metadata), toy_dataset, cfg)
    losses = []
    handle = trainer.training_model.register_forward_hook(
        lambda module, inputs, output: losses.append(float(output[0].detach())))
    trainer.fit()
    handle.remove()
    step_losses = [sum(losses[i:i+2])/2 for i in range(0, len(losses), 2)]
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == [2, 4]
    for record in records:
        step = record['step']
        assert record['loss'] == pytest.approx(sum(step_losses[step-2:step])/2)
        assert record['step_loss'] == pytest.approx(step_losses[step-1])
    assert trainer.last_loss == pytest.approx(step_losses[-1])


def test_learning_rate_warmup_peak_and_decay_floor(cfg, toy_dataset, tmp_path):
    cfg.train.output_dir = str(tmp_path)
    cfg.train.learning_rate = 1e-4
    cfg.train.min_learning_rate = 1e-6
    cfg.train.warmup_steps = 2
    cfg.train.schedule_steps = 6
    trainer = Trainer(build_model(cfg, toy_dataset.audio_metadata), toy_dataset, cfg)
    rates = [trainer.optimizer.param_groups[0]['lr']]
    for _ in range(8):
        trainer.optimizer.step()
        trainer.scheduler.step()
        rates.append(trainer.optimizer.param_groups[0]['lr'])
    assert rates[0] == pytest.approx(5e-5)
    assert rates[2] == pytest.approx(1e-4)
    assert rates[4] == pytest.approx(5.05e-5)
    assert rates[6:] == pytest.approx([1e-6] * 3)
    assert all(a >= b for a, b in zip(rates[2:], rates[3:]))


def test_periodic_and_final_checkpoints_keep_distinct_steps(cfg, toy_dataset, tmp_path):
    cfg.train.output_dir = str(tmp_path)
    cfg.train.steps = 5
    cfg.train.checkpoint_every = 2
    trainer = Trainer(build_model(cfg, toy_dataset.audio_metadata), toy_dataset, cfg)
    trainer.fit()
    paths = sorted((tmp_path / 'checkpoints').glob('step_*.pt'))
    assert [path.name for path in paths] == ['step_2.pt', 'step_4.pt', 'step_5.pt']
    checkpoints = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
    assert [checkpoint['step'] for checkpoint in checkpoints] == [2, 4, 5]
    assert all('optimizer' in checkpoint and 'scheduler' in checkpoint for checkpoint in checkpoints)
    key = 'motion_generator.output.weight'
    assert not torch.equal(checkpoints[0]['model'][key], checkpoints[1]['model'][key])
