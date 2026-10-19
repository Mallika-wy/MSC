import copy
import torch
from msc.factory import seed_everything, build_model, build_dataset
from msc.training.trainer import Trainer
from msc.training.checkpoint import load_checkpoint


def test_checkpoint_resume_matches_continuous(cfg, tmp_path):
    cfg.train.steps = 3
    cfg.train.output_dir = str(tmp_path / "continuous")
    seed_everything(cfg.train.seed)
    data = build_dataset(cfg)
    full = Trainer(build_model(cfg, data.audio_metadata), data, cfg)
    full.fit()
    split_cfg = copy.deepcopy(cfg)
    split_cfg.train.steps = 2
    split_cfg.train.output_dir = str(tmp_path / "split")
    seed_everything(cfg.train.seed)
    first = Trainer(build_model(split_cfg, data.audio_metadata), data, split_cfg)
    first.fit()
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg.train.resume = str(tmp_path / "split/last.pt")
    resumed_cfg.train.output_dir = str(tmp_path / "resumed")
    resumed = Trainer(build_model(resumed_cfg, data.audio_metadata), data, resumed_cfg)
    resumed.fit()
    assert full.last_loss == resumed.last_loss
    for key, value in full.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], atol=0, rtol=0)
    assert load_checkpoint(tmp_path / "resumed/last.pt")["step"] == 3


def test_gradient_accumulation(cfg, tmp_path):
    cfg.train.steps = 2
    cfg.train.accumulation_steps = 2
    cfg.train.output_dir = str(tmp_path)
    data = build_dataset(cfg)
    trainer = Trainer(build_model(cfg, data.audio_metadata), data, cfg)
    trainer.fit()
    assert trainer.step == 2
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2
