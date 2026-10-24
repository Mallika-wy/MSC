import pytest

from msc.factory import build_model
from msc.training.trainer import Trainer


@pytest.mark.integration
def test_offline_wandb_records_training_and_validation(cfg, toy_dataset, tmp_path, monkeypatch):
    wandb = pytest.importorskip('wandb')
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setenv('WANDB_BASE_URL', 'http://127.0.0.1:9')
    cfg.train.wandb_project = 'msc-offline-test'
    cfg.train.wandb_name = 'two-step-test'
    cfg.train.output_dir = str(tmp_path)
    cfg.train.steps = 2
    trainer = Trainer(build_model(cfg, toy_dataset.audio_metadata), toy_dataset, cfg)
    cfg.train.validation_every = 1
    monkeypatch.setattr(trainer, 'validate', lambda: {'audio': {'loss_flow': .5}})
    original_init = wandb.init
    runs, records, summary = [], [], {}
    original_fit = trainer._fit

    def capture_fit(run):
        loss = original_fit(run)
        summary.update(dict(run.summary))
        return loss

    def capture_init(**kwargs):
        run = original_init(**kwargs)
        original_log = run.log

        def capture_log(data):
            records.append(data)
            original_log(data)

        monkeypatch.setattr(run, 'log', capture_log)
        runs.append(run)
        return run

    monkeypatch.setattr(wandb, 'init', capture_init)
    monkeypatch.setattr(trainer, '_fit', capture_fit)
    trainer.fit()
    assert len(runs) == 1
    assert runs[0].settings.mode == 'offline'
    assert runs[0].name == 'two-step-test'
    assert runs[0].config['train']['batch_size'] == cfg.train.batch_size
    assert [record['optimizer_step'] for record in records] == [1, 1, 2, 2]
    assert 'train/loss' in records[0]
    assert 'train/learning_rate' in records[0]
    assert 'train/gradient_norms/expert' in records[0]
    assert 'train/rank_peak_memory_gib/0' in records[0]
    assert records[1]['val/audio/loss_flow'] == .5
    assert summary['profile/optimizer_steps'] == 2
    files = list(tmp_path.glob('wandb/offline-run-*/run-*.wandb'))
    assert len(files) == 1 and files[0].stat().st_size > 0
    assert wandb.run is None


@pytest.mark.parametrize('project,rank', [(None, 0), ('msc-test', 1)])
def test_disabled_or_worker_rank_does_not_start_wandb(cfg, toy_dataset, tmp_path, monkeypatch, project, rank):
    cfg.train.output_dir = str(tmp_path)
    cfg.train.wandb_project = project
    trainer = Trainer(build_model(cfg, toy_dataset.audio_metadata), toy_dataset, cfg)
    trainer.rank = rank

    def unexpected_init(_):
        pytest.fail('Only enabled rank 0 should initialize W&B')

    def fit_without_tracking(run):
        assert run is None
        return .5

    monkeypatch.setattr('msc.training.trainer.offline_run', unexpected_init)
    monkeypatch.setattr(trainer, '_fit', fit_without_tracking)
    assert trainer.fit() == .5
