from msc.factory import build_model
from msc.training.trainer import Trainer


def test_gradient_accumulation(cfg, toy_dataset, tmp_path):
    cfg.train.steps = 2
    cfg.train.accumulation_steps = 2
    cfg.train.output_dir = str(tmp_path)
    data = toy_dataset
    trainer = Trainer(build_model(cfg, data.audio_metadata), data, cfg)
    trainer.fit()
    assert trainer.step == 2
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 2


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
