import pytest
import torch
from msc.config import load_config
from msc.factory import seed_everything, build_model, build_dataset
from msc.data.collator import collate_motion


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(123)


@pytest.fixture
def cfg():
    return load_config("configs/smoke.yaml")


@pytest.fixture
def batch(cfg):
    data = build_dataset(cfg)
    return collate_motion([data[i] for i in range(4)])


@pytest.fixture
def model(cfg):
    data = build_dataset(cfg)
    return build_model(cfg, data.audio_metadata)
