import pytest
import torch
from msc.config import Config
from msc.factory import seed_everything, build_model
from msc.data.collator import collate_motion
from msc.models.representation import MotionRepresentation


class ToyDataset(torch.utils.data.Dataset):
    """Test-only in-memory batch source; it is not a training backend."""
    def __init__(self, size=32, seed=17):
        self.size, self.seed = size, seed
        self.representation = MotionRepresentation()
        self.audio_metadata = dict(backend="mert_precomputed", feature_type="test", dimension=768,
                                   token_rate_hz=75, preprocessing_version="test")
        self.feature_metadata = {}
        self.identity = f'test:{size}:{seed}'
        self.task_ranges = {}

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(self.seed + index)
        phase = index % 4 * .3
        timeline = torch.arange(-5, 30).float()
        values = .1 * torch.sin(timeline[:, None] / 30 + phase).expand(-1, 125).clone()
        values += torch.linspace(-.03, .03, 125)[None]
        motion = self.representation.denormalize(values)
        length = 23 if index % 5 == 0 else 30
        mask = torch.arange(30) < length
        audio_length = 68 + index % 3 * 4
        audio_times = torch.arange(audio_length) / 75
        return dict(motion_history=motion[:5], motion_future=motion[5:], motion_mask=mask,
                    audio_tokens=torch.randn(audio_length, 768, generator=gen) * .1,
                    audio_mask=(audio_times < length / 30), audio_times=audio_times)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    seed_everything(123)


@pytest.fixture
def cfg():
    cfg = Config()
    cfg.model.hidden_dim = 64
    cfg.model.num_layers = 2
    cfg.model.num_heads = 4
    cfg.model.use_text = False
    cfg.model.losses = dict(joint_velocity=0., joint_acceleration=0.)
    cfg.data.roots = ['test-only']
    cfg.train.device = 'cpu'
    cfg.train.steps = 10
    cfg.train.batch_size = 4
    cfg.train.validation_every = 0
    return cfg


@pytest.fixture
def toy_dataset(cfg):
    return ToyDataset(seed=cfg.train.seed)


@pytest.fixture
def batch(toy_dataset):
    data = toy_dataset
    return collate_motion([data[i] for i in range(4)])


@pytest.fixture
def model(cfg, toy_dataset):
    data = toy_dataset
    model = build_model(cfg, data.audio_metadata)
    # General conditioning tests exercise a model after the initial zero-gate phase.
    with torch.no_grad():
        for layer in model.motion_generator.layers:
            layer.modulation[-1].bias.view(9, -1)[[2, 5, 8]] = .1
    return model
