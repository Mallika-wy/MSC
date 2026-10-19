from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    text_dim: int = 768
    use_text: bool = True
    use_audio: bool = True
    text_backend: str = "t5"
    t5_path: str | None = None
    freeze_text: bool = True
    text_max_length: int = 50
    text_cache_dir: str | None = None
    text_dropout: float = 0.1
    audio_dropout: float = 0.1
    history_dropout: float = 0.1
    # Psi0-style RTC training is opt-in: it can bias a generator toward
    # continuing its previous chunk, so ordinary motion training keeps it off.
    rtc: bool = False
    max_delay: int = 8
    stats_path: str | None = None


@dataclass
class DataConfig:
    backend: str = "synthetic"
    root: str | None = None
    split: str = "train"
    size: int = 32
    limit_size: int | None = None
    num_workers: int = 0


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"
    steps: int = 10
    batch_size: int = 4
    accumulation_steps: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    warmup_steps: int = 0
    schedule_steps: int = 1000
    max_grad_norm: float = 1.0
    checkpoint_every: int = 10
    log_every: int = 1
    output_dir: str = ".runs/smoke"
    resume: str | None = None
    num_threads: int = 2


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        m, d, t = self.model, self.data, self.train
        if d.backend not in {"synthetic", "omg"} or m.text_backend not in {"synthetic", "t5"}:
            raise ValueError("Unknown data or text backend")
        if d.backend == "omg" and not d.root:
            raise ValueError("data.root is required for OMG; downloads are disabled")
        if d.backend == "omg" and m.use_text and m.text_backend != "t5":
            raise ValueError("Real captions require the T5 backend")
        if m.use_text and m.text_backend == "t5" and not m.t5_path:
            raise ValueError("model.t5_path is required for local T5")
        if m.text_cache_dir and not m.freeze_text:
            raise ValueError("Text cache requires frozen T5")
        if t.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        for value in (m.hidden_dim, m.num_layers, m.num_heads, m.text_dim, m.text_max_length,
                      d.size, t.steps, t.batch_size, t.accumulation_steps, t.schedule_steps,
                      t.checkpoint_every, t.log_every, t.num_threads):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Dimensions, batch sizes and step intervals must be positive integers")
        if d.num_workers < 0 or (d.limit_size is not None and d.limit_size <= 0):
            raise ValueError("Invalid data worker count or limit_size")
        if not 0 <= t.warmup_steps < t.schedule_steps or t.learning_rate <= 0 or t.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer/scheduler settings")
        if not 0 <= m.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(m.rtc, bool) or not isinstance(m.max_delay, int) or not 1 <= m.max_delay < 30:
            raise ValueError("model.max_delay must be an integer in [1, 29]")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        if set(raw) - {"model", "data", "train"}:
            raise ValueError("Unknown config section")
        return cls(ModelConfig(**raw.get("model", {})), DataConfig(**raw.get("data", {})),
                   TrainConfig(**raw.get("train", {}))).validate()


def load_config(path, overrides=()):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for override in overrides:
        key, value = override.split("=", 1)
        section, name = key.split(".", 1)
        raw.setdefault(section, {})[name] = yaml.safe_load(value)
    return Config.from_dict(raw)
