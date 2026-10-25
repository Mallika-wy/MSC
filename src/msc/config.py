from dataclasses import dataclass, field, asdict
from pathlib import Path
import math
import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 1024
    num_layers: int = 12
    ffn_dim: int | None = None
    representation: str = 'g1_canonical_125d'
    fps: int = 30
    history_frames: int = 5
    future_frames: int = 30
    condition: dict = field(default_factory=dict)
    losses: dict = field(default_factory=lambda: dict(joint_velocity=0.,joint_acceleration=0.,dynamic_warmup_steps=1000))
    dynamic_stats_path: str | None = None
    num_heads: int = 16
    dropout: float = 0.1
    text_dim: int = 768
    use_text: bool = True
    use_audio: bool = True
    t5_path: str | None = None
    freeze_text: bool = True
    text_max_length: int = 50
    text_dropout: float = 0.1
    audio_dropout: float = 0.1
    history_dropout: float = 0.0
    # Psi0-style RTC training is opt-in: it can bias a generator toward
    # continuing its previous chunk, so ordinary motion training keeps it off.
    rtc: bool = False
    max_delay: int = 8
    stats_path: str | None = None


@dataclass
class DataConfig:
    backend: str = "msc_release"
    split: str = "train"
    num_workers: int = 0
    roots: list[str] = field(default_factory=list)
    label_policy: str = 'train_default'
    alignment_policy: str = 'allow_estimated'
    audio_context_sec: float = .25
    video_context_sec: float = .25
    cache_bytes: int = 134217728
    tasks: list[str] = field(default_factory=lambda: ['text','audio'])
    max_motions_per_task: int | None = None
    max_windows_per_motion: int | None = None
    motion_manifest: str | None = None


@dataclass
class TrainConfig:
    seed: int = 42
    deterministic: bool = True
    compile_expert: bool = False
    bf16_gradient_reduce: bool = False
    device: str = "cpu"
    precision: str = "fp32"
    steps: int = 10
    batch_size: int = 4
    accumulation_steps: int = 1
    learning_rate: float = 0.0003
    min_learning_rate: float = 0.000001
    weight_decay: float = 0.01
    warmup_steps: int = 0
    schedule_steps: int = 1000
    max_grad_norm: float = 1.0
    checkpoint_every: int = 10
    log_every: int = 1
    output_dir: str = ".runs/smoke"
    resume: str | None = None
    num_threads: int = 2
    validation_every: int = 100
    validation_batches: int = 4
    validation_motions_per_task: int | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        m, d, t = self.model, self.data, self.train
        if d.backend != "msc_release":
            raise ValueError("Unknown data backend")
        if m.condition.get('text',{}).get('enabled',m.use_text) and not m.t5_path:
            raise ValueError("model.t5_path is required for local T5")
        if not m.freeze_text:
            raise ValueError('Trimodal encoder training is disabled')
        if (m.representation,m.fps,m.history_frames,m.future_frames) != ('g1_canonical_125d',30,5,30):
            raise ValueError('Only canonical G1 125D / 30 FPS / history 5 / future 30 is supported')
        if d.backend == 'msc_release' and not d.roots:
            raise ValueError('data.roots must identify published Releases')
        if d.split not in {'train','val','test'}:
            raise ValueError('data.split must be train/val/test')
        if d.alignment_policy not in {'allow_estimated','verified_only'}:
            raise ValueError('Unknown alignment policy')
        if d.cache_bytes < 0 or min(d.audio_context_sec,d.video_context_sec) < 0:
            raise ValueError('Cache/context must be nonnegative')
        if not d.tasks or len(d.tasks)!=len(set(d.tasks)) or set(d.tasks)-{'text','audio','video'}:
            raise ValueError('Invalid data tasks')
        if m.condition.get('temporal_bias','relative_distance') not in {'none','relative_distance'}:
            raise ValueError('Unknown temporal bias')
        if m.condition.get('history_dropout',m.history_dropout) != 0:
            raise ValueError('History dropout must be zero')
        if m.condition.get('fusion','token_cross_attention') != 'token_cross_attention':
            raise ValueError('Only token_cross_attention is supported')
        if set(m.condition)-{'fusion','temporal_bias','history_dropout','text','audio','video'}:
            raise ValueError('Unknown condition configuration key')
        for name in ('text','audio','video'):
            c = m.condition.get(name,{})
            if set(c)-{'enabled','backend','frozen','dim','dropout','spatial_tokens'}:
                raise ValueError(f'Unknown {name} condition configuration key')
            if c.get('frozen',True) is not True or not 0 <= c.get('dropout',.1) <= 1:
                raise ValueError('Encoders must be frozen with valid condition dropout')
            if c.get('enabled',{'text':m.use_text,'audio':m.use_audio,'video':False}[name]) is False and d.backend=='msc_release' and name in d.tasks:
                raise ValueError(f'{name} task is sampled but its adapter is disabled')
        if m.condition.get('audio',{}).get('dim',768) != 768:
            raise ValueError('Audio dimension must be 768 for MERT')
        if m.losses.get('flow',1.) != 1.:
            raise ValueError('Only unchanged FM plus joint dynamics losses are supported')
        if set(m.losses)-{'flow','joint_velocity','joint_acceleration','dynamic_normalization','dynamic_warmup_steps'}:
            raise ValueError('Unknown loss configuration key')
        if any(not math.isfinite(m.losses.get(k,0)) or m.losses.get(k,0)<0 for k in ('joint_velocity','joint_acceleration','dynamic_warmup_steps')):
            raise ValueError('Loss weights/warmup must be nonnegative')
        if not isinstance(m.losses.get('dynamic_warmup_steps',1000),int):
            raise ValueError('Dynamic warmup must count integer optimizer steps')
        if m.losses.get('dynamic_normalization','train_stats') != 'train_stats':
            raise ValueError('Dynamics require train_stats normalization')
        for value in (m.ffn_dim,d.max_motions_per_task,d.max_windows_per_motion,t.validation_motions_per_task):
            if value is not None and (not isinstance(value,int) or value<=0):
                raise ValueError('Optional dimensions/subset limits must be positive integers')
        if t.validation_every<0 or t.validation_batches<1:
            raise ValueError('Invalid validation intervals/batch count')
        if t.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        for value in (m.hidden_dim, m.num_layers, m.num_heads, m.text_dim, m.text_max_length,
                      t.steps, t.batch_size, t.accumulation_steps, t.schedule_steps,
                      t.checkpoint_every, t.log_every, t.num_threads):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Dimensions, batch sizes and step intervals must be positive integers")
        if d.num_workers < 0:
            raise ValueError("Invalid data worker count")
        if not 0 <= t.warmup_steps < t.schedule_steps or t.learning_rate <= 0 or t.max_grad_norm <= 0:
            raise ValueError("Invalid optimizer/scheduler settings")
        if not 0 <= t.min_learning_rate <= t.learning_rate:
            raise ValueError("min_learning_rate must be between zero and learning_rate")
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
        names = key.split('.')
        target = raw
        for name in names[:-1]:
            target = target.setdefault(name,{})
        target[names[-1]] = yaml.safe_load(value)
    return Config.from_dict(raw)
