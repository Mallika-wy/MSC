import torch
from torch import nn
from .representation import MotionRepresentation
from .motion_generator import MotionGenerator
from .flow_matching import FlowMatching
from .conditioning.text import TextAdapter
from .conditioning.audio import AudioAdapter


class MSCModel(nn.Module):
    def __init__(self, *, hidden_dim=512, num_layers=12, num_heads=8, dropout=0.1,
                 text_dim=768, text_encoder=None, use_text=True, use_audio=True,
                 text_dropout=0.1, audio_dropout=0.1, history_dropout=0.1,
                 stats_path=None, audio_metadata=None, rtc=False, max_delay=8):
        super().__init__()
        self.representation = MotionRepresentation(stats_path)
        self.motion_generator = MotionGenerator(hidden_dim, num_layers, num_heads, dropout)
        self.flow = FlowMatching()
        self.text_encoder = text_encoder
        self.text_adapter = TextAdapter(text_dim, hidden_dim) if use_text else None
        self.audio_adapter = AudioAdapter(35, hidden_dim) if use_audio else None
        self.text_dropout, self.audio_dropout, self.history_dropout = text_dropout, audio_dropout, history_dropout
        self.audio_metadata = audio_metadata
        self.rtc, self.max_delay = rtc, max_delay
        for p in (text_dropout, audio_dropout, history_dropout):
            if not 0 <= p <= 1:
                raise ValueError("Condition dropout must be in [0,1]")
        if not isinstance(rtc, bool) or not isinstance(max_delay, int) or not 1 <= max_delay < 30:
            raise ValueError("RTC max_delay must be an integer in [1, 29]")

    def conditions(self, batch, *, drop_modalities=False):
        if batch.get("video") is not None:
            raise NotImplementedError("Video conditioning is reserved for a future stage")
        history = batch["motion_history"]
        if history.ndim != 3 or history.shape[1:] != (5, 125) or not torch.isfinite(history).all():
            raise ValueError("history must be finite canonical [B,5,125]")
        b = history.shape[0]
        device = history.device
        generator = self.motion_generator
        history_tokens = generator.history_projector(self.representation.normalize(history))
        if self.training and self.history_dropout:
            keep = torch.rand(b, 1, 1, device=device) >= self.history_dropout
            history_tokens = history_tokens.masked_fill(~keep, 0)
        context = generator.extra_proj(history_tokens)
        mask = torch.ones(b, 5, device=device, dtype=torch.bool)
        if self.text_adapter is not None and not drop_modalities:
            features = batch.get("text_features")
            text_mask = batch.get("text_mask")
            if features is not None and self.text_encoder is not None and not self.text_encoder.frozen:
                raise ValueError("Precomputed text features cannot train T5")
            captions = batch.get("text")
            if features is None and captions is not None and any(s.strip() for s in captions):
                if self.text_encoder is None:
                    raise ValueError("Raw text requires local T5; synthetic mode accepts text_features only")
                features, text_mask = self.text_encoder(captions, device)
            if features is not None:
                if text_mask is None or text_mask.shape != features.shape[:2] or features.shape[0] != b:
                    raise ValueError("text_features requires aligned text_mask")
                text_mask = text_mask.bool().to(device)
                if "has_text" in batch:
                    text_mask = text_mask & batch["has_text"].to(device).bool()[:, None]
                if self.training and self.text_dropout:
                    text_mask = text_mask & (torch.rand(b, 1, device=device) >= self.text_dropout)
                features = features.to(device).masked_fill(~text_mask[..., None], 0)
                if not torch.isfinite(features).all():
                    raise ValueError("Valid text features contain nonfinite values")
                tokens = self.text_adapter(features).masked_fill(~text_mask[..., None], 0)
                context = torch.cat([context, tokens], dim=1)
                mask = torch.cat([mask, text_mask], dim=1)
        cond = {"context": context, "context_mask": mask}
        if self.audio_adapter is not None and not drop_modalities and batch.get("audio_features") is not None:
            audio = batch["audio_features"]
            audio_mask = batch.get("audio_mask")
            if audio.shape != (b, 30, 35) or audio_mask is None or audio_mask.shape != (b, 30):
                raise ValueError("audio requires [B,30,35] features and [B,30] validity mask")
            audio_mask = audio_mask.to(device).bool()
            if "motion_mask" in batch:
                audio_mask = audio_mask & batch["motion_mask"].bool()
            if self.training and self.audio_dropout:
                audio_mask = audio_mask & (torch.rand(b, 1, device=device) >= self.audio_dropout)
            audio = audio.to(device).masked_fill(~audio_mask[..., None], 0)
            if not torch.isfinite(audio).all():
                raise ValueError("Valid audio contains nonfinite values")
            cond["audio"] = self.audio_adapter(audio).masked_fill(~audio_mask[..., None], 0)
            cond["audio_mask"] = audio_mask
        return cond

    def forward(self, noisy_future, time, batch):
        """Predict normalized-space flow velocity; time is continuous in [0,1]."""
        return self.motion_generator(noisy_future, time * self.flow.time_scale,
                                     self.conditions(batch), batch.get("motion_mask"))

    def compute_loss(self, batch, *, noise=None, times=None, rtc_delays=None):
        future = batch["motion_future"]
        mask = batch["motion_mask"].bool()
        if future.shape[1:] != (30, 125) or mask.shape != future.shape[:2]:
            raise ValueError("Expected future [B,30,125] and motion_mask [B,30]")
        if not torch.isfinite(future[mask]).all():
            raise ValueError("Valid motion contains nonfinite values")
        data = self.representation.normalize(future)
        return self.flow.loss(self.motion_generator, data, self.conditions(batch), mask, noise=noise, times=times,
                              rtc_training=self.rtc, max_delay=self.max_delay, delays=rtc_delays)

    @torch.no_grad()
    def sample(self, history, text=None, audio=None, *, audio_mask=None, audio_metadata=None,
               text_features=None, text_mask=None, num_steps=10, cfg_scale=1.0, initial_noise=None,
               previous_motion=None, rtc_mode="off", inference_delay=0, execution_horizon=None,
               rtc_mask_schedule="exponential", rtc_guidance_alpha=0.9):
        if text is not None and self.text_adapter is None:
            raise ValueError("Text conditioning is disabled in this model")
        if audio is not None and self.audio_adapter is None:
            raise ValueError("Audio conditioning is disabled in this model")
        if audio is not None and (self.audio_metadata is None or audio_metadata != self.audio_metadata):
            raise ValueError("Audio provenance must match the checkpoint metadata")
        was_training = self.training
        self.eval()
        try:
            batch = dict(motion_history=history)
            if text is not None:
                if isinstance(text, str):
                    text = [text] * len(history)
                if len(text) != len(history):
                    raise ValueError("One caption per history sample is required")
                batch["text"] = text
            if text_features is not None:
                batch.update(text_features=text_features, text_mask=text_mask)
            if audio is not None:
                batch.update(audio_features=audio, audio_mask=audio_mask)
            cond = self.conditions(batch)
            null = self.conditions(batch, drop_modalities=True) if cfg_scale != 1 else None
            if rtc_mode == "off":
                if previous_motion is not None:
                    raise ValueError("previous_motion requires rtc_mode='train' or 'test_time'")
                result = self.flow.sample(self.motion_generator, cond, num_steps=num_steps,
                    initial_noise=initial_noise, null_conditions=null, cfg_scale=cfg_scale)
            elif rtc_mode == "train":
                if not self.rtc:
                    raise ValueError("rtc_mode='train' requires a checkpoint trained with model.rtc=true")
                if previous_motion is None:
                    raise ValueError("RTC requires previous_motion")
                result = self.flow.sample_training_rtc(self.motion_generator, cond,
                    previous_actions=self.representation.normalize(previous_motion), inference_delay=inference_delay,
                    max_delay=self.max_delay, num_steps=num_steps, initial_noise=initial_noise,
                    null_conditions=null, cfg_scale=cfg_scale)
            elif rtc_mode == "test_time":
                if previous_motion is None:
                    raise ValueError("RTC requires previous_motion")
                result = self.flow.sample_rtc(self.motion_generator, cond,
                    previous_actions=self.representation.normalize(previous_motion), inference_delay=inference_delay,
                    execution_horizon=execution_horizon, mask_schedule=rtc_mask_schedule,
                    guidance_alpha=rtc_guidance_alpha, num_steps=num_steps,
                    initial_noise=initial_noise, null_conditions=null, cfg_scale=cfg_scale)
            else:
                raise ValueError("rtc_mode must be 'off', 'train', or 'test_time'")
            return self.representation.denormalize(result)
        finally:
            self.train(was_training)
