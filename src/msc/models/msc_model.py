"""Frozen features -> one condition memory -> one shared action expert."""
import json
import torch
from torch import nn
from .representation import MotionRepresentation
from .motion_generator import MotionGenerator
from .flow_matching import FlowMatching
from .conditioning.fusion import ConditionBridge
from .losses import dynamics_loss


class MSCModel(nn.Module):
    architecture_version = 'trimodal_flow_v2'

    def __init__(self, *, hidden_dim=512, num_layers=6, num_heads=8, dropout=.1,
                 text_dim=768, text_encoder=None, use_text=True, use_audio=True,
                 text_dropout=.1, audio_dropout=.1, history_dropout=0.,
                 stats_path=None, audio_metadata=None, rtc=False, max_delay=8,
                 video_dim=None, spatial_tokens=None, video_dropout=.1,
                 ffn_dim=None, temporal_bias='relative_distance', losses=None,
                 dynamic_stats_path=None, feature_metadata=None):
        super().__init__()
        if history_dropout != 0:
            raise ValueError('History dropout must be zero: CFG retains history')
        if not isinstance(rtc,bool) or not isinstance(max_delay,int) or not 1 <= max_delay < 30:
            raise ValueError('RTC max_delay must be in [1,29]')
        for p in (text_dropout,audio_dropout,video_dropout):
            if not 0 <= p <= 1:
                raise ValueError('Condition dropout must be in [0,1]')
        self.representation = MotionRepresentation(stats_path)
        self.motion_generator = MotionGenerator(hidden_dim,num_layers,num_heads,dropout,ffn_dim,temporal_bias)
        self.condition_bridge = ConditionBridge(hidden_dim,text_dim if use_text else None,
            768 if use_audio else None,video_dim,spatial_tokens,text_dropout,audio_dropout,video_dropout)
        self.flow, self.text_encoder = FlowMatching(),text_encoder
        if text_encoder is not None and not text_encoder.frozen:
            raise ValueError('Trimodal encoders must be frozen')
        self.audio_metadata, self.feature_metadata = audio_metadata,feature_metadata
        self.rtc, self.max_delay = rtc,max_delay
        self.loss_config = losses or dict(joint_velocity=0.,joint_acceleration=0.,dynamic_warmup_steps=1000)
        self.dynamic_stats_identity = None
        vs, acs = torch.ones(29),torch.ones(29)
        if dynamic_stats_path:
            from data_pipeline.io import sha256_file
            obj = json.loads(open(dynamic_stats_path).read())
            if obj.get('split') != 'train' or obj.get('version') != 1:
                raise ValueError('Dynamics statistics must use Train Split version 1')
            vs,acs = (torch.tensor(obj[k],dtype=torch.float32) for k in ('velocity_scale','acceleration_scale'))
            self.dynamic_stats_identity = obj['dataset_identity']
            self.dynamic_stats_sha256 = sha256_file(dynamic_stats_path)
        else:
            self.dynamic_stats_sha256 = None
        if vs.shape != (29,) or acs.shape != (29,) or not torch.isfinite(vs).all() or not torch.isfinite(acs).all() or (vs<0).any() or (acs<0).any():
            raise ValueError('Invalid dynamics scales')
        self.register_buffer('velocity_scale',vs.clamp_min(1e-4))
        self.register_buffer('acceleration_scale',acs.clamp_min(1e-4))

    @property
    def text_adapter(self):
        return self.condition_bridge.adapters['text'] if 'text' in self.condition_bridge.adapters else None

    @property
    def audio_adapter(self):
        return self.condition_bridge.adapters['audio'] if 'audio' in self.condition_bridge.adapters else None

    def encode_conditions(self,batch):
        history = batch['motion_history']
        if history.ndim != 3 or history.shape[1:] != (5,125) or not torch.isfinite(history).all():
            raise ValueError('history must be finite canonical [B,5,125]')
        batch = dict(batch)
        if batch.get('text_tokens',batch.get('text_features')) is None and batch.get('text') is not None and any(s.strip() for s in batch['text']):
            if self.text_encoder is None:
                raise ValueError('Raw text requires local frozen T5')
            batch['text_tokens'],batch['text_mask'] = self.text_encoder(batch['text'],history.device)
        return self.condition_bridge(batch,self.representation.normalize(history))

    def conditions(self,batch,*,drop_modalities=False):
        memory = self.encode_conditions(batch)
        return memory.null() if drop_modalities else memory

    def forward(self,noisy_future,time,batch):
        return self.motion_generator(noisy_future,time*self.flow.time_scale,self.encode_conditions(batch),batch.get('motion_mask'))

    def compute_loss(self,batch,*,noise=None,times=None,rtc_delays=None,return_metrics=False,step=0):
        future,mask = batch['motion_future'],batch['motion_mask'].bool()
        if future.shape[1:] != (30,125) or mask.shape != future.shape[:2]:
            raise ValueError('Expected future [B,30,125] and motion_mask [B,30]')
        if not mask.any(-1).all():
            raise ValueError('Each sample needs at least one valid future frame')
        if not torch.isfinite(future[mask]).all():
            raise ValueError('Valid motion contains NaN/Inf')
        data = self.representation.normalize(future.masked_fill(~mask[...,None],0))
        details = self.flow.loss(self.motion_generator,data,self.encode_conditions(batch),mask,
            noise=noise,times=times,rtc_training=self.rtc,max_delay=self.max_delay,delays=rtc_delays,return_details=True)
        flow = details['loss']
        vel,acc = flow*0,flow*0
        weights = self.loss_config
        active = step >= weights.get('dynamic_warmup_steps',1000)
        if active and (weights.get('joint_velocity',0) or weights.get('joint_acceleration',0)):
            with torch.autocast(device_type=future.device.type,enabled=False):
                clean = details['x_tau'].float()-details['tau'][...,None].float()*details['pred_velocity'].float()
                physical = self.representation.denormalize(clean)
                vel,acc = dynamics_loss(physical[...,9:38],future.float()[...,9:38],
                    batch['motion_history'].float()[...,9:38],details['mask'],self.velocity_scale,self.acceleration_scale)
        total = flow+weights.get('joint_velocity',0)*vel+weights.get('joint_acceleration',0)*acc
        if return_metrics:
            return total,dict(loss_flow=flow.detach(),loss_vel=vel.detach(),loss_acc=acc.detach(),loss_total=total.detach())
        return total

    @torch.no_grad()
    def sample(self, history, text=None, audio=None, *, audio_mask=None, audio_metadata=None,
               text_features=None, text_mask=None, text_tokens=None, audio_tokens=None,
               audio_times=None, video_tokens=None,
               video_mask=None, video_intervals=None, num_steps=10, cfg_scale=1.0, initial_noise=None,
               previous_motion=None, rtc_mode="off", inference_delay=0, execution_horizon=None,
               rtc_mask_schedule="exponential", rtc_guidance_alpha=0.9):
        if text_tokens is not None:
            if text_features is not None:
                raise ValueError('Supply only text_tokens or legacy text_features')
            text_features = text_tokens
        if audio_tokens is not None:
            if audio is not None:
                raise ValueError('Supply only audio_tokens or legacy audio')
            audio = audio_tokens
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
                batch.update(audio_tokens=audio, audio_mask=audio_mask, audio_times=audio_times)
            if video_tokens is not None:
                batch.update(video_tokens=video_tokens, video_mask=video_mask, video_intervals=video_intervals)
            cond = self.conditions(batch)
            null = cond.null() if cfg_scale != 1 else None
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
