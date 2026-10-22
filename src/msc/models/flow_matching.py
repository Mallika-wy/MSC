"""Linear flow, t=1 noise -> t=0 data; source attribution and licenses in THIRD_PARTY.md."""
import torch


class FlowMatching:
    time_scale = 1000.0
    horizon = 30
    action_dim = 125

    def loss(self, generator, data, conditions, mask, *, noise=None, times=None,
             rtc_training=False, max_delay=8, delays=None, return_details=False):
        data = data.float().masked_fill(~mask[..., None], 0)
        noise = torch.randn_like(data) if noise is None else noise.float()
        if noise.shape != data.shape:
            raise ValueError('noise must match [B,30,125] data')
        noise = noise.masked_fill(~mask[...,None],0)
        if not torch.isfinite(noise).all():
            raise ValueError('Valid flow noise contains NaN/Inf')
        if times is None:
            times = torch.rand(data.shape[0], device=data.device)
        if times.shape not in {(data.shape[0],), data.shape[:2]}:
            raise ValueError("times must have shape [B] or [B,30]")
        if times.ndim == 1:
            times = times[:, None].expand(-1, data.shape[1])
        if not torch.isfinite(times).all() or not ((times >= 0) & (times <= 1)).all():
            raise ValueError("times must be finite values in [0,1]")
        loss_mask = mask
        if rtc_training:
            if not isinstance(max_delay, int) or not 1 <= max_delay < data.shape[1]:
                raise ValueError("max_delay must be an integer in [1, horizon - 1]")
            b, horizon = mask.shape
            if delays is None:
                delays = torch.randint(max_delay, (b,), device=data.device)
            if delays.shape != (b,) or delays.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("RTC delays must be integer [B]")
            # A padded tail must not make every target frame a frozen prefix.
            valid_lengths = mask.sum(-1)
            delays = delays.to(data.device).clamp(min=0)
            delays = torch.minimum(delays, (valid_lengths - 1).clamp(min=0))
            prefix = torch.arange(horizon, device=data.device)[None, :] < delays[:, None]
            times = times.masked_fill(prefix, 0)
            loss_mask = mask & ~prefix
            if not loss_mask.any(-1).all():
                raise ValueError("RTC needs at least one unfrozen valid future frame")
        t = times[..., None]
        x = ((1 - t) * data + t * noise).masked_fill(~mask[..., None], 0)
        target = noise - data
        velocity = generator(x, times * self.time_scale, conditions, mask)
        error = (velocity.float() - target).square().mean(-1).masked_fill(~loss_mask, 0)
        if not loss_mask.any(-1).all():
            raise ValueError("Cannot compute loss on an empty motion")
        loss = (error.sum(-1) / loss_mask.sum(-1)).mean()
        if return_details:
            return dict(loss=loss, x_tau=x, tau=times, pred_velocity=velocity, mask=loss_mask)
        return loss

    @torch.no_grad()
    def sample(self, generator, conditions, *, num_steps=10, initial_noise=None,
               null_conditions=None, cfg_scale=1.0):
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        if not torch.isfinite(torch.tensor(cfg_scale)):
            raise ValueError("cfg_scale must be finite")
        ref = conditions.tokens
        shape = (ref.shape[0], self.horizon, self.action_dim)
        x = torch.randn(shape, device=ref.device) if initial_noise is None else initial_noise.to(ref.device).float().clone()
        if x.shape != shape or not torch.isfinite(x).all():
            raise ValueError("initial_noise must be finite with shape [B,30,125]")
        for i in range(num_steps):
            times = torch.full((shape[0],), (1 - i / num_steps) * self.time_scale, device=x.device)
            velocity = generator(x, times, conditions).float()
            if cfg_scale != 1.0:
                if null_conditions is None:
                    raise ValueError("CFG requires null conditions")
                base = generator(x, times, null_conditions).float()
                velocity = base + cfg_scale * (velocity - base)
            x = x - velocity / num_steps
        return x

    def _validate_rtc_inputs(self, conditions, previous_actions, inference_delay, execution_horizon=None):
        ref = conditions.tokens
        b, horizon, action_dim = ref.shape[0], self.horizon, self.action_dim
        previous_actions = previous_actions.to(ref.device).float()
        if previous_actions.shape != (b, horizon, action_dim) or not torch.isfinite(previous_actions).all():
            raise ValueError("previous_actions must be finite normalized [B,30,125]")
        if isinstance(inference_delay, bool) or not isinstance(inference_delay, int) or not 0 <= inference_delay < horizon:
            raise ValueError(f"inference_delay must be an integer in [0, {horizon - 1}]")
        if execution_horizon is not None and (isinstance(execution_horizon, bool) or not isinstance(execution_horizon, int)):
            raise ValueError("execution_horizon must be an integer or None")
        return ref, previous_actions, inference_delay, execution_horizon

    @torch.no_grad()
    def sample_training_rtc(self, generator, conditions, *, previous_actions, inference_delay,
                            max_delay, num_steps=10, initial_noise=None,
                            null_conditions=None, cfg_scale=1.0):
        """Psi0 train-time RTC: clean previous actions are hard-frozen as a prefix."""
        ref, previous_actions, delay, _ = self._validate_rtc_inputs(
            conditions, previous_actions, inference_delay)
        if not isinstance(max_delay, int) or not 1 <= max_delay < self.horizon:
            raise ValueError(f"max_delay must be an integer in [1, {self.horizon - 1}]")
        delay = min(delay, max_delay - 1, self.horizon - 1)
        if delay == 0:
            return self.sample(generator, conditions, num_steps=num_steps, initial_noise=initial_noise,
                               null_conditions=null_conditions, cfg_scale=cfg_scale)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        shape = (ref.shape[0], self.horizon, self.action_dim)
        x = torch.randn(shape, device=ref.device) if initial_noise is None else initial_noise.to(ref.device).float().clone()
        if x.shape != shape or not torch.isfinite(x).all():
            raise ValueError("initial_noise must be finite with shape [B,30,125]")
        prefix = torch.arange(self.horizon, device=x.device)[None, :] < delay
        for i in range(num_steps):
            times = torch.full((shape[0], self.horizon), (1 - i / num_steps) * self.time_scale, device=x.device)
            times = times.masked_fill(prefix, 0)
            x = torch.where(prefix[..., None], previous_actions, x)
            velocity = generator(x, times, conditions).float()
            if cfg_scale != 1.0:
                if null_conditions is None:
                    raise ValueError("CFG requires null conditions")
                base = generator(x, times, null_conditions).float()
                velocity = base + cfg_scale * (velocity - base)
            x = x - velocity / num_steps
            x = torch.where(prefix[..., None], previous_actions, x)
        return x

    def sample_rtc(self, generator, conditions, *, previous_actions, inference_delay=0,
                   execution_horizon=None, mask_schedule="exponential", guidance_alpha=0.9,
                   num_steps=10, initial_noise=None, null_conditions=None, cfg_scale=1.0):
        """Psi0 test-time RTC using PiGDM-style VJP guidance toward the prior chunk."""
        ref, previous_actions, delay, execution_horizon = self._validate_rtc_inputs(
            conditions, previous_actions, inference_delay, execution_horizon)
        if not isinstance(guidance_alpha, (float, int)) or not torch.isfinite(torch.tensor(guidance_alpha)) or guidance_alpha < 0:
            raise ValueError("guidance_alpha must be a finite nonnegative number")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer")
        horizon = self.horizon
        # The RTC paper requires d <= s <= H-d. Saturate an overloaded inference
        # delay to its largest feasible value instead of failing a live replan.
        delay = min(delay, horizon // 2)
        execution_horizon = max(delay, min(execution_horizon if execution_horizon is not None else max(1, horizon - 2 * delay), horizon - delay))
        weights = self.rtc_mask(horizon, delay, execution_horizon, mask_schedule, ref.device)[None, :, None]
        shape = (ref.shape[0], horizon, self.action_dim)
        x = torch.randn(shape, device=ref.device) if initial_noise is None else initial_noise.to(ref.device).float().clone()
        if x.shape != shape or not torch.isfinite(x).all():
            raise ValueError("initial_noise must be finite with shape [B,30,125]")
        for i in range(num_steps):
            tau = 1 - i / num_steps
            times = torch.full((shape[0],), tau * self.time_scale, device=x.device)
            with torch.enable_grad():
                x_grad = x.detach().requires_grad_(True)
                velocity = generator(x_grad, times, conditions).float()
                if cfg_scale != 1.0:
                    if null_conditions is None:
                        raise ValueError("CFG requires null conditions")
                    base = generator(x_grad, times, null_conditions).float()
                    velocity = base + cfg_scale * (velocity - base)
                pred_x0 = x_grad - tau * velocity
                error = (previous_actions - pred_x0.detach()) * weights
                correction = torch.autograd.grad(pred_x0, x_grad, error, retain_graph=False)[0]
            correction_norm = correction.norm()
            if tau >= 1.0 or correction_norm < 1e-8:
                correction.zero_()
            else:
                correction.mul_(float(guidance_alpha) * error.norm() / (tau * correction_norm))
            x = x - (velocity.detach() - correction.detach()) / num_steps
        return x

    @staticmethod
    def rtc_mask(horizon, delay, execution_horizon, schedule, device):
        if schedule not in {"exponential", "linear", "hard", "simple"}:
            raise ValueError("Unknown RTC mask schedule")
        mask = torch.zeros(horizon, device=device)
        mask[:delay] = 1
        overlap_end = horizon - execution_horizon
        if schedule == "hard" or delay >= overlap_end:
            return mask
        indices = torch.arange(delay, overlap_end, device=device).float()
        if schedule == "linear":
            mask[delay:overlap_end] = 1 - (indices - delay) / (overlap_end - delay)
        elif schedule == "exponential":
            c = (overlap_end - indices) / (overlap_end - delay + 1)
            mask[delay:overlap_end] = c * torch.expm1(c) / torch.expm1(torch.tensor(1.0, device=device))
        else:
            mask[delay:] = torch.exp(-5 * (torch.arange(delay, horizon, device=device) - delay) / (horizon - delay))
        return mask
