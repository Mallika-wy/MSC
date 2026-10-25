"""Persistent offline SONIC execution for full-motion evaluation."""
from collections import deque
import numpy as np
from data_pipeline.motion import sonic
from data_pipeline.motion.quality import resample_motion_at_times


class SonicFeedback:
    def __init__(self, settings, initial_history):
        import mujoco
        self.initial_history = initial_history.copy()
        self.model, self.handles, _ = sonic.load_simulator(settings['sonic_root'])
        self.encoder, self.decoder, self.config = sonic.load_policy(
            settings['encoder'], settings['decoder'], settings['observation_config'],
            settings.get('providers', 'CPUExecutionProvider'))
        self.state = mujoco.MjData(self.model)
        sonic.set_qpos(self.model, self.state, initial_history[-1], self.handles)
        before = self.state.qpos.copy()
        before[self.handles[0]] = initial_history[-2]
        mujoco.mj_differentiatePos(self.model, self.state.qvel, 1 / 30, before, self.state.qpos)
        mujoco.mj_forward(self.model, self.state)
        self.action = np.zeros(29)
        self.history = deque([self.snapshot() for _ in range(10)], maxlen=10)
        self.states = [self.state.qpos[self.handles[0]].copy()]

    def snapshot(self):
        state = self.state
        qi, vi, _ = self.handles
        return (state.qvel[3:6].copy(), (state.qpos[qi[7:]] - sonic.DEFAULT)[sonic.TO_ISAAC],
                state.qvel[vi][sonic.TO_ISAAC].copy(), self.action.copy(),
                sonic.rotation(state.qpos[3:7]).inv().apply([0, 0, -1]))

    def execute(self, plan, frames):
        import mujoco
        model, state = self.model, self.state
        qi, vi, ai = self.handles
        reference, _ = sonic.sample_reference(np.concatenate((self.states[-1][None], plan)))
        velocity = np.gradient(reference[:, 7:], .02, axis=0)
        # Regular chunks align to 50 Hz; only the final partial chunk may
        # overshoot by <20ms, then trajectory() interpolates to exact GT times.
        ticks = int(np.ceil(frames * 50 / 30 - 1e-9))
        for i in range(ticks):
            observation = sonic.encoder_observation(reference, velocity, i, state.qpos[3:7], self.config['encoder'])
            token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: observation[None]})[0]
            observation = sonic.decoder_observation(token, self.history, self.config['observations'])
            self.action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: observation[None]})[0].reshape(29)
            target = sonic.DEFAULT + self.action[sonic.TO_MUJOCO] * sonic.ACTION_SCALE
            for _ in range(round(.02 / model.opt.timestep)):
                torque = sonic.KP * (target - state.qpos[qi[7:]]) - sonic.KD * state.qvel[vi]
                state.ctrl[ai] = np.clip(torque, model.actuator_ctrlrange[ai, 0], model.actuator_ctrlrange[ai, 1])
                mujoco.mj_step(model, state)
            self.history.append(self.snapshot())
            self.states.append(state.qpos[qi].copy())

    def measured_history(self):
        times = np.r_[np.arange(-4, 0) / 30, np.arange(len(self.states)) / 50]
        values = np.concatenate((self.initial_history[:-1], np.asarray(self.states)))
        query = (len(self.states) - 1) / 50 + np.arange(-4, 1) / 30
        return resample_motion_at_times(values, times, query).astype(np.float32)

    def trajectory(self, frames):
        return resample_motion_at_times(np.asarray(self.states), np.arange(len(self.states)) / 50,
                                        np.arange(1, frames + 1) / 30).astype(np.float32)
