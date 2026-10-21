"""Offline GEAR-SONIC v1.1 G1 mode, following the vendored deployment code.

Observation layout and motor constants: third_party/sonic/gear_sonic_deploy/
src/g1/g1_deploy_onnx_ref/{src/g1_deploy_onnx_ref.cpp,include/policy_parameters.hpp}.
No DDS, planner, or real-time wall-clock scheduling is needed for offline data.
"""
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from data_pipeline.robot import G1_JOINT_NAMES
from .quality import resample_motion_at_times
from data_pipeline.storage.store import digest, fingerprint

# IsaacLab indices expressed in MSC / MuJoCo joint order, as in the upstream
# policy_parameters.hpp. Reference and measured states use the inverse mapping.
TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
TO_ISAAC = np.argsort(TO_MUJOCO)
DEFAULT = np.array([-.312, 0, 0, .669, -.363, 0] * 2 + [0, 0, 0] +
                   [.2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0])
ARMATURE = np.array([.025101925, .025101925, .010177520, .025101925,
                     .003609725, .003609725] * 2 +
                    [.010177520, .003609725, .003609725] +
                    [.003609725] * 5 + [.00425] * 2 + [.003609725] * 5 + [.00425] * 2)
EFFORT = np.array([139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25] +
                  [25] * 5 + [5] * 2 + [25] * 5 + [5] * 2)
STIFFNESS = ARMATURE * (20 * np.pi) ** 2
ACTION_SCALE = .25 * EFFORT / STIFFNESS
KP = STIFFNESS.copy()
KD = 4 * ARMATURE * 20 * np.pi
KP[[4, 5, 10, 11, 13, 14]] *= 2
KD[[4, 5, 10, 11, 13, 14]] *= 2

ENCODER_DIMS = dict(encoder_mode_4=4, motion_joint_positions_10frame_step5=290,
    motion_joint_velocities_10frame_step5=290, motion_anchor_orientation_heading_10frame_step5=60,
    motion_anchor_orientation_heading=6, motion_joint_positions_lowerbody_10frame_step5=120,
    motion_joint_velocities_lowerbody_10frame_step5=120, vr_3point_local_target=9,
    vr_3point_local_orn_target=12, smpl_joints_10frame_step1=720,
    smpl_anchor_orientation_heading_10frame_step1=60, motion_joint_positions_wrists_10frame_step1=60)

_shared_policy = None


def version(execution):
    root = Path(execution['sonic_root'])
    source = root / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
    scene = root / 'gear_sonic/data/robot_model/model_data/g1'
    files = [Path(__file__), source, root / 'gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml',
             *sorted(scene.rglob('*.xml')),
             *(Path(execution[key]) for key in ('encoder', 'decoder', 'observation_config'))]
    if execution.get('inference_batch_size', 1) > 1:
        files.append(Path(__file__).with_name('inference.py'))
    return 'sonic:' + fingerprint({str(path): digest(path) for path in files})


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def heading(quat):
    matrix = rotation(quat).as_matrix()
    return Rotation.from_euler('z', np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))


def sample_reference(reference):
    """Cover the original timestamps; hold the last pose for <20 ms if needed."""
    duration = (len(reference) - 1) / 30
    times = np.arange(int(np.ceil(duration * 50 - 1e-9)) + 1) / 50
    values = resample_motion_at_times(reference, np.arange(len(reference)) / 30,
                                    np.minimum(times, duration))
    return values, times


def state_handles(model):
    """Map the 29 body joints even when the official XML also has hand joints."""
    joints = np.array([model.joint(name).id for name in G1_JOINT_NAMES])
    qpos = np.r_[np.arange(7), model.jnt_qposadr[joints]]
    qvel = model.jnt_dofadr[joints]
    actuators = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in joints])
    return qpos, qvel, actuators


def set_qpos(model, state, qpos, handles):
    state.qpos[handles[0]] = qpos


def encoder_observation(reference, velocity, index, base_quat, config):
    future = np.minimum(index + np.arange(10) * 5, len(reference) - 1)
    # Both initial headings coincide because the simulator starts at reference[0].
    relative = heading(base_quat).inv() * rotation(reference[future, 3:7])
    values = dict(encoder_mode_4=np.zeros(4),
                  motion_joint_positions_10frame_step5=reference[future, 7:][:, TO_ISAAC].reshape(-1),
                  motion_joint_velocities_10frame_step5=velocity[future][:, TO_ISAAC].reshape(-1),
                  motion_anchor_orientation_heading_10frame_step5=relative.as_matrix()[..., :2].reshape(-1))
    required = next(mode['required_observations'] for mode in config['encoder_modes'] if mode['mode_id'] == 0)
    return np.concatenate([values[item['name']] if item['name'] in required else np.zeros(ENCODER_DIMS[item['name']])
                           for item in config['encoder_observations'] if item.get('enabled', True)]).astype(np.float32)


def decoder_observation(token, history, observations):
    values = dict(token_state=token.reshape(-1))
    for name, column in [('his_base_angular_velocity_10frame_step1', 0),
                         ('his_body_joint_positions_10frame_step1', 1),
                         ('his_body_joint_velocities_10frame_step1', 2),
                         ('his_last_actions_10frame_step1', 3),
                         ('his_gravity_dir_10frame_step1', 4)]:
        values[name] = np.concatenate([entry[column] for entry in history])
    return np.concatenate([values[item['name']] for item in observations if item.get('enabled', True)]).astype(np.float32)


@lru_cache(maxsize=2)
def load_policy(encoder, decoder, observation_config, providers):
    if _shared_policy is not None:
        return _shared_policy
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    providers = [(name, {'use_tf32': False}) if name == 'CUDAExecutionProvider' else name
                 for name in providers.split(',')]
    sessions = [ort.InferenceSession(path, sess_options=options, providers=providers)
                for path in (encoder, decoder)]
    return *sessions, yaml.safe_load(Path(observation_config).read_text())


@lru_cache(maxsize=2)
def load_simulator(root):
    """Reuse immutable robot assets; every motion still gets fresh MjData."""
    import mujoco
    root = Path(root)
    config = yaml.safe_load((root / 'gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml').read_text())
    xml = (root / config['ROBOT_SCENE']).resolve()
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = config['SIMULATE_DT']
    return model, state_handles(model), xml


def run(reference, execution, rollout):
    """Run synchronous 50 Hz control and record post-control physical states."""
    import mujoco
    encoder, decoder, config = load_policy(execution['encoder'], execution['decoder'],
        execution['observation_config'], execution.get('providers', 'CPUExecutionProvider'))
    model, handles, xml = load_simulator(execution['sonic_root'])
    substeps = int(round(.02 / model.opt.timestep))
    state = mujoco.MjData(model)
    qi, vi, ai = handles
    ref, times = sample_reference(reference)
    velocity = np.gradient(ref[:, 7:], .02, axis=0)
    set_qpos(model, state, ref[0], handles)
    # Initial velocity is part of the reference state, not a warm-up label.
    next_qpos = state.qpos.copy()
    next_qpos[qi] = ref[1]
    if len(ref) > 1:
        mujoco.mj_differentiatePos(model, state.qvel, .02, state.qpos, next_qpos)
    mujoco.mj_forward(model, state)
    last_action = np.zeros(29)

    def snapshot():
        quat = state.qpos[3:7]
        return (state.qvel[3:6].copy(), (state.qpos[qi[7:]] - DEFAULT)[TO_ISAAC],
                state.qvel[vi][TO_ISAAC].copy(), last_action.copy(), rotation(quat).inv().apply([0, 0, -1]))

    history = deque([snapshot() for _ in range(10)], maxlen=10)
    executed = []
    for index in range(len(times) - 1):
        obs = encoder_observation(ref, velocity, index, state.qpos[3:7], config['encoder'])
        token = encoder.run(None, {encoder.get_inputs()[0].name: obs[None]})[0]
        obs = decoder_observation(token, history, config['observations'])
        action = decoder.run(None, {decoder.get_inputs()[0].name: obs[None]})[0].reshape(29)
        target = DEFAULT + action[TO_MUJOCO] * ACTION_SCALE
        for _ in range(substeps):
            torque = KP * (target - state.qpos[qi[7:]]) - KD * state.qvel[vi]
            state.ctrl[ai] = np.clip(torque, model.actuator_ctrlrange[ai, 0], model.actuator_ctrlrange[ai, 1])
            mujoco.mj_step(model, state)
        last_action = action
        history.append(snapshot())
        executed.append(state.qpos[qi].copy())
    np.savez_compressed(rollout, executed_qpos_36=np.asarray(executed, dtype=np.float32),
                        reference_qpos_36=ref[:-1], timestamps=times[1:], fps=np.float32(50), robot_xml=str(xml))
