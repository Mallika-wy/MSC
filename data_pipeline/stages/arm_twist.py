"""G1同侧、同时、持续肘腕组合筛查

这是启发式候选指标，不是人体解剖定律；不单独用负肘角或贴限位判错。
不依赖数据来源、文本、FK或GPU，所有角度按G1自身关节轴定义。
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from omg.robots.g1.constants import G1_JOINT_NAMES


RULE_VERSION = 'g1_same_arm_twist_expanded_v2'


@dataclass(frozen=True)
class ArmTwistConfig:
    near_limit_deg: float = 5.0
    elbow_reverse_deg: float = 40.0
    wrist_bend_deg: float = 45.0
    duration_seconds: float = .3

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Arm-twist thresholds must be finite and positive')


@lru_cache(maxsize=1)
def g1_joint_limits():
    """读取拟合指标所用的G1 URDF限位，不用人类肘腕范围代替机器人限位。"""
    path = Path(__file__).resolve().parents[2] / 'assets/robots/g1/g1_29dof.urdf'
    joints = {joint.attrib['name']: joint for joint in ET.parse(path).getroot().findall('joint')}
    lower = np.array([float(joints[name].find('limit').attrib['lower']) for name in G1_JOINT_NAMES])
    upper = np.array([float(joints[name].find('limit').attrib['upper']) for name in G1_JOINT_NAMES])
    lower.setflags(write=False)
    upper.setflags(write=False)
    return lower, upper


def screen_arm_twist(qpos, lengths=None, *, fps=30., joint_order=G1_JOINT_NAMES,
                     lower_rad=None, upper_rad=None, config=ArmTwistConfig()):
    """返回命中的episode及同侧异常区间，帧区间为[start,end)，不能跨episode凑时长。

    只保留扩大档：同臂至少2关节距限位≤5°，且肘≤−40°或双腕轴均≥45°。
    两个分支分别计连续帧，不将短异常拼接。
    超出限位也算饱和，保持原拟合实现的有符号距离语义和float32角度转换。
    """
    qpos = np.asarray(qpos)
    lengths = np.asarray([len(qpos)] if lengths is None else lengths)
    if (lengths.ndim != 1 or not len(lengths) or not np.issubdtype(lengths.dtype, np.integer)
            or (lengths <= 0).any() or qpos.shape != (int(lengths.sum()), 36)
            or len(joint_order) != 29 or len(set(joint_order)) != 29):
        raise ValueError('Invalid G1 motion/episode layout')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid FPS')
    if (lower_rad is None) != (upper_rad is None):
        raise ValueError('Supply both joint-limit arrays or neither')
    if lower_rad is None:
        if tuple(joint_order) != tuple(G1_JOINT_NAMES):
            raise ValueError('Custom joint order requires matching explicit bounds')
        lower_rad, upper_rad = g1_joint_limits()
    lower_rad, upper_rad = np.asarray(lower_rad), np.asarray(upper_rad)
    if (lower_rad.shape != (29,) or upper_rad.shape != (29,)
            or not np.isfinite([lower_rad, upper_rad]).all() or (lower_rad > upper_rad).any()):
        raise ValueError('Invalid 29D joint bounds')
    offsets = np.r_[0, np.cumsum(lengths)]
    minimum = max(1, int(math.ceil(config.duration_seconds * fps - 1e-9)))
    index = {name: i for i, name in enumerate(joint_order)}
    angles = np.rad2deg(np.asarray(qpos[:, 7:], dtype=np.float32))
    low, high = np.rad2deg(lower_rad), np.rad2deg(upper_rad)
    valid = np.isfinite(qpos).all(axis=1)
    first, last = np.zeros(len(qpos), bool), np.zeros(len(qpos), bool)
    first[offsets[:-1]], last[offsets[1:] - 1] = True, True
    records = {}
    for side in ('left', 'right'):
        parts = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')
        cols = [index[f'{side}_{part}_joint'] for part in parts]
        elbow, pitch, yaw = (angles[:, cols[i]] for i in (3, 5, 6))
        near = np.minimum(angles[:, cols] - low[cols], high[cols] - angles[:, cols]) <= config.near_limit_deg
        pair = near.sum(axis=1) >= 2
        wrist = (abs(pitch) >= config.wrist_bend_deg) & (abs(yaw) >= config.wrist_bend_deg)
        masks = {'limit_pair_and_reverse_elbow': pair & (elbow <= -config.elbow_reverse_deg),
                 'limit_pair_and_double_wrist': pair & wrist}
        for rule, predicate in masks.items():
            mask = predicate & valid
            starts = np.flatnonzero(mask & (~np.r_[False, mask[:-1]] | first))
            ends = np.flatnonzero(mask & (~np.r_[mask[1:], False] | last)) + 1
            for start, end in zip(starts, ends):
                if end - start < minimum:
                    continue
                episode = int(np.searchsorted(offsets, start, side='right') - 1)
                event = dict(rule=rule, side=side, start_frame=int(start - offsets[episode]),
                             end_frame=int(end - offsets[episode]), duration_seconds=float((end-start)/fps),
                             elbow_angle_at_start_deg=float(elbow[start]), wrist_pitch_at_start_deg=float(pitch[start]),
                             wrist_yaw_at_start_deg=float(yaw[start]),
                             near_limit_joints_at_start=[joint_order[c] for c, flag in zip(cols, near[start]) if flag])
                record = records.setdefault(episode, dict(local_episode_index=episode, priority='review', events=[]))
                record['events'].append(event)
    for record in records.values():
        # 与原实现保持一致：每侧、每个谓词保留最长3段，控制报告体积。
        events = record['events']
        record['events'] = [event for side in ('left', 'right') for rule in masks
                            for event in sorted((e for e in events if e['side'] == side and e['rule'] == rule),
                                                key=lambda e: -e['duration_seconds'])[:3]]
    return [records[k] for k in sorted(records)]


def screen_reference(path: Path, policy):
    """重定向结束立刻检查参考动作，命中后不再执行控制器或处理视频。

    每次读取参考缓存重新判断，不依赖历史质检结果，也不加载控制器。
    """
    options = dict(policy.get('arm_twist', {}))
    enabled = options.pop('enabled', True)
    config = ArmTwistConfig(**options)
    result = {'version': RULE_VERSION, 'enabled': enabled, 'thresholds': asdict(config), 'events': []}
    if enabled:
        with np.load(path, allow_pickle=False) as data:
            fps = float(data['fps'])
            hits = screen_arm_twist(data['qpos_36'], fps=fps, config=config)
            times = data['source_timestamps']
            for record in hits:
                for event in record['events']:
                    event['sequence_start_s'] = float(times[event['start_frame']])
                    event['sequence_end_s'] = float(times[event['end_frame']-1]+1/fps)
            result.update(events=hits, fps=fps)
    reasons = ['arm_twist_reference'] if result['events'] else []
    return {'stage': 'retarget', 'accepted': not reasons, 'rejection_reasons': reasons, 'arm_twist': result}
