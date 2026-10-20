"""Packaged G1 identity; shared with the training codec."""
import hashlib
import json
from importlib.resources import files

G1_ASSETS = files('msc').joinpath('assets/robots/g1')
G1_KINEMATICS = G1_ASSETS.joinpath('g1_kinematics.json')
G1_URDF = G1_ASSETS.joinpath('g1_29dof.urdf')
G1_JOINT_NAMES = tuple(json.loads(G1_KINEMATICS.read_text())['joint_order'])


def robot_contract():
    return dict(name='g1_29dof', units='m_rad', up_axis='z', quaternion='wxyz',
                joint_order=list(G1_JOINT_NAMES),
                kinematics_sha256=hashlib.sha256(G1_KINEMATICS.read_bytes()).hexdigest(),
                urdf_sha256=hashlib.sha256(G1_URDF.read_bytes()).hexdigest())
