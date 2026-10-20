"""Discover motions independently of camera availability; preserve raw inputs."""
import json
from pathlib import Path
import zipfile
import wave

from data_pipeline.schemas import Inventory, SequenceRecord, AssetRecord
from data_pipeline.io import sha256_file
from data_pipeline.schemas.records import SourceMotion
from dataclasses import asdict


def discover(root, out, *, cameras=('c01',), limit=None, dry_run=False):
    root = Path(root).resolve()
    if Path(out).resolve().is_relative_to(root):
        raise ValueError('Inventory output must be outside raw source root')
    if (root / 'edge_aistpp/motions').is_dir():
        root = root / 'edge_aistpp'
    archive = root / 'archives/motions_complete.zip'
    ignore = root / 'archives/ignore_list.txt'
    ignored = set(ignore.read_text().split()) if ignore.is_file() else set()
    if (root / 'motions').is_dir():
        members = sorted(p.relative_to(root).as_posix() for p in (root / 'motions').glob('*.pkl') if p.stem not in ignored)
        archive = None
    else:
        with zipfile.ZipFile(archive) as z:
            members = sorted(n for n in z.namelist() if n.startswith('motions/') and n.endswith('.pkl') and Path(n).stem not in ignored)
    if not members:
        raise ValueError(f'No AIST SMPL motion files found under {root}')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        members = members[:limit]
    inv = Inventory(external_roots={'aistpp': str(root)}, provenance={'adapter': 'aistpp_v1',
        'split_policy': 'per_source_sequence', 'ignore_list_present': ignore.is_file()})
    raw = []
    for member in members:
        name = Path(member).stem
        sid = f'aistpp:{name}'
        uri = 'root://aistpp/archives/motions_complete.zip' if archive else f'root://aistpp/{member}'
        motion_member = member if archive else ''
        inv.sequences.append(SequenceRecord(sid, 'aistpp', name, f'aistpp:group:sequence:{name}',
            source_provenance={'motion_uri': uri, 'member': motion_member,
                               'license': 'upstream AIST++ terms; not redistributed', 'adapter': 'aistpp_v1'}))
        raw.append(SourceMotion(f'{sid}:smpl', sid, uri, 'aist_smpl', 60., member=motion_member))
        audio = root / 'wavs' / f'{name}.wav'
        if audio.is_file():
            with wave.open(str(audio)) as wav:
                sample_rate = wav.getframerate()
                duration = wav.getnframes() / sample_rate
            # These are per-sequence excerpts, not necessarily one copy of the full song.
            inv.assets.append(AssetRecord(f'{sid}:audio', 'audio', f'root://aistpp/wavs/{audio.name}',
                f'aistpp:excerpt:{name}', sha256_file(audio), audio.stat().st_size,
                'AIST++ / EDGE upstream terms', duration, sample_rate_hz=sample_rate))
            # Matching names/durations alone do not establish a verified offset.
        for camera in cameras:
            video = root / 'videos' / (name.replace('_cAll_', f'_{camera}_') + '.mp4')
            if video.is_file():
                inv.assets.append(AssetRecord(f'{sid}:{camera}', 'video', f'root://aistpp/videos/{video.name}',
                    f'aistpp:{video.stem}', sha256_file(video), video.stat().st_size,
                    'AIST++ upstream terms', camera_id=camera))
                # No link is fabricated before source offset/coverage is established.
    inv.validate()
    if not dry_run:
        inv.write(out)
        (Path(out) / 'meta/source_motions.json').write_text(json.dumps([asdict(r) for r in raw], indent=2))
    return {'sequences': len(inv.sequences), 'source_motions': len(raw), 'videos': sum(a.modality == 'video' for a in inv.assets),
            'audios': sum(a.modality == 'audio' for a in inv.assets),
            'split_groups': sorted({s.split_group_id for s in inv.sequences}), 'output': str(out)}


def decode_motion(source, motion_path, settings):
    """AIST SMPL60 → world-space joints30; no dataset-specific IK solver."""
    import inspect
    import pickle
    import numpy as np
    import torch
    import smplx
    from scipy.spatial.transform import Rotation
    from data_pipeline.stages.retarget import HumanMotion

    if source.fps != 60:
        raise ValueError('AIST SMPL requires 60 Hz source annotations')
    gender = settings['gender']
    if gender != 'male' and not settings.get('allow_nonofficial_model', False):
        raise ValueError('AIST uses male SMPL; other models require allow_nonofficial_model')
    # 1. Source可以指定ZIP+member，也可以直接指定PKL；不需要解压整个动作包。
    if motion_path.suffix.lower() == '.zip':
        with zipfile.ZipFile(motion_path) as archive:
            raw = archive.read(source.member)
    else:
        raw = motion_path.read_bytes()
    payload = pickle.loads(raw)
    ignore = motion_path.parent.parent / 'archives/ignore_list.txt'
    if ignore.is_file() and Path(source.member or motion_path).stem in ignore.read_text().split():
        raise ValueError('Sequence is in the official ignore list')
    # SMPL每帧24个局部轴角旋转；AIST标注60Hz，取0,2,4...生成30Hz参考。
    pose = np.asarray(payload['smpl_poses']).reshape(-1, 24, 3)
    trans = np.asarray(payload['smpl_trans'])
    scale = float(np.asarray(payload['smpl_scaling']).reshape(-1)[0])
    indices = np.arange(0, len(pose), 2)
    if not np.isfinite(pose).all() or not np.isfinite(trans).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid original SMPL values')
    # 2. 用SMPL前向运动学恢复人体位置；不能把局部轴角直接当成全局关节朝向。
    # 官方AIST使用male；neutral只有显式允许时才作为当前原型近似。
    aliases = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
               'object': object, 'unicode': str, 'str': str}
    added = [name for name in aliases if name not in np.__dict__]
    old_getargspec = getattr(inspect, 'getargspec', None)
    try:
        for name in added:
            setattr(np, name, aliases[name])
        if old_getargspec is None:
            inspect.getargspec = inspect.getfullargspec
        model = smplx.SMPL(model_path=str(Path(settings['smpl_model']).resolve()), gender=gender)
    finally:
        for name in added:
            delattr(np, name)
        if old_getargspec is None:
            delattr(inspect, 'getargspec')
    with torch.no_grad():
        neutral = model(return_verts=True)
        height = float(neutral.vertices[0, :, 1].max() - neutral.vertices[0, :, 1].min())
        joints = []
        for start in range(0, len(indices), 64):
            selected = indices[start:start + 64]
            # AIST拟合的平移带全局scale；标准SMPL米制空间要用trans/scale。
            result = model(global_orient=torch.from_numpy(pose[selected, 0]).float(), body_pose=torch.from_numpy(pose[selected, 1:].reshape(-1, 69)).float(), transl=torch.from_numpy(trans[selected] / scale).float(), return_verts=False)
            joints.append(result.joints[:, :24].numpy())
    positions = np.concatenate(joints)
    # 使用原生SMPL关节名；SMPL-X在腕部之后的索引不同，不能直接照搬完整列表。
    names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
    # 3. AIST世界坐标Y向上，机器人Z向上；Rx(+90°)把(x,y,z)变成(x,-z,y)。
    correction = Rotation.from_euler('x', 90, degrees=True)
    positions = correction.apply(positions.reshape(-1, 3)).reshape(positions.shape)
    # 只把首帧骨盆水平位置设为原点，不删帧、不改时间；逆平移信息写入report。
    horizontal_origin = positions[0, 0].copy()
    horizontal_origin[2] = 0
    positions -= horizontal_origin
    parents = model.parents.numpy()
    local_rot = Rotation.from_rotvec(pose[indices].reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    global_rot = np.empty_like(local_rot)
    # 沿父子链累乘：R_world(child)=R_world(parent)@R_local(child)。
    # 坐标系修正只左乘全局旋转，不对每个局部关节重复施加世界旋转。
    for joint in range(24):
        global_rot[:, joint] = local_rot[:, joint] if parents[joint] < 0 else global_rot[:, parents[joint]] @ local_rot[:, joint]
    global_rot = correction.as_matrix() @ global_rot
    orientations = Rotation.from_matrix(global_rot.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(-1, 24, 4)
    return HumanMotion(tuple(names), positions, orientations, indices / 60, height, 'smplx',
        {'source_frames': len(pose), 'source_scaling': scale, 'smpl_gender': gender,
         'world_rotation': correction.as_matrix().tolist(),
         'source_normalized_horizontal_origin': horizontal_origin.tolist()})
