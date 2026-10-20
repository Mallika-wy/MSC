"""Render the saved physical tracker rollout for human review."""
from pathlib import Path

import numpy as np

from data_pipeline.video.media import encoder, verify_encoded


def save_execution_video(rollout: Path) -> Path:
    """Replay post-control states at native FPS; never advance the simulator."""
    import mujoco
    from omg.tracking.holomotion.runtime import build_g1_state_handles, set_g1_qpos

    rollout = Path(rollout)
    video = rollout.parent / 'execution.mp4'
    with np.load(rollout, allow_pickle=False) as data:
        qpos = data['executed_qpos_36'].copy()
        fps = float(data['fps'].item())
        xml = str(data['robot_xml'].item())
    if qpos.ndim != 2 or qpos.shape[1] != 36 or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError('Invalid tracker rollout for video')
    if video.is_file() and video.stat().st_mtime_ns >= rollout.stat().st_mtime_ns:
        verify_encoded(video, len(qpos), fps)
        return video

    model = mujoco.MjModel.from_xml_path(xml)
    model.vis.global_.offwidth = model.vis.global_.offheight = 512
    model.vis.quality.offsamples = 0
    state = mujoco.MjData(model)
    handles = build_g1_state_handles(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth, camera.elevation, camera.distance = 135., -20., 2.8
    renderer = mujoco.Renderer(model, height=512, width=512)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    temporary = rollout.parent / '.execution.partial.mp4'
    try:
        with encoder(temporary, 512, 512, fps) as stream:
            for frame in qpos:
                set_g1_qpos(model, state, handles, frame)
                mujoco.mj_forward(model, state)
                camera.lookat[:] = [frame[0], frame[1], .65]
                renderer.update_scene(state, camera=camera)
                stream.write(np.ascontiguousarray(renderer.render()[:, :, ::-1]).tobytes())
    finally:
        renderer.close()
    verify_encoded(temporary, len(qpos), fps, decode=True)
    temporary.replace(video)
    return video
