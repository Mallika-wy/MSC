"""把已保存的tracker执行轨迹回放成视频；不重新仿真、不修改动作或质检结论。"""
from pathlib import Path

import numpy as np

from data_pipeline.components.person_video.media import encoder, verify_encoded
from data_pipeline.stages.motion import validate_qpos


def prepare_execution_video(directory: Path, force=False) -> Path:
    """逐帧回放原生控制频率下的执行状态，包含失败/倒地过程。

    execution.mp4是诊断视频，512×512，帧率取rollout（当前50Hz）；不是30Hz训练RGB。
    第i帧对应仿真后的(i+1)/fps时刻，不补参考首帧、不把参考动作冒充执行结果。
    """
    import cv2
    import mujoco
    from omg.tracking.holomotion.runtime import build_g1_state_handles, set_g1_qpos

    directory = Path(directory)
    rollout, output = directory / 'tracker_rollout.npz', directory / 'execution.mp4'
    with np.load(rollout, allow_pickle=False) as data:
        qpos = data['executed_qpos_36'].copy()
        fps = float(data['fps'].item())
        robot_xml = str(data['robot_xml'].item())
    validate_qpos(qpos, (np.arange(len(qpos))+1)/fps)
    if output.is_file() and not force and output.stat().st_mtime_ns >= rollout.stat().st_mtime_ns:
        verify_encoded(output, len(qpos), fps)
        return output

    # 加载本条执行时使用的同一机器人场景，恢复同一关节顺序；只做正向几何更新。
    model = mujoco.MjModel.from_xml_path(robot_xml)
    model.vis.global_.offwidth = model.vis.global_.offheight = 512
    state = mujoco.MjData(model)
    handles = build_g1_state_handles(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth, camera.elevation, camera.distance = 135., -20., 2.8
    renderer = mujoco.Renderer(model, height=512, width=512)
    temporary = directory / '.execution.partial.mp4'
    try:
        with encoder(temporary, 512, 512, fps, crf=22) as stream:
            for index, frame in enumerate(qpos):
                set_g1_qpos(model, state, handles, frame)
                mujoco.mj_forward(model, state)  # 不调用mj_step，不能再次推进物理状态。
                # 只跟随水平位置，固定视线高度，倒地时仍能看清地面与机器人。
                camera.lookat[:] = [frame[0], frame[1], .65]
                renderer.update_scene(state, camera=camera)
                image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                cv2.putText(image, f'Executed | t={(index+1)/fps:.2f}s', (12, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
                stream.write(image.tobytes())
    finally:
        renderer.close()
    verify_encoded(temporary, len(qpos), fps, decode=True)
    temporary.replace(output)
    print(f'  execution-video: {output}', flush=True)
    return output
