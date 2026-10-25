"""Render complete, native-duration GT/reference/(optional) SONIC comparisons."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def wrap_caption(text, font, width):
    """Wrap English at spaces and CJK/long words at character boundaries."""
    lines, line = [], ''
    for character in text.strip():
        if character == '\n':
            lines.append(line.rstrip())
            line = ''
        elif font.getlength(line + character) <= width:
            line += character
        else:
            split = line.rfind(' ')
            if split > 0:
                lines.append(line[:split].rstrip())
                line = line[split + 1:] + character
            else:
                lines.append(line.rstrip())
                line = character
    if line:
        lines.append(line.rstrip())
    return lines or ['']


def caption_panels(captions, width, scale, font_path=None):
    from PIL import Image, ImageDraw, ImageFont
    cjk = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font = ImageFont.truetype(font_path or (str(cjk) if cjk.exists() else 'DejaVuSans.ttf'),
                              max(16, round(20 * scale)))
    margin, line_height = max(12, round(14 * scale)), max(24, round(30 * scale))
    wrapped = {text: wrap_caption(text, font, width - 2 * margin) for text in dict.fromkeys(captions)}
    height = 2 * margin + (max(map(len, wrapped.values())) + 1) * line_height
    height += height % 2
    panels = {}
    for text, lines in wrapped.items():
        panel = Image.new('RGB', (width, height), (20, 25, 32))
        draw = ImageDraw.Draw(panel)
        draw.text((margin, margin), 'TEXT CONDITION:', font=font, fill=(150, 205, 255))
        for index, line in enumerate(lines, 1):
            draw.text((margin, margin + index * line_height), line, font=font, fill=(245, 245, 245))
        panels[text] = np.asarray(panel)[..., ::-1].copy()  # BGR, like the scene panels
    return panels, height


def render_comparison(arrays, output, settings, *, rtc, task, panel_size=512,
                      captions=None, caption_font=None):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import cv2
    import mujoco
    from data_pipeline.motion.sonic import load_simulator, set_qpos
    from data_pipeline.video.media import encoder
    cv2.setNumThreads(1)
    model, handles, _ = load_simulator(settings['sonic_root'])
    model.vis.global_.offwidth = model.vis.global_.offheight = panel_size
    model.vis.quality.offsamples = 0
    names = ['target', 'predicted'] + (['executed'] if 'executed' in arrays else [])
    titles = ['GROUND TRUTH', 'MSC | GENERATED REFERENCE', 'SONIC | ACTUAL EXECUTION']
    trajectories = [arrays[name] for name in names]
    frames = len(arrays['target'])
    states = [mujoco.MjData(model) for _ in names]
    cameras = []
    for values in trajectories:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth, camera.elevation, camera.distance = 135., -18., 3.1
        camera.lookat[:] = [values[0, 0], values[0, 1], .75]
        cameras.append(camera)
    renderer = mujoco.Renderer(model, height=panel_size, width=panel_size)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    output = Path(output)
    temporary = output.with_name('.' + output.stem + '.partial.mp4')
    width = panel_size * len(names)
    scale = panel_size / 512
    caption_images, caption_height = caption_panels(captions, width, scale, caption_font) if captions else ({}, 0)
    try:
        with encoder(temporary, width, panel_size + caption_height, 30, crf=20) as stream:
            for frame in range(frames):
                panels = []
                for values, state, camera in zip(trajectories, states, cameras):
                    set_qpos(model, state, values[frame], handles)
                    mujoco.mj_forward(model, state)
                    camera.lookat[:2] = .95 * camera.lookat[:2] + .05 * values[frame, :2]
                    renderer.update_scene(state, camera=camera)
                    panels.append(renderer.render().copy())
                image = cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR)
                cv2.rectangle(image, (0, 0), (width, round(64 * scale)), (20, 25, 32), -1)
                for index, title in enumerate(titles[:len(names)]):
                    x = index * panel_size + round(10 * scale)
                    cv2.putText(image, title, (x, round(23 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                                .58 * scale, (230, 230, 230), 1, cv2.LINE_AA)
                    phase = 'INITIAL GT HISTORY' if frame < 5 else 'CONTINUOUS GENERATION'
                    text = f'{(frame + 1) / 30:.2f}s / {frames / 30:.2f}s | {phase}'
                    cv2.putText(image, text, (x, round(47 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                                .40 * scale, (170, 220, 255), 1, cv2.LINE_AA)
                footer = f'{task} | RTC={rtc} | 30 FPS, native duration | first 5 frames initialized from GT'
                cv2.putText(image, footer, (round(10 * scale), panel_size - round(12 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, .47 * scale, (245, 245, 245), 1, cv2.LINE_AA)
                if captions:
                    image = np.concatenate((image, caption_images[captions[frame]]), axis=0)
                stream.write(image.tobytes())
                if frame == frames // 2:
                    cv2.imwrite(str(output.with_suffix('.jpg')), image)
    finally:
        renderer.close()
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,avg_frame_rate,duration', '-of', 'json', str(temporary)]))['streams'][0]
    if int(probe['nb_frames']) != frames or probe['avg_frame_rate'] != '30/1':
        raise ValueError('Comparison video frame count/FPS does not match GT')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(temporary),
                    '-threads', '1', '-f', 'null', '-'], check=True)
    temporary.replace(output)
    return probe
