"""Evaluate seeded, distinct motions for their complete GT duration, with optional RTC."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from msc.factory import build_dataset
from msc.sample import load_model
from msc.evaluation.rollout import select_motions, generate_full_motion, trajectory_metrics, motion_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', action='append', help='Release root; repeat for multiple datasets')
    parser.add_argument('--motion-manifest', help='Subset manifest, or none; otherwise inherit checkpoint subset')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--task', choices=['text', 'audio', 'video'], default='text')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-motions', type=int, default=10)
    parser.add_argument('--rtc', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--replan-frames', type=int, default=15)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--cfg-scale', type=float, default=1.)
    parser.add_argument('--rtc-alpha', type=float, default=.9)
    parser.add_argument('--rtc-mask', choices=['exponential', 'linear', 'hard', 'simple'], default='exponential')
    parser.add_argument('--feedback', choices=['generated', 'sonic'], default='generated')
    parser.add_argument('--sonic-config', default='configs/data_pipeline/omg_text.yaml')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    parser.add_argument('--t5-path')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--panel-size', type=int, default=512)
    parser.add_argument('--caption-font', help='Optional TTF/TTC font for caption subtitles')
    parser.add_argument('--output', default='.runs/msc-full-eval')
    args = parser.parse_args()
    if args.num_motions < 1 or args.inference_steps < 1 or not 1 <= args.replan_frames < 30:
        parser.error('Positive motion/step counts and 1 <= replan-frames < 30 are required')
    if args.feedback == 'sonic' and args.replan_frames % 3:
        parser.error('SONIC replanning must use a multiple of 3 frames to align 30 FPS and 50 Hz control')
    if args.panel_size < 256 or args.panel_size % 2:
        parser.error('panel-size must be an even integer >=256')
    return args


def configure_dataset(cfg, args):
    if args.data_root:
        cfg.data.roots = args.data_root
        # A different explicit dataset must not silently inherit the training subset.
        cfg.data.motion_manifest = None
    if args.motion_manifest is not None:
        cfg.data.motion_manifest = None if args.motion_manifest.lower() == 'none' else args.motion_manifest
    cfg.data.split = args.split
    cfg.data.tasks = [args.task]
    cfg.data.max_motions_per_task = None
    cfg.data.max_windows_per_motion = None
    return build_dataset(cfg)


def check_feature_contract(dataset, model, task):
    if task == 'text':
        return
    # Every used encoder/preprocessing descriptor must be known to the model.
    # Comparing only the requested modality allows evaluating one Release from
    # a checkpoint trained on several Releases without weakening provenance.
    def descriptors(metadata):
        return {json.dumps(row, sort_keys=True) for row in metadata.get('features', []) if row[0] == task}
    actual = descriptors(dataset.feature_metadata)
    expected = descriptors(model.feature_metadata or {})
    if not actual or not actual.issubset(expected):
        raise ValueError(f'{task} feature provenance differs from checkpoint')


def write_index(output, report):
    cards = []
    for row in report['results']:
        title = html.escape(row['motion']['motion_id'])
        text = ' | '.join(dict.fromkeys(c['condition'][2] for c in row['conditions'] if c['condition'][0] == 'text'))
        video = f"<video controls preload='metadata' width='100%' src='{row['video']}'></video>" if 'video' in row else ''
        cards.append(f"<article><h2>{row['number']:02d}: {title}</h2><p>{row['duration_s']:.3f}s, {row['frames']} frames</p><p>{html.escape(text)}</p>{video}</article>")
    document = """<!doctype html><meta charset='utf-8'><title>MSC full-motion evaluation</title>
<style>body{background:#151820;color:#ddd;font:17px system-ui;max-width:1550px;margin:30px auto}article{margin:40px 0}video{background:#000}</style>"""
    document += f"<h1>MSC full-motion evaluation</h1><p>RTC={report['rtc']}; feedback={report['feedback']}; seed={report['seed']}</p>"
    document += '<p>左：GT；中：生成参考；使用 SONIC feedback 时右侧为实际执行。视频与 GT 帧数完全一致，只有最初 5 帧共享真实历史。原速 30 FPS，无原音轨。误差指标不包含启动历史。</p>'
    document += '<p>离线同步推理，模拟推理延迟为 0；RTC 使用 test-time 引导。这不是在线实时性能测试。</p>'
    (output / 'index.html').write_text(document + ''.join(cards))


def main():
    args = parse_args()
    torch.set_num_threads(2)
    model, cfg, _ = load_model(args.checkpoint, args.device, args.t5_path)
    model.requires_grad_(False)
    dataset = configure_dataset(cfg, args)
    check_feature_contract(dataset, model, args.task)
    output = Path(args.output)
    if any(output.resolve().is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        raise ValueError('Evaluation output must be outside Release roots')
    output.mkdir(parents=True, exist_ok=True)
    cases = select_motions(dataset, args.num_motions, args.seed)
    selected = [dict(**case.record(), sampling_seed=motion_seed(args.seed, case)) for case in cases]
    (output / 'selection.json').write_text(json.dumps(dict(seed=args.seed, dataset_identity=dataset.identity,
        roots=cfg.data.roots, motion_manifest=cfg.data.motion_manifest, split=args.split,
        task=args.task, selection='Uniform distinct full motions with continuous condition coverage',
        motions=selected), ensure_ascii=False, indent=2))
    settings = None
    controller_factory = None
    if args.video or args.feedback == 'sonic':
        from data_pipeline.config import Settings
        settings = Settings.read(args.sonic_config).tracker
        settings['providers'] = 'CPUExecutionProvider'
    if args.feedback == 'sonic':
        from msc.evaluation.sonic import SonicFeedback
        controller_factory = lambda history: SonicFeedback(settings, history)
    report = dict(checkpoint=args.checkpoint, checkpoint_rtc_training=model.rtc, seed=args.seed,
        rtc=args.rtc, rtc_mode='test_time' if args.rtc else 'off', feedback=args.feedback,
        initialization_gt_frames=5, fps=30, text_max_tokens=cfg.model.text_max_length,
        options=vars(args), results=[],
        timing='Offline synchronous generation; inference_delay=0, not a real-time experiment',
        interpretation='Training samples are reconstruction diagnostics, not held-out generalization. Motion fidelity metrics do not establish semantic correctness.')
    for number, case in enumerate(cases, 1):
        directory = output / f'{number:02d}'
        directory.mkdir(exist_ok=True)
        print(json.dumps(dict(stage='motion_start', number=number, total=len(cases),
                              motion=case.motion.motion_id, frames=case.motion.num_frames)), flush=True)
        arrays, chunks = generate_full_motion(model, dataset, case, seed=args.seed, device=args.device,
            rtc=args.rtc, num_steps=args.inference_steps, cfg_scale=args.cfg_scale,
            replan_frames=args.replan_frames, guidance_alpha=args.rtc_alpha,
            mask_schedule=args.rtc_mask, precision=args.precision, controller_factory=controller_factory)
        np.savez_compressed(directory / 'trajectories.npz', **arrays)
        metrics = dict(reference=trajectory_metrics(arrays['predicted'], arrays['target']))
        held = np.broadcast_to(arrays['target'][4], arrays['target'].shape).copy()
        held[:5] = arrays['target'][:5]
        metrics['hold_last'] = trajectory_metrics(held, arrays['target'])
        if 'executed' in arrays:
            metrics['execution'] = trajectory_metrics(arrays['executed'], arrays['target'])
        row = dict(number=number, **selected[number - 1], generated_frames=len(arrays['target']) - 5,
                   chunks=chunks, metrics=metrics)
        if args.video:
            from msc.evaluation.render import render_comparison
            captions = None
            if case.task == 'text':
                captions = [chunks[0]['condition'][2]] * 5
                for chunk in chunks:
                    captions.extend([chunk['condition'][2]] * chunk['generated_frames'])
            row['video_probe'] = render_comparison(arrays, directory / 'comparison.mp4', settings,
                rtc=args.rtc, task=case.task, panel_size=args.panel_size,
                captions=captions, caption_font=args.caption_font)
            row['video'] = f'{number:02d}/comparison.mp4'
        (directory / 'metrics.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
        report['results'].append(row)
        (output / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        write_index(output, report)
        print(json.dumps(dict(completed=number, total=len(cases), motion=case.motion.motion_id,
                              frames=row['frames'], duration_s=row['duration_s'], metrics=metrics)), flush=True)


if __name__ == '__main__':
    main()
