import argparse
from pathlib import Path
import torch
from msc.config import Config
from msc.factory import build_model, seed_everything
from msc.training.checkpoint import load_checkpoint


def load_model(checkpoint, device="cpu", t5_path=None):
    obj = load_checkpoint(checkpoint)
    cfg = Config.from_dict(obj["config"])
    if t5_path:
        cfg.model.t5_path = t5_path
    # The actual normalization buffers travel in the checkpoint. Sampling does
    # not need the original custom statistics file to remain on disk.
    cfg.model.stats_path = None
    cfg.model.dynamic_stats_path = None
    model = build_model(cfg, obj["metadata"]["audio"],obj['metadata'].get('features'))
    if model.text_encoder and model.text_encoder.fingerprint != obj["metadata"]["text_model_sha256"]:
        raise ValueError("Local T5 does not match checkpoint")
    model.load_state_dict(obj["model"])
    model.representation.stats_sha256 = obj["metadata"]["representation"]["stats_sha256"]
    model.dynamic_stats_sha256 = obj['metadata'].get('dynamic_stats_sha256')
    return model.to(device).eval(), cfg, obj["metadata"]


def main():
    p = argparse.ArgumentParser(description="Generate canonical 125D motion (local files only)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--history", help="Tensor .pt [B,5,125]: four history frames plus current state")
    p.add_argument("--text")
    p.add_argument("--audio", help=".pt dict: features, mask, metadata matching the checkpoint")
    p.add_argument('--video',help='.pt dict: tokens, mask, intervals, feature metadata')
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--previous-motion", help="Tensor .pt [B,30,125] from the previous generated chunk")
    p.add_argument("--rtc-mode", choices=("off", "train", "test_time"), default="off")
    p.add_argument("--rtc-delay", type=int, default=0, help="Executed steps before this replan result arrives")
    p.add_argument("--rtc-execution-horizon", type=int, help="Free tail steps for test-time RTC")
    p.add_argument("--rtc-mask-schedule", choices=("exponential", "linear", "hard", "simple"), default="exponential")
    p.add_argument("--rtc-guidance-alpha", type=float, default=0.9)
    p.add_argument("--device", default="cpu")
    p.add_argument("--t5-path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=".runs/sample.pt")
    args = p.parse_args()
    if not args.history:
        p.error("--history is required")
    torch.set_num_threads(2)
    seed_everything(args.seed)
    model, cfg, metadata = load_model(args.checkpoint, args.device, args.t5_path)
    history = torch.load(args.history, map_location="cpu", weights_only=True)
    if bool(args.previous_motion) != (args.rtc_mode != "off"):
        p.error("--previous-motion is required exactly when --rtc-mode is not off")
    kwargs = {}
    if args.audio:
        audio = torch.load(args.audio, map_location=args.device, weights_only=True)
        kwargs = dict(audio=audio.get('tokens',audio.get('features')), audio_mask=audio["mask"], audio_times=audio['times'],audio_metadata=audio["metadata"])
    if args.video:
        video = torch.load(args.video,map_location=args.device,weights_only=True)
        if video['metadata'] != metadata.get('features'):
            raise ValueError('Video feature provenance differs from checkpoint')
        kwargs.update(video_tokens=video['tokens'],video_mask=video['mask'],video_intervals=video['intervals'])
    previous_motion = (torch.load(args.previous_motion, map_location="cpu", weights_only=True)
                       if args.previous_motion else None)
    motion = model.sample(history.to(args.device), text=args.text, num_steps=args.steps,
                          cfg_scale=args.cfg_scale, previous_motion=(previous_motion.to(args.device)
                          if previous_motion is not None else None), rtc_mode=args.rtc_mode,
                          inference_delay=args.rtc_delay, execution_horizon=args.rtc_execution_horizon,
                          rtc_mask_schedule=args.rtc_mask_schedule,
                          rtc_guidance_alpha=args.rtc_guidance_alpha, **kwargs)
    if not torch.isfinite(motion).all():
        raise FloatingPointError("Nonfinite sample")
    output = Path(args.output)
    if any(output.resolve().is_relative_to(Path(root).resolve()) for root in cfg.data.roots):
        p.error('Sample output must be outside read-only Release roots')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(motion=motion.cpu(), metadata=metadata, seed=args.seed), output)
    print(f"Saved {list(motion.shape)} canonical motion to {output}")


if __name__ == "__main__":
    main()
