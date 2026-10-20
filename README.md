# MSC

MSC is a small, independent PyTorch research baseline for multimodal G1 motion generation. It uses the official OMG 125D canonical motion contract and linear flow matching, with separate text and audio adapters.

The package does not import Psi0 or OMG at runtime, download model weights/data, implement video, or provide robot deployment. The included `_omg` files are the minimum source-derived closure needed for the official data adapter, representation and attention; see [THIRD_PARTY.md](THIRD_PARTY.md).

The native data pipeline refactor is documented in [MSC Dataset v1](docs/msc_dataset_v1.md),
Native releases support
optional modality assets; frozen audio/video extraction and unified training integration remain pending.

## Install

Recommended on this machine: reuse the existing OMG environment. No dependency installation is necessary for the verified commands:

```bash
source /home1/gmh/wangyang/motion/OMG/.venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m msc.train --config configs/smoke.yaml
```

The Python dependencies can also come from a compatible Psi0 environment (PyTorch 2.3+). This shares the environment only, not source imports. An independent installation was also verified; that temporary environment has been removed.

For a new machine, use Python 3.10+ and a PyTorch build appropriate for the hardware:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# Only for local frozen T5: pip install -e '.[text]'
# Only for a local official OMG release: pip install -e '.[data]'
```

No command downloads data or weights. Set paths to assets already on disk.

## Synthetic smoke training

```bash
PYTHONPATH=src python -m msc.train --config configs/smoke.yaml
PYTHONPATH=src python -m msc.sample --checkpoint .runs/smoke/last.pt --synthetic-history
```

For a convergence check:

```bash
PYTHONPATH=src python -m pytest -m slow -q
```

The smoke model is intentionally small. `configs/overfit.yaml` uses a larger model and verifies that a four-sample synthetic set can be fitted.

## Official OMG data

Point `configs/omg.yaml` at a local release and a local T5 directory:

```bash
PYTHONPATH=src python -m msc.train --config configs/omg.yaml \
  data.root=/path/to/OMG-Data model.t5_path=/path/to/t5-base-local \
  train.device=cuda:0
PYTHONPATH=src python -m msc.evaluate --checkpoint .runs/omg/last.pt \
  --data-root /path/to/OMG-Data --split val --t5-path /path/to/t5-base-local
```

The adapter expects the pinned `THU-MARS/OMG-Data` release metadata and manifest. It reads `omg.audio.feature` directly as `[T,35]` and preserves `omg.condition.has_audio`; the official metadata does not state whether those values came from `current35` or `aistpp_librosa35`, so the checkpoint records `feature_type: unknown` and never re-extracts audio.

## Sampling

A history tensor is a local `.pt` file shaped `[B,5,125]` in OMG canonical feature coordinates. It contains four preceding frames and the current frame; MSC predicts the following 30 frames:

```bash
PYTHONPATH=src python -m msc.sample \
  --checkpoint .runs/omg/last.pt \
  --history history.pt --steps 10 --output sample.pt \
  --t5-path /path/to/t5-base-local
```

Pass `--text "walk forward"` for text conditioning and `--audio audio.pt` for audio conditioning. Both flags can be used together.

### Real-time chunking

MSC includes the two Psi0-style RTC paths. Training RTC is deliberately off by
default (`model.rtc=false`), because clean-prefix training can bias a model toward
continuing the previous chunk. Set `model.rtc=true model.max_delay=8` only for a
dedicated RTC checkpoint. That checkpoint can use hard frozen-prefix sampling:

```bash
PYTHONPATH=src python -m msc.sample --checkpoint .runs/rtc/last.pt \
  --history history.pt --previous-motion prior_chunk.pt \
  --rtc-mode train --rtc-delay 4
```

A normal checkpoint can instead use Psi0's test-time PiGDM/VJP guidance, which
keeps the training distribution unchanged:

```bash
PYTHONPATH=src python -m msc.sample --checkpoint .runs/omg/last.pt \
  --history history.pt --previous-motion prior_chunk.pt \
  --rtc-mode test_time --rtc-delay 4 --rtc-execution-horizon 12
```

`prior_chunk.pt` is an unnormalized canonical `[B,30,125]` motion tensor. The
train RTC path freezes its first `rtc-delay` rows exactly. Test-time RTC uses the
paper's exponential overlap mask by default; `--rtc-mask-schedule` and
`--rtc-guidance-alpha` expose its schedule and strength. The caller must shift
the previously generated chunk by the actions already executed and pad its tail
before passing it as `prior_chunk.pt`.

Audio is a `.pt` dictionary containing `features [B,30,35]`, `mask [B,30]`, and the exact metadata dictionary stored in the checkpoint. Audio row `i` conditions predicted motion row `i`. Text is one caption per sample. Sampling writes canonical `[B,30,125]` motion and checkpoint metadata.

## Tests

Fast tests cover representation shapes, mask semantics, flow direction, multimodal branches, optimizer steps, checkpoint/resume, and sampling:

```bash
PYTHONPATH=src python -m pytest -q -m 'not slow and not integration'
```

Opt-in local integration tests use the existing assets without downloads:

```bash
MSC_TEST_DATA_ROOT=/path/to/OMG-Data \
MSC_TEST_T5_PATH=/path/to/t5-base-local \
MSC_TEST_OMG_SOURCE=/path/to/OMG \
PYTHONPATH=src python -m pytest tests/test_integration.py -q
```

The tests validate data/codec numerical parity and one real training/sample step. They do not claim motion quality; that requires a full training run and domain evaluation.

## Current boundaries

Implemented: future-only linear flow matching, G1 125D normalization/codec, history conditioning, frozen or trainable local T5, cached text tokens, precomputed OMG audio, audio mask/dropout, text/audio/combined conditions, single-process BF16/AMP, checkpoint/resume, synthetic and official OMG data adapters.

Reserved for later: video encoder, RTC serving/runtime, distributed training, physical losses, robotics deployment, low-level tracking policies, new audio extraction, and old OMG checkpoint conversion.

## Structure and interfaces

```text
MSC/
├── configs/                  # smoke, overfit, real OMG
├── src/msc/
│   ├── models/               # DiT, flow, representation, MSCModel
│   │   └── conditioning/     # T5/TextAdapter, AudioAdapter, encoder protocol
│   ├── data/                 # synthetic, OMG adapter, collator
│   ├── training/             # trainer, atomic checkpoint/RNG state
│   ├── _omg/                 # attributed source-derived numeric/data primitives
│   ├── assets/               # statistics, kinematics JSON, URDF
│   ├── config.py, factory.py
│   └── train.py, sample.py, evaluate.py, cache_text.py
├── tests/                    # unit, convergence, opt-in integration
├── licenses/, THIRD_PARTY.md
└── TEST_RESULTS.md
```

`MSCModel.compute_loss(batch)` returns a scalar differentiable loss. Batch fields:
`motion_history [B,5,125]` (four historical frames plus the current frame),
`motion_future [B,30,125]`, `motion_mask [B,30]`, optional `text: list[str]`,
`has_text [B]`, `audio_features [B,30,35]`, and `audio_mask [B,30]`. For
pre-encoded text, pass `text_features [B,L,text_dim]`
and `text_mask [B,L]`; static text features are rejected when T5 is trainable.
The synthetic backend provides synthetic tokens, not language embeddings.

All external motion tensors are **unnormalized OMG canonical 125D**. The current
frame is history index 4 and is the canonical anchor for the following 30 frames,
with OMG's joint/link order and row-based rot6d convention. The data adapter constructs this contract;
external callers must do the same. Flow velocity and integration operate in
normalized space. Returned motion is denormalized canonical features; it is not
world coordinates and does not enforce joint limits or FK consistency.

The public velocity call is `model(noisy_future, time, batch)` with continuous
`time [B]` in `[0,1]`. The generator embeds `1000*time`. Training uses
`x_t=(1-t)*data+t*noise`, target `noise-data`; sampling runs exactly N Euler steps
from noise at `t=1` to data at `t=0`. Joint CFG uses `base+s*(conditioned-base)`;
`base` keeps the same history and drops both text/audio. `--cfg-scale` defaults
to 1. No physical auxiliary losses or diffusion scheduler are included.

The defaults are a 512-wide, 12-layer, 8-head model. Smoke uses width 64 for fast
shape/engineering checks. Overfit uses width 256: a 64-wide bottleneck cannot
faithfully reconstruct arbitrary 125D Gaussian velocity targets.

## Resume, text caching and operational details

```bash
python -m msc.train --config configs/smoke.yaml \
  train.resume=.runs/smoke/last.pt train.steps=20

python -m msc.cache_text --t5-path /path/to/t5-base-local \
  --captions captions.txt --cache-dir .runs/t5-cache

python -m msc.train --config configs/omg.yaml \
  data.root=/path/to/OMG-Data model.t5_path=/path/to/t5-base-local \
  model.text_cache_dir=.runs/t5-cache
```

Cache files contain FP32 tokens before the trainable adapter and token masks.
They are keyed by the actual local model/tokenizer files, text and max length;
cache misses are computed locally. Only the explicit caching command writes
cache files. Frozen T5 stays in eval mode; set `model.freeze_text=false` and leave
`model.text_cache_dir=null` to train it.

Training logs go to `metrics.jsonl`; `config.json` and `metadata.json` capture the
run contract. `last.pt` is written atomically at the checkpoint interval and final
optimizer step, and contains model (including T5 if configured), optimizer,
scheduler, scaler, RNG, configuration, data release identity, audio provenance
and normalization buffers. Load only trusted checkpoints. Resume preserves the
sampling sequence and rejects incompatible architecture, preprocessing and
optimizer settings; `train.steps` is the desired total completed steps. The
cosine horizon is the separate `train.schedule_steps`, so extending a smoke run
does not silently change its schedule. Exact resume is verified on CPU with zero
data workers; CUDA kernels need not be bitwise deterministic.

The single-process trainer samples fixed OMG windows uniformly with replacement,
using per-microbatch seeds rather than a permutation of the complete release.
`data.limit_size` limits early episodes and returned windows for a quick local
smoke test; it is not a balanced modality subset. The integration suite explicitly
selects known audio/no-audio/short episodes. The first real-data run can create
Hugging Face Arrow caches, but never downloads the release. The full official
release is large; materialized-cache acceleration and DDP are deferred.

Use `train.precision=bf16` or `fp16` on supported CUDA devices, and `fp32` on CPU.
`train.accumulation_steps` counts microbatches per optimizer update. Nonfinite loss
or gradients raise an error instead of being silently logged as successful steps.

For audio inference, copy the checkpoint's `metadata.audio` dictionary into the
input file and provide features **from that same release**, not re-extracted WAV
features. Unknown extractor identity permits the pinned release only; it does
not make arbitrary unknown 35D features interchangeable. Changing any stored
provenance field is rejected.

Future extensions should start with validated motion-quality metrics and longer
real training, then streaming/RTC against this explicit history interface. See
`TEST_RESULTS.md` for the executed checks and their limits.
