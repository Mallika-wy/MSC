# MSC v3 — Conditional Flow Motion Transformer

MSC uses one shared **296.9M trainable-parameter** Flow Motion Expert for
Text→Motion, Audio→Motion and feature-level Video→Motion. Motion is fixed to
**G1 Canonical 125D, 30 FPS, 5 history frames and 30 future frames**.
Each sample uses one external modality; a batch can mix tasks.

T5 stays frozen and encodes text online. Audio reads offline MERT features.
Video is disabled until real V-JEPA metadata supplies its feature dimensions.
The existing Motion Codec, normalization and published Releases are retained.

- [Model architecture and parameter table](docs/msc_v3.md)
- [Dataset pipeline and commands](docs/msc_dataset_v1.md)
- [Release schema](docs/Release数据格式说明.md)
- [OMG Text release](docs/OMG_Release数据格式说明.md)
- [Source and license notices](THIRD_PARTY.md)

## Training

The base configuration is [msc_v3_2xa6000.yaml](configs/msc_v3_2xa6000.yaml):
12 layers, hidden 1024, 16 heads, FFN 3072; two A6000 GPUs, BF16, batch **1152 per GPU**,
global batch **2304**, no accumulation and **30,000 optimizer steps**.
Sampling uses the existing shuffled window blocks without task-specific batches
or a fixed task schedule. Stopping is step-based.

Use Python >=3.10, PyTorch >=2.7 for the compiled dual-GPU setup, and install
`pip install -e '.[text,data,test,tracking]'` in the selected environment. The local `.venv`
is the default interpreter; set `MSC_PYTHON` to use another environment.
The configuration requires four published Releases and local `model/t5-base-local`.

Compute dynamics statistics for the exact Train pool before starting:

```bash
bash train.sh --stats
bash train.sh
```

Training reads only prebuilt Release indices and offline audio/video features.
Outputs go to the launcher's `OUTPUT_DIR`; only rank 0 saves and validates.
Changing the Train pool requires recomputing matching statistics.
Runtime outputs and statistics are local generated files, not included in the repository.
See [dual A6000 setup and measured performance](docs/msc_v3_2xa6000.md).

Edit the grouped settings at the top of `train.sh` for GPU selection, model size,
dataset roots/tasks, batch size, steps, optimizer, losses, validation, checkpoints,
RTC and W&B. Defaults preserve the configuration above. Environment variables
override script defaults, and final `section.key=value` arguments take precedence
over both. These launcher settings override their corresponding YAML values.
GPU process count defaults to the selected GPU count; `SCHEDULE_STEPS` defaults
to `STEPS`. `--stats` uses the same effective settings as training.

Learning rate warms up to `LEARNING_RATE` and then follows cosine decay to
`MIN_LEARNING_RATE` (default `0.000001`). It stays at that floor after
`SCHEDULE_STEPS`; the warmup can start below the decay floor.
Each save writes `OUTPUT_DIR/checkpoints/step_XXX.pt`, retaining earlier steps.
The launcher currently saves every 5000 optimizer steps and always saves the
final step. Checkpoints contain model, optimizer, scheduler and RNG state.

```bash
BATCH_SIZE=512 LEARNING_RATE=0.00005 STEPS=2000 WARMUP_STEPS=100 \
  bash train.sh train.wandb_name=short-run
```

W&B is enabled in **offline mode** through `WANDB_PROJECT`; no login is
required. Only rank 0 writes a run under the output directory's `wandb/` folder.
Training, per-task losses, learning rate, gradients, throughput, per-rank memory
and validation metrics use optimizer steps as the chart axis. Configuration and
the final profile are also saved. Code and model weights are not uploaded.
Use `train.wandb_name=my-run` to name a run or `train.wandb_project=null` to disable it:

```bash
bash train.sh train.wandb_name=msc-v3-first-run
```

## Tests and parameter audit

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests data_pipeline/tests -q
MSC_TEST_T5_PATH=model/t5-base-local PYTHONPATH=src .venv/bin/python \
  -m pytest tests -q -m integration
PYTHONPATH=src .venv/bin/python -m msc.models.audit
```

CUDA, real T5 and simulator integration tests skip when their required environment
is unavailable. Synthetic video checks validate the feature interface, not real video quality.

## Sampling, RTC and evaluation

```bash
PYTHONPATH=src .venv/bin/python -m msc.sample --checkpoint /path/to/checkpoints/step_30000.pt \
  --history /path/to/history.pt --text 'walk forward' \
  --device cuda:0 --output .runs/sample.pt
PYTHONPATH=src .venv/bin/python -m msc.evaluate --checkpoint /path/to/checkpoints/step_30000.pt \
  --split val --device cuda:0 --output .runs/msc-v3-evaluation
```

History is `[B,5,125]`. Audio feature inputs contain `tokens [B,L,768]`, `mask`,
`times` and matching encoder metadata. Video inputs contain `tokens [B,L,K,D]`,
`mask [B,L,K]`, `intervals [B,L,2]` and verified metadata. Media times are relative
to future start after the Release TimeMap mapping. Video K/D are never guessed.

RTC is opt-in through `model.rtc=true`. Sampling supports `--rtc-mode train`
for an RTC-trained model and `--rtc-mode test_time` for VJP guidance; both use
`--previous-motion` and `--rtc-delay`. CFG uses `--cfg-scale` and retains history.
Real video training and full-trajectory quality evaluation remain pending.
