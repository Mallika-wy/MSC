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

The single training configuration is [msc_v3_2xa6000.yaml](configs/msc_v3_2xa6000.yaml):
12 layers, hidden 1024, 16 heads, FFN 3072; two A6000 GPUs, BF16, batch **1152 per GPU**,
global batch **2304**, no accumulation and **30,000 optimizer steps**.
Sampling uses the existing shuffled window blocks without task-specific batches
or a fixed task schedule. Stopping is step-based.

Use Python >=3.10, PyTorch >=2.7 for the compiled dual-GPU setup, and install
`pip install -e '.[text,data,test]'` in the selected environment. The local `.venv`
is the default interpreter; set `MSC_PYTHON` to use another environment.
The configuration requires four published Releases and local `model/t5-base-local`.

Compute dynamics statistics for the exact Train pool before starting:

```bash
PYTHONPATH=src .venv/bin/python -m msc.data.stats \
  --config configs/msc_v3_2xa6000.yaml --workers 8 \
  --output .runs/msc-v3-dynamics-stats.json
bash train.sh
```

Training reads only prebuilt Release indices and offline audio/video features.
Outputs go to `.runs/msc-v3-epoch1-b1152`; only rank 0 saves and validates.
Changing the Train pool requires recomputing matching statistics.
Runtime outputs and statistics are local generated files, not included in the repository.
See [dual A6000 setup and measured performance](docs/msc_v3_2xa6000.md).

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
PYTHONPATH=src .venv/bin/python -m msc.sample --checkpoint /path/to/last.pt \
  --history /path/to/history.pt --text 'walk forward' \
  --device cuda:0 --output .runs/sample.pt
PYTHONPATH=src .venv/bin/python -m msc.evaluate --checkpoint /path/to/last.pt \
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
