# MSC: shared trimodal flow action expert

MSC v2 generates canonical G1 motion with a shared action expert conditioned on
frozen T5, MERT and V-JEPA features. Motion remains **canonical G1 125D, 30 FPS, 5 history
frames and 30 future frames**. Encoders are frozen; the trainable model consists
of independent modality adapters, a history adapter and six shared action blocks.

Training reads published MSC Releases and offline features. It never writes to
Data Pipeline inputs, work directories or Releases. Video currently has synthetic
interface coverage; real V-JEPA features and three-task results are **Pending**.

Motion geometry and canonical conversion live in `msc.motion` (`codec`,
`kinematics`, `rotations`); attention components live in `msc.models.attention`.
The representation identifier is `g1_canonical_125d`. Only newly trained
checkpoint format version 3 is supported; old checkpoints are not migrated.
Source/license attribution is retained in `THIRD_PARTY.md`.

Implementation, milestone acceptance, measured results and remaining prerequisites:
[trimodal implementation report](docs/trimodal_implementation.md).
Dataset and existing SONIC documentation: [MSC v1 dataset](docs/msc_dataset_v1.md).

## Environment and smoke checks

```bash
source .venv/bin/activate
# For a new environment: pip install -e '.[data,text,test]'
PYTHONPATH=src python -m pytest tests -q -m 'not integration'
PYTHONPATH=src python -m msc.train --config configs/smoke.yaml \
  train.steps=2 train.output_dir=.runs/trimodal-smoke
PYTHONPATH=src python -m msc.sample \
  --checkpoint .runs/trimodal-smoke/last.pt --synthetic-history \
  --output .runs/trimodal-smoke/sample.pt
```

The default real model has 6 blocks, width 512, 8 heads and FFN width 2048.
`model.condition` configures the independent adapters and temporal attention bias.
`model.losses` configures joint velocity/acceleration residual supervision.
History stays valid in dropout and CFG null branches. Checkpoints use architecture
`trimodal_flow_v2`, checkpoint version 3, and reject older weights.

## Prepare frozen text and training statistics

Supply a **published** Text Release, a local T5 checkpoint and the Audio Release.
The Text pipeline's work/cache directories are not training datasets.

```bash
PYTHONPATH=src python -m msc.cache_text \
  --t5-path /path/to/local-t5 --release-root /path/to/text/release \
  --cache-dir .runs/trimodal-t5-cache --device cuda:1
PYTHONPATH=src python -m msc.data.stats \
  --config configs/trimodal_release.yaml \
  --output .runs/trimodal-dynamics-stats.json \
  model.text_cache_dir=.runs/trimodal-t5-cache
PYTHONPATH=src python -m msc.train --config configs/trimodal_release.yaml \
  model.text_cache_dir=.runs/trimodal-t5-cache \
  model.dynamic_stats_path=.runs/trimodal-dynamics-stats.json \
  train.device=cuda:1
```

Set `data.roots` in the config to the actual Releases before running these commands.
Statistics use only the configured Train Split and selected official label. Recompute
them when the selected data/window pools change. Training from precomputed tokens
does not load T5, MERT or V-JEPA weights.

Missing features, missing valid task windows, ambiguous labels, wrong fingerprints
and cross-split source/song leakage produce explicit errors. Shared songs cannot
be moved between splits by the training code.

## Controlled short experiments

```bash
PYTHONPATH=src python -m msc.training.experiments \
  --config configs/trimodal_audio_overfit.yaml \
  --output .runs/trimodal-audio-short \
  --sonic-config configs/data_pipeline/aistpp_motion.yaml
```

The runner performs 200 GPU optimizer steps per available task. When real train
and validation pools are available, it runs three 1000-step arms: FM, FM+velocity,
and FM+velocity+acceleration. The latter two enable auxiliary supervision after
100 optimizer steps. Arms share initialization, task/window sequence, flow noise,
tau and evaluation seeds; the report records initialization and sampler hashes.
CPU-only runs report the full experiment as Pending.

The current AIST++ Release contains Train only; independent validation comparisons
remain Pending. Previous training outputs have been removed. Train fresh checkpoints
with the current native implementation before sampling or evaluation.

## Sampling and evaluation

An Audio input `.pt` contains `tokens [B,L,768]`, `times [B,L]`, `mask [B,L]`
and `metadata` matching checkpoint audio provenance. Text token inputs contain
`tokens`, `mask`, `metadata`; Video inputs contain `tokens [B,L,K,D]`, `intervals
[B,L,2]`, `mask [B,L,K]`, and checkpoint feature `metadata`. Times are relative
to the future window start. Use `--t5-path` for optional live frozen text encoding
from a compatible offline checkpoint.

```bash
PYTHONPATH=src python -m msc.sample --checkpoint /path/to/last.pt \
  --history /path/to/history.pt --audio /path/to/audio.pt \
  --device cuda:1 --output .runs/trimodal-sample.pt
PYTHONPATH=src python -m msc.execute --sample .runs/trimodal-sample.pt \
  --initial-state /path/to/anchor-qpos36.npz \
  --config configs/data_pipeline/aistpp_motion.yaml --output .runs/trimodal-execution
PYTHONPATH=src python -m msc.evaluate --checkpoint /path/to/last.pt \
  --split val --device cuda:1 --output .runs/trimodal-evaluation \
  --sonic-config configs/data_pipeline/aistpp_motion.yaml --rollout-chunks 3
```

Evaluation records per-task losses, physical dynamic residuals, motion amplitudes,
root motion, FK MPJPE, Jerk, chunk boundaries and matched/shuffled/missing-condition
controls. Continuous rollouts re-anchor generated history and can run the existing
SONIC tracker. Condition sensitivity alone does not establish semantic alignment.

Third-party source attribution and licensing is documented in [THIRD_PARTY.md](THIRD_PARTY.md).
