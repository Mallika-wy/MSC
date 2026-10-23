# MSC: shared trimodal flow action expert

MSC v2 generates canonical G1 motion with a shared action expert conditioned on
frozen T5, MERT and V-JEPA features. Motion remains **canonical G1 125D, 30 FPS, 5 history
frames and 30 future frames**. Encoders are frozen; the trainable model consists
of independent modality adapters, a history adapter and six shared action blocks.

Training reads published MSC Releases and offline features. It never writes to
Data Pipeline inputs, work directories or Releases. Real V-JEPA features and
three-task results are **Pending**.

Motion geometry and canonical conversion live in `msc.motion` (`codec`,
`kinematics`, `rotations`); attention components live in `msc.models.attention`.
The representation identifier is `g1_canonical_125d`. Only newly trained
checkpoint format version 3 is supported; old checkpoints are not migrated.
Source/license attribution is retained in `THIRD_PARTY.md`.

Implementation, milestone acceptance, measured results and remaining prerequisites:
[trimodal implementation report](docs/trimodal_implementation.md).
Dataset and existing SONIC documentation: [MSC v1 dataset](docs/msc_dataset_v1.md).

## Environment checks

```bash
source .venv/bin/activate
# For a new environment: pip install -e '.[data,text,test]'
PYTHONPATH=src python -m pytest tests -q -m 'not integration'
```

The default real model has 6 blocks, width 512, 8 heads and FFN width 2048.
`model.condition` configures the independent adapters and temporal attention bias.
`model.losses` configures joint velocity/acceleration residual supervision.
History stays valid in dropout and CFG null branches. Checkpoints use architecture
`trimodal_flow_v2`, checkpoint version 3, and reject older weights.

## Train from the published releases

`configs/trimodal_release.yaml` lists the four published releases. Text is read
from annotations and encoded online with frozen `model/t5-base-local`; audio
uses the MERT features already bundled in each audio release. No T5 token cache
is required. Pipeline work/cache directories are not training datasets.

Published releases include `training/summary.json` and shared Arrow window
indices per split. Dataset startup maps these directly and does not load the six
metadata tables or repeat publication checks. Publication/validation builds or
refreshes the index. To migrate existing releases and audit them together:

```bash
PYTHONPATH=src .venv/bin/python -m data_pipeline.cli index --dataset \
  data/processed/omg-text/release \
  data/processed/audio/aistpp/release \
  data/processed/audio/finedance/release \
  data/processed/audio/gdance/release
```

Rebuild after changing metadata, the selected label, or audio/video context
lengths. Training requires a published index and never rebuilds it automatically.

```bash
PYTHONPATH=src .venv/bin/python -m msc.data.stats \
  --config configs/trimodal_release.yaml \
  --output .runs/trimodal-dynamics-stats.json
bash train.sh
```

Set the release paths and local T5 path before running. Statistics use only the
configured Train split and selected motion label. Recompute them when the data
or selected window pools change. The statistics step reads motion windows but
neither loads nor runs T5/MERT.

Training shuffles and traverses every eligible window once per epoch, without
task weights or replacement. Shuffling uses bounded blocks to avoid a full
permutation of the large OMG index, and checkpoints resume at the exact next
batch. The incomplete final batch of an epoch is kept. `train.steps` remains the
optimizer-step limit, so a run can stop partway through an epoch. `data.tasks`
selects available condition types only; it does not set sampling probabilities.

Text/audio batches can be mixed. Missing conditions have false masks; only
present conditions influence the model. Per-condition validation remains
available for comparing text and audio behavior.

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

AIST++ and FineDance now contain song-disjoint Train/Val/Test splits; independent validation comparisons
remain Pending. Previous training outputs have been removed. Train fresh checkpoints
with the current native implementation before sampling or evaluation.

## Sampling and evaluation

An Audio input `.pt` contains `tokens [B,L,768]`, `times [B,L]`, `mask [B,L]`
and `metadata` matching checkpoint audio provenance. Text is supplied as a raw
caption with `--text`; Video inputs contain `tokens [B,L,K,D]`, `intervals
[B,L,2]`, `mask [B,L,K]`, and checkpoint feature `metadata`. Times are relative
to the future window start. Use `--t5-path` to override the checkpoint's local T5
directory for online frozen text encoding.

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
