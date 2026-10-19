# Source and license notices

MSC ports a small dependency closure from the local OMG Flow Matching worktree,
commit `510bc0d`, rather than depending on the OMG or Psi0 packages at runtime.
The original worktree and main checkout are not modified.

| MSC component | Source | Adaptation |
| --- | --- | --- |
| `_omg/attention.py`, `models/motion_generator.py` | `src/omg/generation/denoisers/transformer.py` | Retain RoPE/QK-normalized attention, future-only motion, history cross-attention and audio FiLM; remove human-reference/local-control and alternate injection modes. |
| `_omg/feature_codec.py`, `_omg/kinematics.py` | `src/omg/motion/feature_codec.py`, `src/omg/robots/g1/kinematics.py` | Preserve joint/link order, canonicalization, rot6d conventions, pure Torch FK; resolve packaged assets. |
| `_omg/lerobot_dataset.py`, `_omg/windowing.py`, `_omg/tensor.py` | Corresponding OMG data/core modules | Retain release checks and window/padding semantics; remove downloads and statistics/materialization helpers; validate frame timestamps. |
| `models/representation.py` | `src/omg/motion/representation.py` | Minimal normalization with persistent mean/std and statistics identity. |
| `models/flow_matching.py` | `src/omg/generation/flow_matching/process.py` | Uniform linear flow, masked FP32 loss, Euler sampling and joint CFG; omit RTC. |
| Text/audio conditioning | OMG `conditions/t5.py` and `models/motion_generator.py` | Local-only optional T5, independent adapters, frozen-token cache, explicit masks. |
| `assets/` | OMG `assets/stats/g1_125d_stats.json`, G1 kinematics JSON and URDF | Small required assets packaged with MSC; no meshes or runtime simulators. |

OMG copyright: Copyright (c) 2026 MARS Lab, Tsinghua University and contributors.
The complete MIT notice is in `licenses/OMG-MIT.txt`.

`_omg/rotation_conversions.py` preserves its upstream header:
Copyright (c) Facebook, Inc. and its affiliates; Copyright (c) 2026 NVIDIA
CORPORATION & AFFILIATES; SPDX `BSD-3-Clause AND Apache-2.0`, derived from
ACTOR/PyTorch3D. See `licenses/PyTorch3D-BSD-3-Clause.txt` and
`licenses/Apache-2.0.txt`.

Psi0's `src/psi/trainers/trainer.py`, `posttrain.py`, `models/psi0.py` and
configuration modules were inspected for training organization and the flow
convention. MSC's trainer is written separately; no Qwen, VLM, action-format,
Accelerate or robot deployment code is included. Psi0's supplied license notice
is retained in `licenses/Psi0-Apache-2.0.txt` (Copyright 2026 PSI-lab, USC).

Pretrained T5 weights and official OMG datasets are external user-supplied assets,
not redistributed in this package.
