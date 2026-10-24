# Source and license notices

MSC ports a small dependency closure from the local OMG Flow Matching worktree,
commit `510bc0d`, rather than depending on the OMG or Psi0 packages at runtime.
The original worktree and main checkout are not modified. MSC organizes these
implementations as native motion geometry and model components. The obsolete
source-specific dataset adapter and its window/tensor utilities have been removed;
the current dataset backend reads MSC Releases.

| MSC component | Source | Adaptation |
| --- | --- | --- |
| `models/attention.py`, `models/motion_generator.py` | `src/omg/generation/denoisers/transformer.py` | Retain RoPE/QK-normalized self-attention; MSC now uses a shared expert and structured multimodal cross-attention. |
| `motion/codec.py`, `motion/kinematics.py` | `src/omg/motion/feature_codec.py`, `src/omg/robots/g1/kinematics.py` | Preserve joint/link order, canonicalization, rot6d conventions, pure Torch FK; resolve packaged assets. |
| `models/representation.py` | `src/omg/motion/representation.py` | Minimal normalization with persistent mean/std and statistics identity. |
| `models/flow_matching.py` | `src/omg/generation/flow_matching/process.py` | Uniform linear flow, masked FP32 loss, Euler sampling and CFG; RTC remains opt-in. |
| Text/audio conditioning | OMG `conditions/t5.py` and `models/motion_generator.py` | Frozen online T5, offline audio features, independent adapters and explicit masks. |
| `assets/` | OMG `assets/stats/g1_125d_stats.json`, G1 kinematics JSON and URDF | Small required assets packaged with MSC; no meshes or runtime simulators. |

OMG copyright: Copyright (c) 2026 MARS Lab, Tsinghua University and contributors.
The complete MIT notice is in `licenses/OMG-MIT.txt`.

`motion/rotations.py` preserves its upstream header:
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

The offline motion pipeline vendors GMR source and Unitree G1 assets in
`third_party/gmr` at revision `bb1bbe40774794fceb2a7c579a3464a28e68c844`
(MIT; see `third_party/gmr/LICENSE`). SONIC deployment constants, observation
reference code, MuJoCo configuration, and G1 simulator assets are in
`third_party/sonic` at revision `45c03facbf848e0c743091b7d9c17952096ff908`
(see its `LICENSE` and `legal/` notices). SONIC model weights and the NVIDIA Open
Model License are kept locally in ignored `model/sonic/`; the weights are not
distributed by this repository. The G1 meshes needed by SONIC's scene were
copied from the vendored GMR G1 assets because the upstream repository stores
them via Git LFS. The offline Python runner uses the official model and ports
only its G1 observation, motor mapping, and control loop; it does not require
the upstream ROS/DDS/TensorRT deployment stack.
Only the upstream policy constants header and main control-loop source are kept
as C++ references; deployment builds, transports and deployment tests are omitted.

The vendored GMR IK calls pass `damping`, `safety_break`, and `limits` explicitly
for compatibility with the installed Mink API, reusing the existing joint limits.
