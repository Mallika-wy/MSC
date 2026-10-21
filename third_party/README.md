# Local third-party sources

`gmr/` contains the GMR Python package and Unitree G1 assets needed for
retargeting. `sonic/` contains the GEAR-SONIC v1.1 G1 deployment reference,
observation configuration, MuJoCo scene, and meshes needed for offline
execution. Their upstream revisions and licenses are recorded in each
subdirectory and in `../THIRD_PARTY.md`.

The pipeline resolves both paths from its YAML files under `configs/data_pipeline`.
It does not import code from sibling GMR, OMG, or SONIC checkouts. ONNX weights
are stored locally under `../model/sonic/` and excluded from Git.
