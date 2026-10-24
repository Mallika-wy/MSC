# Vendored GEAR-SONIC references

Upstream: https://github.com/NVlabs/GR00T-WholeBodyControl
Revision: see [UPSTREAM_REVISION](UPSTREAM_REVISION).

MSC uses its own Python offline runner in `data_pipeline/motion/sonic.py`.
This directory retains:

- G1 MuJoCo scene, robot descriptions and referenced meshes;
- `g1_29dof_sonic_model12.yaml` and `observation_config.yaml`;
- upstream `policy_parameters.hpp` and `g1_deploy_onnx_ref.cpp` as the source
  reference for motor constants, observations and the control loop;
- upstream licenses and third-party notices in `LICENSE` and `legal/`.

The unrelated C++ deployment build, ROS/DDS/ZMQ transports and deployment tests
are omitted. The retained C++ files are references, not a standalone build target.
Local encoder/decoder weights remain under ignored `model/sonic/`.
