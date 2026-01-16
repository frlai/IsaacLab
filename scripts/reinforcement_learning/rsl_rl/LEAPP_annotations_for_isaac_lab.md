# LEAPP Export for Isaac Lab

Export RSL-RL reinforcement learning pipelines as portable processing graphs using [LEAPP](https://gitlab-master.nvidia.com/Isaac/leapp).

## Exported Artifacts

| File | Description |
|------|-------------|
| `observation_manager.pt` | Observation processing (TorchScript) |
| `policy.onnx` | Policy network (ONNX) |
| `action_manager.pt` | Action processing (TorchScript) |
| `<task_name>.yaml` | Pipeline configuration and metadata |
| `<task_name>.png` | Visualization of the processing graph |

The YAML file includes semantic metadata (joint names, units, etc.) extracted from IO descriptors. For details on the YAML format, see the [LEAPP documentation](https://gitlab-master.nvidia.com/Isaac/leapp/-/blob/main/docs/0_getting_started.md).

## Usage

### 1. Install LEAPP

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/Isaac/leapp.git
cd leapp
git checkout develop
pip install -e .
```

### 2. Export a Policy

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/export.py \
    --task Isaac-Reach-Franka-v0 \
    --use_pretrained_checkpoint \
    --headless
```

> **Note:** Export runs with a single environment instance.

### 3. View Results

Artifacts are saved to `./<task_name>/`.



sample exported `Isaac-Reach-Franka-v0.yaml`:

```yaml
models:
  observation_manager:
    inputs:
    - name: joint_pos
      dtype: float32
      shape: [1, 9]
      type: tensor
    - name: joint_vel
      dtype: float32
      shape: [1, 9]
      type: tensor
    - name: ee_pose
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: last_actions
      dtype: float32
      shape: [1, 7]
      type: tensor
    outputs:
    - name: obs_policy
      dtype: float32
      shape: [1, 32]
      type: tensor
    parameters:
      model_path: observation_manager.pt
      md5sum: 3e44b3d2942d5fc3c6a88f28ef3d7b5a
      sha256sum: 8d5761e8830be584ec09863775e9bef135e2ad081bcad3029ac1ab50a7fcf819
      device: cuda
      backend: torch
  policy:
    inputs:
    - name: obs_policy
      dtype: float32
      shape: [1, 32]
      type: tensor
    outputs:
    - name: actions
      dtype: float32
      shape: [1, 7]
      type: tensor
    parameters:
      model_path: policy.onnx
      md5sum: 848384ae8e4d22052d6e87719d8cb42c
      sha256sum: e69cf132746a570e504eb23071ad9cddd146eafaa787adf5bc0951ed948a4bcc
      device: cuda
      backend: onnx
  action_manager:
    inputs:
    - name: actions
      dtype: float32
      shape: [1, 7]
      type: tensor
    outputs:
    - name: arm_action
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: arm_action_kp_gains
      dtype: float32
      shape: [1, 7]
      type: tensor
    - name: arm_action_kd_gains
      dtype: float32
      shape: [1, 7]
      type: tensor
    parameters:
      model_path: action_manager.pt
      md5sum: cbbed1862042f23bd285da4c0ddaa946
      sha256sum: 20d31b74ab7dc686f29b4b973ef43b9950076dc5a3da40f417839d581c5b328e
      device: cuda
      backend: torch

pipeline:
  data_flow:
    observation_manager/obs_policy: [policy/obs_policy]
    policy/actions: [action_manager/actions]
  feedback_flow:
    policy/actions: [observation_manager/last_actions]
  inputs:
    observation_manager: [joint_pos, joint_vel, ee_pose]
  outputs:
    action_manager: [arm_action, arm_action_kp_gains, arm_action_kd_gains]

system information:
  cuda version: '12.8'
  leapp version: 0.3.0
  os: Linux
  python version: 3.11.14
  torch version: 2.7.0+cu128

semantic:
  actions:
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    leapp_mapping:
    - arm_action
    name: joint_position_action
  observations:
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    - panda_finger_joint1
    - panda_finger_joint2
    leapp_mapping:
    - joint_pos
    name: joint_pos_rel
    units: rad
  - joint_names:
    - panda_joint1
    - panda_joint2
    - panda_joint3
    - panda_joint4
    - panda_joint5
    - panda_joint6
    - panda_joint7
    - panda_finger_joint1
    - panda_finger_joint2
    leapp_mapping:
    - joint_vel
    name: joint_vel_rel
    units: rad/s
  - leapp_mapping:
    - ee_pose
    name: generated_commands
  - leapp_mapping:
    - last_actions
    name: last_action
  scene:
    decimation: 2
    dt: 0.03333333333333333
    physics_dt: 0.016666666666666666
```
