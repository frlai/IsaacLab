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
