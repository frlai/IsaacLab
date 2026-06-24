# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs import mdp
from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulationData
from isaaclab.utils import math as math_utils
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher, _effective_joint_gains
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum
from isaaclab.utils.leapp.proxy import _DataProxy, _EnvProxy


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


def _make_articulation_data() -> tuple[MockArticulationData, torch.Tensor]:
    """Create mock articulation data with a non-identity root orientation."""
    data = MockArticulationData(num_instances=2, num_joints=0, num_bodies=1, device="cpu")
    root_pose_w = torch.zeros(2, 7, dtype=torch.float32)
    root_pose_w[:, 6] = 1.0
    root_pose_w[1, 3] = math.sin(math.pi / 4.0)
    root_pose_w[1, 6] = math.cos(math.pi / 4.0)
    data.set_root_link_pose_w(root_pose_w)
    return data, root_pose_w


def _make_explicit_pd_asset():
    """Create a minimal articulation-like asset with explicit actuator PD gains."""
    default_kp = torch.zeros(2, 4)
    default_kd = torch.zeros(2, 4)
    actuators = {
        "hip_knee": SimpleNamespace(
            joint_indices=[0, 2],
            stiffness=torch.tensor([[11.0, 22.0], [33.0, 44.0]]),
            damping=torch.tensor([[1.1, 2.2], [3.3, 4.4]]),
        ),
        "ankle": SimpleNamespace(
            joint_indices=[1],
            stiffness=torch.tensor([[55.0], [66.0]]),
            damping=torch.tensor([[5.5], [6.6]]),
        ),
    }
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_stiffness=SimpleNamespace(torch=default_kp),
            default_joint_damping=SimpleNamespace(torch=default_kd),
        ),
        actuators=actuators,
        joint_names=["joint_0", "joint_1", "joint_2", "joint_3"],
    )
    expected_kp = torch.tensor([[11.0, 55.0, 22.0, 0.0], [33.0, 66.0, 44.0, 0.0]])
    expected_kd = torch.tensor([[1.1, 5.5, 2.2, 0.0], [3.3, 6.6, 4.4, 0.0]])
    return asset, expected_kp, expected_kd


def _capture_leapp_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture LEAPP input annotations while returning their tensor references."""
    annotated_inputs = []

    def _record_input_tensor(task_name, semantics):
        annotated_inputs.append((task_name, semantics))
        return semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", _record_input_tensor)
    return annotated_inputs


def test_effective_joint_gains_use_explicit_actuator_pd_gains():
    """Test explicit actuator gains override zero sim-level joint-drive gains."""
    asset, expected_kp, expected_kd = _make_explicit_pd_asset()

    kp_gains, kd_gains = _effective_joint_gains(asset)

    torch.testing.assert_close(kp_gains, expected_kp)
    torch.testing.assert_close(kd_gains, expected_kd)


def test_action_static_outputs_export_effective_pd_gains_for_selected_joints():
    """Test static LEAPP gain outputs use actuator PD gains and action-term joint ordering."""
    asset, expected_kp, expected_kd = _make_explicit_pd_asset()
    action_manager = SimpleNamespace(
        _terms={
            "joint_pos": SimpleNamespace(
                _asset=SimpleNamespace(_real_asset=asset),
                _joint_ids=[2, 0],
            )
        }
    )
    patcher = ExportPatcher(export_method="onnx-dynamo")
    patcher._action_term_scene_keys["joint_pos"] = "robot"

    static_outputs = patcher._collect_action_static_outputs(action_manager)

    outputs_by_kind = {output.kind: output for output in static_outputs}
    assert set(outputs_by_kind) == {"kp", "kd"}
    assert outputs_by_kind["kp"].name == "joint_pos_kp_gains"
    assert outputs_by_kind["kd"].name == "joint_pos_kd_gains"
    assert outputs_by_kind["kp"].element_names == [["joint_2", "joint_0"]]
    assert outputs_by_kind["kd"].element_names == [["joint_2", "joint_0"]]
    torch.testing.assert_close(outputs_by_kind["kp"].ref, expected_kp[:, [2, 0]])
    torch.testing.assert_close(outputs_by_kind["kd"].ref, expected_kd[:, [2, 0]])


def test_direct_projected_gravity_b_read_preserves_vector3d_input(monkeypatch: pytest.MonkeyPatch):
    """Test direct data proxy reads keep projected gravity as its own semantic input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, _ = _make_articulation_data()

    proxy = _DataProxy(
        data,
        entity_name="robot",
        task_name="Isaac-Velocity-Flat-G1-v0",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"robot_{property_name}",
    )

    assert proxy.projected_gravity_b.torch.shape == (2, 3)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_projected_gravity_b"
    assert semantics.kind == InputKindEnum.VECTOR3D
    assert semantics.extra == {"isaaclab_connection": "state:robot:projected_gravity_b"}


def test_projected_gravity_observation_exports_root_quat_w_input(monkeypatch: pytest.MonkeyPatch):
    """Test the projected-gravity observation is export-lowered through root quaternion."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, root_pose_w = _make_articulation_data()
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene)
    proxy_env = _EnvProxy(env, "Isaac-Velocity-Flat-G1-v0", {}, {})

    term_cfg = SimpleNamespace(func=mdp.projected_gravity, noise="noise")
    obs_manager = SimpleNamespace(_group_obs_term_cfgs={"policy": [term_cfg]}, compute=lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "Isaac-Velocity-Flat-G1-v0"
    patcher._patch_observation_manager(obs_manager, proxy_env)

    projected_gravity_b = term_cfg.func(env)

    expected = math_utils.quat_apply_inverse(
        root_pose_w[:, 3:7],
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).expand(2, 3),
    )
    assert torch.allclose(projected_gravity_b, expected)
    assert term_cfg.noise is None

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_root_quat_w"
    assert semantics.kind == InputKindEnum.BODY_ROTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}
