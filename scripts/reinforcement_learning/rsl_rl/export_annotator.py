# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export annotations for Isaac Lab policies using instance-level patching."""


from __future__ import annotations

import inspect
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from leapp import annotate

from isaaclab.assets.articulation.articulation_data import ArticulationData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@dataclass
class ExportAnnotator:
    """Encapsulates all leapp annotation logic for exporting Isaac Lab policies.

    Usage:
        env = gym.make(...)
        annotator = ExportAnnotator(env)
        annotator.setup()
        # ... run policy ...
        obs_map = annotator.observation_to_articulation_map
        action_map = annotator.action_io_to_term_map
        annotator.cleanup()
    """

    env: ManagerBasedEnv
    task_name: str

    io_descriptor_observations: list[Any] = field(default_factory=list)
    io_descriptor_actions: list[Any] = field(default_factory=list)
    io_descriptor_scene: dict[str, Any] = field(default_factory=dict)

    # Mappings built during execution
    observation_to_articulation_map: dict[str, set[str]] = field(default_factory=dict)
    action_io_to_term_map: dict[str, list[str]] = field(default_factory=dict)

    # Original methods for restoration
    _original_compute_group: callable = field(default=None, repr=False)
    _original_process_action: callable = field(default=None, repr=False)

    # ArticulationData patching state
    _articulation_originals: dict[str, property] = field(default_factory=dict, repr=False)
    _articulation_annotating: dict[str, property] = field(default_factory=dict, repr=False)
    _annotations_active: bool = field(default=False, repr=False)
    # Cache for annotated tensors within a single compute_group call
    # Prevents duplicate input tensors when same property is accessed multiple times
    _annotated_tensor_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    # Articulation properties to annotate
    OBSERVATION_PROPERTIES: frozenset[str] = frozenset({
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_b",
        "root_ang_vel_b",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "projected_gravity_b",
        "body_pose_w",
        "body_quat_w",
        "joint_pos",
        "joint_vel",
    })

    def setup(self):
        """Set up all annotations. Call after env is created."""
        self._collect_io_descriptors()
        self._disable_observation_noise()
        self._prepare_articulation_annotations()
        self._patch_observation_functions()
        self._patch_observation_manager()
        self._patch_action_manager()

    def cleanup(self):
        """Restore all original methods and properties."""
        self._restore_observation_functions()
        self._restore_observation_manager()
        self._restore_action_manager()
        self._remove_articulation_annotations()

    # ──────────────────────────────────────────────────────────────────
    # IO Descriptor Collection (before patching)
    # ──────────────────────────────────────────────────────────────────

    def _collect_io_descriptors(self):
        outs = self.env.unwrapped.get_IO_descriptors
        self.io_descriptor_observations = outs["observations"]["policy"]
        self.io_descriptor_actions = outs["actions"]
        self.io_descriptor_scene = outs["scene"]

        # Build action IO descriptor name -> term name mapping
        # e.g., 'joint_position_action' -> ['arm_action']
        action_manager = self.env.env.unwrapped.action_manager
        for term_name, term in action_manager._terms.items():
            try:
                io_name = term.IO_descriptor.name
                self.action_io_to_term_map[io_name] = [term_name]
            except Exception:
                pass  # Skip if IO descriptor not available

    # ──────────────────────────────────────────────────────────────────
    # Observation Manager
    # ──────────────────────────────────────────────────────────────────

    def _disable_observation_noise(self):
        """Disable noise/corruption for deterministic export.

        Since we patch after env creation, we need to set term_cfg.noise = None
        directly on each term config (not just the group config).
        """
        obs_manager = self.env.env.unwrapped.observation_manager

        # Disable noise on each individual term config
        for _, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            for term_cfg in term_cfgs:
                term_cfg.noise = None

    def _patch_observation_functions(self):
        """Patch observation functions inside the observation manager's term configs.

        These functions (last_action, generated_commands) don't access ArticulationData
        properties, so they need separate patching to record their mappings and annotate
        their outputs.

        We patch the term_cfg.func directly because the observation manager stores
        references to these functions at creation time.
        """
        obs_manager = self.env.env.unwrapped.observation_manager

        # Store original functions for restoration: (group_name, term_idx) -> original_func
        self._original_obs_funcs: dict[tuple[str, int], callable] = {}

        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            for term_idx, term_cfg in enumerate(term_cfgs):
                original_func = term_cfg.func
                func_name = getattr(original_func, "__name__", None)

                if func_name == "last_action":
                    self._original_obs_funcs[(group_name, term_idx)] = original_func
                    term_cfg.func = self._make_patched_last_action(original_func)

                elif func_name == "generated_commands":
                    self._original_obs_funcs[(group_name, term_idx)] = original_func
                    term_cfg.func = self._make_patched_generated_commands(original_func, term_cfg)

    def _make_patched_last_action(self, original_func):
        """Create a patched version of last_action that records mappings."""

        def patched_last_action(env, action_name=None, **kwargs):
            result = original_func(env, action_name, **kwargs)
            self._record_articulation_access("last_action", "last_actions")
            result = annotate.input_tensors({"last_actions": result}, node_name=self.task_name)
            return result

        patched_last_action.__name__ = original_func.__name__
        return patched_last_action

    def _make_patched_generated_commands(self, original_func, term_cfg):
        """Create a patched version of generated_commands that records mappings."""
        # Get the command_name from term_cfg.params if available
        command_name_from_cfg = term_cfg.params.get("command_name")

        def patched_generated_commands(env, command_name=None, **kwargs):
            result = original_func(env, command_name, **kwargs)
            # Use command_name parameter, or fall back to config, or default
            leapp_input_name = command_name or command_name_from_cfg or "commands"
            self._record_articulation_access("generated_commands", leapp_input_name)
            result = annotate.input_tensors({leapp_input_name: result}, node_name=self.task_name)
            return result

        patched_generated_commands.__name__ = original_func.__name__
        return patched_generated_commands

    def _restore_observation_functions(self):
        """Restore original observation functions in term configs."""
        if not hasattr(self, "_original_obs_funcs"):
            return

        obs_manager = self.env.env.unwrapped.observation_manager

        for (group_name, term_idx), original_func in self._original_obs_funcs.items():
            obs_manager._group_obs_term_cfgs[group_name][term_idx].func = original_func

    def _patch_observation_manager(self):
        """Patch the observation manager instance's compute_group method."""
        obs_manager = self.env.env.unwrapped.observation_manager
        self._original_compute_group = obs_manager.compute_group

        def patched_compute_group(*args, **kwargs):
            self._apply_articulation_annotations()
            try:
                return self._original_compute_group(*args, **kwargs)
            finally:
                self._remove_articulation_annotations()

        obs_manager.compute_group = patched_compute_group

    def _restore_observation_manager(self):
        """Restore original compute_group method."""
        if self._original_compute_group:
            self.env.env.unwrapped.observation_manager.compute_group = self._original_compute_group

    # ──────────────────────────────────────────────────────────────────
    # Action Manager
    # ──────────────────────────────────────────────────────────────────

    def _patch_action_manager(self):
        """Patch the action manager instance's process_action method."""
        action_manager = self.env.env.unwrapped.action_manager
        self._original_process_action = action_manager.process_action

        def patched_process_action(action: torch.Tensor):

            # Register raw_actions buffers for tracing
            for term_name, term in action_manager._terms.items():
                if hasattr(term, "_raw_actions") and term._raw_actions is not None:
                    buffers = annotate.register_buffer(self.task_name, {"raw_actions": term._raw_actions})
                    term._raw_actions = buffers["raw_actions"]

            self._original_process_action(action)
            # this is stored differently inside the original process action method that would loose tracing. this step preserves it.
            action_manager._action = action.clone()

            tensors, static_values = self._collect_action_outputs(action_manager)
            tensors["last_action"] = action_manager._action
            annotate.output_tensors(self.task_name, tensors, static_outputs=static_values, export_with="onnx")

        action_manager.process_action = patched_process_action

    def _collect_action_outputs(self, action_manager) -> tuple[dict, dict]:
        """Collect action tensors and static values from all terms."""
        tensors = {}
        static_values = {}

        for term_name, term in action_manager._terms.items():
            tensors[term_name] = term.processed_actions

            # Handle variable impedance (dynamic gains)
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                tensors[f"{term_name}_kp_gains"] = torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1)
                tensors[f"{term_name}_kd_gains"] = torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1)
                continue

            # Collect static gains
            asset = getattr(term, "_asset", None)
            if asset and hasattr(asset, "data"):
                self._collect_static_gains(term_name, asset.data, getattr(term, "_joint_ids", None), static_values)

        return tensors, static_values

    def _collect_static_gains(self, term_name: str, data, joint_ids, static_values: dict):
        """Extract static kp/kd gains from asset data."""
        if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
            gains = data.default_joint_stiffness
            static_values[f"{term_name}_kp_gains"] = gains[:, joint_ids] if joint_ids else gains

        if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
            gains = data.default_joint_damping
            static_values[f"{term_name}_kd_gains"] = gains[:, joint_ids] if joint_ids else gains

    def _restore_action_manager(self):
        """Restore original process_action method."""
        if self._original_process_action:
            self.env.env.unwrapped.action_manager.process_action = self._original_process_action

    # ──────────────────────────────────────────────────────────────────
    # ArticulationData Property Annotations
    # ──────────────────────────────────────────────────────────────────

    def _prepare_articulation_annotations(self):
        """Prepare annotating versions of ArticulationData properties."""
        for prop_name in self.OBSERVATION_PROPERTIES:
            original_prop = getattr(ArticulationData, prop_name, None)
            if not isinstance(original_prop, property) or original_prop.fget is None:
                continue

            self._articulation_originals[prop_name] = original_prop
            self._articulation_annotating[prop_name] = self._make_annotating_property(original_prop, prop_name)

    def _make_annotating_property(self, original: property, prop_name: str) -> property:
        """Create an annotating version of an ArticulationData property."""
        original_fget = original.fget
        assert original_fget is not None  # Checked in _prepare_articulation_annotations

        def annotating_fget(data_self):
            result = original_fget(data_self)
            obs_name = self._find_calling_observation()
            if obs_name:
                self._record_articulation_access(obs_name, prop_name)

            if isinstance(result, torch.Tensor):
                # Check if this property was already annotated in this compute_group call
                if prop_name in self._annotated_tensor_cache:
                    # Return a clone of the cached tensor to avoid duplicate input annotations
                    return self._annotated_tensor_cache[prop_name].clone()

                # First access - annotate and cache
                result = annotate.input_tensors({prop_name: result}, node_name=self.task_name)
                self._annotated_tensor_cache[prop_name] = result

            return result

        return property(fget=annotating_fget, fset=original.fset, fdel=original.fdel, doc=original.__doc__)

    def _apply_articulation_annotations(self):
        """Temporarily apply annotating properties."""
        if not self._annotations_active:
            # Clear the tensor cache at the start of each compute_group call
            self._annotated_tensor_cache.clear()
            for prop_name, prop in self._articulation_annotating.items():
                setattr(ArticulationData, prop_name, prop)
            self._annotations_active = True

    def _remove_articulation_annotations(self):
        """Restore original properties."""
        if self._annotations_active:
            for prop_name, prop in self._articulation_originals.items():
                setattr(ArticulationData, prop_name, prop)
            self._annotations_active = False
            # Clear the tensor cache when done
            self._annotated_tensor_cache.clear()

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _record_articulation_access(self, obs_name: str, prop_name: str):
        """Record that an observation accessed an articulation property."""
        if obs_name not in self.observation_to_articulation_map:
            self.observation_to_articulation_map[obs_name] = set()
        self.observation_to_articulation_map[obs_name].add(prop_name)

    def _find_calling_observation(self) -> str | None:
        """Walk the stack to find the observation function that triggered access.

        Returns the IO descriptor name if available, otherwise the function name.
        """
        for frame_info in inspect.stack():
            if "isaaclab/envs/mdp/observations" in frame_info.filename:
                func_name = frame_info.function
                if func_name.startswith("_"):
                    continue

                # Try to get the IO descriptor name from the function's descriptor
                # The function object should be in the frame's global namespace
                frame_globals = frame_info.frame.f_globals
                if func_name in frame_globals:
                    func = frame_globals[func_name]
                    if hasattr(func, "_descriptor") and hasattr(func._descriptor, "name"):
                        return func._descriptor.name

                # Fallback to function name (which is what descriptor.name is set to anyway)
                return func_name
        return None

    # ──────────────────────────────────────────────────────────────────
    # Public API for accessing mappings
    # ──────────────────────────────────────────────────────────────────

    @property
    def get_semantic(self) -> dict[str, Any]:
        observations = []
        for k in self.io_descriptor_observations:
            obs_name = k["name"]
            observation = {
                "name": obs_name,
            }
            # Add the leapp input names this observation maps to (copy list to avoid YAML anchors)
            if obs_name in self.observation_to_articulation_map:
                observation["leapp_mapping"] = list(self.observation_to_articulation_map[obs_name])
            if "joint_names" in k:
                observation["joint_names"] = k["joint_names"]
            if "units" in k["extras"]:
                observation["units"] = k["extras"]["units"]
            observations.append(observation)

        actions = []
        for k in self.io_descriptor_actions:
            action_name = k["name"]
            action = {
                "name": action_name,
            }
            if action_name in self.action_io_to_term_map:
                action["leapp_mapping"] = list(self.action_io_to_term_map[action_name])
            if "joint_names" in k:
                action["joint_names"] = k["joint_names"]
            if "units" in k["extras"]:
                action["units"] = k["extras"]["units"]
            actions.append(action)

        scene = self.io_descriptor_scene

        return {
            "observations": observations,
            "actions": actions,
            "scene": scene,
        }
