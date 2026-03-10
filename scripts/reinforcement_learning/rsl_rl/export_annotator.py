# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export annotations for Isaac Lab policies using instance-level patching."""


from __future__ import annotations

import inspect
import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.utils.leapp_semantics import resolve_leapp_element_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# class ObservationPatcher:

# class ActionPatcher:


@dataclass
class ExportAnnotator:
    """Encapsulates all leapp annotation logic for exporting Isaac Lab policies.

    Usage:
        env = gym.make(...)
        annotator = ExportAnnotator(env)
        annotator.setup()
        # ... run policy ...
        annotator.cleanup()
    """

    env: ManagerBasedEnv
    task_name: str

    # Original methods for restoration
    _original_compute_group: callable = field(default=None, repr=False)
    _original_process_action: callable = field(default=None, repr=False)
    _original_apply_action: callable = field(default=None, repr=False)

    # ArticulationData patching state
    _articulation_originals: dict[str, property] = field(default_factory=dict, repr=False)
    _articulation_annotating: dict[str, property] = field(default_factory=dict, repr=False)
    _annotations_active: bool = field(default=False, repr=False)

    # Action writer patching state
    _action_write_originals: dict[str, callable] = field(default_factory=dict, repr=False)
    _action_write_annotating: dict[str, callable] = field(default_factory=dict, repr=False)
    _action_write_annotations_active: bool = field(default=False, repr=False)

    # Cache for annotated tensors within a single compute_group call
    # Prevents duplicate input tensors when same property is accessed multiple times
    _annotated_tensor_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    _action_output_cache: list[TensorSemantics] = field(default_factory=list, repr=False)
    _active_action_term_name: str | None = field(default=None, repr=False)
    _pending_action_output_export: bool = field(default=False, repr=False)

    def setup(self):
        """Set up all annotations. Call after env is created."""
        self._setup_observation_annotations()
        self._prepare_action_write_annotations()
        self._patch_action_manager()

    def cleanup(self):
        """Restore all original methods and properties."""
        self._restore_observation_annotations()
        self._restore_action_manager()
        self._remove_action_write_annotations()

    # ──────────────────────────────────────────────────────────────────
    # Observation Annotations
    # ──────────────────────────────────────────────────────────────────

    def _setup_observation_annotations(self):
        """Set up all observation-side annotations."""
        self._disable_observation_noise()
        self._prepare_articulation_annotations()
        self._patch_observation_functions()
        self._patch_observation_manager()

    def _restore_observation_annotations(self):
        """Restore all observation-side patches and temporary annotations."""
        self._restore_observation_functions()
        self._restore_observation_manager()
        self._remove_articulation_annotations()

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

        # find and patch all other known non-articulation data properties
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
        """Create a patched version of last_action for LEAPP tracing."""

        def patched_last_action(env, action_name=None, **kwargs):
            result = original_func(env, action_name, **kwargs)
            result = annotate.input_tensors(self.task_name, {"last_actions": result})
            return result

        patched_last_action.__name__ = original_func.__name__
        return patched_last_action

    def _make_patched_generated_commands(self, original_func, term_cfg):
        """Create a patched version of generated_commands for LEAPP tracing."""
        # Get the command_name from term_cfg.params if available
        command_name_from_cfg = term_cfg.params.get("command_name")

        def patched_generated_commands(env, command_name=None, **kwargs):
            result = original_func(env, command_name, **kwargs)
            # Use command_name parameter, or fall back to config, or default
            leapp_input_name = command_name or command_name_from_cfg or "commands"
            command_cfg = None
            try:
                command_cfg = env.command_manager.get_term(leapp_input_name).cfg
            except (AttributeError, KeyError):
                # Keep export working even if the observation term doesn't point to a registered command term.
                command_cfg = None

            semantics = TensorSemantics(
                name=leapp_input_name,
                ref=result,
                kind=getattr(command_cfg, "cmd_hint", None),
                element_names=getattr(command_cfg, "element_names", None),
            )
            result = annotate.input_tensors(self.task_name, semantics)
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
    # ArticulationData Property Annotations
    # ──────────────────────────────────────────────────────────────────

    def _prepare_articulation_annotations(self):
        """Prepare annotating versions of ArticulationData properties."""
        for prop_name in self._get_semantic_articulation_properties():
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

            if isinstance(result, torch.Tensor):
                # Check if this property was already annotated in this compute_group call
                if prop_name in self._annotated_tensor_cache:
                    # Return a clone of the cached tensor to avoid duplicate input annotations
                    return self._annotated_tensor_cache[prop_name].clone()

                sem = self._make_property_semantics(prop_name, data_self, result)

                # First access - annotate and cache
                result = annotate.input_tensors(self.task_name, sem)
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
    # Action Manager
    # ──────────────────────────────────────────────────────────────────

    def _patch_action_manager(self):
        """Patch the action manager instance's action processing methods."""
        action_manager = self.env.env.unwrapped.action_manager
        self._original_process_action = action_manager.process_action
        self._original_apply_action = action_manager.apply_action

        def patched_process_action(action: torch.Tensor):
            # Register raw_actions buffers for tracing
            for term_name, term in action_manager._terms.items():
                if hasattr(term, "_raw_actions") and term._raw_actions is not None:
                    term._raw_actions = annotate.register_buffer(self.task_name, {"raw_actions": term._raw_actions})

            self._original_process_action(action)
            # this is stored differently inside the original process action method that would loose tracing. this step preserves it.
            action_manager._action = action.clone()
            self._pending_action_output_export = True

        def patched_apply_action():
            if not self._pending_action_output_export:
                return self._original_apply_action()

            original_term_apply_actions: dict[str, callable] = {}
            self._action_output_cache.clear()
            self._apply_action_write_annotations()

            try:
                for term_name, term in action_manager._terms.items():
                    original_term_apply_actions[term_name] = term.apply_actions
                    term.apply_actions = self._make_patched_term_apply_actions(term.apply_actions, term_name)

                self._original_apply_action()

                self._action_output_cache.extend(self._collect_action_outputs(action_manager))
                self._action_output_cache.append(TensorSemantics(name="last_action", ref=action_manager._action))
                static_values = self._collect_action_static_outputs(action_manager)
                annotate.output_tensors(
                    self.task_name,
                    self._action_output_cache,
                    static_outputs=static_values,
                    export_with="onnx",
                )
                self._pending_action_output_export = False
            finally:
                for term_name, original_apply_actions in original_term_apply_actions.items():
                    action_manager._terms[term_name].apply_actions = original_apply_actions
                self._active_action_term_name = None
                self._remove_action_write_annotations()
                self._action_output_cache.clear()

        action_manager.process_action = patched_process_action
        action_manager.apply_action = patched_apply_action

    def _make_patched_term_apply_actions(self, original_func, term_name: str):
        """Wrap an action term's apply call to keep the current term context."""

        def patched_apply_actions():
            self._active_action_term_name = term_name
            try:
                return original_func()
            finally:
                self._active_action_term_name = None

        return patched_apply_actions

    def _collect_action_outputs(self, action_manager) -> list[TensorSemantics]:
        """Collect non-writer action tensors that should still be exported."""
        tensors: list[TensorSemantics] = []

        for term_name, term in action_manager._terms.items():
            # Handle variable impedance (dynamic gains)
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kp_gains",
                        ref=torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1),
                        kind="kp",
                    )
                )
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kd_gains",
                        ref=torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1),
                        kind="kp",
                    )
                )
        return tensors

    def _collect_action_static_outputs(self, action_manager) -> dict:
        """Collect static values from action terms."""
        static_values = {}
        for term_name, term in action_manager._terms.items():
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                continue
            asset = getattr(term, "_asset", None)
            if asset and hasattr(asset, "data"):
                self._collect_static_gains(term_name, asset.data, getattr(term, "_joint_ids", None), static_values)
        return static_values

    def _collect_static_gains(self, term_name: str, data, joint_ids, static_values: dict):
        """Extract static kp/kd gains from asset data."""
        if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
            gains = data.default_joint_stiffness
            static_values[f"{term_name}_kp_gains"] = gains[:, joint_ids] if joint_ids else gains

        if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
            gains = data.default_joint_damping
            static_values[f"{term_name}_kd_gains"] = gains[:, joint_ids] if joint_ids else gains

    def _restore_action_manager(self):
        """Restore original action manager methods."""
        if self._original_process_action:
            self.env.env.unwrapped.action_manager.process_action = self._original_process_action
        if self._original_apply_action:
            self.env.env.unwrapped.action_manager.apply_action = self._original_apply_action

    # ──────────────────────────────────────────────────────────────────
    # Action Write Annotations
    # ──────────────────────────────────────────────────────────────────

    def _prepare_action_write_annotations(self):
        """Prepare annotating versions of low-level action writer methods."""
        for method_name in self._get_semantic_action_write_methods():
            original_method = getattr(Articulation, method_name, None)
            if original_method is None:
                continue

            self._action_write_originals[method_name] = original_method
            self._action_write_annotating[method_name] = self._make_annotating_action_write_method(
                original_method, method_name
            )

    def _make_annotating_action_write_method(self, original_func, method_name: str):
        """Create an annotating version of a low-level action writer."""
        signature = inspect.signature(original_func)

        def annotating_method(asset_self, *args, **kwargs):
            result = original_func(asset_self, *args, **kwargs)
            bound_args = signature.bind_partial(asset_self, *args, **kwargs)
            target = bound_args.arguments.get("target")

            if not isinstance(target, torch.Tensor):
                return result

            output_name = self._get_action_output_name(method_name)
            semantics = getattr(self._action_write_originals[method_name], "_leapp_semantics", None)
            joint_ids = bound_args.arguments.get("joint_ids")
            tensor_target: torch.Tensor = target
            target_snapshot = tensor_target.clone()
            self._action_output_cache.append(
                TensorSemantics(
                    name=output_name,
                    ref=target_snapshot,
                    kind=semantics.kind if semantics is not None else None,
                    element_names=resolve_leapp_element_names(
                        semantics, self._make_joint_name_context(asset_self, joint_ids)
                    ),
                )
            )

            return result

        annotating_method.__name__ = original_func.__name__
        return annotating_method

    def _apply_action_write_annotations(self):
        """Temporarily apply annotating action writer methods."""
        if not self._action_write_annotations_active:
            for method_name, method in self._action_write_annotating.items():
                setattr(Articulation, method_name, method)
            self._action_write_annotations_active = True

    def _remove_action_write_annotations(self):
        """Restore original low-level action writer methods."""
        if self._action_write_annotations_active:
            for method_name, method in self._action_write_originals.items():
                setattr(Articulation, method_name, method)
            self._action_write_annotations_active = False

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_semantic_action_write_methods(self) -> frozenset[str]:
        """Collect Articulation methods that advertise LEAPP semantics."""
        methods = set()
        for method_name in dir(Articulation):
            method = getattr(Articulation, method_name, None)
            if callable(method) and hasattr(method, "_leapp_semantics"):
                methods.add(method_name)
        return frozenset(methods)

    def _get_action_output_name(self, method_name: str) -> str:
        """Return a stable output name for the current action write."""
        base_name = self._active_action_term_name or method_name
        output_name = base_name
        existing_names = {tensor.name for tensor in self._action_output_cache}
        if output_name in existing_names:
            output_name = f"{base_name}_{method_name}"
        suffix = 2
        while output_name in existing_names:
            output_name = f"{base_name}_{method_name}_{suffix}"
            suffix += 1
        return output_name

    def _make_joint_name_context(self, asset_self: Articulation, joint_ids):
        """Create a lightweight context for resolving runtime joint name subsets."""
        return type(
            "JointNameContext",
            (),
            {"joint_names": asset_self.joint_names, "_joint_ids": joint_ids},
        )()

    def _get_semantic_articulation_properties(self) -> frozenset[str]:
        """Collect ArticulationData properties that advertise LEAPP semantics."""
        properties = set()
        for prop_name in dir(ArticulationData):
            prop = getattr(ArticulationData, prop_name, None)
            if isinstance(prop, property) and prop.fget is not None and hasattr(prop.fget, "_leapp_semantics"):
                properties.add(prop_name)
        return frozenset(properties)

    def _make_property_semantics(
        self, prop_name: str, data_self: ArticulationData, tensor: torch.Tensor
    ) -> TensorSemantics:
        """Create semantic metadata for raw ArticulationData inputs."""
        semantics = getattr(self._articulation_originals[prop_name].fget, "_leapp_semantics", None)
        return TensorSemantics(
            name=prop_name,
            ref=tensor,
            kind=semantics.kind if semantics is not None else None,
            element_names=resolve_leapp_element_names(semantics, data_self),
        )
