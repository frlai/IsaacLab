# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import torch

from leapp import annotate
from leapp.leapp_graph.traced_tensor import TracedTensor

from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.envs.mdp import observations
from isaaclab.managers.action_manager import ActionManager
from isaaclab.managers.observation_manager import ObservationManager

# Global storage for original and annotating ArticulationData properties
_articulation_data_originals = {}
_articulation_data_annotating = {}

# Global mapping built during execution: observation_function_name -> [articulation_property_names]
OBSERVATION_TO_ARTICULATION_MAP: dict[str, list[str]] = {}


def _record_articulation_access(observation_name: str, articulation_property: str):
    """Record that an observation function accessed an ArticulationData property."""
    if observation_name not in OBSERVATION_TO_ARTICULATION_MAP:
        OBSERVATION_TO_ARTICULATION_MAP[observation_name] = []
    if articulation_property not in OBSERVATION_TO_ARTICULATION_MAP[observation_name]:
        OBSERVATION_TO_ARTICULATION_MAP[observation_name].append(articulation_property)


def get_observation_to_articulation_map() -> dict[str, list[str]]:
    """Get a copy of the observation-to-articulation mapping.

    Returns:
        A dictionary mapping observation function names to lists of
        ArticulationData property names (leapp input names) they access.

        Example:
            {
                'base_lin_vel': ['root_lin_vel_b'],
                'joint_pos_rel': ['joint_pos'],
                'last_action': ['last_actions'],
                'generated_commands': ['commands'],
            }
    """
    return OBSERVATION_TO_ARTICULATION_MAP.copy()


def _find_calling_observation_function() -> str | None:
    """Walk up the call stack to find the observation function that triggered this access."""
    for frame_info in inspect.stack():
        # Look for frames in the observations module
        if "isaaclab/envs/mdp/observations" in frame_info.filename:
            func_name = frame_info.function
            # Skip internal/wrapper functions
            if not func_name.startswith("_"):
                return func_name

        # Also check for custom observation functions in user code
        # Could look for functions with _has_descriptor attribute
        frame_locals = frame_info.frame.f_locals
        if "self" in frame_locals and hasattr(frame_locals.get("self"), "_has_descriptor"):
            return frame_info.function

    return None


def _setup_articulation_data_annotations():
    """
    Prepares annotating versions of ArticulationData properties without applying them.

    The annotations will be temporarily applied only during compute_group calls,
    avoiding conflicts with rewards, terminations, commands, and actions that also
    access these properties.
    """

    # All observation properties - we can include all of them now since annotations
    # are only active during compute_group
    observation_properties = {
        # Root state (position, orientation, velocities)
        "root_pos_w",  # base_pos_z, root_pos_w
        "root_quat_w",  # root_quat_w
        "root_lin_vel_b",  # base_lin_vel
        "root_ang_vel_b",  # base_ang_vel
        "root_lin_vel_w",  # root_lin_vel_w
        "root_ang_vel_w",  # root_ang_vel_w
        "projected_gravity_b",  # 'projected_gravity_b',
        # Body state
        "body_pose_w",  # body_pose_w
        "body_quat_w",  # body_projected_gravity_b
        # Joint state
        "joint_pos",  # joint_pos, joint_pos_rel, joint_pos_limit_normalized
        "joint_vel",  # joint_vel, joint_vel_rel
    }

    for prop_name in observation_properties:
        attr = getattr(ArticulationData, prop_name, None)

        # Skip if attribute doesn't exist or isn't a property
        if attr is None or not isinstance(attr, property):
            raise ValueError(f"Attribute {prop_name} does not exist or is not a property")

        # Skip properties without a getter
        if attr.fget is None:
            raise ValueError(f"Attribute {prop_name} does not have a getter")

        # Store the original property
        _articulation_data_originals[prop_name] = attr

        # Create annotating getter
        original_fget = attr.fget

        def make_annotating_fget(original, prop_name):
            """Factory function to properly capture variables in closure."""

            def annotating_fget(self):
                result = original(self)

                # Find which observation function called us and record the mapping
                observation_name = _find_calling_observation_function()
                if observation_name:
                    _record_articulation_access(observation_name, prop_name)

                if isinstance(result, torch.Tensor):
                    result = annotate.input_tensors({prop_name: result}, node_name="observation_manager")
                return result

            return annotating_fget

        annotating_fget = make_annotating_fget(original_fget, prop_name)
        annotating_fget.__doc__ = original_fget.__doc__

        # Create annotating property
        annotating_property = property(fget=annotating_fget, fset=attr.fset, fdel=attr.fdel, doc=attr.__doc__)
        _articulation_data_annotating[prop_name] = annotating_property

    print(f"Prepared {len(_articulation_data_originals)} ArticulationData properties for temporary annotation")


def _apply_articulation_annotations():
    """Temporarily applies annotating versions of ArticulationData properties."""
    for prop_name, annotating_prop in _articulation_data_annotating.items():
        setattr(ArticulationData, prop_name, annotating_prop)


def _remove_articulation_annotations():
    """Restores original ArticulationData properties."""
    for prop_name, original_prop in _articulation_data_originals.items():
        setattr(ArticulationData, prop_name, original_prop)


def configure_for_export():
    """
    Configures the environment managers for deterministic export.

    This patches ObservationManager to disable noise/corruption during export.
    Random operations like torch.rand_like in noise models cause validation
    failures because they produce different values on each run.

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported and before
    annotate_observation_manager() and annotate_action_manager().
    """
    from isaaclab.managers.manager_term_cfg import ObservationGroupCfg

    # Patch ObservationManager._prepare_terms to force disable noise
    # This ensures deterministic outputs for export validation
    original_prepare_terms = ObservationManager._prepare_terms

    def patched_prepare_terms(self):
        # Force disable corruption on all observation groups before preparing terms
        # Iterate over config items the same way _prepare_terms does
        if isinstance(self.cfg, dict):
            group_cfg_items = self.cfg.items()
        else:
            group_cfg_items = self.cfg.__dict__.items()

        for group_name, group_cfg in group_cfg_items:
            if group_cfg is None:
                continue
            if isinstance(group_cfg, ObservationGroupCfg):
                group_cfg.enable_corruption = False

        # Call original _prepare_terms
        return original_prepare_terms(self)

    ObservationManager._prepare_terms = patched_prepare_terms
    print("Configured for export: Disabled observation noise/corruption for deterministic export")


def annotate_observation_manager():
    """
    Patches observation-related functions and classes to annotate inputs/outputs.

    This patches:
    - ArticulationData properties (temporarily, only during compute_group)
    - Observation functions at the module level (last_action, generated_commands, etc.)
    - ObservationManager.compute_group to annotate outputs

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported.
    """

    # Prepare (but don't apply) ArticulationData annotations
    _setup_articulation_data_annotations()

    # Patch last_action observation function
    original_last_action = observations.last_action

    def patched_last_action(env, action_name=None, **kwargs):
        # Pass through kwargs (including 'inspect' for IO descriptors)
        result = original_last_action(env, action_name, **kwargs)
        # Record the mapping for this custom observation
        _record_articulation_access("last_action", "last_actions")
        result = annotate.input_tensors({"last_actions": result}, node_name="observation_manager")
        return result

    # Preserve original signature and IO descriptor to pass manager validation checks
    patched_last_action.__signature__ = inspect.signature(original_last_action)
    if hasattr(original_last_action, "_descriptor"):
        patched_last_action._descriptor = original_last_action._descriptor
        patched_last_action._has_descriptor = original_last_action._has_descriptor

    # Patch generated_commands observation function
    original_generated_commands = observations.generated_commands

    def patched_generated_commands(env, command_name=None, **kwargs):
        # Pass through kwargs (including 'inspect' for IO descriptors)
        result = original_generated_commands(env, command_name, **kwargs)
        # Record the mapping for this custom observation (use command_name as the leapp input name)
        leapp_input_name = command_name if command_name else "commands"
        _record_articulation_access("generated_commands", leapp_input_name)
        result = annotate.input_tensors({leapp_input_name: result}, node_name="observation_manager")
        return result

    # Preserve original signature and IO descriptor to pass manager validation checks
    patched_generated_commands.__signature__ = inspect.signature(original_generated_commands)
    if hasattr(original_generated_commands, "_descriptor"):
        patched_generated_commands._descriptor = original_generated_commands._descriptor
        patched_generated_commands._has_descriptor = original_generated_commands._has_descriptor

    # Apply observation function patches at module level
    # Note: Observation functions that use ArticulationData properties (base_pos_z, root_pos_w,
    # root_quat_w, body_projected_gravity_b) don't need patching since the underlying
    # ArticulationData properties are temporarily annotated during compute_group.
    observations.last_action = patched_last_action
    observations.generated_commands = patched_generated_commands

    # Patch ObservationManager.compute_group to:
    # 1. Temporarily apply ArticulationData annotations before computing
    # 2. Annotate outputs
    # 3. Restore original ArticulationData properties after computing
    original_compute_group = ObservationManager.compute_group

    def patched_compute_group(self, *args, **kwargs):
        # Apply ArticulationData annotations only during observation computation
        _apply_articulation_annotations()
        try:
            output = original_compute_group(self, *args, **kwargs)
            annotate.output_tensors("observation_manager", output, export_with="torch", use_trace=True)
            if isinstance(output, TracedTensor):
                return output.tensor
            else:
                return output
        finally:
            # Always restore original properties, even if an exception occurs
            _remove_articulation_annotations()

    ObservationManager.compute_group = patched_compute_group


def annotate_action_manager():
    """
    Patches ActionManager.process_action to annotate action inputs/outputs.

    Also patches OperationalSpaceController.set_command for variable impedance tracing.

    Collects static values (default_joint_stiffness and default_joint_damping)
    from action terms that have them. For variable impedance controllers, the gains
    are captured as dynamic outputs instead.

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported.
    """

    # Patch OperationalSpaceController.set_command for variable impedance modes
    # This must be done before ActionManager.process_action is patched
    original_osc_set_command = OperationalSpaceController.set_command

    def patched_osc_set_command(
        self,
        command: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_task_frame_pose_b: torch.Tensor | None = None,
    ):
        # For variable impedance modes, register gain buffers before in-place assignment
        if self.cfg.impedance_mode in ["variable_kp", "variable"]:
            # Register the gain tensors as buffers so in-place assignment is traced
            self._motion_p_gains_task, self._motion_d_gains_task = annotate.register_buffer(
                "action_manager",
                {
                    "motion_p_gains_task": self._motion_p_gains_task,
                    "motion_d_gains_task": self._motion_d_gains_task,
                },
            )

        # Call original set_command - in-place assignments will now be traced
        return original_osc_set_command(self, command, current_ee_pose_b, current_task_frame_pose_b)

    OperationalSpaceController.set_command = patched_osc_set_command

    # Patch ActionManager.process_action at class level
    original_process_action = ActionManager.process_action

    def patched_process_action(self, action: torch.Tensor):
        action = annotate.input_tensors({"actions": action}, node_name="action_manager")

        # Register _raw_actions buffers for each action term before processing
        # This enables tracing through in-place assignments like: self._raw_actions[:] = actions
        for term_name, term_ in self._terms.items():
            if hasattr(term_, "_raw_actions") and term_._raw_actions is not None:
                buffers = annotate.register_buffer(
                    "action_manager",
                    {"raw_actions": term_._raw_actions},
                )
                term_._raw_actions = buffers["raw_actions"]

        original_process_action(self, action)
        annotate.mirror_leapp_tags(action, self._action)
        tensors = {}
        static_values = {}

        for term_name, term_ in self._terms.items():
            tensors[term_name] = term_.processed_actions

            # Check for dynamic gains from OperationalSpaceControllerAction
            osc = getattr(term_, "_osc", None)
            if osc is not None and hasattr(osc, "cfg"):
                if osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                    # Dynamic gains - these are now traced due to register_buffer
                    # Extract diagonal elements (the actual gain values)
                    tensors[f"{term_name}_kp_gains"] = torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1)
                    tensors[f"{term_name}_kd_gains"] = torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1)
                    continue  # Skip static gain collection for this term

            # Collect static values (kp/kd gains) if available
            asset = getattr(term_, "_asset", None)
            if asset is not None and hasattr(asset, "data"):
                data = asset.data
                joint_ids = getattr(term_, "_joint_ids", None)

                # Get default_joint_stiffness (kp gains)
                if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
                    if joint_ids is not None:
                        static_values[f"{term_name}_kp_gains"] = data.default_joint_stiffness[:, joint_ids]
                    else:
                        static_values[f"{term_name}_kp_gains"] = data.default_joint_stiffness

                # Get default_joint_damping (kd gains)
                if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
                    if joint_ids is not None:
                        static_values[f"{term_name}_kd_gains"] = data.default_joint_damping[:, joint_ids]
                    else:
                        static_values[f"{term_name}_kd_gains"] = data.default_joint_damping

        annotate.output_tensors("action_manager", tensors, static_outputs=static_values, export_with="torch")

    ActionManager.process_action = patched_process_action

    print("Patched action manager: ActionManager.process_action, OperationalSpaceController.set_command")


def add_leapp_annotations():
    """
    Adds all leapp annotations for exporting Isaac Lab policies.

    This is the main entry point that patches:
    - ObservationManager and related observation functions
    - ActionManager.process_action (includes OperationalSpaceController.set_command)

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported.
    """
    # Configure for deterministic export first (disables noise/corruption)
    configure_for_export()
    annotate_observation_manager()
    annotate_action_manager()
    print("All leapp annotations added")
