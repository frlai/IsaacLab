# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from leapp import annotate
from leapp.leapp_graph.traced_tensor import TracedTensor

from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.envs.mdp import observations
from isaaclab.managers.action_manager import ActionManager
from isaaclab.managers.observation_manager import ObservationManager

# Global storage for original and annotating ArticulationData properties
_articulation_data_originals = {}
_articulation_data_annotating = {}


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
        "applied_torque",  # joint_effort
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

        def make_annotating_fget(original, name):
            """Factory function to properly capture variables in closure."""

            def annotating_fget(self):
                result = original(self)
                if isinstance(result, torch.Tensor):
                    result = annotate.input_tensors({name: result}, node_name="observation_manager")
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

    def patched_last_action(env, action_name=None):
        result = original_last_action(env, action_name)
        result = annotate.input_tensors({"last_actions": result}, node_name="observation_manager")
        return result

    # Patch generated_commands observation function
    original_generated_commands = observations.generated_commands

    def patched_generated_commands(env, command_name=None):
        result = original_generated_commands(env, command_name)
        result = annotate.input_tensors({"commands": result}, node_name="observation_manager")
        return result

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

    Also collects static values (default_joint_stiffness and default_joint_damping)
    from action terms that have them.

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported.
    """

    # Patch ActionManager.process_action at class level
    original_process_action = ActionManager.process_action

    def patched_process_action(self, action: torch.Tensor):
        action = annotate.input_tensors({"actions": action}, node_name="action_manager")
        original_process_action(self, action)
        annotate.mirror_leapp_tags(action, self._action)
        tensors = {}
        static_values = {}

        for term_name, term_ in self._terms.items():
            tensors[term_name] = term_.processed_actions

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

    print("Patched action manager: ActionManager.process_action")


def add_leapp_annotations():
    """
    Adds all leapp annotations for exporting Isaac Lab policies.

    This is the main entry point that patches:
    - ObservationManager and related observation functions
    - ActionManager.process_action

    IMPORTANT: Must be called BEFORE isaaclab_tasks is imported.
    """
    annotate_observation_manager()
    annotate_action_manager()
    print("All leapp annotations added")
