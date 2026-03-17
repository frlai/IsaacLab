# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from isaaclab.utils.leapp_semantics import leapp_tensor_semantics


@dataclass
class FrameTransformerData:
    """Data container for the frame transformer sensor."""

    _target_frame_names: list[str] = None
    """Target frame names (this denotes the order in which that frame data is ordered).

    The frame names are resolved from the :attr:`FrameTransformerCfg.FrameCfg.name` field.
    This does not necessarily follow the order in which the frames are defined in the config due to
    the regex matching.
    """

    _target_pos_source: torch.Tensor = None
    """Position of the target frame(s) relative to source frame.

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """

    _target_quat_source: torch.Tensor = None
    """Orientation of the target frame(s) relative to source frame quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """

    _target_pos_w: torch.Tensor = None
    """Position of the target frame(s) after offset (in world frame).

    Shape is (N, M, 3), where N is the number of environments, and M is the number of target frames.
    """

    _target_quat_w: torch.Tensor = None
    """Orientation of the target frame(s) after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, M, 4), where N is the number of environments, and M is the number of target frames.
    """

    _source_pos_w: torch.Tensor = None
    """Position of the source frame after offset (in world frame).

    Shape is (N, 3), where N is the number of environments.
    """

    _source_quat_w: torch.Tensor = None
    """Orientation of the source frame after offset (in world frame) quaternion (w, x, y, z).

    Shape is (N, 4), where N is the number of environments.
    """

    @property
    def target_frame_names(self) -> list[str] | None:
        """Target frame names in the same order as the target-frame tensors."""
        return self._target_frame_names

    @property
    @leapp_tensor_semantics(
        kind="state/frame_transformer/target_position_source", element_names=[None, None, ["x", "y", "z"]]
    )
    def target_pos_source(self) -> torch.Tensor:
        """Position of the target frame(s) relative to source frame."""
        return self._target_pos_source

    @property
    @leapp_tensor_semantics(
        kind="state/frame_transformer/target_rotation_source",
        element_names=[None, None, ["qw", "qx", "qy", "qz"]],
    )
    def target_quat_source(self) -> torch.Tensor:
        """Orientation of the target frame(s) relative to source frame quaternion (w, x, y, z)."""
        return self._target_quat_source

    @property
    @leapp_tensor_semantics(
        kind="state/frame_transformer/target_position_world", element_names=[None, None, ["x", "y", "z"]]
    )
    def target_pos_w(self) -> torch.Tensor:
        """Position of the target frame(s) after offset in world frame."""
        return self._target_pos_w

    @property
    @leapp_tensor_semantics(
        kind="state/frame_transformer/target_rotation_world",
        element_names=[None, None, ["qw", "qx", "qy", "qz"]],
    )
    def target_quat_w(self) -> torch.Tensor:
        """Orientation of the target frame(s) after offset in world frame quaternion (w, x, y, z)."""
        return self._target_quat_w

    @property
    @leapp_tensor_semantics(kind="state/frame_transformer/source_position_world", element_names=[None, ["x", "y", "z"]])
    def source_pos_w(self) -> torch.Tensor:
        """Position of the source frame after offset in world frame."""
        return self._source_pos_w

    @property
    @leapp_tensor_semantics(
        kind="state/frame_transformer/source_rotation_world",
        element_names=[None, ["qw", "qx", "qy", "qz"]],
    )
    def source_quat_w(self) -> torch.Tensor:
        """Orientation of the source frame after offset in world frame quaternion (w, x, y, z)."""
        return self._source_quat_w
