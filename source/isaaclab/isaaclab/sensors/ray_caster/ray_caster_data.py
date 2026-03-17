# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass

from isaaclab.utils.leapp_semantics import leapp_tensor_semantics


@dataclass
class RayCasterData:
    """Data container for the ray-cast sensor."""

    _pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.
    """
    _quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.
    """
    _ray_hits_w: torch.Tensor = None
    """The ray hit positions in the world frame.

    Shape is (N, B, 3), where N is the number of sensors, B is the number of rays
    in the scan pattern per sensor.
    """

    @property
    @leapp_tensor_semantics(kind="state/sensor/position", element_names_source="xyz")
    def pos_w(self) -> torch.Tensor:
        """Position of the sensor origin in world frame."""
        return self._pos_w

    @property
    @leapp_tensor_semantics(kind="state/sensor/rotation", element_names_source="quat_wxyz")
    def quat_w(self) -> torch.Tensor:
        """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame."""
        return self._quat_w

    @property
    @leapp_tensor_semantics(kind="state/sensor/ray_hit_position")
    def ray_hits_w(self) -> torch.Tensor:
        """The ray hit positions in the world frame."""
        return self._ray_hits_w
