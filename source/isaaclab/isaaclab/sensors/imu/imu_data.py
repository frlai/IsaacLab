# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.utils.leapp_semantics import leapp_tensor_semantics


@dataclass
class ImuData:
    """Data container for the Imu sensor."""

    _pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    _quat_w: torch.Tensor = None
    """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame.

    Shape is (N, 4), where ``N`` is the number of environments.
    """

    _projected_gravity_b: torch.Tensor = None
    """Gravity direction unit vector projected on the imu frame.

    Shape is (N,3), where ``N`` is the number of environments.
    """

    _lin_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    _ang_vel_b: torch.Tensor = None
    """IMU frame angular velocity relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    _lin_acc_b: torch.Tensor = None
    """IMU frame linear acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    _ang_acc_b: torch.Tensor = None
    """IMU frame angular acceleration relative to the world expressed in IMU frame.

    Shape is (N, 3), where ``N`` is the number of environments.
    """

    @property
    @leapp_tensor_semantics(kind="state/sensor/position", element_names_source="xyz")
    def pos_w(self) -> torch.Tensor:
        """Position of the sensor origin in world frame."""
        return self._pos_w

    @property
    @leapp_tensor_semantics(kind="state/sensor/rotation", element_names_source="quat_wxyz")
    def quat_w(self) -> torch.Tensor:
        """Orientation of the sensor origin in quaternion ``(w, x, y, z)`` in world frame."""
        return self._quat_w

    @property
    @leapp_tensor_semantics(kind="state/imu/projected_gravity", element_names_source="xyz")
    def projected_gravity_b(self) -> torch.Tensor:
        """Gravity direction unit vector projected on the imu frame."""
        return self._projected_gravity_b

    @property
    @leapp_tensor_semantics(kind="state/imu/linear_velocity", element_names_source="xyz")
    def lin_vel_b(self) -> torch.Tensor:
        """IMU frame linear velocity relative to the world expressed in IMU frame."""
        return self._lin_vel_b

    @property
    @leapp_tensor_semantics(kind="state/imu/angular_velocity", element_names_source="xyz")
    def ang_vel_b(self) -> torch.Tensor:
        """IMU frame angular velocity relative to the world expressed in IMU frame."""
        return self._ang_vel_b

    @property
    @leapp_tensor_semantics(kind="state/imu/linear_acceleration", element_names_source="xyz")
    def lin_acc_b(self) -> torch.Tensor:
        """IMU frame linear acceleration relative to the world expressed in IMU frame."""
        return self._lin_acc_b

    @property
    @leapp_tensor_semantics(kind="state/imu/angular_acceleration", element_names_source="xyz")
    def ang_acc_b(self) -> torch.Tensor:
        """IMU frame angular acceleration relative to the world expressed in IMU frame."""
        return self._ang_acc_b
