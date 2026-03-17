# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

import torch
from dataclasses import dataclass

from isaaclab.utils.leapp_semantics import leapp_tensor_semantics


@dataclass
class ContactSensorData:
    """Data container for the contact reporting sensor."""

    _body_names: list[str] | None = None
    _pos_w: torch.Tensor | None = None
    """Position of the sensor origin in world frame.

    Shape is (N, 3), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.

    """

    _contact_pos_w: torch.Tensor | None = None
    """Average of the positions of contact points between sensor body and filter prim in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Collision pairs not in contact will result in NaN.

    Note:

        * If the :attr:`ContactSensorCfg.track_contact_points` is False, then this quantity is None.
        * If the :attr:`ContactSensorCfg.track_contact_points` is True, a ValueError will be raised if:

          * If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
          * If the :attr:`ContactSensorCfg.max_contact_data_per_prim` is not specified or less than 1.
            will not be calculated.

    """

    _friction_forces_w: torch.Tensor | None = None
    """Sum of the friction forces between sensor body and filter prim in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Collision pairs not in contact will result in NaN.

    Note:

        * If the :attr:`ContactSensorCfg.track_friction_forces` is False, then this quantity is None.
        * If the :attr:`ContactSensorCfg.track_friction_forces` is True, a ValueError will be raised if:

          * The :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty.
          * The :attr:`ContactSensorCfg.max_contact_data_per_prim` is not specified or less than 1.

    """

    _quat_w: torch.Tensor | None = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, 4), where N is the number of sensors.

    Note:
        If the :attr:`ContactSensorCfg.track_pose` is False, then this quantity is None.
    """

    _net_forces_w: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, B, 3), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    _net_forces_w_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, B, 3), where N is the number of sensors, T is the configured history length
    and B is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    _force_matrix_w: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, B, M, 3), where N is the number of sensors, B is number of bodies in each sensor
    and M is the number of filtered bodies.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """

    _force_matrix_w_history: torch.Tensor | None = None
    """The normal contact forces filtered between the sensor bodies and filtered bodies in world frame.

    Shape is (N, T, B, M, 3), where N is the number of sensors, T is the configured history length,
    B is number of bodies in each sensor and M is the number of filtered bodies.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        If the :attr:`ContactSensorCfg.filter_prim_paths_expr` is empty, then this quantity is None.
    """

    _last_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air before the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    _current_air_time: torch.Tensor | None = None
    """Time spent (in s) in the air since the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    _last_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact before the last detach.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    _current_contact_time: torch.Tensor | None = None
    """Time spent (in s) in contact since the last contact.

    Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        If the :attr:`ContactSensorCfg.track_air_time` is False, then this quantity is None.
    """

    @property
    @leapp_tensor_semantics(kind="state/sensor/position", element_names_source="body_xyz")
    def pos_w(self) -> torch.Tensor | None:
        """Position of the sensor origin in world frame."""
        return self._pos_w

    @property
    @leapp_tensor_semantics(kind="state/sensor/rotation", element_names_source="body_quat")
    def quat_w(self) -> torch.Tensor | None:
        """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame."""
        return self._quat_w

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/contact_position")
    def contact_pos_w(self) -> torch.Tensor | None:
        """Average contact positions in world frame."""
        return self._contact_pos_w

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/friction_force")
    def friction_forces_w(self) -> torch.Tensor | None:
        """Friction forces between sensor body and filter prim in world frame."""
        return self._friction_forces_w

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/net_force", element_names_source="body_xyz")
    def net_forces_w(self) -> torch.Tensor | None:
        """Net normal contact forces in world frame."""
        return self._net_forces_w

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/net_force_history")
    def net_forces_w_history(self) -> torch.Tensor | None:
        """History of net normal contact forces in world frame."""
        return self._net_forces_w_history

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/force_matrix")
    def force_matrix_w(self) -> torch.Tensor | None:
        """Filtered contact force matrix in world frame."""
        return self._force_matrix_w

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/force_matrix_history")
    def force_matrix_w_history(self) -> torch.Tensor | None:
        """History of filtered contact force matrices in world frame."""
        return self._force_matrix_w_history

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/last_air_time", element_names_source="body_names")
    def last_air_time(self) -> torch.Tensor | None:
        """Time spent in the air before the last contact."""
        return self._last_air_time

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/current_air_time", element_names_source="body_names")
    def current_air_time(self) -> torch.Tensor | None:
        """Time spent in the air since the last detach."""
        return self._current_air_time

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/last_contact_time", element_names_source="body_names")
    def last_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact before the last detach."""
        return self._last_contact_time

    @property
    @leapp_tensor_semantics(kind="state/contact_sensor/current_contact_time", element_names_source="body_names")
    def current_contact_time(self) -> torch.Tensor | None:
        """Time spent in contact since the last contact."""
        return self._current_contact_time
