# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export annotations for Isaac Lab policies using proxy-based patching.

Observation and action annotation share a single set of annotating getters
and a unified dedup cache so that a state property (e.g. ``joint_pos``)
read by both an observation term and an action term resolves to one LEAPP
input edge.

- Observation term functions see an _EnvProxy whose scene returns
  _ArticulationProxy objects with annotating data getters.

- Action terms have their ``_asset`` attribute replaced with an
  _ArticulationWriteProxy that intercepts ``_leapp_semantics``-decorated
  write methods **and** routes ``.data`` reads through the same annotating
  data proxy used by observations.

Cache lifecycle (assuming single-env play-mode export):

    compute_group()          clear cache → obs terms populate cache
    policy inference         TracedTensors propagate through NN
    process_action()         register_buffer for raw_actions
    apply_action() [tracing] reuse cached TracedTensors for state reads,
                             capture write outputs, call output_tensors(),
                             then clear cache
    apply_action() [decim.]  clear cache → fresh reads for simulation
    ...
    compute_group()          clear cache → fresh reads for next obs
"""

from __future__ import annotations

import inspect
import torch
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab.sensors.frame_transformer.frame_transformer_data import FrameTransformerData
from isaaclab.sensors.imu.imu_data import ImuData
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera_data import MultiMeshRayCasterCameraData
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.sensors.tacsl_sensor.visuotactile_sensor_data import VisuoTactileSensorData
from isaaclab.utils.leapp_semantics import resolve_leapp_element_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Reuse the generic joint-name resolver for kp/kd outputs by providing the
# same ``element_names_source`` contract as articulation getters/writers.
_GAIN_JOINT_SEMANTICS = type(
    "GainJointSemantics",
    (),
    {"element_names": None, "element_names_source": "joint_names"},
)()

_ANNOTATED_DATA_CLASSES = (
    ArticulationData,
    CameraData,
    ContactSensorData,
    FrameTransformerData,
    ImuData,
    MultiMeshRayCasterCameraData,
    MultiMeshRayCasterData,
    RayCasterData,
    VisuoTactileSensorData,
)


# ══════════════════════════════════════════════════════════════════
# Shared data proxy
# ══════════════════════════════════════════════════════════════════


def _lookup_annotating_getter(
    annotating_getters_by_type: dict[type, dict[str, callable]], real_data: Any, name: str
) -> callable | None:
    """Return the annotating getter for a property on the given data object, if any."""
    for data_cls in type(real_data).__mro__:
        getter = annotating_getters_by_type.get(data_cls, {}).get(name)
        if getter is not None:
            return getter
    return None


def _has_annotated_getters(annotating_getters_by_type: dict[type, dict[str, callable]], real_data: Any) -> bool:
    """Return True when the data object's class hierarchy exposes any annotated getters."""
    return any(annotating_getters_by_type.get(data_cls) for data_cls in type(real_data).__mro__)


class _DataProxy:
    """Proxy around a real data object that intercepts annotated property reads.

    The real data object may be an ``ArticulationData`` instance or any sensor
    data class whose properties carry ``_leapp_semantics``. The proxy calls the
    annotating getter (which records the tensor with LEAPP) and caches the
    result for deduplication. Consumers within the same annotation pass
    (observation terms **and** action terms) receive the same TracedTensor for
    repeated reads from the same underlying data object.

    All other attribute access is forwarded transparently to the real object.
    """

    def __init__(
        self,
        real_data: Any,
        annotating_getters_by_type: dict[type, dict[str, callable]],
        cache: dict,
        input_name_resolver: callable,
    ):
        object.__setattr__(self, "_real_data", real_data)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_input_name_resolver", input_name_resolver)

    def __getattr__(self, name):
        """Intercept annotated properties; forward everything else."""
        real_data = object.__getattribute__(self, "_real_data")
        getter = _lookup_annotating_getter(
            object.__getattribute__(self, "_annotating_getters_by_type"), real_data, name
        )
        if getter is not None:
            cache = object.__getattribute__(self, "_cache")
            cache_key = (id(real_data), name)
            if cache_key in cache:
                return cache[cache_key].clone()
            input_name = object.__getattribute__(self, "_input_name_resolver")(name)
            result = getter(real_data, input_name)
            cache[cache_key] = result
            return result
        return getattr(real_data, name)


# ══════════════════════════════════════════════════════════════════
# Observation-side proxies
# ══════════════════════════════════════════════════════════════════


class _EntityProxy:
    """Proxy around a real scene entity that returns a ``_DataProxy`` for ``.data``.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(self, real_entity: Any, data_proxy: _DataProxy):
        object.__setattr__(self, "_real_entity", real_entity)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the annotating data proxy instead of the real data object."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Forward all non-data attribute access to the real scene entity."""
        return getattr(object.__getattribute__(self, "_real_entity"), name)


class _EntityMappingProxy:
    """Proxy around a mapping of scene entities that lazily wraps data-producing entries."""

    def __init__(self, real_mapping, annotating_getters_by_type: dict[type, dict[str, callable]], cache: dict):
        object.__setattr__(self, "_real_mapping", real_mapping)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_proxied", {})

    def __getitem__(self, key):
        """Return a proxied entity when it exposes annotated data properties."""
        proxied = object.__getattribute__(self, "_proxied")
        if key in proxied:
            return proxied[key]
        real_mapping = object.__getattribute__(self, "_real_mapping")
        entity = real_mapping[key]
        data = getattr(entity, "data", None)
        if data is None:
            return entity
        annotating_getters_by_type = object.__getattribute__(self, "_annotating_getters_by_type")
        if not _has_annotated_getters(annotating_getters_by_type, data):
            return entity
        data_proxy = _DataProxy(
            data,
            annotating_getters_by_type,
            object.__getattribute__(self, "_cache"),
            input_name_resolver=lambda prop_name: f"{key}_{prop_name}",
        )
        proxy = _EntityProxy(entity, data_proxy)
        proxied[key] = proxy
        return proxy

    def get(self, key, default=None):
        """Return a proxied entity when present, default otherwise."""
        real_mapping = object.__getattribute__(self, "_real_mapping")
        if key not in real_mapping:
            return default
        return self[key]

    def __iter__(self):
        return iter(object.__getattribute__(self, "_real_mapping"))

    def __len__(self):
        return len(object.__getattribute__(self, "_real_mapping"))

    def __getattr__(self, name):
        """Forward all other mapping access to the real mapping."""
        return getattr(object.__getattribute__(self, "_real_mapping"), name)


class _SceneProxy:
    """Proxy around the real InteractiveScene.

    When an observation term looks up a scene entity by name, this proxy lazily
    wraps entities whose ``.data`` object exposes ``_leapp_semantics``-decorated
    properties. This covers articulations and sensors through both
    ``scene["name"]`` and ``scene.sensors["name"]`` access paths.
    """

    def __init__(self, real_scene, annotating_getters_by_type: dict[type, dict[str, callable]], cache: dict):
        object.__setattr__(self, "_real_scene", real_scene)
        object.__setattr__(self, "_annotating_getters_by_type", annotating_getters_by_type)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_proxied", {})
        object.__setattr__(self, "_sensor_mapping_proxy", None)

    def _maybe_proxy_entity(self, key: str, entity: Any):
        """Return a proxy for entities whose data object has annotated getters."""
        proxied = object.__getattribute__(self, "_proxied")
        if key in proxied:
            return proxied[key]

        data = getattr(entity, "data", None)
        if data is None:
            return entity

        annotating_getters_by_type = object.__getattribute__(self, "_annotating_getters_by_type")
        if not _has_annotated_getters(annotating_getters_by_type, data):
            return entity

        cache = object.__getattribute__(self, "_cache")
        data_proxy = _DataProxy(
            data,
            annotating_getters_by_type,
            cache,
            input_name_resolver=(
                (lambda prop_name: f"ego_{prop_name}")
                if isinstance(entity, Articulation)
                else (lambda prop_name: f"{key}_{prop_name}")
            ),
        )
        proxy = _EntityProxy(entity, data_proxy)
        proxied[key] = proxy
        return proxy

    def __getitem__(self, key):
        """Return a proxied entity when it exposes annotated data getters."""
        real_scene = object.__getattribute__(self, "_real_scene")
        entity = real_scene[key]
        return self._maybe_proxy_entity(key, entity)

    @property
    def sensors(self):
        """Return a mapping proxy for scene sensors."""
        sensor_mapping_proxy = object.__getattribute__(self, "_sensor_mapping_proxy")
        if sensor_mapping_proxy is None:
            real_scene = object.__getattribute__(self, "_real_scene")
            sensor_mapping_proxy = _EntityMappingProxy(
                real_scene.sensors,
                object.__getattribute__(self, "_annotating_getters_by_type"),
                object.__getattribute__(self, "_cache"),
            )
            object.__setattr__(self, "_sensor_mapping_proxy", sensor_mapping_proxy)
        return sensor_mapping_proxy

    def __getattr__(self, name):
        """Forward all other scene access to the real scene."""
        return getattr(object.__getattribute__(self, "_real_scene"), name)


class _EnvProxy:
    """Proxy around the real env that returns a _SceneProxy for ``.scene``.

    All other attribute access (``num_envs``, ``command_manager``, etc.)
    is forwarded transparently to the real env.
    """

    def __init__(self, real_env, scene_proxy: _SceneProxy):
        object.__setattr__(self, "_real_env", real_env)
        object.__setattr__(self, "_scene_proxy", scene_proxy)

    @property
    def scene(self):
        """Return the scene proxy instead of the real scene."""
        return object.__getattribute__(self, "_scene_proxy")

    def __getattr__(self, name):
        """Forward all non-scene attribute access to the real env."""
        return getattr(object.__getattribute__(self, "_real_env"), name)


class _ManagerTermProxy(ManagerTermBase):
    """Proxy a class-based manager term while preserving its lifecycle methods.

    Observation manager terms can be stateful ``ManagerTermBase`` instances that
    expose ``reset()`` and ``serialize()`` in addition to being callable. This
    proxy preserves that interface while swapping the env argument passed into
    ``__call__`` for the observation-side proxy env.
    """

    def __init__(self, target: ManagerTermBase, proxy_env: _EnvProxy):
        super().__init__(target.cfg, target._env)
        self._target = target
        self._proxy_env = proxy_env

    @property
    def __name__(self) -> str:
        """Expose the wrapped term name for compatibility and debugging."""
        return getattr(self._target, "__name__", self._target.__class__.__name__)

    def reset(self, env_ids=None) -> None:
        """Forward resets to the wrapped term instance."""
        self._target.reset(env_ids=env_ids)

    def serialize(self) -> dict:
        """Forward serialization to the wrapped term instance."""
        return self._target.serialize()

    def __call__(self, *args, **kwargs):
        """Call the wrapped term with the proxy env in place of the real env."""
        if args:
            args = (self._proxy_env, *args[1:])
        else:
            args = (self._proxy_env,)
        return self._target(*args, **kwargs)

    def __getattr__(self, name):
        """Forward all other attribute access to the wrapped term instance."""
        return getattr(self._target, name)


# ══════════════════════════════════════════════════════════════════
# Action-side proxy
# ══════════════════════════════════════════════════════════════════


class _ArticulationWriteProxy:
    """Proxy around a real Articulation for action terms.

    Intercepts ``_leapp_semantics``-decorated write methods **and** routes
    ``.data`` reads through a shared ``_DataProxy`` so that
    action-side state reads (e.g. ``self._asset.data.joint_pos`` inside
    ``RelativeJointPositionAction``) participate in LEAPP annotation and
    share the dedup cache with observation-side reads.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(
        self,
        real_asset: Articulation,
        term_name: str,
        output_cache: list[TensorSemantics],
        annotating_methods: dict[str, callable],
        data_proxy: _DataProxy,
    ):
        object.__setattr__(self, "_real_asset", real_asset)
        object.__setattr__(self, "_term_name", term_name)
        object.__setattr__(self, "_output_cache", output_cache)
        object.__setattr__(self, "_annotating_methods", annotating_methods)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the shared annotating data proxy."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Return an annotating wrapper for _leapp_semantics methods; forward everything else."""
        methods = object.__getattribute__(self, "_annotating_methods")
        if name in methods:
            real_asset = object.__getattribute__(self, "_real_asset")
            term_name = object.__getattribute__(self, "_term_name")
            output_cache = object.__getattribute__(self, "_output_cache")
            original_method = getattr(real_asset, name)
            return methods[name](real_asset, original_method, term_name, output_cache)
        return getattr(object.__getattribute__(self, "_real_asset"), name)


# ══════════════════════════════════════════════════════════════════
# ExportPatcher
# ══════════════════════════════════════════════════════════════════


class ExportPatcher:
    """Unified patcher that annotates observation inputs and action outputs for LEAPP export.

    Builds a single set of annotating getters from ``ArticulationData`` and a
    shared dedup cache, then wires them into both:

    - The observation proxy chain (``_EnvProxy`` → ``_SceneProxy`` →
      ``_EntityProxy`` → ``_DataProxy``) for state reads
      by observation term functions.
    - The ``_ArticulationWriteProxy`` on each action term, which intercepts
      target writes **and** routes ``.data`` reads through the same
      ``_DataProxy`` / cache.

    This ensures that a property like ``joint_pos`` read by both an
    observation term and ``RelativeJointPositionAction.apply_actions()``
    resolves to a single LEAPP input edge rather than being silently baked
    in as a constant.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self._annotated_tensor_cache: dict[tuple[int, str], torch.Tensor] = {}
        self._action_output_cache: list[TensorSemantics] = []
        self._pending_action_output_export: bool = False
        self._uses_last_action_state: bool = False

    def setup(self, env):
        """Patch observation and action managers on the unwrapped env."""
        unwrapped = env.env.unwrapped

        annotating_getters = self._build_annotating_getters()
        annotating_write_methods = self._build_annotating_write_methods()
        cache = self._annotated_tensor_cache

        scene_proxy = _SceneProxy(unwrapped.scene, annotating_getters, cache)
        proxy_env = _EnvProxy(unwrapped, scene_proxy)

        self._patch_observation_manager(unwrapped.observation_manager, proxy_env)
        self._patch_action_manager(
            unwrapped.action_manager,
            annotating_getters,
            cache,
            annotating_write_methods,
        )

    # ── Scanning ──────────────────────────────────────────────────

    def _build_annotating_getters(self) -> dict[type, dict[str, callable]]:
        """Scan articulation and sensor data classes for ``_leapp_semantics`` properties.

        Returns a dict mapping data class type to a dict of
        ``property_name -> callable(data_self, input_name) -> annotated_tensor``.
        """
        getters: dict[type, dict[str, callable]] = {}
        for data_cls in _ANNOTATED_DATA_CLASSES:
            class_getters: dict[str, callable] = {}
            for prop_name in dir(data_cls):
                prop = getattr(data_cls, prop_name, None)
                if isinstance(prop, property) and prop.fget and hasattr(prop.fget, "_leapp_semantics"):
                    class_getters[prop_name] = self._make_annotating_getter(prop.fget, prop_name)
            if class_getters:
                getters[data_cls] = class_getters
        return getters

    def _make_annotating_getter(self, original_fget, prop_name: str):
        """Create an annotating getter callable for a single annotated data property.

        The returned callable invokes the real getter, then registers the result
        as a LEAPP input tensor with the property's semantic metadata and the
        caller-supplied public input name.
        """
        task_name = self.task_name

        def getter(data_self, input_name: str):
            result = original_fget(data_self)
            if not isinstance(result, torch.Tensor):
                return result
            semantics_meta = getattr(original_fget, "_leapp_semantics", None)
            sem = TensorSemantics(
                name=input_name,
                ref=result,
                kind=semantics_meta.kind if semantics_meta else None,
                element_names=resolve_leapp_element_names(semantics_meta, data_self),
            )
            return annotate.input_tensors(task_name, sem)

        return getter

    def _build_annotating_write_methods(self) -> dict[str, callable]:
        """Scan Articulation for ``_leapp_semantics`` methods and build interceptors.

        Returns a dict mapping method name to a factory callable.  The factory takes
        ``(real_asset, original_bound_method, term_name, output_cache)`` and returns
        a callable that the proxy returns in ``__getattr__``.
        """
        methods: dict[str, callable] = {}
        for method_name in dir(Articulation):
            method = getattr(Articulation, method_name, None)
            if callable(method) and hasattr(method, "_leapp_semantics"):
                methods[method_name] = self._make_write_interceptor_factory(method, method_name)
        return methods

    def _make_write_interceptor_factory(self, original_unbound, method_name: str):
        """Create a factory that produces bound annotating wrappers for a single write method.

        The factory is called by ``_ArticulationWriteProxy.__getattr__`` each time the
        method is accessed.  It returns a callable that:

        1. Calls the real method on the real asset.
        2. Inspects the ``target`` argument.
        3. Records a ``TensorSemantics`` entry in the shared output cache.
        """
        signature = inspect.signature(original_unbound)
        semantics = getattr(original_unbound, "_leapp_semantics", None)

        def factory(real_asset: Articulation, original_bound, term_name: str, output_cache: list):

            def interceptor(*args, **kwargs):
                result = original_bound(*args, **kwargs)
                bound_args = signature.bind_partial(real_asset, *args, **kwargs)
                target = bound_args.arguments.get("target")

                if isinstance(target, torch.Tensor):
                    tensor_target: torch.Tensor = target
                    output_name = _unique_output_name(term_name, method_name, output_cache)
                    joint_ids = bound_args.arguments.get("joint_ids")
                    output_cache.append(
                        TensorSemantics(
                            name=output_name,
                            ref=tensor_target.clone(),
                            kind=semantics.kind if semantics is not None else None,
                            element_names=resolve_leapp_element_names(
                                semantics,
                                _JointNameContext(real_asset.joint_names, joint_ids),
                            ),
                        )
                    )

                return result

            return interceptor

        return factory

    # ── Observation manager patches ───────────────────────────────

    def _patch_observation_manager(self, obs_manager, proxy_env):
        """Patch observation terms to use annotating proxies and disable noise."""
        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            for term_cfg in term_cfgs:
                original_func = term_cfg.func
                func_name = getattr(original_func, "__name__", None)

                if func_name == "last_action":
                    self._uses_last_action_state = True
                    term_cfg.func = self._wrap_last_action(original_func)
                elif func_name == "generated_commands":
                    term_cfg.func = self._wrap_generated_commands(original_func, term_cfg)
                else:
                    term_cfg.func = self._wrap_with_proxy(original_func, proxy_env)

                term_cfg.noise = None

        original_compute_group = obs_manager.compute_group
        cache = self._annotated_tensor_cache

        def patched_compute_group(*args, **kwargs):
            """Clear the tensor dedup cache, then run the real compute_group."""
            cache.clear()
            return original_compute_group(*args, **kwargs)

        obs_manager.compute_group = patched_compute_group

    # ── Action manager patches ────────────────────────────────────

    def _patch_action_manager(self, action_manager, annotating_getters, cache, annotating_write_methods):
        """Patch action terms with write+read proxies and patch manager methods."""
        for term_name, term in action_manager._terms.items():
            asset = getattr(term, "_asset", None)
            if isinstance(asset, Articulation):
                real_asset: Articulation = asset
                data_proxy = _DataProxy(
                    real_asset.data,
                    annotating_getters,
                    cache,
                    input_name_resolver=lambda prop_name: f"ego_{prop_name}",
                )
                term._asset = _ArticulationWriteProxy(
                    real_asset=real_asset,
                    term_name=term_name,
                    output_cache=self._action_output_cache,
                    annotating_methods=annotating_write_methods,
                    data_proxy=data_proxy,
                )

        self._patch_action_manager_methods(action_manager)

    def _patch_action_manager_methods(self, action_manager):
        """Patch ``process_action`` and ``apply_action`` on the action manager instance.

        ``process_action`` registers raw_action buffers for LEAPP tracing and
        preserves the action tensor clone.

        ``apply_action`` coordinates the cache and output lifecycle:

        - **Tracing pass** (first ``apply_action`` after ``process_action``):
          The cache still holds TracedTensors populated by ``compute_group``.
          Action terms that read state (e.g. ``RelativeJointPositionAction``
          reading ``joint_pos``) get those TracedTensors from the cache,
          keeping the LEAPP graph connected.  After ``output_tensors()`` the
          cache is cleared so subsequent decimation sub-steps read fresh values.

        - **Non-tracing passes** (remaining decimation sub-steps and all
          subsequent iterations): The cache is cleared **before** running
          action terms so every ``.data`` read returns the current simulator
          value, preserving simulation correctness.
        """
        original_process = action_manager.process_action
        original_apply = action_manager.apply_action
        task_name = self.task_name
        cache = self._annotated_tensor_cache

        def patched_process_action(action: torch.Tensor):
            """Register raw_action buffers, call real process_action, preserve action clone."""
            for term_name, term in action_manager._terms.items():
                if hasattr(term, "_raw_actions") and term._raw_actions is not None:
                    term._raw_actions = annotate.register_buffer(task_name, {"raw_actions": term._raw_actions})

            original_process(action)
            action_manager._action = action.clone()
            self._pending_action_output_export = True

        def patched_apply_action():
            """Coordinate cache lifecycle and LEAPP output annotation."""
            if not self._pending_action_output_export:
                cache.clear()
                return original_apply()

            # Tracing pass: cache still holds TracedTensors from compute_group.
            self._action_output_cache.clear()
            original_apply()

            self._action_output_cache.extend(self._collect_action_outputs(action_manager))
            if self._uses_last_action_state:
                annotate.update_state(task_name, {"last_action": action_manager._action})
            static_values = self._collect_action_static_outputs(action_manager)
            annotate.output_tensors(
                task_name,
                self._action_output_cache,
                static_outputs=static_values,
                export_with="onnx",
            )
            self._pending_action_output_export = False
            self._action_output_cache.clear()
            cache.clear()
            return None

        action_manager.process_action = patched_process_action
        action_manager.apply_action = patched_apply_action

    # ── Observation term wrappers ─────────────────────────────────

    @staticmethod
    def _wrap_with_proxy(original_func, proxy_env):
        """Wrap a term function so it receives the proxy env instead of the real env."""

        if isinstance(original_func, ManagerTermBase):
            return _ManagerTermProxy(original_func, proxy_env)

        def wrapped(*args, **kwargs):
            if args:
                args = (proxy_env, *args[1:])
            else:
                args = (proxy_env,)
            return original_func(*args, **kwargs)

        wrapped.__name__ = getattr(original_func, "__name__", "unknown")
        return wrapped

    def _wrap_last_action(self, original_func):
        """Wrap ``last_action`` as a LEAPP state tensor.

        ``last_action`` is feedback state, not a regular dangling input.  We
        therefore register it through ``annotate.state_tensors(...)`` on the
        observation side and update it through ``annotate.update_state(...)``
        after the traced action pass.
        """
        task_name = self.task_name

        def wrapped(env, action_name=None, **kwargs):
            result = original_func(env, action_name, **kwargs)
            return annotate.state_tensors(task_name, {"last_action": result})

        wrapped.__name__ = original_func.__name__
        return wrapped

    def _wrap_generated_commands(self, original_func, term_cfg):
        """Wrap the ``generated_commands`` observation term to annotate its output as a LEAPP input.

        Resolves command semantics (kind, element_names) from the command manager
        configuration when available.
        """
        task_name = self.task_name
        command_name_from_cfg = term_cfg.params.get("command_name")

        def wrapped(env, command_name=None, **kwargs):
            result = original_func(env, command_name, **kwargs)
            leapp_input_name = command_name or command_name_from_cfg or "commands"
            command_cfg = None
            with suppress(AttributeError, KeyError):
                command_cfg = env.command_manager.get_term(leapp_input_name).cfg
            sem = TensorSemantics(
                name=leapp_input_name,
                ref=result,
                kind=getattr(command_cfg, "cmd_hint", None),
                element_names=getattr(command_cfg, "element_names", None),
            )
            return annotate.input_tensors(task_name, sem)

        wrapped.__name__ = original_func.__name__
        return wrapped

    # ── Output collection ─────────────────────────────────────────

    @staticmethod
    def _collect_action_outputs(action_manager) -> list[TensorSemantics]:
        """Collect non-writer action tensors that should be exported (e.g. OSC dynamic gains)."""
        tensors: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                asset = getattr(term, "_asset", None)
                real_asset = getattr(asset, "_real_asset", asset)
                joint_ids = getattr(term, "_joint_ids", None)
                joint_name_context = None
                if real_asset is not None and hasattr(real_asset, "joint_names"):
                    joint_name_context = _JointNameContext(real_asset.joint_names, joint_ids)
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kp_gains",
                        ref=torch.diagonal(osc._motion_p_gains_task, dim1=-2, dim2=-1),
                        kind="kp",
                        element_names=(
                            resolve_leapp_element_names(
                                _GAIN_JOINT_SEMANTICS,
                                joint_name_context,
                            )
                            if joint_name_context is not None
                            else None
                        ),
                    )
                )
                tensors.append(
                    TensorSemantics(
                        name=f"{term_name}_kd_gains",
                        ref=torch.diagonal(osc._motion_d_gains_task, dim1=-2, dim2=-1),
                        kind="kd",
                        element_names=(
                            resolve_leapp_element_names(
                                _GAIN_JOINT_SEMANTICS,
                                joint_name_context,
                            )
                            if joint_name_context is not None
                            else None
                        ),
                    )
                )
        return tensors

    @staticmethod
    def _collect_action_static_outputs(action_manager) -> list[TensorSemantics]:
        """Collect static kp/kd gain values from action terms for export metadata."""
        static_values: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                continue
            asset = getattr(term, "_asset", None)
            real_asset = getattr(asset, "_real_asset", asset)
            if real_asset and hasattr(real_asset, "data"):
                data = real_asset.data
                joint_ids = getattr(term, "_joint_ids", None)
                joint_name_context = None
                if hasattr(real_asset, "joint_names"):
                    joint_name_context = _JointNameContext(real_asset.joint_names, joint_ids)
                if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
                    gains = data.default_joint_stiffness
                    static_values.append(
                        TensorSemantics(
                            name=f"{term_name}_kp_gains",
                            ref=gains[:, joint_ids] if joint_ids else gains,
                            kind="kp",
                            element_names=(
                                resolve_leapp_element_names(
                                    _GAIN_JOINT_SEMANTICS,
                                    joint_name_context,
                                )
                                if joint_name_context is not None
                                else None
                            ),
                        )
                    )
                if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
                    gains = data.default_joint_damping
                    static_values.append(
                        TensorSemantics(
                            name=f"{term_name}_kd_gains",
                            ref=gains[:, joint_ids] if joint_ids else gains,
                            kind="kd",
                            element_names=(
                                resolve_leapp_element_names(
                                    _GAIN_JOINT_SEMANTICS,
                                    joint_name_context,
                                )
                                if joint_name_context is not None
                                else None
                            ),
                        )
                    )
        return static_values


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


class _JointNameContext:
    """Lightweight stand-in for resolving runtime joint name subsets in ``resolve_leapp_element_names``."""

    __slots__ = ("joint_names", "_joint_ids")

    def __init__(self, joint_names: list[str], joint_ids):
        self.joint_names = joint_names
        self._joint_ids = joint_ids


def _unique_output_name(term_name: str, method_name: str, output_cache: list[TensorSemantics]) -> str:
    """Return a stable, unique output name for an action write entry.

    Prefers ``term_name``, falls back to ``term_name_method_name``, and appends a
    numeric suffix if even that collides.
    """
    existing = {t.name for t in output_cache}
    candidate = term_name
    if candidate in existing:
        candidate = f"{term_name}_{method_name}"
    suffix = 2
    while candidate in existing:
        candidate = f"{term_name}_{method_name}_{suffix}"
        suffix += 1
    return candidate


# ══════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════


def patch_env_for_export(env: ManagerBasedEnv, task_name: str) -> None:
    """Patch the env's observation and action managers for LEAPP export.

    This is a thin public entry point around ``ExportPatcher``.  It mutates
    the provided env instance in-place so that:

    - Observation terms route through proxy objects that annotate
      ``ArticulationData`` reads.
    - Action terms route through proxy objects that annotate both
      ``ArticulationData`` reads **and** ``Articulation`` write methods.

    State reads are deduplicated across observation and action paths via a
    shared cache, so a property like ``joint_pos`` that is read by both an
    observation term and a relative-position action term appears as a single
    LEAPP input edge.

    The underlying env, scene, assets, and tensors remain shared with the rest
    of the pipeline; only the manager call paths are redirected.
    """
    patcher = ExportPatcher(task_name)
    patcher.setup(env)
