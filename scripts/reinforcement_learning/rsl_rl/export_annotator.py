# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export annotations for Isaac Lab policies using proxy-based patching.

Observation and action annotation is achieved by routing calls through lightweight
proxy objects rather than globally patching class methods. This means:

- Observation term functions see an _EnvProxy whose scene returns _ArticulationProxy
  objects with annotating data getters. The real env / data is shared and unmodified.

- Action terms have their ``_asset`` attribute replaced with an _ArticulationWriteProxy
  that intercepts ``_leapp_semantics``-decorated write methods and records outputs.
  The real Articulation class is never patched.
"""

from __future__ import annotations

import inspect
import torch
from contextlib import suppress
from typing import TYPE_CHECKING

from leapp import annotate
from leapp.utils.tensor_description import TensorSemantics

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.utils.leapp_semantics import resolve_leapp_element_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ══════════════════════════════════════════════════════════════════
# Observation-side proxies
# ══════════════════════════════════════════════════════════════════


class _ArticulationDataProxy:
    """Proxy around a real ArticulationData that intercepts annotated property reads.

    For properties whose getter carries ``_leapp_semantics``, the proxy calls
    the annotating getter (which records the tensor with LEAPP) and caches the
    result for deduplication within a single ``compute_group`` call.

    All other attribute access is forwarded transparently to the real object.
    """

    def __init__(self, real_data: ArticulationData, annotating_getters: dict[str, callable], cache: dict):
        object.__setattr__(self, "_real_data", real_data)
        object.__setattr__(self, "_annotating_getters", annotating_getters)
        object.__setattr__(self, "_cache", cache)

    def __getattr__(self, name):
        """Intercept annotated properties; forward everything else."""
        getters = object.__getattribute__(self, "_annotating_getters")
        if name in getters:
            cache = object.__getattribute__(self, "_cache")
            if name in cache:
                return cache[name].clone()
            real_data = object.__getattribute__(self, "_real_data")
            result = getters[name](real_data)
            cache[name] = result
            return result
        return getattr(object.__getattribute__(self, "_real_data"), name)


class _ArticulationProxy:
    """Proxy around a real Articulation that returns _ArticulationDataProxy for ``.data``.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(self, real_asset: Articulation, data_proxy: _ArticulationDataProxy):
        object.__setattr__(self, "_real_asset", real_asset)
        object.__setattr__(self, "_data_proxy", data_proxy)

    @property
    def data(self):
        """Return the annotating data proxy instead of the real ArticulationData."""
        return object.__getattribute__(self, "_data_proxy")

    def __getattr__(self, name):
        """Forward all non-data attribute access to the real asset."""
        return getattr(object.__getattribute__(self, "_real_asset"), name)


class _SceneProxy:
    """Proxy around the real InteractiveScene.

    When an observation term looks up an asset by name, this proxy lazily wraps
    Articulation entities in _ArticulationProxy so their data getters annotate.
    Non-Articulation entities are returned as-is.
    """

    def __init__(self, real_scene, annotating_getters: dict[str, callable], cache: dict):
        object.__setattr__(self, "_real_scene", real_scene)
        object.__setattr__(self, "_annotating_getters", annotating_getters)
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_proxied", {})

    def __getitem__(self, key):
        """Return an ArticulationProxy for Articulation entities, real entity otherwise."""
        proxied = object.__getattribute__(self, "_proxied")
        if key in proxied:
            return proxied[key]
        real_scene = object.__getattribute__(self, "_real_scene")
        entity = real_scene[key]
        if isinstance(entity, Articulation):
            getters = object.__getattribute__(self, "_annotating_getters")
            cache = object.__getattribute__(self, "_cache")
            data_proxy = _ArticulationDataProxy(entity.data, getters, cache)
            proxy = _ArticulationProxy(entity, data_proxy)
            proxied[key] = proxy
            return proxy
        return entity

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


# ══════════════════════════════════════════════════════════════════
# Action-side proxy
# ══════════════════════════════════════════════════════════════════


class _ArticulationWriteProxy:
    """Proxy around a real Articulation that intercepts ``_leapp_semantics`` write methods.

    When an action term calls e.g. ``self._asset.set_joint_position_target(target, joint_ids)``,
    this proxy:

    1. Calls the real method on the real asset (so the simulation sees the write).
    2. Snapshots the ``target`` tensor and records a ``TensorSemantics`` entry in the
       shared output cache.

    All other attribute access is forwarded transparently to the real asset.
    """

    def __init__(
        self,
        real_asset: Articulation,
        term_name: str,
        output_cache: list[TensorSemantics],
        annotating_methods: dict[str, callable],
    ):
        object.__setattr__(self, "_real_asset", real_asset)
        object.__setattr__(self, "_term_name", term_name)
        object.__setattr__(self, "_output_cache", output_cache)
        object.__setattr__(self, "_annotating_methods", annotating_methods)

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
# ObservationPatcher
# ══════════════════════════════════════════════════════════════════


class ObservationPatcher:
    """Permanently patches observation term functions to annotate their inputs via proxies.

    Instead of globally patching ArticulationData properties and toggling them on/off,
    this scans for ``_leapp_semantics``-decorated properties once, builds annotating
    getters, and routes observation term calls through lightweight proxy objects that
    share the same underlying env and tensor state.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self._annotated_tensor_cache: dict[str, torch.Tensor] = {}

    def setup(self, obs_manager):
        """Patch all observation terms to use annotating proxies.

        For each term in the observation manager:
        - Normal terms get their ``env`` argument swapped for a proxy env.
        - ``last_action`` and ``generated_commands`` get dedicated wrappers.
        - Noise is disabled on every term for deterministic export.

        A thin wrapper on ``compute_group`` clears the dedup cache between calls.
        """
        real_env = obs_manager._env

        annotating_getters = self._build_annotating_getters()
        scene_proxy = _SceneProxy(real_env.scene, annotating_getters, self._annotated_tensor_cache)
        proxy_env = _EnvProxy(real_env, scene_proxy)

        for group_name, term_cfgs in obs_manager._group_obs_term_cfgs.items():
            for term_cfg in term_cfgs:
                original_func = term_cfg.func
                func_name = getattr(original_func, "__name__", None)

                if func_name == "last_action":
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

    # ── Scanning ──────────────────────────────────────────────────

    def _build_annotating_getters(self) -> dict[str, callable]:
        """Scan ArticulationData for ``_leapp_semantics`` properties and build annotating getters.

        Returns a dict mapping property name to a callable ``(data_self) -> annotated_tensor``.
        """
        getters: dict[str, callable] = {}
        for prop_name in dir(ArticulationData):
            prop = getattr(ArticulationData, prop_name, None)
            if isinstance(prop, property) and prop.fget and hasattr(prop.fget, "_leapp_semantics"):
                getters[prop_name] = self._make_annotating_getter(prop.fget, prop_name)
        return getters

    def _make_annotating_getter(self, original_fget, prop_name: str):
        """Create an annotating getter callable for a single ArticulationData property.

        The returned callable invokes the real getter, then registers the result
        as a LEAPP input tensor with the property's semantic metadata.
        """
        task_name = self.task_name

        def getter(data_self):
            result = original_fget(data_self)
            if not isinstance(result, torch.Tensor):
                return result
            semantics_meta = getattr(original_fget, "_leapp_semantics", None)
            sem = TensorSemantics(
                name=prop_name,
                ref=result,
                kind=semantics_meta.kind if semantics_meta else None,
                element_names=resolve_leapp_element_names(semantics_meta, data_self),
            )
            return annotate.input_tensors(task_name, sem)

        return getter

    # ── Term wrappers ─────────────────────────────────────────────

    @staticmethod
    def _wrap_with_proxy(original_func, proxy_env):
        """Wrap a term function so it receives the proxy env instead of the real env.

        This is the generic wrapper for observation terms that read ArticulationData
        properties. By substituting the env, the entire downstream chain
        (env.scene[name].data.property) goes through the proxy.
        """

        def wrapped(env, **kwargs):
            return original_func(proxy_env, **kwargs)

        wrapped.__name__ = getattr(original_func, "__name__", "unknown")
        return wrapped

    def _wrap_last_action(self, original_func):
        """Wrap the ``last_action`` observation term to annotate its output as a LEAPP input."""
        task_name = self.task_name

        def wrapped(env, action_name=None, **kwargs):
            result = original_func(env, action_name, **kwargs)
            return annotate.input_tensors(task_name, {"last_actions": result})

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


# ══════════════════════════════════════════════════════════════════
# ActionPatcher
# ══════════════════════════════════════════════════════════════════


class ActionPatcher:
    """Permanently patches action terms to annotate their outputs via proxies.

    1. Scans Articulation for ``_leapp_semantics``-decorated methods once.
    2. Replaces each action term's ``_asset`` with an ``_ArticulationWriteProxy`` that
       intercepts those methods and records output semantics.
    3. Patches ``process_action`` and ``apply_action`` on the action manager instance
       to coordinate buffer registration and the single ``annotate.output_tensors`` call.

    No Articulation class methods are ever modified.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self._action_output_cache: list[TensorSemantics] = []
        self._pending_action_output_export: bool = False

    def setup(self, action_manager):
        """Patch all action terms and the action manager for LEAPP annotation.

        For each action term with an Articulation asset, replaces ``term._asset``
        with an ``_ArticulationWriteProxy``. Then patches ``process_action`` and
        ``apply_action`` on the manager instance.
        """
        annotating_methods = self._build_annotating_write_methods()

        for term_name, term in action_manager._terms.items():
            asset = getattr(term, "_asset", None)
            if isinstance(asset, Articulation):
                term._asset = _ArticulationWriteProxy(
                    real_asset=asset,
                    term_name=term_name,
                    output_cache=self._action_output_cache,
                    annotating_methods=annotating_methods,
                )

        self._patch_manager_methods(action_manager)

    # ── Scanning ──────────────────────────────────────────────────

    def _build_annotating_write_methods(self) -> dict[str, callable]:
        """Scan Articulation for ``_leapp_semantics`` methods and build interceptors.

        Returns a dict mapping method name to a factory callable. The factory takes
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
        method is accessed. It returns a callable that:

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

    # ── Manager patches ───────────────────────────────────────────

    def _patch_manager_methods(self, action_manager):
        """Patch ``process_action`` and ``apply_action`` on the action manager instance.

        ``process_action`` registers raw_action buffers for LEAPP tracing and
        preserves the action tensor clone.

        ``apply_action`` coordinates the output cache lifecycle: clear before,
        collect and export after.
        """
        original_process = action_manager.process_action
        original_apply = action_manager.apply_action
        task_name = self.task_name

        def patched_process_action(action: torch.Tensor):
            """Register raw_action buffers, call real process_action, preserve action clone."""
            for term_name, term in action_manager._terms.items():
                if hasattr(term, "_raw_actions") and term._raw_actions is not None:
                    term._raw_actions = annotate.register_buffer(task_name, {"raw_actions": term._raw_actions})

            original_process(action)
            action_manager._action = action.clone()
            self._pending_action_output_export = True

        def patched_apply_action():
            """Clear cache, call real apply_action, collect outputs, call annotate.output_tensors."""
            if not self._pending_action_output_export:
                return original_apply()

            self._action_output_cache.clear()
            original_apply()

            self._action_output_cache.extend(self._collect_action_outputs(action_manager))
            self._action_output_cache.append(TensorSemantics(name="last_action", ref=action_manager._action))
            static_values = self._collect_action_static_outputs(action_manager)
            annotate.output_tensors(
                task_name,
                self._action_output_cache,
                static_outputs=static_values,
                export_with="onnx",
            )
            self._pending_action_output_export = False
            self._action_output_cache.clear()
            return None

        action_manager.process_action = patched_process_action
        action_manager.apply_action = patched_apply_action

    # ── Output collection ─────────────────────────────────────────

    @staticmethod
    def _collect_action_outputs(action_manager) -> list[TensorSemantics]:
        """Collect non-writer action tensors that should be exported (e.g. OSC dynamic gains)."""
        tensors: list[TensorSemantics] = []
        for term_name, term in action_manager._terms.items():
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

    @staticmethod
    def _collect_action_static_outputs(action_manager) -> dict:
        """Collect static kp/kd gain values from action terms for export metadata."""
        static_values: dict = {}
        for term_name, term in action_manager._terms.items():
            osc = getattr(term, "_osc", None)
            if osc and hasattr(osc, "cfg") and osc.cfg.impedance_mode in ["variable", "variable_kp"]:
                continue
            asset = getattr(term, "_asset", None)
            real_asset = getattr(asset, "_real_asset", asset)
            if real_asset and hasattr(real_asset, "data"):
                data = real_asset.data
                joint_ids = getattr(term, "_joint_ids", None)
                if hasattr(data, "default_joint_stiffness") and data.default_joint_stiffness is not None:
                    gains = data.default_joint_stiffness
                    static_values[f"{term_name}_kp_gains"] = gains[:, joint_ids] if joint_ids else gains
                if hasattr(data, "default_joint_damping") and data.default_joint_damping is not None:
                    gains = data.default_joint_damping
                    static_values[f"{term_name}_kd_gains"] = gains[:, joint_ids] if joint_ids else gains
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

    This is a thin public entry point around ``ObservationPatcher`` and
    ``ActionPatcher``. It mutates the provided env instance in-place so that:

    - Observation terms route through proxy objects that annotate
      ``ArticulationData`` reads.
    - Action terms route through proxy objects that annotate
      ``Articulation`` write methods.

    The underlying env, scene, assets, and tensors remain shared with the rest
    of the pipeline; only the manager call paths are redirected.
    """
    unwrapped = env.env.unwrapped

    obs_patcher = ObservationPatcher(task_name)
    obs_patcher.setup(unwrapped.observation_manager)

    action_patcher = ActionPatcher(task_name)
    action_patcher.setup(unwrapped.action_manager)
