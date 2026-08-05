import csv
import datetime
import importlib
import inspect
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from adapters.tool_registry import ToolRegistry
from agent.experiment_executor import ExperimentExecuteAgent
from core_tool.tool_utils import SayCapture
from runtime.agent_factory import build_microscope_executor_cfg, build_system_executor_cfg, build_user_executor_cfg
from storage.managers import HistoryManager, StorageManager
from tool.base import BaseTool, _slugify_tool_name
from tooling.doc_paths import DEFAULT_USER_TOOL_DOCS_DIR
from tooling.manifest import (
    SYSTEM_TOOL_ROLES,
    ToolManifest,
    ToolManifestError,
    UserToolManifestEntry,
    import_string,
    load_tool_manifest,
    resolve_default_tool_manifest_path,
)


logger = logging.getLogger(__name__)

DEFAULT_HELPERS = {"np", "cv", "math", "datetime", "time", "csv", "json", "plt", "say"}
TOOL_DOCS_DIR = DEFAULT_USER_TOOL_DOCS_DIR


@dataclass(frozen=True)
class ResolvedUserTool:
    tool_id: str
    class_path: str
    tool_cls: type[BaseTool]
    planning_hint: str = ""
    execution_hint: str = ""


@dataclass(frozen=True)
class RuntimeToolAssembly:
    tool_registry: ToolRegistry
    env_bindings: Dict[str, Any]


class _MissingOptionalModule:
    def __init__(self, module_name: str) -> None:
        self._module_name = module_name

    def __getattr__(self, attr_name: str) -> Any:
        raise ImportError(
            f"Optional module '{self._module_name}' is required to access '{attr_name}', "
            "but it is not installed in the current environment."
        )

    def __repr__(self) -> str:
        return f"<missing optional module {self._module_name}>"


class _GeneratedTimeProxy:
    def __init__(self, module: Any, *, max_sleep_seconds: float = 0.0) -> None:
        self._module = module
        self._max_sleep_seconds = max(float(max_sleep_seconds), 0.0)

    def sleep(self, seconds: float = 0.0) -> None:
        if self._max_sleep_seconds <= 0:
            return
        self._module.sleep(min(max(float(seconds), 0.0), self._max_sleep_seconds))

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self._module, attr_name)


def build_var_map(env_obj: Any) -> Dict[str, Any]:
    methods = env_obj.get_public_methods()
    return {name: getattr(env_obj, name) for name in methods if hasattr(env_obj, name)}


def _load_optional_runtime_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.info(
            "Optional runtime helper '%s' is unavailable: %s: %s",
            module_name,
            type(exc).__name__,
            exc,
        )
        return _MissingOptionalModule(module_name)


def build_fixed_vars(required_helpers: Any = None, say_callable: Any = None) -> Dict[str, Any]:
    if callable(required_helpers) and say_callable is None:
        say_callable = required_helpers
        required_helpers = {"cv", "plt"}
    helper_names = set(DEFAULT_HELPERS) | set(required_helpers or [])
    generated_time_helper: Any = time
    if os.environ.get("EIMS_DISABLE_GENERATED_SLEEP", "").lower() in {"1", "true", "yes", "on"}:
        generated_time_helper = _GeneratedTimeProxy(
            time,
            max_sleep_seconds=float(os.environ.get("EIMS_MAX_GENERATED_SLEEP_SECONDS", "0") or 0),
        )
    values = {
        "np": np,
        "cv": _load_optional_runtime_module("cv2"),
        "math": math,
        "datetime": datetime,
        "time": generated_time_helper,
        "csv": csv,
        "json": importlib.import_module("json"),
        "plt": _load_optional_runtime_module("matplotlib.pyplot"),
        "say": say_callable,
    }
    return {name: values[name] for name in helper_names if name in values}


def _read_tool_doc_artifact(tool_id: str, suffix: str) -> str:
    artifact_path = TOOL_DOCS_DIR / f"{_slugify_tool_name(tool_id)}.{suffix}"
    try:
        return artifact_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _resolve_user_execution_prompt(tool_cls: type[BaseTool], tool_id: str, execution_hint: str) -> str:
    artifact_text = _read_tool_doc_artifact(tool_id, "executor_prompt.txt")
    if artifact_text:
        return artifact_text
    return tool_cls.get_execution_prompt_context(tool_id=tool_id, execution_hint=execution_hint)


def _extract_user_tool_planner_sections(summary_text: str) -> tuple[str, str]:
    text = str(summary_text or "").strip()
    if not text:
        return "", ""

    task_marker = "\n\n# Task Example\n"
    capability_marker = "\n\n# Capability Summary\n"

    example_text = ""
    if task_marker in text:
        text, example_text = text.split(task_marker, 1)
        example_text = example_text.strip()

    submodule_text = text
    if capability_marker in text:
        submodule_text = text.split(capability_marker, 1)[0]

    return submodule_text.strip(), example_text.strip()


def _instantiate_system_tool_env(
    tool_cls: Any,
    constructor_kind: str,
    system_config: Any,
    detection_targets: Dict[str, Dict[str, Any]],
    output_dir: str,
    storage_manager: StorageManager,
) -> Any:
    def _supports_kwarg(name: str) -> bool:
        signature = inspect.signature(tool_cls.__init__)
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD or param.name == name
            for param in signature.parameters.values()
        )

    kwargs: Dict[str, Any] = {}
    if _supports_kwarg("system_config"):
        kwargs["system_config"] = system_config
    if _supports_kwarg("detection_targets"):
        kwargs["detection_targets"] = {key: dict(value) for key, value in detection_targets.items()}

    if constructor_kind == "microscope":
        return tool_cls(system_config.CONFIG_PATH, system_config.MM_DIR, output_dir, storage_manager, **kwargs)
    if constructor_kind == "storage_output":
        return tool_cls(storage_manager, output_dir, **kwargs)
    if constructor_kind == "no_args":
        return tool_cls()
    raise ToolManifestError(f"Unsupported constructor kind '{constructor_kind}'")


def _validate_user_tool_class(class_path: str) -> type[BaseTool]:
    tool_obj = import_string(class_path)
    if not inspect.isclass(tool_obj) or not issubclass(tool_obj, BaseTool):
        raise ToolManifestError(f"User tool '{class_path}' must resolve to a BaseTool subclass")
    if not tool_obj.get_public_methods():
        raise ToolManifestError(f"User tool '{class_path}' must expose at least one @tool_func method")
    return tool_obj


def _resolve_user_tool_entry(
    entry: UserToolManifestEntry,
) -> Optional[ResolvedUserTool]:
    if not entry.enabled:
        return None
    tool_cls = _validate_user_tool_class(entry.class_path)
    planning_hint = entry.planning_hint or tool_cls.get_planning_hint()
    execution_hint = entry.execution_hint or tool_cls.get_execution_hint()
    return ResolvedUserTool(
        tool_id=entry.tool_id,
        class_path=entry.class_path,
        tool_cls=tool_cls,
        planning_hint=planning_hint,
        execution_hint=execution_hint,
    )


def resolve_user_tools(manifest: ToolManifest) -> list[ResolvedUserTool]:
    resolved: list[ResolvedUserTool] = []
    for entry in manifest.user_tools:
        if not entry.enabled:
            continue
        item = _resolve_user_tool_entry(entry)
        if item is not None:
            resolved.append(item)

    reserved_ids = {
        manifest.system_tools[role].tool_id
        for role in SYSTEM_TOOL_ROLES
    } | {
        manifest.system_tools[role].platform_name
        for role in SYSTEM_TOOL_ROLES
    }
    seen_tool_ids: dict[str, str] = {}
    for item in resolved:
        if item.tool_id in reserved_ids:
            raise ToolManifestError(f"User tool_id '{item.tool_id}' conflicts with a reserved system tool identifier")
        if item.tool_id in seen_tool_ids:
            raise ToolManifestError(
                f"Duplicate resolved user tool_id '{item.tool_id}' from '{seen_tool_ids[item.tool_id]}' and '{item.class_path}'"
            )
        seen_tool_ids[item.tool_id] = item.class_path
    return resolved


def _instantiate_user_tool_env(tool_cls: type[BaseTool], output_dir: str, storage_manager: StorageManager) -> Any:
    signature = inspect.signature(tool_cls.__init__)
    params = [
        param
        for param in signature.parameters.values()
        if param.name != "self" and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    supported_values = {
        "storage_manager": storage_manager,
        "output_dir": output_dir,
    }
    kwargs: Dict[str, Any] = {}
    for param in params:
        if param.name in supported_values:
            kwargs[param.name] = supported_values[param.name]
            continue
        if param.default is inspect._empty:
            raise ToolManifestError(
                f"User tool '{tool_cls.__module__}:{tool_cls.__name__}' uses unsupported required constructor parameter '{param.name}'"
            )
    return tool_cls(**kwargs)


def _build_executor_execution_context(
    *,
    role: str,
    role_mode: str,
    env_obj: Any,
    system_config: Any,
    session_dir: str,
    output_dir: str,
    storage_manager: StorageManager,
    say_capture: SayCapture,
) -> Dict[str, Any]:
    in_process_timeout = float(os.environ.get("EIMS_IN_PROCESS_EXECUTOR_TIMEOUT_SECONDS", "30") or 30)
    if role != "image_analysis" or role_mode != "real":
        return {"timeout_seconds": in_process_timeout}
    return {
        "use_fiji_subprocess": True,
        "session_dir": session_dir,
        "output_dir": output_dir,
        "workdir": output_dir,
        "storage_manager": storage_manager,
        "say_capture": say_capture,
        "artifact_emitter_getter": getattr(env_obj, "get_interaction_artifact_listener", lambda: None),
        "timeout_seconds": float(getattr(system_config, "fiji_executor_timeout_seconds", 300.0) or 300.0),
        "startup_retry_times": int(getattr(system_config, "fiji_executor_startup_retry_times", 2) or 2),
        "startup_retry_backoff_seconds": float(
            getattr(system_config, "fiji_executor_startup_retry_backoff_seconds", 2.0) or 2.0
        ),
    }


def _render_user_tool_planner_sections(resolved_user_tools: list[ResolvedUserTool]) -> tuple[str, str]:
    if not resolved_user_tools:
        return "", ""
    submodule_blocks: list[str] = []
    example_blocks: list[str] = []
    for item in resolved_user_tools:
        artifact_text = _read_tool_doc_artifact(item.tool_id, "planner_summary.txt")
        artifact_submodule, artifact_example = _extract_user_tool_planner_sections(artifact_text)
        submodule_blocks.append(
            artifact_submodule
            or item.tool_cls.get_planning_submodule_block(tool_id=item.tool_id, planning_hint=item.planning_hint)
        )
        example_blocks.append(
            artifact_example
            or item.tool_cls.get_planning_example_block(tool_id=item.tool_id)
        )
    return "\n".join(block.strip() for block in submodule_blocks if block.strip()), "\n\n".join(
        block.strip() for block in example_blocks if block.strip()
    )


def inject_user_tool_planner_prompt(prompt_text: str, resolved_user_tools: list[ResolvedUserTool]) -> str:
    required_placeholders = ("{{USER_TOOL_SUBMODULES}}", "{{USER_TOOL_EXAMPLES}}")
    missing = [placeholder for placeholder in required_placeholders if placeholder not in prompt_text]
    if missing:
        raise ToolManifestError(
            "Planner prompt template is missing required user-tool placeholders: "
            + ", ".join(missing)
        )
    submodule_text, example_text = _render_user_tool_planner_sections(resolved_user_tools)
    rendered = prompt_text.replace("{{USER_TOOL_SUBMODULES}}", submodule_text)
    rendered = rendered.replace("{{USER_TOOL_EXAMPLES}}", example_text)
    return rendered


def _discover_generated_user_tool_ids() -> set[str]:
    discovered: set[str] = set()
    if not TOOL_DOCS_DIR.exists():
        return discovered
    for path in TOOL_DOCS_DIR.glob("*.planner_summary.txt"):
        name = path.name
        if not name.endswith(".planner_summary.txt"):
            continue
        discovered.add(name[: -len(".planner_summary.txt")])
    for path in TOOL_DOCS_DIR.glob("*.executor_prompt.txt"):
        name = path.name
        if not name.endswith(".executor_prompt.txt"):
            continue
        discovered.add(name[: -len(".executor_prompt.txt")])
    return discovered


def _validate_runtime_user_tool_consistency(
    *,
    manifest: ToolManifest,
    resolved_user_tools: list[ResolvedUserTool],
    planner_prompt_text: str,
    tool_registry: ToolRegistry,
) -> None:
    enabled_manifest_ids = [entry.tool_id for entry in manifest.user_tools if entry.enabled]
    resolved_tool_ids = [item.tool_id for item in resolved_user_tools]
    if enabled_manifest_ids != resolved_tool_ids:
        raise ToolManifestError(
            "Enabled user tools in the manifest do not match the resolved runtime user tools: "
            f"manifest={enabled_manifest_ids}, resolved={resolved_tool_ids}"
        )

    registry_tools = tool_registry.list_tools()
    registry_user_tool_ids = sorted(
        item["platform"]
        for item in registry_tools
        if str(item.get("role") or "") == "user_tool"
    )
    if sorted(resolved_tool_ids) != registry_user_tool_ids:
        raise ToolManifestError(
            "Resolved runtime user tools do not match the executor registry: "
            f"resolved={sorted(resolved_tool_ids)}, registry={registry_user_tool_ids}"
        )

    generated_tool_ids = _discover_generated_user_tool_ids()
    leaked_tool_ids = [
        tool_id
        for tool_id in sorted(generated_tool_ids - set(resolved_tool_ids))
        if re.search(rf"(^|\n)###\s+{re.escape(tool_id)}(\s|$)", planner_prompt_text)
    ]
    if leaked_tool_ids:
        raise ToolManifestError(
            "Planner prompt contains stale user-tool documentation for tools that are not enabled in the manifest: "
            + ", ".join(leaked_tool_ids)
        )

    missing_tool_ids = [
        tool_id
        for tool_id in resolved_tool_ids
        if not re.search(rf"(^|\n)###\s+{re.escape(tool_id)}(\s|$)", planner_prompt_text)
    ]
    if missing_tool_ids:
        raise ToolManifestError(
            "Enabled user tools are missing from the injected planner prompt: "
            + ", ".join(missing_tool_ids)
        )

    logger.info(
        "Runtime user-tool self-check passed. manifest=%s enabled_user_tools=%s",
        resolve_default_tool_manifest_path(),
        resolved_tool_ids,
    )


def _normalize_mode(value: Any, *, allowed: tuple[str, ...], default: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _role_mode_map(agent_config: Any) -> dict[str, str]:
    return {
        "microscope": _normalize_mode(getattr(agent_config, "microscope_mode", "demo"), allowed=("demo", "real"), default="demo"),
        "image_analysis": _normalize_mode(
            getattr(agent_config, "image_analysis_mode", "mock"),
            allowed=("real", "mock"),
            default="mock",
        ),
        "segmentation": _normalize_mode(
            getattr(agent_config, "segmentation_mode", "mock"),
            allowed=("real", "mock"),
            default="mock",
        ),
    }


def _resolve_system_tool_class_path(spec: Any, *, role: str, role_mode: str) -> str:
    if role == "microscope":
        return spec.real_class_path
    if role_mode == "real":
        return spec.real_class_path
    if spec.mock_class_path:
        return spec.mock_class_path
    raise ToolManifestError(f"System tool '{role}' is missing a mock_class_path for mode '{role_mode}'")


def assemble_runtime_tools(
    *,
    runtime: Dict[str, Any],
    llm_client: Any,
    history_manager: HistoryManager,
    storage_manager: StorageManager,
    say_capture: SayCapture,
    session_dir: str,
    output_dir: str,
    shared_lmps: Dict[str, Dict[str, Any]],
) -> RuntimeToolAssembly:
    agent_config = runtime["agent"]
    system_config = runtime["system"]
    detection_targets = runtime["detection_targets"]
    manifest = load_tool_manifest()
    role_modes = _role_mode_map(agent_config)

    fgen_cfg = shared_lmps["fgen"]
    fixed_vars = build_fixed_vars({"cv", "plt"}, say_capture.say)
    tool_registry = ToolRegistry()
    env_bindings: Dict[str, Any] = {}

    for role in SYSTEM_TOOL_ROLES:
        spec = manifest.system_tools[role]
        role_mode = role_modes[role]
        tool_cls = import_string(_resolve_system_tool_class_path(spec, role=role, role_mode=role_mode))
        env_obj = _instantiate_system_tool_env(
            tool_cls,
            spec.constructor_kind,
            system_config,
            detection_targets,
            output_dir,
            storage_manager,
        )
        executor = ExperimentExecuteAgent(
            spec.platform_name,
            build_microscope_executor_cfg(agent_config, spec.prompt_source, system_config)
            if role == "microscope"
            else build_system_executor_cfg(agent_config, spec.prompt_source),
            fgen_cfg,
            fixed_vars,
            build_var_map(env_obj),
            llm_client,
            history_manager,
            execution_context=_build_executor_execution_context(
                role=role,
                role_mode=role_mode,
                env_obj=env_obj,
                system_config=system_config,
                session_dir=session_dir,
                output_dir=output_dir,
                storage_manager=storage_manager,
                say_capture=say_capture,
            ),
        )
        tool_registry.register_platform(spec.platform_name, env_obj, executor, port_kind=spec.port_kind)
        env_bindings[spec.tool_id] = env_obj

    resolved_user_tools = resolve_user_tools(manifest)
    shared_lmps["Task_manger"] = dict(shared_lmps["Task_manger"])
    shared_lmps["Task_manger"]["prompt_text"] = inject_user_tool_planner_prompt(
        shared_lmps["Task_manger"].get("prompt_text", ""),
        resolved_user_tools,
    )
    for item in resolved_user_tools:
        env_obj = _instantiate_user_tool_env(item.tool_cls, output_dir, storage_manager)
        prompt_text = _resolve_user_execution_prompt(item.tool_cls, item.tool_id, item.execution_hint)
        executor = ExperimentExecuteAgent(
            item.tool_id,
            build_user_executor_cfg(agent_config, prompt_text),
            fgen_cfg,
            fixed_vars,
            build_var_map(env_obj),
            llm_client,
            history_manager,
        )
        tool_registry.register_tool(
            item.tool_id,
            env_obj,
            executor,
            role="user_tool",
            validate_role=False,
            expose_public_callables=False,
        )

    _validate_runtime_user_tool_consistency(
        manifest=manifest,
        resolved_user_tools=resolved_user_tools,
        planner_prompt_text=shared_lmps["Task_manger"]["prompt_text"],
        tool_registry=tool_registry,
    )

    return RuntimeToolAssembly(
        tool_registry=tool_registry,
        env_bindings=env_bindings,
    )
