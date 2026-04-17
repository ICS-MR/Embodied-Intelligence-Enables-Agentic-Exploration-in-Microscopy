import csv
import datetime
import importlib
import inspect
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from adapters.llm_clients import build_openai_clients
from adapters.tool_registry import ToolRegistry
from agent.experiment_checker import ExperimentCheckAgent
from agent.experiment_executor import ExperimentExecuteAgent
from agent.experiment_planner import ExperimentPlanAgent
from core_tool.tool_utils import SayCapture
from services.task_orchestrator import TaskOrchestrator
from tool.base import BaseTool, _slugify_tool_name
from utils.tool_doc_paths import DEFAULT_USER_TOOL_DOCS_DIR
from utils.memory_manager import HistoryManager, StorageManager
from utils.runtime_models import RuntimeContext
from utils.runtime_session import create_runtime_session_paths
from utils.runtime_text import (
    rewrite_task_plan_for_confirmation,
    stream_plan_preview_for_confirmation,
    stream_task_execution_summary,
    summarize_checker_issue,
    summarize_checker_success,
    summarize_my_spoken_messages,
    summarize_spoken_messages,
    summarize_step_completion,
    summarize_task_execution,
)
from utils.tool_manifest import (
    SYSTEM_TOOL_ROLES,
    ToolManifest,
    ToolManifestError,
    UserToolManifestEntry,
    import_string,
    load_tool_manifest,
)


logger = logging.getLogger(__name__)

REAL_HARDWARE_REQUIREMENTS = {
    "core_tool.microscope": [
        ("cv2", None),
        ("numpy", None),
        ("aicsimageio.types", "PhysicalPixelSizes"),
        ("aicsimageio.writers", "OmeTiffWriter"),
        ("pymmcore_plus", "CMMCorePlus"),
        ("torch", None),
        ("mmdet.apis", "init_detector"),
        ("mmdet.apis", "inference_detector"),
    ],
    "core_tool.fiji": [
        ("torch", None),
        ("imagej", None),
        ("scyjava", "jimport"),
        ("tifffile", None),
        ("cv2", None),
        ("mmdet.apis", "init_detector"),
        ("mmdet.apis", "inference_detector"),
        ("aicsimageio", "AICSImage"),
    ],
    "core_tool.cellpose_tool": [
        ("numpy", None),
        ("pandas", None),
        ("matplotlib.pyplot", None),
        ("skimage", None),
        ("tifffile", None),
        ("cellpose.models", None),
        ("cellpose.core", None),
    ],
}

DEFAULT_HELPERS = {"np", "cv", "math", "datetime", "time", "csv", "json", "plt", "say"}
SYSTEM_ROLE_ATTRS = {
    "microscope": "env_olympus",
    "image_analysis": "env_imagej",
    "segmentation": "env_cellpose",
}
TOOL_DOCS_DIR = DEFAULT_USER_TOOL_DOCS_DIR


@dataclass(frozen=True)
class ResolvedUserTool:
    tool_id: str
    class_path: str
    tool_cls: type[BaseTool]
    planning_hint: str = ""
    execution_hint: str = ""


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


def load_runtime_config() -> Dict[str, Any]:
    agent_config = importlib.reload(importlib.import_module("config.agent_config"))
    system_config = importlib.reload(importlib.import_module("config.system_config"))
    task_config = importlib.reload(importlib.import_module("config.task_config"))
    return {
        "agent": agent_config,
        "system": system_config,
        "task": task_config,
    }


def _check_dependency(module_name: str, attr_name: str | None) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"{module_name}: {type(exc).__name__}: {exc}"
    if attr_name and not hasattr(module, attr_name):
        return f"{module_name} is missing required attribute '{attr_name}'"
    return None


def validate_real_hardware_stack() -> list[str]:
    issues: list[str] = []
    for module_name, requirements in REAL_HARDWARE_REQUIREMENTS.items():
        module_issues: list[str] = []
        for dependency_name, attr_name in requirements:
            problem = _check_dependency(dependency_name, attr_name)
            if problem is not None:
                module_issues.append(f"{module_name} requires {problem}")
        if module_issues:
            issues.extend(module_issues)
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            issues.append(f"{module_name} import failed: {type(exc).__name__}: {exc}")
    return issues


def get_tool_classes(simulation_mode: bool):
    manifest = load_tool_manifest()
    microscope_cls = import_string(manifest.system_tools["microscope"].class_path_for_mode(simulation_mode))
    imagej_cls = import_string(manifest.system_tools["image_analysis"].class_path_for_mode(simulation_mode))
    cellpose_cls = import_string(manifest.system_tools["segmentation"].class_path_for_mode(simulation_mode))
    return microscope_cls, imagej_cls, cellpose_cls


def build_clients(agent_module: Any) -> Tuple[Any, Any]:
    bundle = build_openai_clients(agent_module)
    return bundle.llm_client, bundle.vlm_client


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


def _build_fixed_vars(required_helpers: Any = None, say_callable: Any = None) -> Dict[str, Any]:
    if callable(required_helpers) and say_callable is None:
        say_callable = required_helpers
        required_helpers = {"cv", "plt"}
    helper_names = set(DEFAULT_HELPERS) | set(required_helpers or [])
    values = {
        "np": np,
        "cv": _load_optional_runtime_module("cv2"),
        "math": math,
        "datetime": datetime,
        "time": time,
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

def _build_shared_lmps(agent_module: Any) -> Dict[str, Dict[str, Any]]:
    if hasattr(agent_module, "build_shared_lmp_configs"):
        return agent_module.build_shared_lmp_configs()
    return dict(agent_module.cfg_tabletop.get("lmps", {}))


def _build_system_executor_cfg(agent_module: Any, prompt_source: str) -> Dict[str, Any]:
    if hasattr(agent_module, "build_executor_lmp_config"):
        return agent_module.build_executor_lmp_config(prompt_source, append_sandbox_guidance=True)
    raise ToolManifestError("Agent config module does not expose build_executor_lmp_config")


def _build_user_executor_cfg(agent_module: Any, tool_cls: type[BaseTool], tool_id: str, execution_hint: str) -> Dict[str, Any]:
    prompt_text = _resolve_user_execution_prompt(tool_cls, tool_id, execution_hint)
    if hasattr(agent_module, "build_executor_lmp_config_from_text"):
        return agent_module.build_executor_lmp_config_from_text(prompt_text, append_sandbox_guidance=True)
    raise ToolManifestError("Agent config module does not expose build_executor_lmp_config_from_text")


def _instantiate_system_tool_env(
    tool_cls: Any,
    constructor_kind: str,
    system_module: Any,
    output_dir: str,
    storage_manager: StorageManager,
) -> Any:
    if constructor_kind == "microscope":
        return tool_cls(system_module.CONFIG_PATH, system_module.MM_DIR, output_dir, storage_manager)
    if constructor_kind == "storage_output":
        return tool_cls(storage_manager, output_dir)
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


def _resolve_user_tools(manifest: ToolManifest) -> list[ResolvedUserTool]:
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
    simulation_mode: bool,
    system_module: Any,
    session_dir: str,
    output_dir: str,
    storage_manager: StorageManager,
    say_capture: SayCapture,
) -> Dict[str, Any]:
    if role != "image_analysis" or simulation_mode:
        return {}
    return {
        "use_fiji_subprocess": True,
        "session_dir": session_dir,
        "output_dir": output_dir,
        "workdir": output_dir,
        "storage_manager": storage_manager,
        "say_capture": say_capture,
        "timeout_seconds": float(getattr(system_module, "fiji_executor_timeout_seconds", 300.0) or 300.0),
        "startup_retry_times": int(getattr(system_module, "fiji_executor_startup_retry_times", 2) or 2),
        "startup_retry_backoff_seconds": float(
            getattr(system_module, "fiji_executor_startup_retry_backoff_seconds", 2.0) or 2.0
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


def _inject_user_tool_planner_prompt(prompt_text: str, resolved_user_tools: list[ResolvedUserTool]) -> str:
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


def _needs_real_hardware_validation(manifest: ToolManifest) -> bool:
    return any(entry.validate_real_stack for entry in manifest.system_tools.values())


def _build_runtime_context_from_manifest(
    runtime: Dict[str, Any],
    simulation_mode: bool,
    llm_client: Any,
    vlm_client: Any,
) -> RuntimeContext:
    agent_module = runtime["agent"]
    system_module = runtime["system"]
    task_module = runtime["task"]
    manifest = load_tool_manifest()
    shared_lmps = _build_shared_lmps(agent_module)

    if not simulation_mode and _needs_real_hardware_validation(manifest):
        issues = validate_real_hardware_stack()
        if issues:
            detail = "\n - ".join(issues)
            raise RuntimeError(f"Real hardware stack validation failed:\n - {detail}")

    session_id, session_dir, session_output_dir = create_runtime_session_paths(
        task_module.HISTORY_DIR,
        task_module.OUTPUT_DIR,
    )
    history_manager = HistoryManager(session_dir)
    storage_manager = StorageManager(session_dir, session_output_dir)
    say_capture = SayCapture()
    fixed_vars = _build_fixed_vars({"cv", "plt"}, say_capture.say)
    fgen_cfg = shared_lmps["fgen"]

    tool_registry = ToolRegistry()
    env_bindings: Dict[str, Any] = {}

    for role in SYSTEM_TOOL_ROLES:
        spec = manifest.system_tools[role]
        tool_cls = import_string(spec.class_path_for_mode(simulation_mode))
        env_obj = _instantiate_system_tool_env(
            tool_cls,
            spec.constructor_kind,
            system_module,
            session_output_dir,
            storage_manager,
        )
        executor = ExperimentExecuteAgent(
            spec.platform_name,
            _build_system_executor_cfg(agent_module, spec.prompt_source),
            fgen_cfg,
            fixed_vars,
            build_var_map(env_obj),
            llm_client,
            history_manager,
            execution_context=_build_executor_execution_context(
                role=role,
                simulation_mode=simulation_mode,
                system_module=system_module,
                session_dir=session_dir,
                output_dir=session_output_dir,
                storage_manager=storage_manager,
                say_capture=say_capture,
            ),
        )
        tool_registry.register_platform(spec.platform_name, env_obj, executor, port_kind=spec.port_kind)
        env_bindings[spec.tool_id] = env_obj

    resolved_user_tools = _resolve_user_tools(manifest)
    shared_lmps["Task_manger"] = dict(shared_lmps["Task_manger"])
    shared_lmps["Task_manger"]["prompt_text"] = _inject_user_tool_planner_prompt(
        shared_lmps["Task_manger"].get("prompt_text", ""),
        resolved_user_tools,
    )
    for item in resolved_user_tools:
        env_obj = _instantiate_user_tool_env(item.tool_cls, session_output_dir, storage_manager)
        executor = ExperimentExecuteAgent(
            item.tool_id,
            _build_user_executor_cfg(agent_module, item.tool_cls, item.tool_id, item.execution_hint),
            fgen_cfg,
            fixed_vars,
            build_var_map(env_obj),
            llm_client,
            history_manager,
        )
        tool_registry.register_tool(item.tool_id, env_obj, executor, role="user_tool", validate_role=False, expose_public_callables=False)

    task_manager = ExperimentPlanAgent(
        "Task_manager",
        shared_lmps["Task_manger"],
        llm_client,
        history_manager,
        clarify_tag=bool(getattr(agent_module, "clarify_enabled", False)),
    )
    checker = ExperimentCheckAgent(
        shared_lmps["checker"],
        llm_client,
        vlm_client,
        session_output_dir,
        history_manager,
    )

    runtime_context = RuntimeContext(
        session_id=session_id,
        session_dir=session_dir,
        output_dir=session_output_dir,
        runtime=runtime,
        llm_client=llm_client,
        vlm_client=vlm_client,
        say_capture=say_capture,
        env_olympus=env_bindings.get("microscope_operation"),
        env_imagej=env_bindings.get("image_analysis"),
        env_cellpose=env_bindings.get("cell_segmentation"),
        storage_manager=storage_manager,
        history_manager=history_manager,
        task_manager=task_manager,
        tool_registry=tool_registry,
        checker=checker,
        task_orchestrator=None,
    )
    runtime_context.task_orchestrator = TaskOrchestrator(
        runtime_context,
        summarize_spoken_messages=summarize_spoken_messages,
        summarize_my_spoken_messages=summarize_my_spoken_messages,
        summarize_step_completion=summarize_step_completion,
        summarize_checker_issue=summarize_checker_issue,
        summarize_checker_success=summarize_checker_success,
        summarize_task_execution=summarize_task_execution,
        rewrite_plan_for_confirmation=rewrite_task_plan_for_confirmation,
        stream_plan_for_confirmation=stream_plan_preview_for_confirmation,
        stream_task_execution_summary=stream_task_execution_summary,
    )
    return runtime_context


def initialize_system_components(simulation_mode: Optional[bool] = None) -> RuntimeContext:
    runtime = load_runtime_config()
    agent_module = runtime["agent"]
    if simulation_mode is not None:
        agent_module.Simulation_mode = simulation_mode

    llm_client, vlm_client = build_clients(agent_module)
    return _build_runtime_context_from_manifest(runtime, bool(agent_module.Simulation_mode), llm_client, vlm_client)







