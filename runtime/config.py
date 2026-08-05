from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict

from bootstrap.config import ModelConfig, RuntimeSettings, TaskRuntimeConfig
from bootstrap.microscope_semantics import render_microscope_prompt_template
from prompts.shared.function_generation import prompt_fgen
from prompts.checker.quality_checker import (
    instruction_prompt_with_no_target,
    instruction_prompt_without_no_target,
    prompt_no_target,
    prompt_out_of_focus,
    prompt_over_exposed,
    prompt_quality_check,
)
from prompts.planner.task_manager_stateful_with_brightness import prompt_manger
from prompts.planner.task_manager_stateful_without_brightness import prompt_manger as prompt_manger_no_brightness

_EXECUTOR_SANDBOX_GUIDANCE = """
# Sandbox and Execution Safety
- You must only use the provided API functions and safe built-in control flow.
- Do not call `open()`, `exit()`, `quit()`, `eval()`, `exec()`, `compile()`, `input()`, or similar unrestricted runtime/file APIs.
- Do not write files directly; use the platform save APIs that are explicitly provided in the prompt.
- If a step cannot continue, report it with `say("[ERROR] ...")` and raise `RuntimeError(...)` when needed.
- Never attempt to terminate the Python interpreter or bypass the execution sandbox.
""".strip()


def import_prompt_text(prompt_source: str) -> str:
    if ":" not in prompt_source:
        raise ValueError(f"Prompt source must use 'module.submodule:attribute' format, got: {prompt_source}")
    module_name, attr_name = prompt_source.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        value = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Prompt source '{prompt_source}' could not be resolved") from exc
    if not isinstance(value, str):
        raise TypeError(f"Prompt source '{prompt_source}' did not resolve to a string prompt")
    return value


def _resolve_microscope_prompt_source(prompt_source: str, system_config: Any) -> str:
    full_prompt_source = "prompts.executor.microscope_with_brightness:prompt_olympus"
    no_brightness_prompt_source = "prompts.executor.microscope_without_brightness:prompt_olympus"
    if prompt_source != full_prompt_source:
        return prompt_source

    transmitted_light = dict(getattr(system_config, "transmitted_light", {}) or {})
    brightness_device = str(transmitted_light.get("device") or "").strip()
    brightness_property = str(transmitted_light.get("intensity_property") or "").strip()
    if brightness_device and brightness_property:
        return full_prompt_source
    return no_brightness_prompt_source


def _has_transmitted_light_brightness_control(system_config: Any) -> bool:
    transmitted_light = dict(getattr(system_config, "transmitted_light", {}) or {})
    brightness_device = str(transmitted_light.get("device") or "").strip()
    brightness_property = str(transmitted_light.get("intensity_property") or "").strip()
    return bool(brightness_device and brightness_property)


def _resolve_planner_prompt_text(system_config: Any) -> str:
    if _has_transmitted_light_brightness_control(system_config):
        return prompt_manger
    return prompt_manger_no_brightness


def build_executor_lmp_config_from_text(
    model_config: ModelConfig,
    prompt_text: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_prompt_text = str(prompt_text)
    if append_sandbox_guidance and _EXECUTOR_SANDBOX_GUIDANCE not in resolved_prompt_text:
        resolved_prompt_text = f"{resolved_prompt_text.rstrip()}\n\n{_EXECUTOR_SANDBOX_GUIDANCE}\n"
    cfg = {
        "prompt_text": resolved_prompt_text,
        "engine": model_config.model_name,
        "seed": model_config.llm_seed,
        "max_tokens": 5120,
        "temperature": 0,
        "query_prefix": "#",
        "query_suffix": ".",
        "stop": [],
        "maintain_session": False,
        "debug_mode": False,
        "include_context": True,
        "has_return": False,
        "return_val_name": "ret_val",
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def build_executor_lmp_config(
    model_config: ModelConfig,
    prompt_source: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return build_executor_lmp_config_from_text(
        model_config,
        import_prompt_text(prompt_source),
        append_sandbox_guidance=append_sandbox_guidance,
        overrides=overrides,
    )


def build_microscope_executor_lmp_config(
    model_config: ModelConfig,
    prompt_source: str,
    system_config: Any,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_prompt_source = _resolve_microscope_prompt_source(prompt_source, system_config)
    prompt_text = render_microscope_prompt_template(import_prompt_text(resolved_prompt_source), system_config)
    return build_executor_lmp_config_from_text(
        model_config,
        prompt_text,
        append_sandbox_guidance=append_sandbox_guidance,
        overrides=overrides,
    )


def build_planner_lmp_config(model_config: ModelConfig, system_config: Any | None = None) -> Dict[str, Any]:
    return {
        "prompt_text": _resolve_planner_prompt_text(system_config) if system_config is not None else prompt_manger,
        "engine": model_config.model_name,
        "system_config": system_config,
        "seed": model_config.llm_seed,
        "max_tokens": int(os.getenv("EIMS_PLANNER_MAX_TOKENS", "12000")),
        "temperature": 0,
        "query_prefix": "# ",
        "query_suffix": ".",
        "stop": "#",
        "maintain_session": True,
        "debug_mode": False,
        "include_context": True,
        "has_return": False,
        "return_val_name": "ret_val",
    }


def build_skill_resolver_config() -> Dict[str, Any]:
    return {
        "skill_dirs": [str(Path("user_skills") / "planning")],
        "skill_max_files": 20,
        "skill_max_chars_per_file": 2000,
        "skill_max_selected": 2,
        "skill_route_max_tokens": 512,
        "skill_route_temperature": 0,
    }


def build_fgen_lmp_config(model_config: ModelConfig) -> Dict[str, Any]:
    return {
        "prompt_text": prompt_fgen,
        "engine": model_config.model_name,
        "seed": model_config.llm_seed,
        "max_tokens": 1024,
        "temperature": 0,
        "query_prefix": "# define function: ",
        "query_suffix": ".",
        "stop": [],
        "maintain_session": False,
        "debug_mode": False,
        "include_context": True,
    }


def build_checker_lmp_config(model_config: ModelConfig) -> Dict[str, Any]:
    return {
        "prompt_no_target": prompt_no_target,
        "prompt_over_exposed": prompt_over_exposed,
        "prompt_out_of_focus": prompt_out_of_focus,
        "prompt_quality_check": prompt_quality_check,
        "instruction_prompt_with_no_target": instruction_prompt_with_no_target,
        "instruction_prompt_without_no_target": instruction_prompt_without_no_target,
        "engine": model_config.model_name,
        "vlm_engine": model_config.vlm_model_name,
        "seed": model_config.llm_seed,
        "max_tokens": 1024,
        "temperature": 0,
        "vlm_max_tokens": 1024,
        "vlm_temperature": 0,
        "query_prefix": "# define function: ",
        "query_suffix": ".",
        "stop": [],
        "maintain_session": False,
        "debug_mode": False,
        "include_context": True,
    }


def build_shared_lmp_configs(model_config: ModelConfig, system_config: Any | None = None) -> Dict[str, Dict[str, Any]]:
    return {
        "Task_manger": build_planner_lmp_config(model_config, system_config),
        "fgen": build_fgen_lmp_config(model_config),
        "checker": build_checker_lmp_config(model_config),
    }


def build_runtime_config(settings: RuntimeSettings) -> Dict[str, Any]:
    return {
        "agent": settings.model,
        "system": settings.system,
        "task": TaskRuntimeConfig(),
        "detection_targets": {
            str(key): dict(value)
            for key, value in settings.detection_targets.items()
        },
        "settings": settings,
    }
