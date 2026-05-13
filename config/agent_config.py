import importlib
from pathlib import Path
from typing import Any, Dict

from bootstrap.config import load_model_config
from prompts.fgen import prompt_fgen
from prompts.prompt_check import (
    instruction_prompt_with_no_target,
    instruction_prompt_without_no_target,
    prompt_no_target,
    prompt_out_of_focus,
    prompt_over_exposed,
    prompt_quality_check,
)
from prompts.task_manager_full_stateful import prompt_manger

_config = load_model_config()

Simulation_mode = _config.Simulation_mode
clarify_enabled = _config.clarify_enabled
checker_enabled = _config.checker_enabled
emit_skill_routing = _config.emit_skill_routing
openai_api_key = _config.openai_api_key
base_url = _config.base_url
model_name = _config.model_name
vlm_api_key = _config.vlm_api_key
vlm_base_url = _config.vlm_base_url
vlm_model_name = _config.vlm_model_name
CROSS_ENCODER_MODEL_PATH = _config.CROSS_ENCODER_MODEL_PATH
cross_encoder_model_path = CROSS_ENCODER_MODEL_PATH
task_similarity_threshold = _config.task_similarity_threshold

SANDBOX_EXECUTOR_GUIDANCE = """

# Sandbox Constraints
- No direct file, system, network, import, reflection, or dynamic execution APIs: `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `delattr`, `os`, `sys`, `pathlib`, `subprocess`, `socket`, `shutil`, `requests`, `httpx`, `urllib`, `builtins`.
- Use platform APIs for save/load and prefer them over custom implementations.
""".strip()

SANDBOX_FGEN_GUIDANCE = """
# Sandbox Constraints
- No direct file, system, network, import, reflection, or dynamic execution APIs: `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`, `setattr`, `delattr`, `os`, `sys`, `pathlib`, `subprocess`, `socket`, `shutil`, `requests`, `httpx`, `urllib`, `builtins`.
""".strip()


def _append_sandbox_guidance(prompt_text: str, guidance: str) -> str:
    return f"{prompt_text.rstrip()}\n\n{guidance}\n"


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


def build_executor_lmp_config_from_text(
    prompt_text: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    resolved_prompt_text = prompt_text
    if append_sandbox_guidance:
        resolved_prompt_text = _append_sandbox_guidance(resolved_prompt_text, SANDBOX_EXECUTOR_GUIDANCE)

    cfg = {
        "prompt_text": resolved_prompt_text,
        "engine": model_name,
        "max_tokens": 51200,
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
    prompt_source: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt_text = import_prompt_text(prompt_source)
    return build_executor_lmp_config_from_text(
        prompt_text,
        append_sandbox_guidance=append_sandbox_guidance,
        overrides=overrides,
    )


def build_planner_lmp_config() -> Dict[str, Any]:
    return {
        "prompt_text": prompt_manger,
        "engine": model_name,
        "max_tokens": 51200,
        "temperature": 0,
        "query_prefix": "# ",
        "query_suffix": ".",
        "stop": "#",
        "maintain_session": True,
        "debug_mode": False,
        "include_context": True,
        "has_return": False,
        "return_val_name": "ret_val",
        "skill_dirs": [str(Path("user_skills") / "planning")],
        "skill_top_k": 3,
        "skill_max_files": 20,
        "skill_max_chars_per_file": 2000,
        "skill_max_selected": 2,
        "skill_route_max_tokens": 512,
        "skill_route_temperature": 0,
        "emit_skill_routing": emit_skill_routing,
    }


def build_fgen_lmp_config() -> Dict[str, Any]:
    return {
        "prompt_text": _append_sandbox_guidance(prompt_fgen, SANDBOX_FGEN_GUIDANCE),
        "engine": model_name,
        "max_tokens": 1024,
        "temperature": 0,
        "query_prefix": "# define function: ",
        "query_suffix": ".",
        "stop": [],
        "maintain_session": False,
        "debug_mode": False,
        "include_context": True,
    }


def build_checker_lmp_config() -> Dict[str, Any]:
    return {
        "prompt_no_target": prompt_no_target,
        "prompt_over_exposed": prompt_over_exposed,
        "prompt_out_of_focus": prompt_out_of_focus,
        "prompt_quality_check": prompt_quality_check,
        "instruction_prompt_with_no_target": instruction_prompt_with_no_target,
        "instruction_prompt_without_no_target": instruction_prompt_without_no_target,
        "engine": model_name,
        "vlm_engine": vlm_model_name,
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


def build_shared_lmp_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "Task_manger": build_planner_lmp_config(),
        "fgen": build_fgen_lmp_config(),
        "checker": build_checker_lmp_config(),
    }


cfg_tabletop = {
    "lmps": build_shared_lmp_configs(),
}

