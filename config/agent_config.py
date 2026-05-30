import importlib
from pathlib import Path
import os
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
skill_mode = _config.skill_mode
openai_api_key = _config.openai_api_key
base_url = _config.base_url
model_name = _config.model_name
llm_seed = _config.llm_seed
vlm_api_key = _config.vlm_api_key
vlm_base_url = _config.vlm_base_url
vlm_model_name = _config.vlm_model_name
CROSS_ENCODER_MODEL_PATH = _config.CROSS_ENCODER_MODEL_PATH
cross_encoder_model_path = CROSS_ENCODER_MODEL_PATH
task_similarity_threshold = _config.task_similarity_threshold

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

    cfg = {
        "prompt_text": resolved_prompt_text,
        "engine": model_name,
        "seed": llm_seed,
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
        "seed": llm_seed,
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


def build_fgen_lmp_config() -> Dict[str, Any]:
    return {
        "prompt_text": prompt_fgen,
        "engine": model_name,
        "seed": llm_seed,
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
        "seed": llm_seed,
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

