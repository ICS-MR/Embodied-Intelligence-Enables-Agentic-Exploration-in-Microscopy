from typing import Any, Dict

from bootstrap.config import load_model_config
from runtime.config import (
    build_checker_lmp_config as _build_checker_lmp_config,
    build_executor_lmp_config as _build_executor_lmp_config,
    build_executor_lmp_config_from_text as _build_executor_lmp_config_from_text,
    build_fgen_lmp_config as _build_fgen_lmp_config,
    build_planner_lmp_config as _build_planner_lmp_config,
    build_shared_lmp_configs as _build_shared_lmp_configs,
    build_skill_resolver_config,
    import_prompt_text,
)


_config = load_model_config()

microscope_mode = _config.microscope_mode
image_analysis_mode = _config.image_analysis_mode
segmentation_mode = _config.segmentation_mode
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
knowledge_base_path = _config.knowledge_base_path


def build_executor_lmp_config_from_text(
    prompt_text: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _build_executor_lmp_config_from_text(
        _config,
        prompt_text,
        append_sandbox_guidance=append_sandbox_guidance,
        overrides=overrides,
    )


def build_executor_lmp_config(
    prompt_source: str,
    *,
    append_sandbox_guidance: bool = True,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return _build_executor_lmp_config(
        _config,
        prompt_source,
        append_sandbox_guidance=append_sandbox_guidance,
        overrides=overrides,
    )


def build_planner_lmp_config() -> Dict[str, Any]:
    return _build_planner_lmp_config(_config)


def build_fgen_lmp_config() -> Dict[str, Any]:
    return _build_fgen_lmp_config(_config)


def build_checker_lmp_config() -> Dict[str, Any]:
    return _build_checker_lmp_config(_config)


def build_shared_lmp_configs() -> Dict[str, Dict[str, Any]]:
    return _build_shared_lmp_configs(_config)


cfg_tabletop = {
    "lmps": build_shared_lmp_configs(),
}
