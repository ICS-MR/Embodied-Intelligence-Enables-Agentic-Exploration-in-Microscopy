from typing import Any, Dict, Tuple

from adapters.llm_clients import build_openai_clients
from agent.code_repair import CodeRepairAgent
from agent.experiment_checker import ExperimentCheckAgent
from agent.experiment_planner import ExperimentPlanAgent
from agent.plan_trace_checker import PlanTraceChecker
from storage.managers import HistoryManager
from runtime.config import (
    build_executor_lmp_config,
    build_executor_lmp_config_from_text,
    build_microscope_executor_lmp_config,
    build_shared_lmp_configs,
)


def build_clients(agent_config: Any) -> Tuple[Any, Any]:
    bundle = build_openai_clients(agent_config)
    return bundle.llm_client, bundle.vlm_client


def build_shared_lmps(agent_config: Any, system_config: Any | None = None) -> Dict[str, Dict[str, Any]]:
    return build_shared_lmp_configs(agent_config, system_config)


def build_system_executor_cfg(agent_config: Any, prompt_source: str) -> Dict[str, Any]:
    return build_executor_lmp_config(agent_config, prompt_source, append_sandbox_guidance=True)


def build_microscope_executor_cfg(agent_config: Any, prompt_source: str, system_config: Any) -> Dict[str, Any]:
    return build_microscope_executor_lmp_config(
        agent_config,
        prompt_source,
        system_config,
        append_sandbox_guidance=True,
    )


def build_user_executor_cfg(agent_config: Any, prompt_text: str) -> Dict[str, Any]:
    return build_executor_lmp_config_from_text(agent_config, prompt_text, append_sandbox_guidance=True)


def build_planner_and_checker(
    *,
    runtime: Dict[str, Any],
    shared_lmps: Dict[str, Dict[str, Any]],
    llm_client: Any,
    vlm_client: Any,
    output_dir: str,
    history_manager: HistoryManager,
) -> tuple[ExperimentPlanAgent, ExperimentCheckAgent, PlanTraceChecker, CodeRepairAgent]:
    agent_config = runtime["agent"]
    runtime_settings = runtime.get("settings")
    clarify_enabled = bool(getattr(agent_config, "clarify_enabled", False))
    if runtime_settings is not None and hasattr(runtime_settings, "model"):
        clarify_enabled = bool(getattr(runtime_settings.model, "clarify_enabled", clarify_enabled))

    task_manager = ExperimentPlanAgent(
        "Task_manager",
        shared_lmps["Task_manger"],
        llm_client,
        history_manager,
        clarify_tag=clarify_enabled,
    )
    checker = ExperimentCheckAgent(
        shared_lmps["checker"],
        llm_client,
        vlm_client,
        output_dir,
        history_manager,
    )
    plan_trace_checker = PlanTraceChecker(history_manager)
    code_repair_agent = CodeRepairAgent(history_manager)
    return task_manager, checker, plan_trace_checker, code_repair_agent
