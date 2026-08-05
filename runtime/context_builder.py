from typing import Any, Dict

from core_tool.tool_utils import SayCapture
from services.task_orchestrator import TaskOrchestrator
from storage.managers import HistoryManager, StorageManager
from runtime.agent_factory import build_planner_and_checker, build_shared_lmps
from runtime.models import RuntimeContext
from runtime.session import create_runtime_session_paths
from runtime.text import (
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
from runtime.tool_factory import assemble_runtime_tools


def build_runtime_context(
    runtime: Dict[str, Any],
    llm_client: Any,
    vlm_client: Any,
) -> RuntimeContext:
    agent_config = runtime["agent"]
    system_config = runtime["system"]
    task_config = runtime["task"]
    shared_lmps = build_shared_lmps(agent_config, system_config)

    session_id, session_dir, session_output_dir = create_runtime_session_paths(
        task_config.HISTORY_DIR,
        task_config.OUTPUT_DIR,
    )
    history_manager = HistoryManager(session_dir)
    storage_manager = StorageManager(session_dir, session_output_dir)
    say_capture = SayCapture()

    tool_assembly = assemble_runtime_tools(
        runtime=runtime,
        llm_client=llm_client,
        history_manager=history_manager,
        storage_manager=storage_manager,
        say_capture=say_capture,
        session_dir=session_dir,
        output_dir=session_output_dir,
        shared_lmps=shared_lmps,
    )
    task_manager, checker, plan_trace_checker, code_repair_agent = build_planner_and_checker(
        runtime=runtime,
        shared_lmps=shared_lmps,
        llm_client=llm_client,
        vlm_client=vlm_client,
        output_dir=session_output_dir,
        history_manager=history_manager,
    )

    runtime_context = RuntimeContext(
        session_id=session_id,
        session_dir=session_dir,
        output_dir=session_output_dir,
        runtime=runtime,
        llm_client=llm_client,
        vlm_client=vlm_client,
        say_capture=say_capture,
        env_olympus=tool_assembly.env_bindings.get("microscope_operation"),
        env_imagej=tool_assembly.env_bindings.get("image_analysis"),
        env_cellpose=tool_assembly.env_bindings.get("cell_segmentation"),
        storage_manager=storage_manager,
        history_manager=history_manager,
        task_manager=task_manager,
        tool_registry=tool_assembly.tool_registry,
        checker=checker,
        plan_trace_checker=plan_trace_checker,
        code_repair_agent=code_repair_agent,
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
