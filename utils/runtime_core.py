from typing import Any, Callable, Dict, List, Optional, Tuple

from bootstrap.config import StartupConfig, load_startup_config
from services.task_orchestrator import TaskPlan, TaskRequest
from utils.cli_logging import get_cli_logger
from utils.memory_manager import StorageManager
from utils.runtime_factory import (
    build_clients,
    build_var_map,
    get_tool_classes,
    initialize_system_components,
    load_runtime_config,
)
from utils.runtime_models import RuntimeContext, SystemComponents
from utils.runtime_text import (
    rewrite_task_plan_for_confirmation,
    stream_plan_preview_for_confirmation,
    stream_task_execution_summary,
    summarize_checker_issue,
    summarize_my_spoken_messages,
    summarize_spoken_messages,
    summarize_step_completion,
    summarize_task_execution,
)


planner_logger = get_cli_logger("PLANNER")


def _call_if_available(obj: Any, method_name: str, *args: Any) -> None:
    if hasattr(obj, method_name):
        getattr(obj, method_name)(*args)


def initialize_microscope(env_olympus: Any) -> None:
    env_olympus.initialize()


def apply_startup_state(env_olympus: Any, startup_config: Optional[StartupConfig] = None) -> None:
    startup = startup_config or load_startup_config()
    _call_if_available(env_olympus, "set_objective", startup.objective)
    if hasattr(env_olympus, "remember_brightfield_brightness"):
        env_olympus.remember_brightfield_brightness(startup.brightness)
    _call_if_available(env_olympus, "set_channel", startup.channel)
    _call_if_available(env_olympus, "set_exposure", startup.exposure)
    _call_if_available(env_olympus, "set_brightness", startup.brightness)
    _call_if_available(env_olympus, "set_z_position", startup.z_position)


def setup_microscope(env_olympus: Any, startup_config: Optional[StartupConfig] = None) -> None:
    initialize_microscope(env_olympus)
    apply_startup_state(env_olympus, startup_config)


def process_instruction(
    system_components: RuntimeContext,
    command: str,
    human_mode: bool = True,
) -> Tuple[bool, List[Dict[str, Any]], Optional[Dict[str, int]]]:
    planner_logger.info("Received instruction: %s", command)
    request = TaskRequest(user_command=command, human_mode=human_mode)
    plan = system_components.task_orchestrator.plan(request)
    if plan.ready and plan.steps:
        planner_logger.info("Generated %s planned step(s)", len(plan.steps))
    return plan.ready, plan.steps or [], plan.tokens


def check_results(
    storage_manager: StorageManager,
    original_instruction: List[Dict[str, Any]],
    original_x_y: Any,
    checker: Any,
) -> Tuple[bool, List[Dict[str, Any]], bool]:
    try:
        meta_file_temp = storage_manager.read_cache()
        if not meta_file_temp:
            return True, [], False
        if not any(
            info.get("created_by") == "microscope" and info.get("file_type") == "ome-tiff"
            for info in meta_file_temp.values()
        ):
            return True, [], False

        checker.batch_check_from_json(meta_file_temp)
        has_no_target_error = checker.has_any_no_target()
        if has_no_target_error:
            cache_filenames = list(meta_file_temp.keys())
            storage_manager.batch_delete_files(
                filenames=cache_filenames,
                delete_physical=True,
                remove_meta=True,
            )

        unified_instruction = checker.generate_task_unified_instruction(
            original_x_y,
            original_instruction=original_instruction,
        )
        all_images_normal = checker.all_results_defect_free()
        return all_images_normal, unified_instruction or [], has_no_target_error
    finally:
        checker.clear_history_results()


def run_task(
    system_components: RuntimeContext,
    lmp_steps: List[Dict[str, Any]],
    on_robot_action: Optional[Callable[[str], None]] = None,
) -> None:
    system_components.task_orchestrator._run_task(lmp_steps, on_robot_action=on_robot_action)


def run_task_with_validation(
    system_components: RuntimeContext,
    original_lmp_steps: List[Dict[str, Any]],
    on_robot_action: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, int]:
    plan = TaskPlan(
        task_id="compatibility-plan",
        session_id="default",
        user_command="compatibility execution",
        steps=original_lmp_steps,
        ready=bool(original_lmp_steps),
    )
    result = system_components.task_orchestrator.execute(plan, on_robot_action=on_robot_action)
    return result.success, result.retry_times


def release_resources(system_components: RuntimeContext) -> None:
    env_olympus = system_components.env_olympus
    env_imagej = system_components.env_imagej

    if env_olympus and hasattr(env_olympus, "shutdown"):
        env_olympus.shutdown()

    if env_imagej and hasattr(env_imagej, "fiji_shutdown"):
        env_imagej.fiji_shutdown()

    system_components.storage_manager.clear_cache()
