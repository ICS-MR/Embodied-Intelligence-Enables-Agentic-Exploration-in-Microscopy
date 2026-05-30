import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config.system_config import objective_labels
from services.skill_resolver import SkillResolutionRequest, SkillResolver
from utils.runtime_text import format_planner_failure_message


@dataclass
class TaskRequest:
    user_command: str
    human_mode: bool = True
    session_id: str = "default"
    planner_context: str = ""


@dataclass
class TaskPlan:
    task_id: str
    session_id: str
    user_command: str
    status: str = "error"
    question: str = ""
    selected_skills: List[str] = field(default_factory=list)
    skill_reason: str = ""
    active_templates: List[Dict[str, Any]] = field(default_factory=list)
    planner_raw_response: str = ""
    skill_routing_raw_response: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    display_text: str = ""
    ready: bool = False
    tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None


@dataclass
class TaskResult:
    task_id: str
    session_id: str
    user_command: str
    steps: List[Dict[str, Any]]
    success: bool
    retry_times: int
    summary: str
    step_summaries: List[str] = field(default_factory=list)
    checker_warnings: List[str] = field(default_factory=list)
    checker_summary: str = ""
    error: Optional[str] = None


@dataclass
class StepExecutionReport:
    step: Dict[str, Any]
    spoken_messages: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class CheckResult:
    checked: bool
    all_images_normal: bool
    revised_steps: List[Dict[str, Any]] = field(default_factory=list)
    has_no_target_error: bool = False


CHANNEL_SEMANTICS = {
    "1-NONE": "Brightfield",
    "2-U-FUNA": "DAPI / 405 nm",
    "3-U-FBNA": "FITC / 488 nm",
    "4-U-FGNA": "TRITC / 640 nm",
}


def format_objective_semantic(objective_label: str) -> str:
    magnification = objective_labels.get(str(objective_label).strip())
    if magnification is None:
        return "Unknown"
    return f"{magnification}x objective"


class TaskOrchestrator:
    def __init__(
        self,
        runtime_context: Any,
        *,
        summarize_spoken_messages: Callable[[Any, str, List[str]], str],
        summarize_my_spoken_messages: Callable[[Any, str, List[str]], str],
        summarize_step_completion: Callable[[Any, str, Dict[str, Any], List[str]], str],
        summarize_checker_issue: Callable[..., str],
        summarize_checker_success: Callable[[Any, str, str], str],
        summarize_task_execution: Callable[[Any, str, str, List[Dict[str, Any]]], str],
        rewrite_plan_for_confirmation: Callable[[Any, str, str, List[Dict[str, Any]]], str],
        stream_plan_for_confirmation: Callable[[Any, str, str, List[Dict[str, Any]], Callable[[str], None]], str],
        stream_task_execution_summary: Callable[[Any, str, str, List[Dict[str, Any]], Callable[[str], None]], str],
    ) -> None:
        self.runtime_context = runtime_context
        self._summarize_spoken_messages = summarize_spoken_messages
        self._summarize_my_spoken_messages = summarize_my_spoken_messages
        self._summarize_step_completion = summarize_step_completion
        self._summarize_checker_issue = summarize_checker_issue
        self._summarize_checker_success = summarize_checker_success
        self._summarize_task_execution = summarize_task_execution
        self._rewrite_plan_for_confirmation = rewrite_plan_for_confirmation
        self._stream_plan_for_confirmation = stream_plan_for_confirmation
        self._stream_task_execution_summary = stream_task_execution_summary
        agent_module = self.runtime_context.runtime["agent"]
        raw_skill_mode = str(getattr(agent_module, "skill_mode", "") or "").strip().lower()
        self._skill_mode = raw_skill_mode if raw_skill_mode in {"enabled", "disabled"} else "disabled"
        self._skill_enabled = self._skill_mode == "enabled"
        if hasattr(agent_module, "build_skill_resolver_config"):
            skill_resolver_cfg = agent_module.build_skill_resolver_config()
        else:
            skill_resolver_cfg = {}
        self._skill_resolver = None
        if self._skill_enabled:
            self._skill_resolver = SkillResolver(
                client=self.runtime_context.llm_client,
                model_name=agent_module.model_name,
                seed=getattr(agent_module, "llm_seed", None),
                history_manager=self.runtime_context.history_manager,
                skill_dirs=skill_resolver_cfg.get("skill_dirs"),
                skill_max_files=skill_resolver_cfg.get("skill_max_files", 20),
                skill_max_chars_per_file=skill_resolver_cfg.get("skill_max_chars_per_file", 2000),
                skill_max_selected=skill_resolver_cfg.get("skill_max_selected", 2),
                skill_route_max_tokens=skill_resolver_cfg.get("skill_route_max_tokens", 512),
                skill_route_temperature=skill_resolver_cfg.get("skill_route_temperature", 0),
                resolution_max_tokens=skill_resolver_cfg.get("resolution_max_tokens", 4096),
            )


    def _capture_microscope_state(self) -> Dict[str, Any]:
        channel = self.runtime_context.env_olympus.get_channel()
        objective = self.runtime_context.env_olympus.get_objective()
        return {
            "objective": objective,
            "objective_semantic": format_objective_semantic(objective),
            "channel": channel,
            "channel_semantic": CHANNEL_SEMANTICS.get(channel, "Unknown"),
            "exposure": self.runtime_context.env_olympus.get_exposure(),
            "brightness": self.runtime_context.env_olympus.get_brightness(),
        }

    def _resolved_planner_context(self) -> str:
        return (
            "The user request below was produced by an upstream resolver as a complete task instruction. "
            "Treat it as authoritative, and do not ask again for workflow parameters that have already been resolved."
        )

    def _merge_usage(self, *usages: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        merged: Dict[str, int] = {}
        for usage in usages:
            if not usage:
                continue
            for key, value in usage.items():
                merged[key] = merged.get(key, 0) + int(value)
        return merged or None

    def record_confirmed_plan_history(self, plan: TaskPlan) -> None:
        if not plan.ready or not plan.steps:
            return

        task_manager = getattr(self.runtime_context, "task_manager", None)
        if task_manager is None or not hasattr(task_manager, "remember_planned_task"):
            return

        task_manager.remember_planned_task(
            plan.user_command,
            self._capture_microscope_state(),
            plan.steps,
        )
    def plan(self, request: TaskRequest) -> TaskPlan:
        microscope_state = self._capture_microscope_state()
        planner_result = None
        planner_query = request.user_command
        skill_routing_raw_response = ""
        merged_tokens: Optional[Dict[str, int]] = None
        if self._skill_enabled and self._skill_resolver is not None:
            resolution_result = self._skill_resolver.resolve(
                SkillResolutionRequest(
                    user_request=request.user_command,
                    system_state=microscope_state,
                    clarification_history=request.planner_context,
                )
            )
            skill_routing_raw_response = str(resolution_result.routing_raw_response or "")
            merged_tokens = resolution_result.usage
            if resolution_result.status == "ask_user":
                task_id = str(uuid.uuid4())
                return TaskPlan(
                    task_id=task_id,
                    session_id=request.session_id,
                    user_command=request.user_command,
                    status="ask_user",
                    question=resolution_result.question,
                    selected_skills=list(resolution_result.selected_skills),
                    skill_reason=resolution_result.reason,
                    active_templates=list(resolution_result.active_templates),
                    planner_raw_response=resolution_result.raw_response,
                    skill_routing_raw_response=skill_routing_raw_response,
                    ready=False,
                    tokens=merged_tokens,
                    error=resolution_result.error,
                )
            if resolution_result.status == "ready_for_planner":
                planner_query = resolution_result.resolved_task_instruction or request.user_command
                planner_result = self.runtime_context.task_manager(
                    planner_query,
                    microscope_state,
                    self._resolved_planner_context(),
                )
                merged_tokens = self._merge_usage(merged_tokens, planner_result.tokens)
            else:
                task_id = str(uuid.uuid4())
                return TaskPlan(
                    task_id=task_id,
                    session_id=request.session_id,
                    user_command=request.user_command,
                    status="error",
                    question="",
                    selected_skills=list(resolution_result.selected_skills),
                    skill_reason=resolution_result.reason,
                    active_templates=list(resolution_result.active_templates),
                    planner_raw_response=resolution_result.raw_response,
                    skill_routing_raw_response=skill_routing_raw_response,
                    ready=False,
                    tokens=merged_tokens,
                    error=resolution_result.error or "Skill resolver failed before planner execution.",
                )

        if planner_result is None:
            planner_result = self.runtime_context.task_manager(
                request.user_command,
                microscope_state,
                request.planner_context,
            )
        task_manager = self.runtime_context.task_manager
        task_id = str(uuid.uuid4())
        tokens = self._merge_usage(merged_tokens, planner_result.tokens)

        if planner_result.ready and planner_result.steps:
            return TaskPlan(
                task_id=task_id,
                session_id=request.session_id,
                user_command=planner_query,
                status="final_plan",
                planner_raw_response=planner_result.raw_response,
                skill_routing_raw_response=skill_routing_raw_response,
                steps=planner_result.steps,
                display_text="",
                ready=True,
                tokens=tokens,
            )

        if planner_result.status == "ask_user":
            clarify_enabled = bool(getattr(task_manager, "_clarify_enabled", False))
            if clarify_enabled:
                return TaskPlan(
                    task_id=task_id,
                    session_id=request.session_id,
                    user_command=request.user_command,
                    status="ask_user",
                    question=planner_result.question,
                    planner_raw_response=planner_result.raw_response,
                    skill_routing_raw_response=skill_routing_raw_response,
                    ready=False,
                    tokens=tokens,
                    error=planner_result.error,
                )
            return TaskPlan(
                task_id=task_id,
                session_id=request.session_id,
                user_command=request.user_command,
                status="error",
                question="",
                planner_raw_response=planner_result.raw_response,
                skill_routing_raw_response=skill_routing_raw_response,
                ready=False,
                tokens=tokens,
                error=planner_result.error or "Planner returned disallowed status 'ask_user' while Clarify is disabled.",
            )

        return TaskPlan(
            task_id=task_id,
            session_id=request.session_id,
            user_command=request.user_command,
            status=planner_result.status or "error",
            question=planner_result.question,
            planner_raw_response=planner_result.raw_response,
            skill_routing_raw_response=skill_routing_raw_response,
            ready=False,
            tokens=tokens,
            error=planner_result.error or "Unable to generate an executable plan.",
        )

    def present_plan(self, plan: TaskPlan) -> str:
        if plan.display_text.strip():
            return plan.display_text.strip()
        if plan.status == "ask_user" and plan.question.strip():
            return plan.question.strip()
        if not plan.steps:
            return format_planner_failure_message(plan, prefers_zh=False)

        try:
            rewritten = self._rewrite_plan_for_confirmation(
                self.runtime_context.llm_client,
                self.runtime_context.runtime["agent"].model_name,
                plan.user_command,
                plan.steps,
            )
            if rewritten and rewritten.strip():
                return rewritten.strip()
        except Exception:
            pass

        lines = ["I have prepared a brief plan:"]
        for index, step in enumerate(plan.steps, start=1):
            command = str(step.get("command", "")).strip() or "run this step"
            lines.append(f"{index}. {command}")
        return "\n".join(lines)

    def stream_plan_preview(self, plan: TaskPlan, on_delta: Callable[[str], None]) -> str:
        return self._stream_plan_for_confirmation(
            self.runtime_context.llm_client,
            self.runtime_context.runtime["agent"].model_name,
            plan.user_command,
            plan.steps,
            on_delta,
        )

    def stream_task_summary(
        self,
        plan: TaskPlan,
        on_delta: Callable[[str], None],
        *,
        steps: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return self._stream_task_execution_summary(
            self.runtime_context.llm_client,
            self.runtime_context.runtime["agent"].model_name,
            plan.user_command,
            steps if steps is not None else plan.steps,
            on_delta,
        )

    def _checker_enabled(self) -> bool:
        runtime_agent = self.runtime_context.runtime.get("agent")
        return bool(getattr(runtime_agent, "checker_enabled", True))

    def execute(
        self,
        plan: TaskPlan,
        on_robot_action: Optional[Callable[[str], None]] = None,
        on_step_summary: Optional[Callable[[str], None]] = None,
        on_checker_warning: Optional[Callable[[str], None]] = None,
        summarize_completion: bool = True,
    ) -> TaskResult:
        if not plan.ready or not plan.steps:
            return TaskResult(
                task_id=plan.task_id,
                session_id=plan.session_id,
                user_command=plan.user_command,
                steps=[],
                success=False,
                retry_times=0,
                summary="No execution steps were generated.",
                error=plan.error or "Plan not ready.",
            )

        self.record_confirmed_plan_history(plan)
        runtime_agent = self.runtime_context.runtime['agent']
        runtime_task = self.runtime_context.runtime["task"]
        retry_count = 0
        original_steps = [step.copy() for step in plan.steps]
        current_steps = [step.copy() for step in plan.steps]
        step_summaries: List[str] = []
        checker_warnings: List[str] = []

        while retry_count < runtime_task.MAX_RETRY_TIMES:
            retry_count += 1
            original_x_y = self.runtime_context.env_olympus.get_x_y_position()

            try:
                reports = self._run_task(
                    current_steps,
                    on_robot_action=on_robot_action,
                    on_step_summary=on_step_summary,
                )
                step_summaries.extend(report.summary for report in reports if report.summary)
            except Exception as exc:
                if retry_count >= runtime_task.MAX_RETRY_TIMES:
                    return TaskResult(
                        task_id=plan.task_id,
                        session_id=plan.session_id,
                        user_command=plan.user_command,
                        steps=current_steps,
                        success=False,
                        retry_times=retry_count,
                        summary=f"Task execution failed: {exc}",
                        step_summaries=step_summaries,
                        checker_warnings=checker_warnings,
                        checker_summary="\n".join(checker_warnings),
                        error=str(exc),
                    )
                time.sleep(runtime_task.RETRY_INTERVAL)
                current_steps = [step.copy() for step in original_steps]
                continue

            if not self._checker_enabled():
                self.runtime_context.storage_manager.commit_cache()
                summary = ""
                if summarize_completion:
                    summary = self._summarize_task_execution(
                        self.runtime_context.llm_client,
                        runtime_agent.model_name,
                        plan.user_command,
                        current_steps,
                    )
                return TaskResult(
                    task_id=plan.task_id,
                    session_id=plan.session_id,
                    user_command=plan.user_command,
                    steps=current_steps,
                    success=True,
                    retry_times=retry_count,
                    summary=summary,
                    step_summaries=step_summaries,
                    checker_warnings=checker_warnings,
                    checker_summary="",
                )

            check_result = self._check_results(current_steps, original_x_y)
            if not check_result.checked or check_result.all_images_normal:
                if check_result.checked and check_result.all_images_normal and on_checker_warning is not None:
                    checker_success = self._summarize_checker_success(
                        self.runtime_context.llm_client,
                        runtime_agent.model_name,
                        plan.user_command,
                    )
                    if checker_success:
                        on_checker_warning(checker_success)
                self.runtime_context.storage_manager.commit_cache()
                summary = ""
                if summarize_completion:
                    summary = self._summarize_task_execution(
                        self.runtime_context.llm_client,
                        runtime_agent.model_name,
                        plan.user_command,
                        current_steps,
                    )
                return TaskResult(
                    task_id=plan.task_id,
                    session_id=plan.session_id,
                    user_command=plan.user_command,
                    steps=current_steps,
                    success=True,
                    retry_times=retry_count,
                    summary=summary,
                    step_summaries=step_summaries,
                    checker_warnings=checker_warnings,
                    checker_summary="\n".join(checker_warnings),
                )

            warning_summary = self._summarize_checker_issue(
                self.runtime_context.llm_client,
                runtime_agent.model_name,
                plan.user_command,
                check_result.revised_steps,
                has_no_target_error=check_result.has_no_target_error,
            )
            if warning_summary:
                checker_warnings.append(warning_summary)
                if on_checker_warning is not None:
                    on_checker_warning(warning_summary)

            if retry_count >= runtime_task.MAX_RETRY_TIMES:
                break

            current_steps = [step.copy() for step in (check_result.revised_steps or original_steps)]
            time.sleep(runtime_task.RETRY_INTERVAL)

        return TaskResult(
            task_id=plan.task_id,
            session_id=plan.session_id,
            user_command=plan.user_command,
            steps=current_steps,
            success=False,
            retry_times=retry_count,
            summary="Task execution failed after reaching the maximum number of retries.",
            step_summaries=step_summaries,
            checker_warnings=checker_warnings,
            checker_summary="\n".join(checker_warnings),
            error="Maximum retry limit reached.",
        )

    def _run_task(
        self,
        lmp_steps: List[Dict[str, Any]],
        on_robot_action: Optional[Callable[[str], None]] = None,
        on_step_summary: Optional[Callable[[str], None]] = None,
    ) -> List[StepExecutionReport]:
        storage_manager = self.runtime_context.storage_manager
        env_olympus = self.runtime_context.env_olympus
        runtime_agent = self.runtime_context.runtime["agent"]
        step_reports: List[StepExecutionReport] = []

        storage_manager.clear_cache()
        for step in sorted(lmp_steps, key=lambda item: item["subtask_index"]):
            self.runtime_context.say_capture.clear()
            meta_file = storage_manager.read_log(True)
            context = f"# Saved documents:\n {meta_file}"
            module_name = step["module"]
            command = step["command"]

            if module_name == "Microscope Operation Platform":
                current_objective = env_olympus.get_objective()
                env_info = (
                    f"Current xy_position:{env_olympus.get_x_y_position()}, "
                    f"z_position:{env_olympus.get_z_position()}, "
                    f"exposure_time:{env_olympus.get_exposure()}, "
                    f"objective:{current_objective} "
                    f"({format_objective_semantic(current_objective)}), "
                    f"dichroic:{env_olympus.get_channel()} "
                    f"({CHANNEL_SEMANTICS.get(env_olympus.get_channel(), 'Unknown')}), "
                    f"brightness:{env_olympus.get_brightness()}"
                )
                context += f"\n# Current environment:{env_info}"

            module_instance = self.runtime_context.tool_registry.get_executor(module_name)
            if module_instance is None:
                raise ValueError(f"Unknown module: {module_name}")

            self.runtime_context.say_capture.set_listener(on_robot_action)
            try:
                module_instance.run(command, context)
            finally:
                self.runtime_context.say_capture.set_listener(None)

            spoken_messages = self.runtime_context.say_capture.get_messages()
            step_summary = self._summarize_step_completion(
                self.runtime_context.llm_client,
                runtime_agent.model_name,
                step,
                spoken_messages,
            )
            if step_summary and on_step_summary is not None:
                on_step_summary(step_summary)
            step_reports.append(
                StepExecutionReport(
                    step=step.copy(),
                    spoken_messages=spoken_messages,
                    summary=step_summary,
                )
            )

        return step_reports

    def _check_results(
        self,
        original_instruction: List[Dict[str, Any]],
        original_x_y: Any,
    ) -> CheckResult:
        checker = self.runtime_context.checker
        storage_manager = self.runtime_context.storage_manager
        try:
            meta_file_temp = storage_manager.read_cache()
            if not meta_file_temp:
                return CheckResult(checked=False, all_images_normal=True)
            if not any(
                info.get("created_by") == "microscope" and info.get("file_type") == "ome-tiff"
                for info in meta_file_temp.values()
            ):
                return CheckResult(checked=False, all_images_normal=True)

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
            return CheckResult(
                checked=True,
                all_images_normal=all_images_normal,
                revised_steps=unified_instruction or [],
                has_no_target_error=has_no_target_error,
            )
        finally:
            checker.clear_history_results()







