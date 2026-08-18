import time
import uuid
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from bootstrap.microscope_semantics import (
    channel_display_name,
    channel_semantic_for_label,
    objective_display_name,
    objective_semantic_for_label,
)
from runtime.config import _has_transmitted_light_brightness_control
from services.skill_resolver import SkillResolutionRequest, SkillResolver
from runtime.config import build_skill_resolver_config
from runtime.text import StepSummaryResult, format_planner_failure_message
from agent.code_repair import CodeRepairContext
from agent.plan_trace_checker import PlanTraceContext


logger = logging.getLogger(__name__)

CONSOLIDATED_WORKFLOW_PREFIX = "Consolidated workflow specification for replanning:"


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
    clarification_details: Dict[str, Any] = field(default_factory=dict)


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
    summary_used_fallback: bool = False
    summary_error: str = ""


class StepExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        step: Dict[str, Any],
        original_exception: Exception,
        executor: Any = None,
        saved_documents: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(str(original_exception))
        self.step = step
        self.original_exception = original_exception
        self.executor = executor
        self.saved_documents = saved_documents or {}


@dataclass
class ExecutionTrace:
    task_id: str
    attempt: int
    step: Dict[str, Any]
    module: str
    command: str
    exception_type: str
    exception_message: str
    generated_code: str = ""
    executor_query: str = ""
    saved_documents: Dict[str, Any] = field(default_factory=dict)
    cache_documents: Dict[str, Any] = field(default_factory=dict)
    executor_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    checked: bool
    all_images_normal: bool
    revised_steps: List[Dict[str, Any]] = field(default_factory=list)
    has_no_target_error: bool = False


class TaskOrchestrator:
    def __init__(
        self,
        runtime_context: Any,
        *,
        summarize_spoken_messages: Callable[[Any, str, List[str]], str],
        summarize_my_spoken_messages: Callable[[Any, str, List[str]], str],
        summarize_step_completion: Callable[[Any, str, Dict[str, Any], List[str]], StepSummaryResult],
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
        skill_resolver_cfg = build_skill_resolver_config()
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
        system_config = self.runtime_context.runtime["system"]
        state = {
            "objective": objective,
            "objective_semantic": objective_semantic_for_label(objective, system_config),
            "objective_display": objective_display_name(objective, system_config),
            "channel": channel,
            "channel_semantic": channel_semantic_for_label(channel, system_config),
            "channel_display": channel_display_name(channel, system_config),
            "exposure": self.runtime_context.env_olympus.get_exposure(),
        }
        if _has_transmitted_light_brightness_control(system_config):
            state["brightness"] = self.runtime_context.env_olympus.get_brightness()
        return state

    def _resolved_planner_context(self) -> str:
        return (
            "The user request below was produced by an upstream resolver as a complete task instruction. "
            "Treat it as authoritative, and do not ask again for workflow parameters that have already been resolved."
        )

    def _planner_context_after_skill_resolution(self, clarification_history: str) -> str:
        base_context = self._resolved_planner_context()
        normalized_history = str(clarification_history or "").strip()
        if not normalized_history:
            return base_context
        return "\n\n".join(
            [
                base_context,
                normalized_history,
                "Treat the resolved clarification history above as authoritative. Do not ask again about those same resolved choices.",
            ]
        )

    def _direct_planner_context(self, request: TaskRequest) -> str:
        if str(request.user_command or "").lstrip().startswith(CONSOLIDATED_WORKFLOW_PREFIX):
            return ""
        return request.planner_context

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
                    self._planner_context_after_skill_resolution(request.planner_context),
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
                self._direct_planner_context(request),
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
                    clarification_details=dict(getattr(planner_result, "clarification_details", {}) or {}),
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
            clarification_details=dict(getattr(planner_result, "clarification_details", {}) or {}),
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
        except Exception as exc:
            logger.warning("Plan presentation rewrite failed; using structured plan fallback: %s", exc, exc_info=True)

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
                logger.exception(
                    "Task step execution attempt failed. attempt=%s max_attempts=%s",
                    retry_count,
                    runtime_task.MAX_RETRY_TIMES,
                )
                execution_trace = self._build_execution_trace(
                    task_id=plan.task_id,
                    attempt=retry_count,
                    exc=exc,
                    fallback_steps=current_steps,
                )
                self.runtime_context.storage_manager.clear_cache()

                code_advice = self._diagnose_code_repair(execution_trace)
                if code_advice is not None and getattr(code_advice, "checked", False):
                    advice_payload = code_advice.to_dict() if hasattr(code_advice, "to_dict") else {}
                    if bool(advice_payload.get("recoverable")) and bool(advice_payload.get("retry_same_step")):
                        repair_instruction = str(
                            advice_payload.get("repair_instruction")
                            or advice_payload.get("reason")
                            or ""
                        ).strip()
                        if repair_instruction:
                            checker_warnings.append(repair_instruction)
                            if on_checker_warning is not None:
                                on_checker_warning(repair_instruction)
                        current_steps = [step.copy() for step in current_steps]
                        if retry_count >= runtime_task.MAX_RETRY_TIMES:
                            break
                        time.sleep(runtime_task.RETRY_INTERVAL)
                        continue

                plan_diagnosis = self._diagnose_plan_trace(
                    plan=plan,
                    current_steps=current_steps,
                    execution_trace=execution_trace,
                )
                if plan_diagnosis is not None and getattr(plan_diagnosis, "checked", False):
                    diagnosis_payload = plan_diagnosis.to_dict() if hasattr(plan_diagnosis, "to_dict") else {}
                    planner_feedback = str(
                        diagnosis_payload.get("planner_feedback")
                        or diagnosis_payload.get("reason")
                        or ""
                    ).strip()
                    if planner_feedback:
                        checker_warnings.append(planner_feedback)
                        if on_checker_warning is not None:
                            on_checker_warning(planner_feedback)

                    if not bool(diagnosis_payload.get("recoverable")):
                        detail = f"{execution_trace.exception_type}: {execution_trace.exception_message}"
                        return TaskResult(
                            task_id=plan.task_id,
                            session_id=plan.session_id,
                            user_command=plan.user_command,
                            steps=current_steps,
                            success=False,
                            retry_times=retry_count,
                            summary=self._format_execution_failure_summary(
                                execution_trace,
                                outcome="The plan trace checker determined that automatic replanning would not resolve the failure.",
                                checker_feedback=planner_feedback,
                            ),
                            step_summaries=step_summaries,
                            checker_warnings=checker_warnings,
                            checker_summary="\n".join(checker_warnings),
                            error=detail,
                        )

                    replanned_steps = self._replan_after_trace_failure(
                        plan=plan,
                        current_steps=current_steps,
                        execution_trace=execution_trace,
                        planner_feedback=planner_feedback,
                    )
                    if replanned_steps:
                        current_steps = [step.copy() for step in replanned_steps]
                    else:
                        detail = f"{execution_trace.exception_type}: {execution_trace.exception_message}"
                        return TaskResult(
                            task_id=plan.task_id,
                            session_id=plan.session_id,
                            user_command=plan.user_command,
                            steps=current_steps,
                            success=False,
                            retry_times=retry_count,
                            summary=self._format_execution_failure_summary(
                                execution_trace,
                                outcome="Automatic replanning did not produce executable steps.",
                                checker_feedback=planner_feedback,
                            ),
                            step_summaries=step_summaries,
                            checker_warnings=checker_warnings,
                            checker_summary="\n".join(checker_warnings),
                            error=detail,
                        )
                    if retry_count >= runtime_task.MAX_RETRY_TIMES:
                        break
                    time.sleep(runtime_task.RETRY_INTERVAL)
                    continue

                if retry_count >= runtime_task.MAX_RETRY_TIMES:
                    original_exception = exc.original_exception if isinstance(exc, StepExecutionError) else exc
                    detail = f"{type(original_exception).__name__}: {original_exception}"
                    return TaskResult(
                        task_id=plan.task_id,
                        session_id=plan.session_id,
                        user_command=plan.user_command,
                        steps=current_steps,
                        success=False,
                        retry_times=retry_count,
                        summary=f"Task execution failed: {detail}",
                        step_summaries=step_summaries,
                        checker_warnings=checker_warnings,
                        checker_summary="\n".join(checker_warnings),
                        error=detail,
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

    @staticmethod
    def _format_execution_failure_summary(
        execution_trace: ExecutionTrace,
        *,
        outcome: str,
        checker_feedback: str = "",
    ) -> str:
        step_index = execution_trace.step.get("subtask_index")
        step_label = f"step {step_index}" if step_index is not None else "the current step"
        if execution_trace.module:
            step_label += f" ({execution_trace.module})"

        detail = f"{execution_trace.exception_type}: {execution_trace.exception_message}"
        summary = f"Task execution failed at {step_label}. Cause: {detail}. {outcome}"
        feedback = str(checker_feedback or "").strip()
        if feedback:
            summary += f" Checker feedback: {feedback}"
        return summary

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
        ordered_steps = sorted(lmp_steps, key=lambda item: item["subtask_index"])
        frap_phase_active = False
        frap_handoff_state: Optional[Dict[str, Any]] = None

        try:
            for step in ordered_steps:
                self.runtime_context.say_capture.clear()
                meta_file = storage_manager.read_log(True)
                context = f"# Saved documents:\n {meta_file}"
                module_name = step["module"]
                command = step["command"]

                is_frap_step = self._is_frap_module(module_name)
                if is_frap_step:
                    module_name = "frap"
                if is_frap_step and not frap_phase_active:
                    try:
                        frap_handoff_state = self._begin_frap_handoff()
                    except Exception as exc:
                        raise StepExecutionError(
                            step=step.copy(),
                            original_exception=exc,
                            executor=None,
                            saved_documents=meta_file,
                        ) from exc
                    frap_phase_active = True
                elif not is_frap_step and frap_phase_active:
                    try:
                        self._end_frap_handoff(frap_handoff_state)
                    except Exception as exc:
                        raise StepExecutionError(
                            step=step.copy(),
                            original_exception=exc,
                            executor=None,
                            saved_documents=meta_file,
                        ) from exc
                    finally:
                        frap_phase_active = False
                        frap_handoff_state = None

                if module_name == "Microscope Operation Platform":
                    current_objective = env_olympus.get_objective()
                    system_config = self.runtime_context.runtime["system"]
                    current_channel = env_olympus.get_channel()
                    env_parts = [
                        f"Current xy_position:{env_olympus.get_x_y_position()}, "
                        f"z_position:{env_olympus.get_z_position()}, "
                        f"exposure_time:{env_olympus.get_exposure()}, "
                        f"objective:{current_objective} "
                        f"({objective_display_name(current_objective, system_config)}; "
                        f"semantic={objective_semantic_for_label(current_objective, system_config) or 'unknown'}), "
                        f"dichroic:{current_channel} "
                        f"({channel_display_name(current_channel, system_config)}; "
                        f"semantic={channel_semantic_for_label(current_channel, system_config) or 'unknown'})",
                    ]
                    if _has_transmitted_light_brightness_control(system_config):
                        env_parts.append(f"brightness:{env_olympus.get_brightness()}")
                    env_info = ", ".join(env_parts)
                    context += f"\n# Current environment:{env_info}"

                module_instance = self.runtime_context.tool_registry.get_executor(module_name)
                if module_instance is None:
                    raise StepExecutionError(
                        step=step.copy(),
                        original_exception=ValueError(f"Unknown module: {module_name}"),
                        executor=None,
                        saved_documents=meta_file,
                    )

                self.runtime_context.say_capture.set_listener(on_robot_action)
                try:
                    module_instance.run(command, context)
                except Exception as exc:
                    raise StepExecutionError(
                        step=step.copy(),
                        original_exception=exc,
                        executor=module_instance,
                        saved_documents=meta_file,
                    ) from exc
                finally:
                    self.runtime_context.say_capture.set_listener(None)

                spoken_messages = self.runtime_context.say_capture.get_messages()
                step_summary_result = self._summarize_step_completion(
                    self.runtime_context.llm_client,
                    runtime_agent.model_name,
                    step,
                    spoken_messages,
                )
                step_summary = step_summary_result.text
                if step_summary_result.used_fallback:
                    logger.warning(
                        "Step summary used fallback. module=%s subtask_index=%s reason=%s",
                        step.get("module", "Unknown"),
                        step.get("subtask_index", "?"),
                        step_summary_result.error or "unknown",
                    )
                if step_summary and on_step_summary is not None:
                    on_step_summary(step_summary)
                step_reports.append(
                    StepExecutionReport(
                        step=step.copy(),
                        spoken_messages=spoken_messages,
                        summary=step_summary,
                        summary_used_fallback=step_summary_result.used_fallback,
                        summary_error=step_summary_result.error,
                    )
                )
        except Exception as exc:
            if frap_phase_active:
                try:
                    self._end_frap_handoff(frap_handoff_state)
                except Exception as cleanup_exc:
                    detail = RuntimeError(
                        f"{type(exc).__name__}: {exc}; additionally failed to release FRAP/restore Micro-Manager: {cleanup_exc}"
                    )
                    if isinstance(exc, StepExecutionError):
                        raise StepExecutionError(
                            step=exc.step,
                            original_exception=detail,
                            executor=exc.executor,
                            saved_documents=exc.saved_documents,
                        ) from exc
                    raise detail from exc
            raise
        else:
            if frap_phase_active:
                last_step = ordered_steps[-1] if ordered_steps else {}
                try:
                    self._end_frap_handoff(frap_handoff_state)
                except Exception as exc:
                    raise StepExecutionError(
                        step=last_step.copy(),
                        original_exception=exc,
                        executor=None,
                        saved_documents=storage_manager.read_log(True),
                    ) from exc

        return step_reports

    @staticmethod
    def _is_frap_module(module_name: Any) -> bool:
        return str(module_name or "").strip().lower() == "frap"

    def _begin_frap_handoff(self) -> Optional[Dict[str, Any]]:
        env_olympus = getattr(self.runtime_context, "env_olympus", None)
        if env_olympus is None:
            return None

        capture_state = getattr(env_olympus, "capture_handoff_state", None)
        release = getattr(env_olympus, "release_for_handoff", None)
        if not callable(capture_state) or not callable(release):
            raise RuntimeError("Microscope controller does not support FRAP handoff.")
        state = capture_state()
        release()
        return state

    def _end_frap_handoff(self, handoff_state: Optional[Dict[str, Any]]) -> None:
        errors: List[str] = []
        frap_binding = self.runtime_context.tool_registry.get_tool("frap")
        frap_env = getattr(frap_binding, "env", None) if frap_binding is not None else None
        release_session = getattr(frap_env, "release_session", None)
        if callable(release_session):
            try:
                release_session()
            except Exception as exc:
                errors.append(f"FRAP release: {exc}")

        env_olympus = getattr(self.runtime_context, "env_olympus", None)
        restore = getattr(env_olympus, "restore_after_handoff", None)
        if handoff_state is not None:
            if not callable(restore):
                errors.append("Micro-Manager restore: microscope controller does not support handoff restore")
            else:
                try:
                    restore(handoff_state)
                except Exception as exc:
                    errors.append(f"Micro-Manager restore: {exc}")

        if errors:
            raise RuntimeError("; ".join(errors))

    def _build_execution_trace(
        self,
        *,
        task_id: str,
        attempt: int,
        exc: Exception,
        fallback_steps: List[Dict[str, Any]],
    ) -> ExecutionTrace:
        if isinstance(exc, StepExecutionError):
            failed_step = dict(exc.step)
            original_exception = exc.original_exception
            executor = exc.executor
            saved_documents = dict(exc.saved_documents)
        else:
            failed_step = dict(fallback_steps[-1]) if fallback_steps else {}
            original_exception = exc
            executor = None
            saved_documents = self._safe_read_saved_documents(include_temp=True)

        executor_record = {}
        if executor is not None:
            record = getattr(executor, "last_execution_record", {})
            if isinstance(record, dict):
                executor_record = dict(record)

        cache_documents = self._safe_read_cache_documents()
        trace = ExecutionTrace(
            task_id=task_id,
            attempt=attempt,
            step=failed_step,
            module=str(failed_step.get("module") or ""),
            command=str(failed_step.get("command") or ""),
            exception_type=type(original_exception).__name__,
            exception_message=str(original_exception),
            generated_code=str(executor_record.get("generated_code") or ""),
            executor_query=str(executor_record.get("query") or ""),
            saved_documents=saved_documents,
            cache_documents=cache_documents,
            executor_record=executor_record,
        )
        history_manager = getattr(self.runtime_context, "history_manager", None)
        if history_manager is not None:
            history_manager.record_interaction(
                agent_name="Runtime",
                event_type="step_execution_failed",
                message="A task step failed during execution.",
                payload=trace.to_dict(),
            )
        return trace

    def _diagnose_code_repair(self, execution_trace: ExecutionTrace) -> Any:
        code_repair_agent = getattr(self.runtime_context, "code_repair_agent", None)
        diagnose = getattr(code_repair_agent, "diagnose", None)
        if not callable(diagnose):
            return None
        try:
            return diagnose(
                CodeRepairContext(
                    step=dict(execution_trace.step),
                    exception_type=execution_trace.exception_type,
                    exception_message=execution_trace.exception_message,
                    generated_code=execution_trace.generated_code,
                    executor_query=execution_trace.executor_query,
                    executor_context=str(execution_trace.executor_record.get("context") or ""),
                    executor_record=dict(execution_trace.executor_record),
                )
            )
        except Exception:
            logger.exception("Code repair advisor failed while diagnosing generated-code error.")
            return None

    def _diagnose_plan_trace(
        self,
        *,
        plan: TaskPlan,
        current_steps: List[Dict[str, Any]],
        execution_trace: ExecutionTrace,
    ) -> Any:
        plan_trace_checker = getattr(self.runtime_context, "plan_trace_checker", None)
        diagnose = getattr(plan_trace_checker, "diagnose", None)
        if not callable(diagnose):
            return None
        try:
            return diagnose(
                PlanTraceContext(
                    user_request=plan.user_command,
                    current_plan=self._summarize_plan_steps(current_steps, execution_trace.step),
                    failed_step=dict(execution_trace.step),
                    exception_type=execution_trace.exception_type,
                    exception_message=execution_trace.exception_message,
                    saved_documents=dict(execution_trace.saved_documents),
                    cache_documents=dict(execution_trace.cache_documents),
                    detection_targets=dict(self.runtime_context.runtime.get("detection_targets", {})),
                )
            )
        except Exception:
            logger.exception("Plan trace checker failed while diagnosing planning trajectory error.")
            return None

    def _summarize_plan_steps(
        self,
        current_steps: List[Dict[str, Any]],
        failed_step: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        failed_index = failed_step.get("subtask_index")
        summary: List[Dict[str, Any]] = []
        for step in current_steps:
            step_index = step.get("subtask_index")
            if step_index == failed_index:
                status = "failed"
            elif failed_index is not None and step_index is not None and step_index < failed_index:
                status = "success"
            else:
                status = "not_started"
            summary.append(
                {
                    "subtask_index": step_index,
                    "module": step.get("module"),
                    "command": step.get("command"),
                    "status": status,
                }
            )
        return summary

    def _replan_after_trace_failure(
        self,
        *,
        plan: TaskPlan,
        current_steps: List[Dict[str, Any]],
        execution_trace: ExecutionTrace,
        planner_feedback: str,
    ) -> List[Dict[str, Any]]:
        context = self._build_repair_planner_context(
            current_steps=current_steps,
            execution_trace=execution_trace,
            planner_feedback=planner_feedback,
        )
        planner_result = self.runtime_context.task_manager(
            plan.user_command,
            self._capture_microscope_state(),
            context,
        )
        if getattr(planner_result, "ready", False) and getattr(planner_result, "steps", None):
            return [step.copy() for step in planner_result.steps]
        logger.warning(
            "Planner did not produce executable steps after trace repair feedback. error=%s",
            getattr(planner_result, "error", ""),
        )
        return []

    def _build_repair_planner_context(
        self,
        *,
        current_steps: List[Dict[str, Any]],
        execution_trace: ExecutionTrace,
        planner_feedback: str,
    ) -> str:
        payload = {
            "repair_reason": planner_feedback,
            "failed_step": {
                "subtask_index": execution_trace.step.get("subtask_index"),
                "module": execution_trace.module,
                "command": execution_trace.command,
            },
            "failure_summary": {
                "exception_type": execution_trace.exception_type,
                "exception_message": execution_trace.exception_message,
            },
            "current_plan": self._summarize_plan_steps(current_steps, execution_trace.step),
            "artifact_summary": {
                "saved_documents": self._summarize_artifacts(execution_trace.saved_documents),
                "cache_documents": self._summarize_artifacts(execution_trace.cache_documents),
            },
        }
        return (
            "The previous execution attempt failed because the plan or instruction trajectory was incomplete. "
            "Revise the task steps to satisfy the feedback below. Do not debug generated Python code here.\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
        )

    def _summarize_artifacts(self, documents: Dict[str, Any]) -> List[Dict[str, Any]]:
        artifacts: List[Dict[str, Any]] = []
        for name, info in documents.items():
            if not isinstance(info, dict):
                artifacts.append({"name": name})
                continue
            artifacts.append(
                {
                    "name": name,
                    "file_type": info.get("file_type"),
                    "created_by": info.get("created_by"),
                    "description": info.get("description"),
                }
            )
        return artifacts

    def _safe_read_saved_documents(self, *, include_temp: bool) -> Dict[str, Any]:
        storage_manager = self.runtime_context.storage_manager
        read_log = getattr(storage_manager, "read_log", None)
        if not callable(read_log):
            return {}
        try:
            result = read_log(include_temp)
        except Exception:
            logger.warning("Failed to read saved documents for execution trace.", exc_info=True)
            return {}
        return dict(result) if isinstance(result, dict) else {}

    def _safe_read_cache_documents(self) -> Dict[str, Any]:
        storage_manager = self.runtime_context.storage_manager
        read_cache = getattr(storage_manager, "read_cache", None)
        if not callable(read_cache):
            return {}
        try:
            result = read_cache()
        except Exception:
            logger.warning("Failed to read cache documents for execution trace.", exc_info=True)
            return {}
        return dict(result) if isinstance(result, dict) else {}

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







