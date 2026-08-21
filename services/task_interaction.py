import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from services.task_orchestrator import TaskPlan, TaskRequest
from interfaces.interaction_flow import (
    build_consolidated_workflow_request,
    combine_clarification_context,
    interpret_clarification_feedback,
    interpret_plan_feedback,
    is_debug_plan_request,
    pick_text,
    prefers_chinese,
)
from runtime.text import format_raw_planner_debug


MaybeAwaitable = Any | Awaitable[Any]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _format_clarification_display(plan: TaskPlan, fallback_text: str) -> str:
    details = getattr(plan, "clarification_details", {}) or {}
    if not isinstance(details, dict):
        return fallback_text

    consistency = details.get("consistency_result") or {}
    if not isinstance(consistency, dict):
        consistency = {}
    violation = details.get("violation_result") or {}
    if not isinstance(violation, dict):
        violation = {}

    reason = _clean_text(details.get("reason"))
    summary = _clean_text(consistency.get("summary"))
    differences = _string_list(consistency.get("differences"))
    question = _clean_text(plan.question) or fallback_text
    has_violation = bool(violation.get("has_violation", False))

    if not (reason or summary or differences or has_violation):
        return fallback_text

    if has_violation:
        lines = ["I found a planning detail that needs confirmation."]
        rationale = reason
        differences = []
    else:
        lines = ["I found a difference between the candidate plans that needs confirmation."]
        rationale = summary or reason
    if rationale:
        lines.extend(["", f"Rationale: {rationale}"])
    if differences:
        lines.extend(["", "Key differences:"])
        lines.extend(f"- {item}" for item in differences[:4])
    lines.extend(["", f"Question: {question}"])
    return "\n".join(lines)


@dataclass(frozen=True)
class InteractionOutcome:
    status: str
    plan: Optional[TaskPlan] = None
    summary: str = ""

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed" and self.plan is not None


@dataclass(frozen=True)
class TaskInteractionPorts:
    plan: Callable[[TaskRequest], MaybeAwaitable]
    stream_plan_preview: Callable[[TaskPlan], MaybeAwaitable]
    prompt_user: Callable[[str, str, str], MaybeAwaitable]
    send_robot_message: Callable[[str], MaybeAwaitable]
    emit_skill_summary: Callable[[TaskPlan, bool], MaybeAwaitable]
    record_user_input: Callable[[str, str, str, str, str], MaybeAwaitable]
    log_planner_tokens: Callable[[dict[str, int]], MaybeAwaitable] | None = None
    after_replan_notice: Callable[[], MaybeAwaitable] | None = None


async def _maybe_await(value: MaybeAwaitable) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class TaskInteractionSession:
    def __init__(self, ports: TaskInteractionPorts) -> None:
        self.ports = ports

    async def request_plan_confirmation(self, original_command: str) -> InteractionOutcome:
        original_command = original_command.strip()
        current_command = original_command
        revisions: list[str] = []
        clarification_entries: list[str] = []
        prefers_zh = prefers_chinese(original_command)

        while True:
            plan = await _maybe_await(
                self.ports.plan(
                    TaskRequest(
                        user_command=current_command,
                        human_mode=True,
                        planner_context=combine_clarification_context(clarification_entries),
                    )
                )
            )
            if plan.tokens and self.ports.log_planner_tokens is not None:
                await _maybe_await(self.ports.log_planner_tokens(plan.tokens))
            await _maybe_await(self.ports.emit_skill_summary(plan, prefers_zh))

            if plan.ready and plan.steps:
                await _maybe_await(self.ports.stream_plan_preview(plan))
                reply = await self._prompt_with_debug(
                    plan,
                    pick_text(
                        prefers_zh,
                        "Reply with 'confirm' or 'continue' to execute, 'cancel' to stop, type 'debug_plan' to inspect the raw planner output, or type a revision:",
                        "Reply with 'confirm' or 'continue' to execute, 'cancel' to stop, type 'debug_plan' to inspect the raw planner output, or type a revision:",
                    ),
                    prompt_record_text="plan_ready_confirmation",
                    command_snapshot=current_command,
                    prefers_zh=prefers_zh,
                )
                decision = interpret_plan_feedback(
                    reply,
                    plan_ready=True,
                    original_command=original_command,
                    revisions=revisions,
                )
                if decision.action == "confirm":
                    return InteractionOutcome(status="confirmed", plan=plan)
                if decision.action == "cancel":
                    message = pick_text(
                        prefers_zh,
                        "Okay, I will not execute this plan for now. You can revise it and ask me again later.",
                        "Okay, I will not execute this plan for now. You can revise it and ask me again later.",
                    )
                    await _maybe_await(self.ports.send_robot_message(message))
                    return InteractionOutcome(status="cancelled", plan=plan, summary=message)
                if decision.action == "empty":
                    await _maybe_await(
                        self.ports.send_robot_message(
                            pick_text(
                                prefers_zh,
                                "I have not received any revision yet. You can confirm, cancel, or type a revision directly.",
                                "I have not received any revision yet. You can confirm, cancel, or type a revision directly.",
                            )
                        )
                    )
                    continue

                revisions = decision.revisions
                current_command = decision.current_command
                await _maybe_await(
                    self.ports.send_robot_message(
                        pick_text(
                            prefers_zh,
                            "Received. I will reorganize the plan based on your revision.",
                            "Received. I will reorganize the plan based on your revision.",
                        )
                    )
                )
                await self._after_replan_notice()
                continue

            if getattr(plan, "status", "") == "ask_user":
                prompt_text = str(plan.question or "").strip() or pick_text(
                    prefers_zh,
                    "I need one key detail before I can continue planning.",
                    "I need one key detail before I can continue planning.",
                )
                visible_prompt_text = _format_clarification_display(plan, prompt_text)
                await _maybe_await(self.ports.send_robot_message(visible_prompt_text))
                reply = await self._prompt_with_debug(
                    plan,
                    pick_text(
                        prefers_zh,
                        "Answer the question, type 'debug_plan' to inspect the raw planner output, or type 'cancel':",
                        "Answer the question, type 'debug_plan' to inspect the raw planner output, or type 'cancel':",
                    ),
                    prompt_record_text=visible_prompt_text,
                    command_snapshot=current_command,
                    prefers_zh=prefers_zh,
                    prompt_mode="clarification",
                )
                decision = interpret_clarification_feedback(
                    reply,
                    entries=clarification_entries,
                    planner_question=prompt_text,
                )
                if decision.action == "cancel":
                    message = pick_text(
                        prefers_zh,
                        "Okay, I will pause this task for now.",
                        "Okay, I will pause this task for now.",
                    )
                    await _maybe_await(self.ports.send_robot_message(message))
                    return InteractionOutcome(status="cancelled", plan=plan, summary=message)
                if decision.action == "confirm_without_plan":
                    await _maybe_await(
                        self.ports.send_robot_message(
                            pick_text(
                                prefers_zh,
                                "I still do not have an executable plan yet, so I cannot start. Please answer the question first.",
                                "I still do not have an executable plan yet, so I cannot start. Please answer the question first.",
                            )
                        )
                    )
                    continue
                if decision.action == "empty":
                    await _maybe_await(
                        self.ports.send_robot_message(
                            pick_text(
                                prefers_zh,
                                "I have not received any new detail yet. Please answer the question first or type 'cancel'.",
                                "I have not received any new detail yet. Please answer the question first or type 'cancel'.",
                            )
                        )
                    )
                    continue

                clarification_entries = decision.entries
                current_command = build_consolidated_workflow_request(original_command, clarification_entries)
                await _maybe_await(
                    self.ports.send_robot_message(
                        pick_text(
                            prefers_zh,
                            "Received. I will replan using that resolved workflow detail.",
                            "Received. I will replan with that new detail.",
                        )
                    )
                )
                await self._after_replan_notice()
                continue

            if getattr(plan, "status", "") == "unsupported":
                unsupported_text = pick_text(
                    prefers_zh,
                    "The current system cannot execute this request. Here is the original planner output:",
                    "The current system cannot execute this request. Here is the original planner output:",
                )
                detail = str(plan.planner_raw_response or plan.error or "Unsupported request.")
                await _maybe_await(self.ports.send_robot_message(unsupported_text))
                await _maybe_await(self.ports.send_robot_message(detail))
                return InteractionOutcome(status="unsupported", plan=plan, summary=detail)

            await _maybe_await(
                self.ports.send_robot_message(
                    pick_text(
                        prefers_zh,
                        "I still cannot turn this into an executable plan. You can add more detail or type 'cancel'.",
                        "I still cannot turn this into an executable plan. You can add more detail or type 'cancel'.",
                    )
                )
            )
            reply = await self._prompt_with_debug(
                plan,
                pick_text(
                    prefers_zh,
                    "Please add a revision, type 'debug_plan' to inspect the raw planner output, or type 'cancel':",
                    "Please add a revision, type 'debug_plan' to inspect the raw planner output, or type 'cancel':",
                ),
                prompt_record_text="plan_revision_request",
                command_snapshot=current_command,
                prefers_zh=prefers_zh,
                prompt_mode="revision",
            )
            decision = interpret_plan_feedback(
                reply,
                plan_ready=False,
                original_command=original_command,
                revisions=revisions,
            )
            if decision.action == "cancel":
                message = pick_text(
                    prefers_zh,
                    "Okay, I will pause this task for now.",
                    "Okay, I will pause this task for now.",
                )
                await _maybe_await(self.ports.send_robot_message(message))
                return InteractionOutcome(status="cancelled", plan=plan, summary=message)
            if decision.action == "confirm_without_plan":
                await _maybe_await(
                    self.ports.send_robot_message(
                        pick_text(
                            prefers_zh,
                            "I still do not have an executable updated plan, so I cannot start yet. Please keep revising it.",
                            "I still do not have an executable updated plan, so I cannot start yet. Please keep revising it.",
                        )
                    )
                )
                continue
            if decision.action == "empty":
                continue

            revisions = decision.revisions
            current_command = decision.current_command
            await _maybe_await(
                self.ports.send_robot_message(
                    pick_text(
                        prefers_zh,
                        "Received. I will keep replanning.",
                        "Received. I will keep replanning.",
                    )
                )
            )
            await self._after_replan_notice()

    async def _prompt_with_debug(
        self,
        plan: TaskPlan,
        prompt_text: str,
        *,
        prompt_record_text: str,
        command_snapshot: str,
        prefers_zh: bool,
        prompt_mode: str = "plan_confirmation",
    ) -> str:
        while True:
            reply = await _maybe_await(
                self.ports.prompt_user(prompt_text, command_snapshot, prompt_mode)
            )
            await _maybe_await(
                self.ports.record_user_input(
                    reply,
                    "plan_feedback",
                    prompt_record_text,
                    command_snapshot,
                    prompt_mode,
                )
            )
            if is_debug_plan_request(reply):
                message = format_raw_planner_debug(plan, prefers_zh=prefers_zh)
                if message:
                    await _maybe_await(self.ports.send_robot_message(message))
                continue
            return reply

    async def _after_replan_notice(self) -> None:
        if self.ports.after_replan_notice is not None:
            await _maybe_await(self.ports.after_replan_notice())
