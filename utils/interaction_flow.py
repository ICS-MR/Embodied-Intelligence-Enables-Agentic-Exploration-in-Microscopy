from dataclasses import dataclass
import re
from typing import List, Literal, Optional


NEGATIVE_CONFIRMATION_KEYWORDS = (
    "cancel",
    "stop",
    "abort",
    "exit",
    "no",
)

POSITIVE_CONFIRMATION_KEYWORDS = (
    "yes",
    "ok",
    "okay",
    "continue",
    "confirm",
    "run",
    "start",
    "execute",
)

DEBUG_PLAN_KEYWORDS = (
    "debugplan",
    "debug_plan",
    "rawplan",
    "raw_plan",
    "showrawplan",
    "show_raw_plan",
    "plannerraw",
    "planner_raw",
)


def prefers_chinese(text: str) -> bool:
    del text
    return False


def pick_text(prefers_zh: bool, zh_text: str, en_text: str) -> str:
    return zh_text if prefers_zh else en_text


def parse_confirmation(text: str) -> Optional[bool]:
    lowered = text.strip().lower()
    normalized = "".join(
        ch for ch in lowered
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"
    )
    if not normalized:
        return None

    english_tokens = re.findall(r"[a-z]+", lowered)
    is_short_english_reply = bool(english_tokens) and len(english_tokens) <= 3
    is_short_cjk_reply = not english_tokens and len(normalized) <= 12

    if is_short_english_reply:
        if any(keyword in english_tokens for keyword in NEGATIVE_CONFIRMATION_KEYWORDS if keyword.isascii()):
            return False
        if any(keyword in english_tokens for keyword in POSITIVE_CONFIRMATION_KEYWORDS if keyword.isascii()):
            return True

    if is_short_cjk_reply:
        if any(keyword in normalized for keyword in NEGATIVE_CONFIRMATION_KEYWORDS if not keyword.isascii()):
            return False
        if any(keyword in normalized for keyword in POSITIVE_CONFIRMATION_KEYWORDS if not keyword.isascii()):
            return True

    return None


def is_debug_plan_request(text: str) -> bool:
    normalized = "".join(ch for ch in text.strip().lower() if not ch.isspace())
    return normalized in DEBUG_PLAN_KEYWORDS


def combine_replanned_command(base_command: str, revisions: List[str]) -> str:
    if not revisions:
        return base_command
    parts = [base_command.strip()]
    for index, revision in enumerate(revisions, start=1):
        parts.append(f"Supplemental user instruction {index}: {revision}")
    return "\n\n".join(part for part in parts if part)


def format_clarification_response(question: str, answer: str) -> str:
    normalized_answer = answer.strip()
    if not normalized_answer:
        return ""
    return normalized_answer


def combine_clarification_context(entries: List[str]) -> str:
    if not entries:
        return ""
    parts = [
        "Resolved workflow parameters and clarification details collected so far:",
    ]
    for index, entry in enumerate(entries, start=1):
        normalized_entry = entry.strip()
        if not normalized_entry:
            continue
        parts.append(f"{index}. {normalized_entry}")
    return "\n".join(parts)


def build_consolidated_workflow_request(base_command: str, entries: List[str]) -> str:
    normalized_command = base_command.strip()
    if not entries:
        return normalized_command

    parts = [
        "Consolidated workflow specification for replanning:",
        "",
        "Original workflow intent:",
        normalized_command,
        "",
        "Resolved workflow parameters and execution constraints:",
    ]
    for index, entry in enumerate(entries, start=1):
        normalized_entry = entry.strip()
        if not normalized_entry:
            continue
        parts.append(f"{index}. {normalized_entry}")
    parts.extend(
        [
            "",
            "Planner instruction:",
            "- Treat the resolved workflow parameters above as authoritative clarified requirements.",
            "- Do not ask again for parameters that are already resolved above.",
            "- Rewrite the clarified workflow into a complete executable experiment plan directly unless a genuinely new blocking ambiguity still remains.",
        ]
    )
    return "\n".join(parts)


def summarize_clarification_request(question: str, answer: str) -> str:
    normalized_question = question.strip()
    normalized_answer = answer.strip()
    if not normalized_question:
        return normalized_answer
    return (
        f"Resolved clarification for request \"{normalized_question}\": {normalized_answer}"
    )


@dataclass(frozen=True)
class PlanFeedbackDecision:
    action: Literal["confirm", "cancel", "empty", "revise", "confirm_without_plan"]
    reply: str
    revisions: List[str]
    current_command: str


@dataclass(frozen=True)
class ClarificationFeedbackDecision:
    action: Literal["cancel", "empty", "revise", "confirm_without_plan"]
    reply: str
    entries: List[str]
    planner_context: str


def interpret_plan_feedback(
    reply: str,
    *,
    plan_ready: bool,
    original_command: str,
    revisions: List[str],
    planner_question: str = "",
) -> PlanFeedbackDecision:
    decision = parse_confirmation(reply)
    if decision is False:
        return PlanFeedbackDecision(
            action="cancel",
            reply=reply,
            revisions=list(revisions),
            current_command=combine_replanned_command(original_command, revisions),
        )

    if decision is True:
        action: Literal["confirm", "confirm_without_plan"] = "confirm" if plan_ready else "confirm_without_plan"
        return PlanFeedbackDecision(
            action=action,
            reply=reply,
            revisions=list(revisions),
            current_command=combine_replanned_command(original_command, revisions),
        )

    revision = reply.strip()
    if not revision:
        return PlanFeedbackDecision(
            action="empty",
            reply=reply,
            revisions=list(revisions),
            current_command=combine_replanned_command(original_command, revisions),
        )

    normalized_revision = format_clarification_response(planner_question, revision)
    updated_revisions = [*revisions, normalized_revision]
    return PlanFeedbackDecision(
        action="revise",
        reply=reply,
        revisions=updated_revisions,
        current_command=combine_replanned_command(original_command, updated_revisions),
    )


def interpret_clarification_feedback(
    reply: str,
    *,
    entries: List[str],
    planner_question: str = "",
) -> ClarificationFeedbackDecision:
    decision = parse_confirmation(reply)
    if decision is False:
        return ClarificationFeedbackDecision(
            action="cancel",
            reply=reply,
            entries=list(entries),
            planner_context=combine_clarification_context(entries),
        )

    if decision is True:
        return ClarificationFeedbackDecision(
            action="confirm_without_plan",
            reply=reply,
            entries=list(entries),
            planner_context=combine_clarification_context(entries),
        )

    revision = reply.strip()
    if not revision:
        return ClarificationFeedbackDecision(
            action="empty",
            reply=reply,
            entries=list(entries),
            planner_context=combine_clarification_context(entries),
        )

    normalized_entry = summarize_clarification_request(planner_question, revision)
    updated_entries = [*entries, normalized_entry]
    return ClarificationFeedbackDecision(
        action="revise",
        reply=reply,
        entries=updated_entries,
        planner_context=combine_clarification_context(updated_entries),
    )
