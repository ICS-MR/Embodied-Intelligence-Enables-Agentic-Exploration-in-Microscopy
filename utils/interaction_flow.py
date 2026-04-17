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
    normalized_question = question.strip()
    normalized_answer = answer.strip()
    if not normalized_question:
        return normalized_answer
    return (
        "Clarification exchange:\n"
        f"Planner question: {normalized_question}\n"
        f"User answer: {normalized_answer}"
    )


@dataclass(frozen=True)
class PlanFeedbackDecision:
    action: Literal["confirm", "cancel", "empty", "revise", "confirm_without_plan"]
    reply: str
    revisions: List[str]
    current_command: str


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
