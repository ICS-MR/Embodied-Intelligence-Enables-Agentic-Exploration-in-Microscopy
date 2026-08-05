import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class PlanTraceContext:
    user_request: str
    current_plan: List[Dict[str, Any]]
    failed_step: Dict[str, Any]
    exception_type: str
    exception_message: str
    saved_documents: Dict[str, Any] = field(default_factory=dict)
    cache_documents: Dict[str, Any] = field(default_factory=dict)
    detection_targets: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanTraceDiagnosis:
    checked: bool
    category: str
    recoverable: bool
    reason: str
    planner_feedback: str = ""
    requires_replan: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PlanTraceChecker:
    """Rule-based first-pass checker for plan and instruction trajectory failures."""

    def __init__(self, history_manager: Any = None) -> None:
        self._history_manager = history_manager

    def diagnose(self, context: PlanTraceContext) -> PlanTraceDiagnosis:
        category = self._classify(context)
        recoverable = category in {
            "missing_input_image",
            "missing_target_json",
            "unknown_module",
            "plan_order_error",
        }
        diagnosis = PlanTraceDiagnosis(
            checked=True,
            category=category,
            recoverable=recoverable,
            reason=self._reason(category, context),
            planner_feedback=self._planner_feedback(category, context),
            requires_replan=recoverable,
        )
        if self._history_manager is not None:
            self._history_manager.record_interaction(
                agent_name="plan_trace_checker",
                event_type="plan_trace_diagnosis",
                message="Plan trace checker diagnosed a planning trajectory failure.",
                payload={
                    "context": context.to_dict(),
                    "diagnosis": diagnosis.to_dict(),
                },
            )
        return diagnosis

    def _classify(self, context: PlanTraceContext) -> str:
        exception_type = str(context.exception_type or "")
        message = str(context.exception_message or "")
        command = str(context.failed_step.get("command") or "")
        module = str(context.failed_step.get("module") or "")
        text = f"{command}\n{message}".lower()

        if "unknown module" in message.lower():
            return "unknown_module"
        if self._mentions_unknown_detection_target(text, context.detection_targets):
            return "unknown_detection_target"
        if exception_type in {"FileNotFoundError", "OSError"}:
            if self._looks_like_target_json_failure(text):
                return "missing_target_json"
            if module == "Image Analysis Platform" or self._looks_like_image_failure(text):
                return "missing_input_image"
        if self._looks_like_missing_prerequisite(text):
            return "plan_order_error"
        return "tool_runtime_failure"

    def _mentions_unknown_detection_target(self, text: str, detection_targets: Dict[str, Any]) -> bool:
        if "target" not in text and "class" not in text and "detect" not in text:
            return False

        configured = {str(key).lower() for key in detection_targets.keys()}
        for value in detection_targets.values():
            if isinstance(value, dict):
                target_name = value.get("target_class_name")
                if target_name:
                    configured.add(str(target_name).lower())

        patterns = (
            r"unknown (?:detection )?target[:\s'\"]+([A-Za-z0-9_\- ]+)",
            r"target class[:\s'\"]+([A-Za-z0-9_\- ]+)",
            r"detect(?:ion)? target[:\s'\"]+([A-Za-z0-9_\- ]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .'\"\n\r\t").lower()
            if candidate and candidate not in configured:
                return True
        return False

    def _looks_like_target_json_failure(self, text: str) -> bool:
        return "target position loading" in text or ("json" in text and ("target" in text or "location" in text))

    def _looks_like_image_failure(self, text: str) -> bool:
        return any(term in text for term in ("image import", "load_image", "ome.tif", "ome-tiff", ".tif", ".tiff"))

    def _looks_like_missing_prerequisite(self, text: str) -> bool:
        return any(
            term in text
            for term in (
                "no saved document",
                "no registered",
                "missing prerequisite",
                "must acquire",
                "before running",
            )
        )

    def _reason(self, category: str, context: PlanTraceContext) -> str:
        if category == "missing_input_image":
            return "The failed step requires an input image, but the plan did not provide a usable registered image first."
        if category == "missing_target_json":
            return "The failed step requires target-position JSON, but the plan did not provide a usable target JSON first."
        if category == "unknown_module":
            return "The plan references a module that is not registered in the runtime tool registry."
        if category == "unknown_detection_target":
            return "The plan requested a detection target that is not configured."
        if category == "plan_order_error":
            return "The failed step appears to require a prerequisite step that was missing or ordered incorrectly."
        detail = str(context.exception_message or "").strip()
        return f"The failure does not clearly indicate a recoverable planning trajectory issue: {detail}"

    def _planner_feedback(self, category: str, context: PlanTraceContext) -> str:
        if category == "missing_input_image":
            return "Revise the plan so an image is acquired or a registered existing image is selected before image analysis."
        if category == "missing_target_json":
            return "Revise the plan so target positions are detected and saved as registered JSON before loading or using target positions."
        if category == "unknown_module":
            return "Revise the plan using only modules registered in the current runtime tool registry."
        if category == "unknown_detection_target":
            available = ", ".join(sorted(str(key) for key in context.detection_targets.keys())) or "none"
            return f"The requested detection target is unavailable. Use a configured target or mark the task unsupported. Available targets: {available}."
        if category == "plan_order_error":
            return "Revise the plan so every analysis or movement step has its required acquisition or metadata step before it."
        return "Do not replan automatically unless the planner can identify a missing or incorrectly ordered prerequisite."
