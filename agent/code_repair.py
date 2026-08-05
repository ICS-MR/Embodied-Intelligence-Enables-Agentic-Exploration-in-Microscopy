from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class CodeRepairContext:
    step: Dict[str, Any]
    exception_type: str
    exception_message: str
    generated_code: str = ""
    executor_query: str = ""
    executor_context: str = ""
    executor_record: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CodeRepairAdvice:
    checked: bool
    category: str
    recoverable: bool
    reason: str
    repair_instruction: str = ""
    retry_same_step: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CodeRepairAgent:
    """Rule-based first-pass advisor for generated-code failures."""

    def __init__(self, history_manager: Any = None) -> None:
        self._history_manager = history_manager

    def diagnose(self, context: CodeRepairContext) -> CodeRepairAdvice:
        category = self._classify(context)
        recoverable = category in {
            "sandbox_blocked_operation",
            "name_error",
            "type_error",
            "api_call_error",
            "code_file_reference_error",
        }
        advice = CodeRepairAdvice(
            checked=True,
            category=category,
            recoverable=recoverable,
            reason=self._reason(category, context),
            repair_instruction=self._repair_instruction(category),
            retry_same_step=recoverable,
        )
        if self._history_manager is not None:
            self._history_manager.record_interaction(
                agent_name="code_repair",
                event_type="code_repair_diagnosis",
                message="Code repair advisor diagnosed a generated-code failure.",
                payload={"context": context.to_dict(), "advice": advice.to_dict()},
            )
        return advice

    def _classify(self, context: CodeRepairContext) -> str:
        exception_type = str(context.exception_type or "")
        message = str(context.exception_message or "")
        code = str(context.generated_code or "")
        text = f"{message}\n{code}".lower()

        if self._mentions_forbidden_operation(text):
            return "sandbox_blocked_operation"
        if exception_type == "NameError":
            return "name_error"
        if exception_type == "TypeError":
            return "type_error"
        if exception_type in {"AttributeError", "ImportError"}:
            return "api_call_error"
        if (
            exception_type in {"FileNotFoundError", "OSError"}
            and self._mentions_specific_file_reference(text)
            and self._has_relevant_artifact_context(context)
        ):
            return "code_file_reference_error"
        return "tool_runtime_failure"

    def _mentions_forbidden_operation(self, text: str) -> bool:
        forbidden_terms = ("open(", "exit(", "quit(", "forbidden", "not allowed", "blocked", "unsafe")
        sandbox_terms = ("open", "exit", "quit", "sandbox", "builtin")
        return any(term in text for term in forbidden_terms) and any(term in text for term in sandbox_terms)

    def _mentions_specific_file_reference(self, text: str) -> bool:
        return any(term in text for term in (".ome.tif", ".ome-tiff", ".tif", ".tiff", ".json"))

    def _has_relevant_artifact_context(self, context: CodeRepairContext) -> bool:
        context_text = str(context.executor_context or "").lower()
        return "# saved documents" in context_text and any(
            term in context_text for term in (".ome.tif", ".ome-tiff", ".tif", ".tiff", ".json")
        )

    def _reason(self, category: str, context: CodeRepairContext) -> str:
        if category == "sandbox_blocked_operation":
            return "The generated code attempted an operation blocked by the execution sandbox."
        if category == "name_error":
            return "The generated code referenced a name that is not available in the executor environment."
        if category == "type_error":
            return "The generated code called an available API with incompatible arguments or value types."
        if category == "api_call_error":
            return "The generated code used an API shape that does not match the executor environment."
        if category == "code_file_reference_error":
            return "The generated code referenced a concrete file that was not available at execution time."
        detail = str(context.exception_message or "").strip()
        return f"The tool failed during generated-code execution: {detail}" if detail else "The tool failed during generated-code execution."

    def _repair_instruction(self, category: str) -> str:
        if category == "sandbox_blocked_operation":
            return "Regenerate the same step using only provided platform APIs and allowed builtins."
        if category == "name_error":
            return "Regenerate the same step using only names exposed in the executor prompt and context."
        if category == "type_error":
            return "Regenerate the same step with API arguments matching the documented executor methods."
        if category == "api_call_error":
            return "Regenerate the same step using the documented API names and signatures."
        if category == "code_file_reference_error":
            return "Regenerate the same step by selecting files from registered saved documents instead of inventing paths."
        return "Do not retry as a code-only repair unless the failure can be fixed without changing the plan."
