from typing import Any, Callable, Dict, List
import inspect
import re

# Predefined say() method template.
SAY_METHOD_TEMPLATE = '''def say(message: str):
    """
    Outputs a log message with `[ACTION]`, `[INFO]`, or `[ERROR]` prefix. Ensures consistent logging format.
    """
    print(f'robot says: {message}')'''


def format_method_with_docstring(def_line: str, docstring: str) -> str:
    if not docstring.strip():
        return f"{def_line}\n    pass  # No docstring"
    lines = docstring.splitlines()
    if len(lines) == 1:
        doc_block = f'    """{lines[0]}"""'
    else:
        doc_block = '    """\n' + '\n'.join(f"    {line}" for line in lines) + '\n    """'
    return f"{def_line}\n{doc_block}"


# Mark a function as a tool function.
def tool_func(func: Callable) -> Callable:
    """Decorator that marks a function as a tool function without registering it immediately."""
    func._is_tool_func = True
    return func


def _annotation_name(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _build_param_schema(parameters: List[inspect.Parameter]) -> Dict[str, Dict[str, Any]]:
    schema: Dict[str, Dict[str, Any]] = {}
    for param in parameters:
        schema[param.name] = {
            "type": _annotation_name(param.annotation),
            "required": param.default is inspect._empty,
            "default": None if param.default is inspect._empty else param.default,
        }
    return schema


def _infer_side_effect(attr_name: str) -> bool:
    read_only_prefixes = ("get_", "read_", "load_", "list_", "preview_", "dump_")
    return not attr_name.startswith(read_only_prefixes)


def _slugify_tool_name(value: str) -> str:
    lowered = value.strip().replace("-", "_").replace(" ", "_")
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in lowered)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()


def _docstring_summary(docstring: str, fallback: str) -> str:
    text = (docstring or "").strip()
    if not text:
        return fallback
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return fallback
    sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0].strip()
    return sentence or fallback


def _humanize_tool_method_name(name: str) -> str:
    cleaned = name.replace("_", " ").strip()
    if not cleaned:
        return "Tool Operation"
    return cleaned[:1].upper() + cleaned[1:]


class ToolMeta(type):
    """Metaclass that collects functions marked with @tool_func into a subclass-local registry."""
    def __new__(cls, name, bases, attrs):
        # Create the class.
        subclass = super().__new__(cls, name, bases, attrs)

        # Initialize the subclass-local registry.
        subclass._tool_func_registry = {}

        # Scan all attributes defined on the class.
        for attr_name, attr_value in attrs.items():
            if callable(attr_value) and getattr(attr_value, '_is_tool_func', False):
                # This function was marked with @tool_func.
                try:
                    sig = inspect.signature(attr_value)
                    params = list(sig.parameters.values())

                    # Remove self if this is an instance method.
                    if params and params[0].name == 'self' and params[0].kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ):
                        params = params[1:]
                    
                    filtered_sig = sig.replace(parameters=params)
                    signature_str = f"def {attr_name}{filtered_sig}:"
                except Exception as e:
                    signature_str = f"def {attr_name}() -> <signature_parse_failed>:"
                    print(f"[WARNING] Failed to inspect signature for function {attr_name}: {e}")
                
                docstring = (attr_value.__doc__ or "").strip()
                
                subclass._tool_func_registry[attr_name] = {
                    "signature": signature_str,
                    "docstring": docstring,
                    "parameters": _build_param_schema(params),
                    "returns": _annotation_name(sig.return_annotation),
                    "side_effect": _infer_side_effect(attr_name),
                    "simulation_safe": True,
                }
        
        return subclass


class BaseTool(metaclass=ToolMeta):
    """Base class that provides tool-function discovery and prompt helpers."""

    tool_func = staticmethod(tool_func)
    planning_hint = ""
    execution_hint = ""

    @classmethod
    def _merged_tool_func_registry(cls) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for base in reversed(cls.__mro__):
            registry = getattr(base, "_tool_func_registry", None)
            if not isinstance(registry, dict):
                continue
            for name, payload in registry.items():
                merged[name] = dict(payload)
        return merged
    
    @classmethod
    def get_public_methods(cls) -> List[str]:
        method_names = list(cls._merged_tool_func_registry().keys())
        method_names.sort()
        return method_names

    @classmethod
    def get_tool_methods(cls) -> List[Dict[str, str]]:
        return [
            {
                "name": func_name,
                "signature": func_info["signature"],
                "docstring": func_info["docstring"]
            }
            for func_name, func_info in cls._merged_tool_func_registry().items()
        ]

    @classmethod
    def get_tool_descriptors(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": func_name,
                "signature": func_info["signature"],
                "docstring": func_info["docstring"],
                "parameters": func_info["parameters"],
                "returns": func_info["returns"],
                "side_effect": func_info["side_effect"],
                "simulation_safe": func_info["simulation_safe"],
            }
            for func_name, func_info in cls._merged_tool_func_registry().items()
        ]

    @classmethod
    def get_formatted_tool_methods(cls, inject_say: bool = True) -> str:
        method_strs = []
        for tool_method in cls.get_tool_methods():
            method_str = format_method_with_docstring(
                tool_method["signature"],
                tool_method["docstring"]
            )
            method_strs.append(method_str)

        if inject_say and "def say(" not in "\n".join(method_strs):
            method_strs.append(SAY_METHOD_TEMPLATE)

        return "\n\n".join(method_strs)

    @classmethod
    def get_default_tool_id(cls) -> str:
        return _slugify_tool_name(cls.__name__)

    @classmethod
    def get_execution_hint(cls) -> str:
        return str(getattr(cls, "execution_hint", "") or "").strip()

    @classmethod
    def get_planning_hint(cls) -> str:
        return str(getattr(cls, "planning_hint", "") or "").strip()

    @classmethod
    def get_planning_capability_lines(cls) -> List[str]:
        lines: List[str] = []
        for descriptor in cls.get_tool_descriptors():
            capability_name = _humanize_tool_method_name(descriptor["name"])
            summary = _docstring_summary(
                str(descriptor.get("docstring") or ""),
                f"Use this method when the `{descriptor['name']}` capability is needed.",
            )
            lines.append(f"- {capability_name}: {summary}")
        if not lines:
            lines.append("- Tool Operation: This tool exposes no documented public methods.")
        return lines

    @classmethod
    def get_planning_submodule_block(
        cls,
        *,
        tool_id: str | None = None,
        planning_hint: str = "",
    ) -> str:
        resolved_tool_id = (tool_id or cls.get_default_tool_id()).strip() or cls.get_default_tool_id()
        lines = [f"### {resolved_tool_id}"] + cls.get_planning_capability_lines()
        hint_text = planning_hint.strip() or cls.get_planning_hint()
        if hint_text:
            lines.append(f"- Preferred usage: {hint_text}")
        lines.append("---")
        return "\n".join(lines)

    @classmethod
    def get_planning_example_block(
        cls,
        *,
        tool_id: str | None = None,
        planning_hint: str = "",
    ) -> str:
        resolved_tool_id = (tool_id or cls.get_default_tool_id()).strip() or cls.get_default_tool_id()
        descriptors = cls.get_tool_descriptors()
        first_descriptor = descriptors[0] if descriptors else None
        capability_summary = _docstring_summary(
            str(first_descriptor.get("docstring") or "") if first_descriptor else "",
            "Use the documented tool capability.",
        )
        del planning_hint
        example_input = f"Use the {resolved_tool_id} tool to {capability_summary.lower().rstrip('.')}"
        example_input = example_input.rstrip(".")
        command_summary = capability_summary[:1].upper() + capability_summary[1:]
        if not re.match(r"^[A-Za-z]", command_summary):
            command_summary = f"Use the documented tool capability for {resolved_tool_id}."
        return "\n".join(
            [
                "# Example input:",
                example_input,
                "",
                "# Example output",
                "<Task Ready>",
                '{"Status": "OK"}',
                "</Task Ready>",
                "<Task steps>",
                "[",
                "    {",
                '        "subtask_index": 1,',
                f'        "module": "{resolved_tool_id}",',
                f'        "command": "{command_summary}"',
                "    }",
                "]",
                "</Task steps>",
            ]
        )

    @classmethod
    def get_execution_prompt_context(cls, *, tool_id: str | None = None, execution_hint: str = "") -> str:
        resolved_tool_id = (tool_id or cls.get_default_tool_id()).strip() or cls.get_default_tool_id()
        hint_text = execution_hint.strip() or cls.get_execution_hint()
        sections = [
            f"Tool ID: {resolved_tool_id}",
            f"Tool class: {cls.__name__}",
            "You are generating Python code for this tool only.",
            "Use only the public methods listed below and respect their parameter types, defaults, and docstrings.",
            "Available API methods:",
            cls.get_formatted_tool_methods(inject_say=True),
        ]
        if hint_text:
            sections.append("Execution guidance:\n" + hint_text)
        return "\n\n".join(section for section in sections if section.strip())

    @classmethod
    def get_planning_prompt_context(cls, *, tool_id: str | None = None, planning_hint: str = "") -> str:
        resolved_tool_id = (tool_id or cls.get_default_tool_id()).strip() or cls.get_default_tool_id()
        hint_text = planning_hint.strip() or cls.get_planning_hint()
        descriptor_lines = cls.get_planning_capability_lines()

        sections = [
            f"Tool ID: {resolved_tool_id}",
            f"Tool class: {cls.__name__}",
            "Use the tool name above directly in the planner `module` field.",
            "Capabilities:",
            "\n".join(descriptor_lines),
        ]
        if hint_text:
            sections.append("Planning guidance:\n" + hint_text)
        return "\n\n".join(section for section in sections if section.strip())
    
