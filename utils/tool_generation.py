from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from openai import OpenAI

from tool.base import BaseTool, _slugify_tool_name
from utils.tool_doc_paths import DEFAULT_USER_TOOL_DOCS_DIR
from utils.tool_manifest import discover_tool_candidates, import_string, load_tool_manifest


class Config:
    PROMPTS: Dict[str, str] = {
        "system_role": """
You are a Python programming assistant skilled in understanding code, function definitions, and generating relevant examples.
Output format:
# Example input
Task instruction based on the function
# Example output
Function usage example
""",
        "docstring": """You are a professional Python developer. Generate a concise and accurate Google-style docstring in English for the following function.
Requirements:
- Include Args and Returns if applicable
- Do NOT include function definition, triple quotes, code fences, or extra explanations
- Return ONLY the docstring content

Function signature:
{signature}

Generated docstring:
""",
        "usage": """You are a Python expert. Based on the following method definitions, generate a concise, correct, and low-risk usage example.

Requirements:
- Use English only.
- Keep the example simple and conservative. Prefer the shortest correct sequence that demonstrates the core API.
- Use only the provided API methods and the built-in `say()` helper when useful.
- Treat the documented API methods as atomic capabilities. If the requested behavior is not a one-shot API, you may compose multiple documented API calls with standard Python computation, control flow, and intermediate data generation.
- Do not import extra libraries unless they are absolutely required by the shown API usage.
- Do not invent files, hardware state, return schemas, or parameters not supported by the method definitions.
- Do not add unnecessary complexity such as retries, helper wrappers, dry-run previews, broad exception handling, or multi-branch workflows unless they are directly required by the API contract.
- If a return value is a dictionary, access only keys that are clearly documented in the method docstring.
- Avoid dangerous or irreversible behavior in the example. Prefer one straightforward happy-path example with minimal validation.
- The example output must be code only. Do not include explanatory prose, headings, markdown fences, tables, separators, or notes before or after the code.
- Use ASCII characters only in the generated code unless non-ASCII is explicitly required by the API contract.
- Output format must strictly follow:
# Example input
<A brief task description>
# Example output
<Directly runnable Python code without any markdown code fences (e.g., no ```python or ```)>.

Method definitions:
{methods}

Generate the usage example:
""",
        "summary": """You are a technical documentation expert. Summarize the core capabilities of the following Python class based on its public methods.
Requirements:
- Use English
- List by functional modules (start each with "- ")
- One sentence per point, concise and clear
- Do NOT repeat method names; abstract their purpose into planner-relevant operational capabilities
- Prefer task-level capability descriptions over GUI, click, window, or other implementation details when possible
- Describe what the tool can accomplish for planning, not merely what each function literally says
- When several low-level methods support one higher-level workflow ability, summarize them as a single higher-level capability
- Keep the abstraction bounded by the documented API. Do not invent capabilities that cannot be achieved either directly by a documented public API or by composing documented public APIs.
- It is allowed to describe a higher-level capability that can be achieved by combining multiple documented atomic capabilities, but make that abstraction operational and realistic rather than speculative.
- When composition matters, prefer describing the resulting accomplishable task capability instead of restating each low-level action separately.
- When documented atomic capabilities can be repeatedly applied over computed point sequences to realize a higher-level spatial or procedural task, it is allowed to summarize that higher-level accomplishable capability explicitly.
- Ignore placeholder texts like 'auto generated docstring failed'
- Do NOT include code, examples, or prefixes

Methods:
{methods}

Capabilities summary:
""",
        "task_example": """You are a task planning expert. Generate a simple, correct, and low-risk planning example in the specified format based on the tool's capabilities.

Tool name: {tool_id}
Capabilities:
{capabilities}

Requirements:
- Generate a realistic, specific, and simple user instruction (# Example input).
- Keep the plan short and conservative. Prefer 1 to 4 steps.
- Use only capabilities that are explicitly documented.
- Treat documented capabilities as atomic building blocks. Do not mark a task unsupported only because there is no one-shot API if the requested outcome can be achieved by composing documented capabilities into a valid short plan.
- Do not invent hidden state such as "current image", "current ROI", or unspecified hardware configuration when the capability description requires an explicit input.
- If the capability involves image input, prefer an explicit OME-TIFF-style path such as `sample.ome.tif` instead of an implicit image reference.
- Do not expose internal implementation details in planner commands when a higher-level tool capability already covers them.
- Output must strictly follow:
<Task Ready>
{{"Status": "OK"}}
</Task Ready>
<Task steps>
[
    {{
        "subtask_index": 1,
        "module": "{tool_id}",
        "command": "Clear, specific action in English, starting with a verb"
    }}
]
</Task steps>
- The 'command' must be based ONLY on the given capabilities
- Both input and command must be in English
- Each command should be high-level, short, and operationally clear.
- ONLY output # Example input and # Example output sections, no extra text

Generate example:
""",
    }


@dataclass(frozen=True)
class GeneratedToolArtifact:
    tool_id: str
    class_path: str
    execution_prompt_path: str
    planning_summary_path: str


@dataclass(frozen=True)
class ToolDocOverride:
    tool_id: str
    planning_hint: str = ""
    execution_hint: str = ""


class LLMAgent:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4.1",
        temperature: float = 0.2,
        timeout: int = 30,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url.strip())
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = Config.PROMPTS["system_role"].strip()

    def generate_required(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt.strip()},
                ],
                temperature=self.temperature,
                timeout=self.timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("LLM returned empty content during tool prompt generation.")
            return content
        except Exception as exc:
            raise RuntimeError(f"Tool prompt generation failed: {exc}") from exc


class ToolDocGenerator:
    def __init__(self, llm_agent: LLMAgent):
        self.llm_agent = llm_agent
        self.prompts = Config.PROMPTS

    def generate_class_usage_example(self, public_methods: str, tool_cls: type[BaseTool]) -> str:
        custom_example = tool_cls.get_usage_example_block(tool_id=tool_cls.get_default_tool_id())
        if custom_example.strip():
            return custom_example.strip()
        prompt = self.prompts["usage"].format(methods=public_methods)
        return self.llm_agent.generate_required(prompt)

    def summarize_class_capabilities(self, public_methods: str, tool_cls: type[BaseTool]) -> str:
        prompt = self.prompts["summary"].format(methods=public_methods)
        raw_summary = self.llm_agent.generate_required(prompt)
        lines = [
            line for line in raw_summary.splitlines()
            if not line.strip().startswith(("Capabilities summary", "Summary", ":", "```", "Note", "Requirement"))
        ]
        cleaned = "\n".join(line for line in lines if line.strip()).strip()
        if not cleaned:
            raise RuntimeError(
                f"Tool prompt generation produced an unusable capability summary for {tool_cls.__name__}."
            )
        return cleaned

    def generate_task_example(self, tool_id: str, capability_summary: str) -> str:
        prompt = self.prompts["task_example"].format(tool_id=tool_id, capabilities=capability_summary)
        return self.llm_agent.generate_required(prompt)

    def _build_abstract_planning_submodule_block(
        self,
        tool_cls: type[BaseTool],
        tool_id: str,
        capability_summary: str,
        planning_hint: str = "",
    ) -> str:
        lines = [f"### {tool_id}"]
        summary_lines = [line.strip() for line in capability_summary.splitlines() if line.strip()]
        if summary_lines:
            lines.extend(summary_lines)
        else:
            lines.extend(tool_cls.get_planning_capability_lines())
        hint_text = planning_hint.strip() or tool_cls.get_planning_hint()
        if hint_text:
            lines.append(f"- Preferred usage: {hint_text}")
        lines.append("---")
        return "\n".join(lines)

    def build_execution_prompt_text(
        self,
        tool_cls: type[BaseTool],
        tool_id: str,
        execution_hint: str = "",
    ) -> str:
        public_methods = tool_cls.get_formatted_tool_methods(inject_say=True)
        usage_example = self.generate_class_usage_example(public_methods, tool_cls)
        return "\n\n".join(
            section
            for section in [
                tool_cls.get_execution_prompt_context(tool_id=tool_id, execution_hint=execution_hint),
                "# Usage Example\n" + usage_example.strip(),
            ]
            if section.strip()
        )

    def build_planning_summary_text(
        self,
        tool_cls: type[BaseTool],
        tool_id: str,
        planning_hint: str = "",
    ) -> str:
        public_methods = tool_cls.get_formatted_tool_methods(inject_say=False)
        capability_summary = self.summarize_class_capabilities(public_methods, tool_cls)
        submodule_block = self._build_abstract_planning_submodule_block(
            tool_cls,
            tool_id,
            capability_summary,
            planning_hint=planning_hint,
        )
        task_example = self.generate_task_example(tool_id, capability_summary)
        return "\n\n".join(
            section
            for section in [
                submodule_block,
                "# Capability Summary\n" + capability_summary.strip(),
                "# Task Example\n" + task_example.strip(),
            ]
            if section.strip()
        )

    def _build_usage_fallback(self, tool_cls: type[BaseTool]) -> str:
        methods = tool_cls.get_public_methods()
        lines = ["# Example input", f"Use {tool_cls.__name__} for a simple task.", "# Example output"]
        lines.append(f"tool = {tool_cls.__name__}(storage_manager=None, output_dir='./output')")
        for method_name in methods[:1]:
            signature = inspect.signature(getattr(tool_cls, method_name))
            args = []
            for param in signature.parameters.values():
                if param.name == "self":
                    continue
                if param.default is inspect._empty:
                    args.append(self._placeholder_argument(param.annotation))
                else:
                    args.append(repr(param.default))
            lines.append(f"tool.{method_name}({', '.join(args)})")
        return "\n".join(lines)

    def _build_capability_fallback(self, tool_cls: type[BaseTool]) -> str:
        items = []
        for descriptor in tool_cls.get_tool_descriptors():
            docstring = str(descriptor.get("docstring") or "").strip()
            if docstring:
                first_line = next((line.strip() for line in docstring.splitlines() if line.strip()), "")
                items.append(f"- {first_line}")
            else:
                items.append(f"- Supports the `{descriptor['name']}` operation.")
        return "\n".join(items) if items else "- No documented capabilities were found."

    def _build_task_example_fallback(self, tool_id: str, capability_summary: str) -> str:
        first_summary = next((line.strip("- ").strip() for line in capability_summary.splitlines() if line.strip()), "use the tool")
        example_input = f"Use the {tool_id} tool to {first_summary.lower().rstrip('.') }."
        steps = [
            {
                "subtask_index": 1,
                "module": tool_id,
                "command": "Run the appropriate tool action based on the documented capability",
            }
        ]
        if tool_id == "frap":
            example_input = (
                "Use the frap tool to enable FRAP mode, extract a cell contour from "
                "`cell_image.ome.tif`, move the laser to the detected centroid, and then disable FRAP mode."
            )
            steps = [
                {
                    "subtask_index": 1,
                    "module": tool_id,
                    "command": "Enable FRAP mode for the session",
                },
                {
                    "subtask_index": 2,
                    "module": tool_id,
                    "command": "Extract the cell contour from `cell_image.ome.tif` to obtain centroid_px coordinates",
                },
                {
                    "subtask_index": 3,
                    "module": tool_id,
                    "command": "Move the laser to the detected centroid coordinates and trigger the FRAP click",
                },
                {
                    "subtask_index": 4,
                    "module": tool_id,
                    "command": "Disable FRAP mode after the click sequence",
                },
            ]
        return "\n".join(
            [
                "# Example input",
                example_input,
                "# Example output",
                "<Task Ready>",
                '{"Status": "OK"}',
                "</Task Ready>",
                "<Task steps>",
                json.dumps(steps, indent=4, ensure_ascii=False),
                "</Task steps>",
            ]
        )

    def _placeholder_argument(self, annotation: object) -> str:
        annotation_text = str(annotation).lower()
        if "int" in annotation_text or "float" in annotation_text:
            return "0"
        if "bool" in annotation_text:
            return "False"
        return repr("example")


class ToolProcessingPipeline:
    def __init__(self, openai_api_key: str, base_url: str, model_name: str):
        if not openai_api_key or openai_api_key == "your-openai-api-key":
            raise EnvironmentError(
                "OPENAI_API_KEY is not configured. Prompt artifact generation requires a real LLM and does not support fallback output."
            )
        self.llm_agent = LLMAgent(openai_api_key, base_url=base_url, model=model_name)
        self.doc_generator = ToolDocGenerator(self.llm_agent)

    def run_pipeline(
        self,
        *,
        allowed_class_paths: Optional[Set[str]] = None,
        output_dir: str | Path = DEFAULT_USER_TOOL_DOCS_DIR,
    ) -> List[GeneratedToolArtifact]:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        artifacts: List[GeneratedToolArtifact] = []
        tool_doc_overrides = self._build_tool_doc_overrides()

        for tool_cls, class_path in self._iter_tool_classes(allowed_class_paths):
            override = tool_doc_overrides.get(
                class_path,
                ToolDocOverride(tool_id=tool_cls.get_default_tool_id()),
            )
            execution_text = self.doc_generator.build_execution_prompt_text(
                tool_cls,
                override.tool_id,
                execution_hint=override.execution_hint,
            )
            planning_text = self.doc_generator.build_planning_summary_text(
                tool_cls,
                override.tool_id,
                planning_hint=override.planning_hint,
            )
            artifact_key = _slugify_tool_name(override.tool_id)

            execution_path = target_dir / f"{artifact_key}.executor_prompt.txt"
            planning_path = target_dir / f"{artifact_key}.planner_summary.txt"
            execution_path.write_text(execution_text + "\n", encoding="utf-8")
            planning_path.write_text(planning_text + "\n", encoding="utf-8")

            artifacts.append(
                GeneratedToolArtifact(
                    tool_id=override.tool_id,
                    class_path=class_path,
                    execution_prompt_path=str(execution_path),
                    planning_summary_path=str(planning_path),
                )
            )
        return artifacts

    def _build_tool_doc_overrides(self) -> Dict[str, ToolDocOverride]:
        manifest = load_tool_manifest()
        return {
            entry.class_path: ToolDocOverride(
                tool_id=entry.tool_id,
                planning_hint=entry.planning_hint,
                execution_hint=entry.execution_hint,
            )
            for entry in manifest.user_tools
            if entry.tool_id.strip()
        }

    def _iter_tool_classes(self, allowed_class_paths: Optional[Set[str]]) -> Iterable[tuple[type[BaseTool], str]]:
        allowed = set(allowed_class_paths or set())
        seen: set[str] = set()

        for candidate in discover_tool_candidates():
            if allowed and candidate.class_path not in allowed:
                continue
            tool_cls = self._load_tool_class(candidate.class_path)
            yield tool_cls, candidate.class_path
            seen.add(candidate.class_path)

        for class_path in sorted(allowed - seen):
            yield self._load_tool_class(class_path), class_path

    def _load_tool_class(self, class_path: str) -> type[BaseTool]:
        tool_obj = import_string(class_path)
        if not isinstance(tool_obj, type) or not issubclass(tool_obj, BaseTool):
            raise TypeError(f"{class_path} is not a BaseTool subclass")
        if not tool_obj.get_public_methods():
            raise ValueError(f"{class_path} does not expose any @tool_func methods")
        return tool_obj
