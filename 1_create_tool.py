import ast
import os
from typing import Set, Optional, Dict, Any
from openai import OpenAI

# === 常量定义 ===
SAY_METHOD_TEMPLATE = '''def say(message: str):
    """
    Outputs a log message with `[ACTION]`, `[INFO]`, or `[ERROR]` prefix. Ensures consistent logging format.
    """
    print(f'robot says: {message}')'''

# === LLM 提示词模板（集中管理）===
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
    "usage": """You are a Python expert. Based on the following method definitions, generate a concise and practical usage example.

Requirements:
- Use English comments.
- Include necessary class instantiation and method calls.
- Cover the main functionalities.
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
- Do NOT repeat method names; abstract their purpose
- Ignore placeholder texts like '自动生成 docstring 失败'
- Do NOT include code, examples, or prefixes

Methods:
{methods}

Capabilities summary:
""",
    "task_example": """You are a task planning expert. Generate a usage example in the specified format based on the tool's capabilities.

Tool name: {class_name}
Capabilities:
{capabilities}

Requirements:
- Generate a realistic, specific user instruction (# Example input)
- Output must strictly follow:
<Task Ready>
{{"Status": "OK"}}
</Task Ready>
<Task steps>
[
    {{
        "subtask_index": 1,
        "module": "{class_name}",
        "command": "Clear, specific action in English, starting with a verb"
    }}
]
</Task steps>
- The 'command' must be based ONLY on the given capabilities
- Both input and command must be in English
- ONLY output # Example input and # Example output sections, no extra text

Generate example:
"""
}


class LLMAgent:
    """LLM Agent compatible with OpenAI API (e.g., OpenAI, Qwen via DashScope)."""

    def __init__(
            self,
            api_key: str,
            base_url: str = "https://api.openai.com/v1",
            model: str = "gpt-3.5-turbo",
            temperature: float = 0.7,
            timeout: int = 30
    ):
        print(f"[INFO] Initializing LLM Agent with model: {model}")
        self.client = OpenAI(api_key=api_key, base_url=base_url.strip())
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": PROMPTS["system_role"]},
            {"role": "user", "content": prompt}
        ]
        try:
            print(f"[INFO] Calling LLM API with prompt (first 50 chars): {prompt[:50]}...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                timeout=self.timeout,
            )
            result = response.choices[0].message.content.strip()
            print(f"[SUCCESS] LLM generation completed (result length: {len(result)} chars)")
            return result
        except Exception as e:
            error_msg = f"[ERROR] LLM API call failed: {str(e)}"
            print(error_msg)
            return f"❌ LLM Error: {e}"


def safe_llm_call(llm: LLMAgent, prompt: str, fallback: str = "LLM generation failed.") -> str:
    """Safely call LLM and return fallback on error or empty result."""
    try:
        result = llm.generate(prompt)
        if not result or "❌" in result or not result.strip():
            print(f"[WARNING] LLM returned empty/invalid result, using fallback: {fallback[:30]}...")
            return fallback
        return result
    except Exception as e:
        print(f"[ERROR] Safe LLM call failed: {str(e)}, using fallback")
        return fallback


def get_all_class_names(folder_path: str) -> Set[str]:
    """Extract all class names from Python files in a folder."""
    print(f"[INFO] Scanning folder '{folder_path}' for Python classes...")
    class_names: Set[str] = set()
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read(), filename=file_path)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_names.add(node.name)
                            print(f"[INFO] Found class: {node.name} in {file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to parse {file_path}: {e}")
    print(f"[INFO] Total classes found: {len(class_names)}")
    return class_names


def parse_function_signature(func_node: ast.FunctionDef) -> str:
    """Convert an AST FunctionDef node into a clean function signature string."""
    args = func_node.args
    new_args = []
    start_idx = 1 if args.args and args.args[0].arg == 'self' else 0
    pos_args = args.args[start_idx:]
    num_defaults = len(args.defaults)

    for i, arg in enumerate(pos_args):
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                arg_str += ": <unknown>"
        if num_defaults > 0 and i >= len(pos_args) - num_defaults:
            default_node = args.defaults[i - (len(pos_args) - num_defaults)]
            try:
                default_val = ast.unparse(default_node)
                arg_str += f" = {default_val}"
            except Exception:
                arg_str += " = <default>"
        new_args.append(arg_str)

    if args.vararg:
        vararg_str = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            try:
                vararg_str += f": {ast.unparse(args.vararg.annotation)}"
            except Exception:
                pass
        new_args.append(vararg_str)

    if args.kwonlyargs:
        if not args.vararg:
            new_args.append("*")
        for kwarg in args.kwonlyargs:
            kw_str = kwarg.arg
            if kwarg.annotation:
                try:
                    kw_str += f": {ast.unparse(kwarg.annotation)}"
                except Exception:
                    kw_str += ": <unknown>"
            kw_idx = args.kwonlyargs.index(kwarg)
            if (args.kw_defaults and
                    kw_idx < len(args.kw_defaults) and
                    args.kw_defaults[kw_idx] is not None):
                try:
                    default_val = ast.unparse(args.kw_defaults[kw_idx])
                    kw_str += f" = {default_val}"
                except Exception:
                    kw_str += " = <default>"
            new_args.append(kw_str)

    if args.kwarg:
        kwarg_str = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            try:
                kwarg_str += f": {ast.unparse(args.kwarg.annotation)}"
            except Exception:
                pass
        new_args.append(kwarg_str)

    signature = "(" + ", ".join(new_args) + ")"
    return_annotation = ""
    if func_node.returns:
        try:
            return_annotation = f" -> {ast.unparse(func_node.returns)}"
        except Exception:
            return_annotation = " -> <unknown>"
    return f"def {func_node.name}{signature}{return_annotation}:"


def extract_or_generate_docstring(
        func_node: ast.FunctionDef,
        signature: str,
        llm: Optional[LLMAgent] = None,
        auto_generate: bool = False
) -> str:
    """Get existing docstring or generate one via LLM."""
    existing = ast.get_docstring(func_node)
    if existing and existing.strip():
        return existing

    if auto_generate and llm:
        print(f"[INFO] Generating docstring for function: {func_node.name}")
        prompt = PROMPTS["docstring"].format(signature=signature)
        return safe_llm_call(
            llm, prompt, fallback="Auto-generated docstring failed."
        )
    elif auto_generate:
        return "LLM client not provided; cannot generate docstring."
    else:
        return ""


def format_method_with_docstring(def_line: str, docstring: str) -> str:
    """Format method definition with docstring block."""
    if not docstring.strip():
        return f"{def_line}\n    pass  # No docstring"

    lines = docstring.splitlines()
    if len(lines) == 1:
        doc_block = f'    """{lines[0]}"""'
    else:
        doc_block = '    """\n' + '\n'.join(f"    {line}" for line in lines) + '\n    """'
    return f"{def_line}\n{doc_block}"


def get_class_public_methods(
        folder_path: str,
        class_name: str,
        llm: Optional[LLMAgent] = None,
        auto_generate_docstring: bool = False,
        inject_say: bool = True
) -> str:
    """Extract all public methods of a given class with optional docstring generation."""
    print(f"[INFO] Extracting public methods for class: {class_name}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder: {folder_path}")

    result_parts = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.py'):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source, filename=file_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                                signature = parse_function_signature(item)
                                docstring = extract_or_generate_docstring(
                                    item, signature, llm, auto_generate_docstring
                                )
                                method_str = format_method_with_docstring(signature, docstring)
                                result_parts.append(method_str)
                                print(f"[INFO] Extracted method: {item.name} from {file_path}")
            except Exception as e:
                print(f"[ERROR] Failed to parse {file_path}: {e}")
                continue

    methods_str = "\n\n".join(result_parts)
    if inject_say and "def say(" not in methods_str:
        print(f"[INFO] Injecting 'say' method into class {class_name}")
        methods_str += "\n\n" + SAY_METHOD_TEMPLATE
    
    if not methods_str.strip():
        print(f"[WARNING] No public methods found for class {class_name}")
    else:
        print(f"[INFO] Extracted {len(result_parts)} public methods for class {class_name}")
    return methods_str


def create_usage(public_methods: str, llm: LLMAgent) -> str:
    print("[INFO] Generating usage example for the class methods")
    prompt = PROMPTS["usage"].format(methods=public_methods)
    return safe_llm_call(llm, prompt)


def generate_class_usage(
        public_methods: str,
        llm: LLMAgent,
        usage_content: Optional[str] = None
) -> str:
    return usage_content if usage_content else create_usage(public_methods, llm)


def summarize_capabilities(public_methods: str, llm: LLMAgent) -> str:
    print("[INFO] Generating capabilities summary for the class")
    prompt = PROMPTS["summary"].format(methods=public_methods)
    raw = safe_llm_call(llm, prompt)
    # Clean system/role prefix lines
    lines = [
        line for line in raw.splitlines()
        if not line.strip().startswith(("Capabilities summary", "Summary", "：", ":", "```", "Note", "Requirement"))
    ]
    summary = "\n".join(lines).strip()
    print(f"[INFO] Capabilities summary generated: {summary[:100]}...")
    return summary


def generate_task_example_from_capabilities(
        class_name: str,
        capability_summary: str,
        llm: LLMAgent
) -> str:
    print(f"[INFO] Generating task example for class: {class_name}")
    prompt = PROMPTS["task_example"].format(
        class_name=class_name,
        capabilities=capability_summary
    )
    example = safe_llm_call(llm, prompt)
    print(f"[INFO] Task example generated (length: {len(example)} chars)")
    return example


def generate_tool_usage_prompt(
        folder: str,
        target_class: str,
        llm: LLMAgent,
        output_file: Optional[str] = None,
        usage_example: Optional[str] = None,
        public_methods: Optional[str] = None,
) -> str:
    print(f"[INFO] Generating tool usage prompt for class: {target_class}")
    if output_file is None:
        output_file = f"{target_class}_usage.py"

    if public_methods is None:
        public_methods = get_class_public_methods(folder, target_class, llm, auto_generate_docstring=True)

    if usage_example is None:
        usage_example = generate_class_usage(public_methods, llm)

    header = f'''import cv2 as cv
import numpy as np
# Prohibit importing other Python libraries.

# Role and Objectives
Role: A professional assistant specialized in generating Python code for controlling {target_class} systems based on user instructions.
Core Objective: Ensure the generated code is secure and complies with hardware constraints.

# Behavioral Constraints
- Language use: English
- Determine parameters based on Behavioral Constraints and Current environment
## Hardware security control
- All motion and imaging commands must include parameter verification.
## Context-Aware
- Fully utilize user-provided file information to avoid assuming non-existent files/parameters
## Decision-Making Mechanism
- No assumptions are allowed.
- Prioritize using the provided API functions to complete tasks. Carefully read the API function definitions before answering to ensure that the tasks can be completed and comply with the function definitions.
- Saving mechanism: Use the provided functions to read and save files.
- Execute each instruction sequentially and completely without skipping, merging, or reordering.
'''

    methods_section = "# API function\n" + public_methods + "\n\n"
    usage_section = "# Usage Example\n" + usage_example.strip()
    full_content = f"prompt_{target_class} = '''{header}{methods_section}{usage_section}'''"

    abs_output_file = os.path.abspath(output_file)
    with open(abs_output_file, 'w', encoding='utf-8') as f:
        f.write(full_content)

    print(f"✅ Prompt file generated successfully: {abs_output_file}")
    return abs_output_file


def update_task_manager(
        target_class: str,
        capability_summary: str,
        task_example: str,
        task_manager_file: str = "task_manager_full_1.py"
) -> None:
    try:
        print(f"[INFO] Updating task manager file: {task_manager_file}")
        with open(task_manager_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        tool_desc_marker = '###Tool Description\n'
        tool_usage_marker = '###Tool usage\n'

        output_lines = []
        for line in lines:
            output_lines.append(line)
            if line.strip() == tool_desc_marker.strip():
                output_lines.append(f"{target_class}\n{capability_summary}\n")
            if line.strip() == tool_usage_marker.strip():
                output_lines.append(f"{task_example}\n")

        with open(task_manager_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
        
        print(f"✅ Task manager file updated successfully: {task_manager_file}")

    except Exception as e:
        print(f"⚠️ Failed to update {task_manager_file}: {e}")


def update_task_manager_with_summary_and_example(
        target_class: str,
        public_methods: str,
        llm: LLMAgent,
        capability_summary: Optional[str] = None,
        task_example: Optional[str] = None,
        task_manager_file: str = "task_manager_full_1.py"
) -> None:
    if capability_summary is None:
        capability_summary = summarize_capabilities(public_methods, llm)

    if task_example is None:
        task_example = generate_task_example_from_capabilities(target_class, capability_summary, llm)

    update_task_manager(target_class, capability_summary, task_example, task_manager_file)


# ================== 主程序 ==================
if __name__ == "__main__":
    print("="*50)
    print("[START] Starting tool usage prompt generation process")
    print("="*50)
    
    try:
        from config.agent_config import openai_api_key, base_url, model_name
    except ImportError:
        print("[ERROR] Could not import config from config.agent_config - please check the file exists")
        raise

    api_key = openai_api_key
    if not api_key:
        raise EnvironmentError("Please set environment variable: OPENAI_API_KEY")

    llm = LLMAgent(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.2,
        timeout=15
    )

    folder = "tool"
    target_class = "Frap"
    task_manager_file = 'prompts/task_manager_full.py'

    try:
        class_names = get_all_class_names(folder)
        if target_class not in class_names:
            raise FileNotFoundError(f"Class '{target_class}' not found in folder '{folder}'")

        public_methods = get_class_public_methods(
            folder=folder,
            class_name=target_class,
            llm=llm,
            auto_generate_docstring=True,
            inject_say=True
        )

        if not public_methods.strip():
            print(f"❌ No public methods found for class '{target_class}'")
            exit(1)

        prompt_file_path = generate_tool_usage_prompt(
            folder=folder,
            target_class=target_class,
            llm=llm,
            public_methods=public_methods
        )

        update_task_manager_with_summary_and_example(
            target_class=target_class,
            public_methods=public_methods,
            llm=llm,
            task_manager_file = task_manager_file
        )
        
        print("\n" + "="*50)
        print("[COMPLETE] All operations completed successfully!")
        print(f"- Prompt file: {prompt_file_path}")
        print(f"- Task manager file: {task_manager_file}")
        print("="*50)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Program failed: {str(e)}")
        exit(1)
