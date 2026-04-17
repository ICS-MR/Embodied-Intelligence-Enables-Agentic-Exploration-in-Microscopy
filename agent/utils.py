import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI

from adapters.llm_clients import create_chat_completion
from utils.cli_logging import get_cli_logger


logger = get_cli_logger("PLANNER")
SAFE_BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
SAFE_IMPORT_MODULES = {
    "collections",
    "csv",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "statistics",
    "time",
}
FORBIDDEN_NAME_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
FORBIDDEN_ROOT_NAMES = {
    "os",
    "sys",
    "pathlib",
    "subprocess",
    "socket",
    "shutil",
    "requests",
    "httpx",
    "urllib",
    "builtins",
}

        
        
def _parse_json_response(content: str) -> Optional[List[Dict]]:
    try:
        content = content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[-1].split("```")[0].strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.warning("JSON parsing failed: %s\nContent:\n%s", e, content)
        return None
    except Exception as e:
        logger.error("Unexpected error during JSON parsing: %s", e)
        return None

def remove_all_imports(code_text: str) -> str:
    if not code_text:
        return ""
    # Import safety is enforced by the AST sandbox so safe imports can remain available.
    return code_text

def extract_python_code(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^\s*'''\s*python\s*'''\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    match = re.search(r"```python\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text

def merge_dicts(dicts: List[Dict]) -> Dict:
    result = {}
    if not validate_param_type(dicts, list, "dicts"):
        return result
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
        else:
            logger.warning("Skipping non-dictionary element: %s", d)
    return result

def validate_param_type(param: Any, expected_type: type, param_name: str) -> bool:
    if not isinstance(param, expected_type):
        logger.warning("Invalid type for %s, expected %s, got %s", param_name, expected_type, type(param))
        return False
    return True

def _collect_assigned_names(tree: ast.AST) -> Set[str]:
    assigned: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    return assigned


def _collect_local_function_names(tree: ast.AST) -> Set[str]:
    local_function_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("__") or node.name in FORBIDDEN_NAME_CALLS:
            continue
        local_function_names.add(node.name)
    return local_function_names


def _attribute_root_name(node: ast.AST) -> Optional[str]:
    current = node
    while isinstance(current, ast.Attribute):
        if current.attr.startswith("__"):
            return "__forbidden__"
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return None


def _describe_import(node: ast.AST) -> str:
    if isinstance(node, ast.ImportFrom):
        module_name = node.module or ""
        imported_names = ",".join(alias.name for alias in node.names)
        return f"{module_name}:{imported_names}" if module_name else imported_names
    if isinstance(node, ast.Import):
        return ",".join(alias.name for alias in node.names)
    return type(node).__name__


def _is_allowed_import(module_name: str) -> bool:
    return module_name in SAFE_IMPORT_MODULES


def _validate_import_node(node: ast.AST) -> tuple[List[str], Set[str]]:
    errors: List[str] = []
    imported_names: Set[str] = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            module_name = alias.name
            if not _is_allowed_import(module_name):
                errors.append(f"Import prohibited: {module_name}")
                continue
            imported_names.add(alias.asname or module_name.split(".", 1)[0])
        return errors, imported_names

    if isinstance(node, ast.ImportFrom):
        if node.level:
            return [f"Import prohibited: {_describe_import(node)}"], imported_names
        module_name = node.module or ""
        if not _is_allowed_import(module_name):
            return [f"Import prohibited: {_describe_import(node)}"], imported_names
        for alias in node.names:
            if alias.name == "*":
                errors.append(f"Import prohibited: {_describe_import(node)}")
                continue
            imported_names.add(alias.asname or alias.name)
        return errors, imported_names

    return errors, imported_names


def _validate_call_target(
    node: ast.Call,
    *,
    allowed_call_names: Set[str],
    allowed_attribute_roots: Set[str],
    assigned_names: Set[str],
    local_function_names: Set[str],
    imported_names: Set[str],
) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        func_name = node.func.id
        if func_name in FORBIDDEN_NAME_CALLS:
            return f"Dangerous function prohibited: {func_name}"
        if (
            func_name not in SAFE_BUILTIN_CALLS
            and func_name not in allowed_call_names
            and func_name not in local_function_names
            and func_name not in imported_names
        ):
            return f"Call target is not whitelisted: {func_name}"
        return None

    if isinstance(node.func, ast.Attribute):
        root_name = _attribute_root_name(node.func)
        if root_name == "__forbidden__":
            return "Dunder attribute access is prohibited"
        if root_name in FORBIDDEN_ROOT_NAMES:
            return f"Forbidden module or object access: {root_name}"
        if (
            root_name
            and root_name not in allowed_attribute_roots
            and root_name not in assigned_names
            and root_name not in imported_names
        ):
            return f"Attribute call root is not whitelisted: {root_name}"
        return None

    return "Dynamic call targets are prohibited"


def _collect_unsafe_operations(
    tree: ast.AST,
    *,
    allowed_call_names: Set[str],
    allowed_attribute_roots: Set[str],
) -> List[str]:
    banned_ops: List[str] = []
    assigned_names = _collect_assigned_names(tree)
    local_function_names = _collect_local_function_names(tree)
    imported_names: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors, allowed_imported_names = _validate_import_node(node)
            banned_ops.extend(errors)
            imported_names.update(allowed_imported_names)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(node, (ast.AsyncWith, ast.Try, ast.Raise, ast.ClassDef, ast.Lambda, ast.Nonlocal)):
            banned_ops.append(f"Unsupported syntax prohibited: {type(node).__name__}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                banned_ops.append("Dunder attribute access is prohibited")
        elif isinstance(node, ast.Call):
            error = _validate_call_target(
                node,
                allowed_call_names=allowed_call_names,
                allowed_attribute_roots=allowed_attribute_roots,
                assigned_names=assigned_names,
                local_function_names=local_function_names,
                imported_names=imported_names,
            )
            if error:
                banned_ops.append(error)
    return banned_ops


def exec_safe(
    code_str: str,
    gvars: Optional[Dict] = None,
    lvars: Optional[Dict] = None,
    *,
    allowed_call_names: Optional[Set[str]] = None,
    allowed_attribute_roots: Optional[Set[str]] = None,
) -> None:
    if not code_str.strip():
        logger.warning("Code to execute is empty")
        return
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise SyntaxError(f"Code syntax error: {e}") from e

    globals_dict = dict(gvars or {})
    initial_local_keys = set(lvars.keys()) if lvars is not None else set()
    locals_dict = dict(lvars or {})
    all_visible_vars = merge_dicts([globals_dict, locals_dict])
    allowed_call_names = set(allowed_call_names or set()) | {
        name for name, value in all_visible_vars.items()
        if callable(value)
    }
    allowed_attribute_roots = set(allowed_attribute_roots or set()) | set(all_visible_vars.keys())
    banned_ops = _collect_unsafe_operations(
        tree,
        allowed_call_names=allowed_call_names,
        allowed_attribute_roots=allowed_attribute_roots,
    )

    if banned_ops:
        raise RuntimeError("Unsafe operations detected:\n" + "\n".join(banned_ops))

    # Execute code
    custom_gvars = merge_dicts([globals_dict, {
        'exec': lambda *a, **k: None,
        'eval': lambda *a, **k: None,
        'compile': lambda *a, **k: None
    }])
    exec_scope = merge_dicts([custom_gvars, locals_dict])
    try:
        exec(code_str, exec_scope, exec_scope)
        if lvars is not None:
            result_keys = initial_local_keys | {
                name for name in exec_scope.keys()
                if name not in custom_gvars
            }
            lvars.clear()
            for name in result_keys:
                lvars[name] = exec_scope[name]
    except Exception as e:
        logger.error("Code execution failed: %s", e)
        raise

def var_exists(name: str, all_vars: Dict) -> bool:
    if not (validate_param_type(all_vars, dict, "all_vars") and name.strip()):
        return False
    try:
        return name in all_vars or any(
            name in frame for frame in all_vars.get('__dict__', {}).values()
            if isinstance(frame, dict)
        )
    except Exception:
        return False

def call_openai_generic(
    client: OpenAI,
    prompt: str,
    system_prompt: str = "",
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    stop_tokens: list = None,
    stream: bool = False
) -> Any:
    stop_tokens = stop_tokens or []
    return create_chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stop=stop_tokens,
        temperature=temperature,
        stream=stream,
    )


def extract_task_ready(input_text: str) -> str:
    match = re.search(r'<Task Ready>(.*?)</Task Ready>', input_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_task_steps(input_text: str) -> str:
    match = re.search(r'<Task steps>(.*?)</Task steps>', input_text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_planner_state(input_text: str) -> str:
    match = re.search(r'<Planner State>(.*?)</Planner State>', input_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def merge_module_tasks(task_list: List[Dict]) -> List[Dict]:
    if not task_list:
        return []
    merged = []
    current = None
    for task in task_list:
        if not isinstance(task, dict) or not task.get('module') or not task.get('command'):
            continue
        if current is None:
            current = {'subtask_index': 1, 'module': task['module'], 'command': task['command']}
        elif current['module'] == task['module']:
            current['command'] += f"; \n#{task['command']}"
        else:
            merged.append(current)
            current = {'subtask_index': len(merged)+1, 'module': task['module'], 'command': task['command']}
    if current:
        merged.append(current)
    return merged


def convert_to_list(input_str: str) -> list:
    payload = input_str.strip()
    if payload.startswith("```json"):
        payload = payload.split("```json", 1)[1].split("```", 1)[0].strip()
    try:
        result_list = json.loads(payload)
        if isinstance(result_list, list) and all(isinstance(item, dict) for item in result_list):
            return result_list
        logger.error("JSON parsing result is not a list of dictionaries")
        return []
    except json.JSONDecodeError as exc:
        logger.error("Strict JSON parsing failed: %s", str(exc)[:160])
        array_match = re.search(r"\[[\s\S]*\]", payload)
        if not array_match:
            return []

        candidate = array_match.group(0).strip()
        try:
            result_list = ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as literal_exc:
            logger.error("Literal fallback parsing failed: %s", str(literal_exc)[:160])
            return []

        if isinstance(result_list, list) and all(isinstance(item, dict) for item in result_list):
            logger.warning("Recovered checker retry plan via Python-literal fallback parsing.")
            return result_list
        logger.error("Literal fallback parsing result is not a list of dictionaries")
        return []



