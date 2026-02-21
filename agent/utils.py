import ast
import json
import re
import time
from typing import List, Dict, Optional, Any
from openai import OpenAI, RateLimitError, APIConnectionError

        
        
def _parse_json_response(content: str) -> Optional[List[Dict]]:
    try:
        content = content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[-1].split("```")[0].strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}\nContent:\n{content}")
        return None
    except Exception as e:
        print(f"Unexpected error during JSON parsing: {e}")
        return None

def remove_all_imports(code_text: str) -> str:
    if not code_text:
        return ""
    lines = code_text.split('\n')
    filtered = [line for line in lines if not (
        line.strip().startswith("import ") or
        (line.strip().startswith("from ") and not line.strip().startswith("from ."))
    )]
    return '\n'.join(filtered)

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
            print(f"Skipping non-dictionary element: {d}")
    return result

def validate_param_type(param: Any, expected_type: type, param_name: str) -> bool:
    if not isinstance(param, expected_type):
        print(f"Invalid type for {param_name}, expected {expected_type}, got {type(param)}")
        return False
    return True

def exec_safe(code_str: str, gvars: Optional[Dict] = None, lvars: Optional[Dict] = None) -> None:
    if not code_str.strip():
        print("Code to execute is empty")
        return
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        raise SyntaxError(f"Code syntax error: {e}") from e

    banned_ops = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module or ",".join(alias.name for alias in node.names)
            banned_ops.append(f"Import prohibited: {mod}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ('exec', 'eval', 'compile', '__import__'):
                banned_ops.append(f"Dangerous function prohibited: {node.func.id}")

    if banned_ops:
        raise RuntimeError("Unsafe operations detected:\n" + "\n".join(banned_ops))

    # Execute code
    custom_gvars = merge_dicts([gvars or {}, {
        'exec': lambda *a, **k: None,
        'eval': lambda *a, **k: None,
        'compile': lambda *a, **k: None
    }])
    try:
        exec(code_str, custom_gvars, lvars or {})
    except Exception as e:
        print(f"Code execution failed: {e}")
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
    while True:
        try:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stop=stop_tokens,
                temperature=temperature,
                stream=stream
            )
        except (RateLimitError, APIConnectionError) as e:
            time.sleep(10)
        except Exception as e:
            raise


def extract_task_ready(input_text: str) -> str:
    match = re.search(r'<Task Ready>(.*?)</Task Ready>', input_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_task_steps(input_text: str) -> str:
    match = re.search(r'<Task steps>(.*?)</Task steps>', input_text, re.DOTALL)
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
    try:
        processed_str = input_str.replace("\\'", "'")
        result_list = eval(processed_str)
        if isinstance(result_list, list) and all(isinstance(item, dict) for item in result_list):
            return result_list
        else:
            print("⚠️ eval parsing result format exception (not a list of dictionaries), triggering JSON fallback solution")
            raise ValueError("Parsing result is not a list of dictionaries")

    except Exception as e:
        # ========== Fallback Solution: Manually handle quotes to comply with JSON specifications ==========
        print(f"⚠️ eval parsing failed: {str(e)[:100]}, attempting JSON fallback solution")
        try:
            # Step 1: Restore escaped single quotes (camera\'s → camera's)
            json_str = input_str.replace("\\'", "'")
            # Step 2: Escape double quotes within strings (avoid conflict with outer JSON double quotes)
            json_str = json_str.replace('"', '\\"')
            # Step 3: Replace outer single quotes with double quotes (comply with JSON's double quote rule for field names)
            json_str = json_str.replace("'", '"')
            # Step 4: Clean up escaped newlines (\\n → \n)
            json_str = json_str.replace("\\n", "\n")
            
            # Step 5: Parse JSON string into a list
            result_list = json.loads(json_str)
            
            # Final validation
            if isinstance(result_list, list):
                return result_list
            else:
                print("❌ JSON fallback solution parsing result is not a list")
                return []

        except Exception as e2:
            print(f"❌ JSON fallback solution also failed: {str(e2)[:100]}")
            return []



