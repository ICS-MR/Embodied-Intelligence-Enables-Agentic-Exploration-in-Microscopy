import json

from typing import List, Dict, Tuple, Optional, Any

from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name
from openai import OpenAI
import re
from agent.utils import (
    _parse_json_response,
    call_openai_generic,
    merge_module_tasks,
    extract_task_steps,
    extract_task_ready
)

from config.agent_config import cross_encoder_model_path, task_similarity_threshold
from agent.clarify import Clarify

def explain_planned_execution(client, lmp_steps: list) -> str:
    """
    Generate a short, natural English preview of what the assistant will do next,
    based on raw lmp_steps, using first-person future tense and ending with a gentle confirmation.
    """
    if not lmp_steps:
        return "No actions planned. Would you like to proceed?"

    # Preserve raw module/command format for LLM understanding
    steps_text = "\n".join(
        f"{i+1}. [{step.get('module', 'Unknown')}] {step.get('command', '').strip()}"
        for i, step in enumerate(lmp_steps)
    )

    prompt = f"""You are a helpful AI lab assistant. Before executing a task, briefly explain in one short, natural English sentence what you will do next.

    Planned steps:
    {steps_text}

    Requirements:
    - Use first-person future tense (e.g., "I will...")
    - Translate technical steps into plain lab actions (e.g., "set_objective 20x" → "switch to the 20x objective")
    - Do NOT mention module names, brackets, or internal command syntax
    - End with a gentle confirmation question like "OK?", "Shall I start?", or "Ready to proceed?"

    Output ONLY the sentence. No extra text, explanations, or formatting."""

    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "I'll execute the planned steps. Ready to proceed?"
    
class TaskManager:
    def __init__(
        self,
        name: str,
        cfg: Dict,
        client: OpenAI,
        historymanager=None,
        fix_bug=False,
        clarify_tag=False
    ):
        self._name = name
        self._cfg = cfg or {}
        self._base_prompt = self._cfg.get('prompt_text', "")
        self._historyManager = historymanager
        self.code_history = []
        self._stop_tokens = list(self._cfg.get('stop', []))
        self.fix_bug = fix_bug
        self.observation_object = None
        self._client = client
        self.clarify = None
        if clarify_tag:
            try:
                self.clarify = Clarify(self._client, self._base_prompt, self._cfg.get('engine'), cross_encoder_model_path, task_similarity_threshold)
            except Exception as e:
                print(f"Failed to initialize clarifier: {e}")

    def clear_history(self):
        self.code_history = []
        self.observation_object = None

    def build_prompt(self, query: str, context: str = '') -> Tuple[str, str, str]:
        prompt_parts = []
        if self._cfg.get('maintain_session') and self.code_history:
            max_hist = self._cfg.get('max_history', 10)
            prompt_parts.append(f"Historical tasks:\n{json.dumps(self.code_history[-max_hist:], indent=2)}")
        if context.strip():
            prompt_parts.append(context)
        query = f"{self._cfg.get('query_prefix', '')}{query}{self._cfg.get('query_suffix', '')}"
        prompt_parts.append(query)
        return self._base_prompt, "\n".join(prompt_parts), query

    def __call__(self, query: str, context: str = '') -> Tuple[bool, str, Any, Optional[Dict[str, int]]]:
        base_prompt, prompt, use_query = self.build_prompt(query, context)
        usage_dict = None

        if self.clarify:
            answer_content = self.clarify.generate_code_solutions(prompt)
        else:
            response = call_openai_generic(
                client=self._client,
                prompt=prompt,
                system_prompt=base_prompt,
                model=self._cfg.get('engine', 'gpt-3.5-turbo'),
                temperature=self._cfg.get('temperature', 0.7),
                stop_tokens=self._stop_tokens,
                stream=False
            )
            answer_content = response.choices[0].message.content or ""
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }


        try:
            ready_content = extract_task_ready(answer_content)
            if not ready_content:
                return False, "", None, usage_dict
            data = json.loads(ready_content)
            if data.get("Status") != 'OK':
                return False, "", None, usage_dict
            steps_content = extract_task_steps(answer_content)
            tasks = _parse_json_response(steps_content)
            if tasks:
                tasks = merge_module_tasks(tasks)
                return True, "", tasks, usage_dict
        except Exception as e:
            print(f"Task parsing failed: {e}")
        return False, "", None, usage_dict

    def _highlight_list(self, items: List) -> str:
        parts = []
        for item in items:
            try:
                if isinstance(item, str):
                    lexer = PythonLexer() if 'def ' in item else get_lexer_by_name('json')
                else:
                    item = json.dumps(item, indent=2)
                    lexer = get_lexer_by_name('json')
                parts.append(highlight(item, lexer, TerminalFormatter()))
            except Exception as e:
                parts.append(str(item))
        return "\n".join(parts)

    def _process_observation_object(self, command: str) -> Tuple[Optional[str], str]:
        if self.observation_object:
            return self.observation_object, command

        system_prompt = (
            "You are an accurate instruction parser. Please extract the unique and specific observation object (such as an object, entity, target, etc.) from the following user instruction.\n"
            "You must strictly return it in the following JSON format, and only return this line without any explanations, prefixes, suffixes or Markdown:"
            '{"object": "Extracted object name"}\n'
            "Example:\n"
            "Instruction: Take a focused photo of the 2D cell section\n"
            '{"object": "2D cell section"}\n\n'
            "Now please process the following instruction:"
        )
        user_prompt = command.strip()

        try:
            response = call_openai_generic(
                client=self._client,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self._cfg.get('engine', 'gpt-3.5-turbo'),
                temperature=0.0,
                stream=False
            )
            raw_content = response.choices[0].message.content or ""
            raw_content = raw_content.strip()

            json_match = re.search(r'\{[^{}]*\}', raw_content, re.DOTALL)
            if not json_match:
                obj_match = re.search(r'"object"\s*:\s*"([^"]*)"', raw_content)
                if obj_match:
                    obj_name = obj_match.group(1).strip()
                    self.observation_object = obj_name if obj_name else None
                else:
                    self.observation_object = None
            else:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    self.observation_object = parsed.get("object", "").strip() or None
                except json.JSONDecodeError:
                    try:
                        fixed_json = json_str.replace("'", '"')
                        parsed = json.loads(fixed_json)
                        self.observation_object = parsed.get("object", "").strip() or None
                    except Exception:
                        self.observation_object = None

            return self.observation_object, command

        except Exception as e:
            self.observation_object = None
            return None, command
    
    def process_instruction(self, command: str) -> Tuple[bool, Any, Optional[Dict[str, int]]]:
        """Interactively process user instructions"""
        obs_obj, instruction = self._process_observation_object(command)
        extent = ""
        while True:
            full_input = f"{instruction}\n{extent}" if extent else instruction
            is_exec, _, tasks, tokens = self.__call__(full_input, f"Observation object: {obs_obj}")
            if is_exec:
                summary_tasks = explain_planned_execution(self._client, tasks)
                print(f'[Robot]{summary_tasks}')
                user_input = input("Supplement instruction (enter 'execute' to start, 'exit' to quit): ").strip()
                print(f'[User]{user_input}')
                execute_keywords = ['execute', 'run', 'start', 'launch', 'perform', 'execute now', 'run task']
                if user_input.lower() in execute_keywords:
                    if self._historyManager:
                        self._historyManager.append(f"Instruction: {command}\nTasks: {tasks}", self._name)
                    self.code_history.append(json.dumps(tasks))
                    return True, tasks, tokens
                elif user_input.lower() == 'exit':
                    return False, None, None
                else:
                    extent = user_input
            else:
                user_input = input("Task not supported, please re-enter ('exit' to quit): ").strip()
                if user_input.lower() == 'exit':
                    return False, None, None
                instruction = user_input
                extent = ""
