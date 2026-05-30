import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name

from adapters.llm_clients import create_chat_completion
from agent.clarifier import Clarify
from agent.utils import (
    _parse_json_response,
    extract_planner_state,
    extract_task_ready,
    extract_task_steps,
    merge_module_tasks,
)
from config.agent_config import cross_encoder_model_path, task_similarity_threshold
from utils.cli_logging import get_cli_logger


logger = get_cli_logger("PLANNER")


@dataclass
class PlannerResult:
    status: str
    question: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    ready: bool = False
    tokens: Optional[Dict[str, int]] = None
    error: Optional[str] = None
    raw_response: str = ""


def explain_planned_execution(client, model_name, lmp_steps: list) -> str:
    """
    Generate a short, natural English preview of what the assistant will do next,
    based on raw lmp_steps, using first-person future tense and ending with a gentle confirmation.
    """
    if not lmp_steps:
        return "No actions planned. Would you like to proceed?"

    steps_text = "\n".join(
        f"{i + 1}. [{step.get('module', 'Unknown')}] {step.get('command', '').strip()}"
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
        response = create_chat_completion(
            client,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "I'll execute the planned steps. Ready to proceed?"


class ExperimentPlanAgent:
    def __init__(
        self,
        name: str,
        cfg: Dict,
        client: OpenAI,
        historymanager=None,
        fix_bug=False,
        clarify_tag=False,
    ):
        self._name = name
        self._cfg = cfg or {}
        self._base_prompt = self._cfg.get("prompt_text", "")
        self._last_planner_raw_response = ""
        self._historyManager = historymanager
        self.code_history = []
        self._stop_tokens = list(self._cfg.get("stop", []))
        self.fix_bug = fix_bug
        self.observation_object = None
        self._client = client
        self.clarify = None
        self._clarify_enabled = bool(clarify_tag)
        self._clarify_method = "clarify"
        if self._clarify_enabled:
            try:
                self.clarify = Clarify(
                    self._client,
                    self._base_prompt,
                    self._cfg.get("engine"),
                    cross_encoder_model_path,
                    task_similarity_threshold,
                    historymanager=self._historyManager,
                )
            except Exception as e:
                raise RuntimeError(f"Clarify initialization failed: {e}") from e

    def set_clarify_method(self, method: str) -> None:
        normalized = str(method or "").strip().lower()
        self._clarify_method = normalized if normalized in {"clarify", "clarify_llm"} else "clarify"

    def clear_history(self):
        self.code_history = []
        self.observation_object = None

    def _make_history_state_snapshot(self, state: Any) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}

        snapshot: Dict[str, Any] = {}
        for key in ("objective", "channel", "exposure", "brightness"):
            if key in state:
                snapshot[key] = state[key]
        return snapshot

    def _build_history_entry(self, command: str, state: Any, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "command": command,
            "state": self._make_history_state_snapshot(state),
            "steps": [step.copy() if isinstance(step, dict) else step for step in steps],
        }

    def remember_planned_task(self, command: str, state: Any, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        entry = self._build_history_entry(command, state, steps)
        self.code_history.append(entry)
        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name=self._name,
                event_type="planning_history_recorded",
                message="Planner stored the confirmed task in session history.",
                payload={
                    "command": command,
                    "state": entry["state"],
                    "steps": entry["steps"],
                },
            )
        return entry

    def _collect_prompt_parts(self, state: Any, context: str = "") -> List[str]:
        prompt_parts: List[str] = []

        if self._cfg.get("maintain_session") and self.code_history:
            max_hist = self._cfg.get("max_history", 10)
            history_items: List[Any] = []
            for item in self.code_history[-max_hist:]:
                if isinstance(item, str):
                    try:
                        history_items.append(json.loads(item))
                    except json.JSONDecodeError:
                        history_items.append(item)
                else:
                    history_items.append(item)
            prompt_parts.append(self._wrap_prompt_section("Historical Tasks", json.dumps(history_items, indent=2)))

        if state is not None:
            try:
                if isinstance(state, dict):
                    state_str = json.dumps(state, indent=2)
                else:
                    state_str = json.dumps(state.__dict__, indent=2) if hasattr(state, "__dict__") else str(state)
                prompt_parts.append(f"Current system state:\n{state_str}")
            except Exception as e:
                logger.warning("Failed to serialize state: %s", e)
                prompt_parts.append(f"Current system state:\n{str(state)}")

        if context.strip():
            prompt_parts.append(context)
        return prompt_parts

    def _serialize_state_text(self, state: Any) -> str:
        try:
            if isinstance(state, dict):
                return json.dumps(state, indent=2)
            if hasattr(state, "__dict__"):
                return json.dumps(state.__dict__, indent=2)
            return str(state)
        except Exception as e:
            logger.warning("Failed to serialize state: %s", e)
            return str(state)

    def _wrap_prompt_section(self, title: str, content: str) -> str:
        normalized = str(content or "").strip()
        if not normalized:
            return ""
        return f"<{title}>\n{normalized}\n</{title}>"

    def _format_query(self, query: str) -> str:
        return f"{self._cfg.get('query_prefix', '')}{query}{self._cfg.get('query_suffix', '')}"

    def _extract_observation_object(self, context: str) -> str:
        normalized = str(context or "").strip()
        prefix = "Observation object:"
        if not normalized.lower().startswith(prefix.lower()):
            return ""
        return normalized[len(prefix):].strip()

    def _extract_imaging_target(self, query: str) -> str:
        match = re.search(r"\bImaging\s+(?:target|object)\s*:\s*([^;\n\r]+)", str(query or ""), re.IGNORECASE)
        return match.group(0).strip() if match else ""

    def _is_internal_planner_context(self, context: str) -> bool:
        normalized = str(context or "").strip()
        return normalized.startswith("The user request below was produced by an upstream resolver")

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        stop_tokens: Optional[List[str]] = None,
        allow_clarify: bool = False,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        del allow_clarify

        response = create_chat_completion(
            self._client,
            model=self._cfg.get("engine", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            seed=self._cfg.get("seed"),
            stop=stop_tokens or [],
            stream=False,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or "", {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    def _merge_usage(
        self,
        *usages: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, int]]:
        merged: Dict[str, int] = {}
        for usage in usages:
            if not usage:
                continue
            for key, value in usage.items():
                merged[key] = merged.get(key, 0) + int(value)
        return merged or None

    def _parse_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        raw_text = (text or "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
        try:
            payload = json.loads(raw_text)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", raw_text)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                return None

    def build_prompt(
        self,
        query: str,
        state: Any,
        context: str = "",
    ) -> Tuple[str, str, str]:
        observation_object = self._extract_observation_object(context) or self._extract_imaging_target(query)
        prompt_parts: List[str] = []
        if self._cfg.get("maintain_session") and self.code_history:
            max_hist = self._cfg.get("max_history", 10)
            history_items: List[Any] = []
            for item in self.code_history[-max_hist:]:
                if isinstance(item, str):
                    try:
                        history_items.append(json.loads(item))
                    except json.JSONDecodeError:
                        history_items.append(item)
                else:
                    history_items.append(item)
            prompt_parts.append(self._wrap_prompt_section("Historical Tasks", json.dumps(history_items, indent=2)))
        prompt_parts.extend(
            [
                self._wrap_prompt_section("Current System State", self._serialize_state_text(state)),
                self._wrap_prompt_section("Observation object", observation_object or "{}"),
            ]
        )
        prompt_parts.append(self._wrap_prompt_section("User Request", self._format_query(query)))
        return self._base_prompt, "\n\n".join(part for part in prompt_parts if part), query

    def _build_resolved_planner_context(self) -> str:
        return (
            "The user request below has already been rewritten into a consolidated workflow specification "
            "by an upstream resolver. Treat the resolved workflow parameters in that request as "
            "authoritative and do not ask again for parameters that are already resolved there unless a "
            "genuine new blocking ambiguity remains."
        )

    def _parse_planner_result(
        self,
        *,
        query: str,
        answer_content: str,
        usage_dict: Optional[Dict[str, int]],
        allow_ask_user: bool = False,
    ) -> PlannerResult:
        planner_state = self._parse_json_object(extract_planner_state(answer_content)) or {}
        state_status = str(planner_state.get("status") or "").strip().lower()
        state_question = str(planner_state.get("question") or "").strip()
        state_reason = str(planner_state.get("reason") or "").strip()

        if not state_status:
            ready_content = extract_task_ready(answer_content)
            if ready_content:
                ready_payload = self._parse_json_object(ready_content) or {}
                ready_status = ready_payload.get("Status", ready_payload.get("status", ""))
                if str(ready_status or "").strip().upper() == "OK":
                    state_status = "final_plan"
            if not state_status and extract_task_steps(answer_content).strip():
                    state_status = "final_plan"
            if not state_status:
                state_status = "ask_user" if state_question else "error"

        if state_status == "ask_user":
            if not allow_ask_user:
                if self._historyManager:
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="planning_disallowed_question",
                        message="Planner attempted ask_user without an active planning template allowing it.",
                        payload={
                            "query": query,
                            "raw_response": answer_content,
                            "usage": usage_dict or {},
                        },
                    )
                return PlannerResult(
                    status="error",
                    ready=False,
                    tokens=usage_dict,
                    error="Planner returned ask_user without an active planning template that allows clarification.",
                    raw_response=answer_content,
                )
            question = state_question or "Please provide the key missing detail needed to continue planning."
            if self._historyManager:
                self._historyManager.record_interaction(
                    agent_name=self._name,
                    event_type="planning_requires_input",
                    message="Planner requested one additional user clarification.",
                    payload={
                        "query": query,
                        "question": question,
                        "reason": state_reason,
                        "usage": usage_dict or {},
                    },
                )
            return PlannerResult(
                status="ask_user",
                question=question,
                ready=False,
                tokens=usage_dict,
                raw_response=answer_content,
            )

        if state_status == "unsupported":
            unsupported_reason = state_reason or "The current system cannot execute this request."
            if self._historyManager:
                self._historyManager.record_interaction(
                    agent_name=self._name,
                    event_type="planning_unsupported",
                    message="Planner reported that the request is unsupported by current system capabilities.",
                    payload={
                        "query": query,
                        "reason": unsupported_reason,
                        "raw_response": answer_content,
                        "usage": usage_dict or {},
                    },
                )
            return PlannerResult(
                status="unsupported",
                question="",
                ready=False,
                tokens=usage_dict,
                error=unsupported_reason,
                raw_response=answer_content,
            )

        if state_status == "final_plan":
            steps_content = extract_task_steps(answer_content)
            if not steps_content.strip():
                missing_steps_error = (
                    "Planner declared final_plan but did not provide a <Task steps> block."
                )
                if self._historyManager:
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="planning_missing_task_steps",
                        message=missing_steps_error,
                        payload={
                            "query": query,
                            "reason": state_reason,
                            "raw_response": answer_content,
                            "usage": usage_dict or {},
                        },
                    )
                return PlannerResult(
                    status="error",
                    ready=False,
                    tokens=usage_dict,
                    error=missing_steps_error,
                    raw_response=answer_content,
                )
            tasks = _parse_json_response(steps_content)
            if tasks:
                tasks = merge_module_tasks(tasks)
                if self._historyManager:
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="planning_result",
                        message="Planner produced executable task steps.",
                        payload={
                            "query": query,
                            "tasks": tasks,
                            "reason": state_reason,
                            "raw_response": answer_content,
                            "usage": usage_dict or {},
                        },
                    )
                return PlannerResult(
                    status="final_plan",
                    steps=tasks,
                    ready=True,
                    tokens=usage_dict,
                    raw_response=answer_content,
                )
            invalid_steps_error = (
                "Planner returned a <Task steps> block, but it could not be parsed as a JSON step list."
            )
            if self._historyManager:
                self._historyManager.record_interaction(
                    agent_name=self._name,
                    event_type="planning_invalid_task_steps",
                    message=invalid_steps_error,
                    payload={
                        "query": query,
                        "reason": state_reason,
                        "raw_response": answer_content,
                        "task_steps_content": steps_content,
                        "usage": usage_dict or {},
                    },
                )
            return PlannerResult(
                status="error",
                ready=False,
                tokens=usage_dict,
                error=invalid_steps_error,
                raw_response=answer_content,
            )

        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name=self._name,
                event_type="planning_not_ready",
                message="Planner did not produce a valid planning state.",
                payload={
                    "query": query,
                    "reason": state_reason,
                    "raw_response": answer_content,
                    "usage": usage_dict or {},
                },
            )
        return PlannerResult(
            status="error",
            ready=False,
            tokens=usage_dict,
            error="Planner did not return a valid state or executable steps.",
            raw_response=answer_content,
        )

    def plan_with_clarify(
        self,
        query: str,
        state: Any,
        context: str = "",
    ) -> PlannerResult:
        if self.clarify is None:
            return PlannerResult(
                status="error",
                ready=False,
                error="Clarify is enabled but was not initialized.",
            )

        base_prompt, prompt, _ = self.build_prompt(
            query,
            state,
            context,
        )
        del base_prompt

        method_name = self._clarify_method if self._clarify_method in {"clarify", "clarify_llm"} else "clarify"
        try:
            decision = self.clarify.generate_planning_decision(
                prompt,
                force_llm_consistency_analysis=(method_name == "clarify_llm"),
            )
        except Exception as exc:
            return PlannerResult(
                status="error",
                ready=False,
                error=f"{method_name} planning failed: {type(exc).__name__}: {exc}",
            )

        answer_content = decision.raw_response
        if decision.status == "final_plan" and not answer_content:
            answer_content = self.clarify.generate_code_solutions(
                prompt,
                force_llm_consistency_analysis=(method_name == "clarify_llm"),
            )
        elif decision.status == "ask_user" and not answer_content:
            answer_content = self.clarify.generate_code_solutions(
                prompt,
                force_llm_consistency_analysis=(method_name == "clarify_llm"),
            )

        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name=self._name,
                event_type=f"{method_name}_planning_response",
                message=f"{method_name} acted as the primary planner for the current request.",
                payload={
                    "method": method_name,
                    "query": query,
                    "context": context,
                    "status": decision.status,
                    "question": decision.question,
                    "reason": decision.reason,
                    "candidate_solutions": decision.candidate_solutions,
                    "consistency_result": decision.consistency_result,
                    "violation_result": decision.violation_result,
                    "raw_response": answer_content,
                    "error": decision.error,
                },
            )

        if decision.status == "error":
            return PlannerResult(
                status="error",
                ready=False,
                error=decision.error or f"{method_name} planning failed.",
                raw_response=answer_content,
            )

        return self._parse_planner_result(
            query=query,
            answer_content=answer_content,
            usage_dict=None,
            allow_ask_user=True,
        )

    def plan(
        self,
        query: str,
        state: Any,
        context: str = "",
        *,
        allow_ask_user: bool = False,
    ) -> PlannerResult:
        base_prompt, prompt, _ = self.build_prompt(
            query,
            state,
            context,
        )
        answer_content, usage_dict = self._chat_completion(
            system_prompt=base_prompt,
            prompt=prompt,
            temperature=self._cfg.get("temperature", 0.7),
            max_tokens=self._cfg.get("max_tokens"),
            stop_tokens=self._stop_tokens,
            allow_clarify=False,
        )

        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name=self._name,
                event_type="planning_llm_response",
                message="Planner generated a candidate task plan response.",
                payload={
                    "query": query,
                    "context": context,
                    "allow_ask_user": allow_ask_user,
                    "system_prompt": base_prompt,
                    "prompt": prompt,
                    "raw_response": answer_content,
                    "usage": usage_dict or {},
                },
            )

        return self._parse_planner_result(
            query=query,
            answer_content=answer_content,
            usage_dict=usage_dict,
            allow_ask_user=allow_ask_user,
        )

    def __call__(self, query: str, state: Any, context: str = "") -> PlannerResult:
        if self._clarify_enabled:
            planner_result = self.plan_with_clarify(query, state, context)
            self._last_planner_raw_response = planner_result.raw_response
            return planner_result

        planner_result = self.plan(query, state, context)
        self._last_planner_raw_response = planner_result.raw_response
        return planner_result

    def _highlight_list(self, items: List) -> str:
        parts = []
        for item in items:
            try:
                if isinstance(item, str):
                    lexer = PythonLexer() if "def " in item else get_lexer_by_name("json")
                else:
                    item = json.dumps(item, indent=2)
                    lexer = get_lexer_by_name("json")
                parts.append(highlight(item, lexer, TerminalFormatter()))
            except Exception:
                parts.append(str(item))
        return "\n".join(parts)

    def _process_observation_object(self, command: str) -> Tuple[Optional[str], str]:
        if self.observation_object:
            return self.observation_object, command

        system_prompt = (
            "You are an accurate instruction parser. Please extract the unique and specific observation object "
            "(such as an object, entity, target, etc.) from the following user instruction.\n"
            "You must strictly return it in the following JSON format, and only return this line without any "
            'explanations, prefixes, suffixes or Markdown:{"object": "Extracted object name"}\n'
            "Example:\n"
            "Instruction: Take a focused photo of the 2D cell section\n"
            '{"object": "2D cell section"}\n\n'
            "Now please process the following instruction:"
        )
        user_prompt = command.strip()

        try:
            raw_content, _ = self._chat_completion(
                system_prompt=system_prompt,
                prompt=user_prompt,
                temperature=0.0,
                max_tokens=128,
                stop_tokens=[],
                allow_clarify=False,
            )
            raw_content = raw_content.strip()

            json_match = re.search(r"\{[^{}]*\}", raw_content, re.DOTALL)
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
        except Exception:
            self.observation_object = None
            return None, command

    def run(self, command: str, state) -> Tuple[bool, Any, Optional[Dict[str, int]]]:
        """Interactively process user instructions."""
        obs_obj, instruction = self._process_observation_object(command)
        extent = ""
        while True:
            full_input = f"{instruction}\n{extent}" if extent else instruction
            result = self.__call__(full_input, state, f"Observation object: {obs_obj}")
            if result.ready and result.steps:
                summary_tasks = explain_planned_execution(
                    self._client,
                    self._cfg.get("engine", "gpt-3.5-turbo"),
                    result.steps,
                )
                logger.info("%s", summary_tasks)
                user_input = input("Supplement instruction (enter 'execute' to start, 'exit' to quit): ").strip()
                logger.info("User confirmation: %s", user_input)
                execute_keywords = ["execute", "run", "start", "launch", "perform", "execute now", "run task"]
                if user_input.lower() in execute_keywords:
                    if self._historyManager:
                        self._historyManager.record_interaction(
                            agent_name=self._name,
                            event_type="planning_confirmed",
                            message="User confirmed the generated plan.",
                            payload={"instruction": command, "tasks": result.steps, "user_input": user_input},
                        )
                    self.remember_planned_task(command, state, result.steps)
                    return True, result.steps, result.tokens
                if user_input.lower() == "exit":
                    if self._historyManager:
                        self._historyManager.record_interaction(
                            agent_name=self._name,
                            event_type="planning_cancelled",
                            message="User cancelled after planner preview.",
                            payload={"instruction": command},
                        )
                    return False, None, None
                if self._historyManager:
                    self._historyManager.record_interaction(
                        agent_name=self._name,
                        event_type="planning_revision",
                        message="User requested plan revision.",
                        payload={"instruction": command, "revision": user_input},
                    )
                extent = user_input
                continue

            if result.status == "ask_user":
                user_input = input(f"{result.question} ('exit' to quit): ").strip()
                if user_input.lower() == "exit":
                    return False, None, None
                instruction = f"{instruction}\n{user_input}"
                extent = ""
                continue

            user_input = input("Task not supported, please re-enter ('exit' to quit): ").strip()
            if user_input.lower() == "exit":
                return False, None, None
            instruction = user_input
            extent = ""
