from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from adapters.llm_clients import create_chat_completion
from agent.utils import _parse_json_response, extract_task_steps, merge_module_tasks

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    SentenceTransformer = None
    _SENTENCE_TRANSFORMER_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMER_IMPORT_ERROR = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:
    cosine_similarity = None
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


@dataclass
class ClarifyPlanningDecision:
    status: str
    question: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""
    reason: str = ""
    error: str = ""


class Clarify:
    def __init__(self, client: OpenAI, base_prompt: str, model_name: str, semantic_model_dir, threshold):
        self._base_prompt = base_prompt
        self._model_name = model_name
        self.threshold = threshold
        self.PROMPT = "Represent this sentence for semantic similarity comparison: "
        self._client = client
        self._semantic_model = self._load_semantic_model(semantic_model_dir)

        self._ZERO_WIDTH_CHARS = [
            "\u200B",
            "\u200C",
            "\u200D",
            "\u2060",
            "\uFEFF",
        ]

    def _load_semantic_model(self, semantic_model_dir: str):
        missing: List[str] = []
        if SentenceTransformer is None:
            detail = type(_SENTENCE_TRANSFORMER_IMPORT_ERROR).__name__ if _SENTENCE_TRANSFORMER_IMPORT_ERROR else "unknown"
            missing.append(f"sentence-transformers ({detail})")
        if cosine_similarity is None:
            detail = type(_SKLEARN_IMPORT_ERROR).__name__ if _SKLEARN_IMPORT_ERROR else "unknown"
            missing.append(f"scikit-learn ({detail})")
        if missing:
            raise RuntimeError(
                "Clarify initialization failed because required dependencies are unavailable: "
                + ", ".join(missing)
            )

        model_path = Path(str(semantic_model_dir)).expanduser()
        if not model_path.exists():
            raise RuntimeError(
                f"Clarify initialization failed because semantic model path does not exist: {model_path}"
            )

        try:
            return SentenceTransformer(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Clarify initialization failed while loading semantic model '{model_path}': {type(exc).__name__}: {exc}"
            ) from exc

    def _query_llm(
        self,
        prompt: str,
        _temperature: float,
        num_outputs: int = 1,
        base_prompt: str = "You are a helpful coding assistant.",
        use_perturbation: bool = True,
    ) -> List[str]:
        results: List[str] = []
        for call_index in range(num_outputs):
            last_error: Optional[Exception] = None
            for _attempt in range(3):
                try:
                    if use_perturbation:
                        temperature = 1.5
                        top_p = 0.75
                        seed_value = 4242 + call_index
                        zw_suffix = "".join(
                            self._ZERO_WIDTH_CHARS[i % len(self._ZERO_WIDTH_CHARS)]
                            for i in range(5)
                        )
                        perturbed_prompt = prompt + zw_suffix + (" " * 3) + ("\n" * 2)
                    else:
                        temperature = _temperature
                        top_p = 1.0
                        seed_value = 42 + call_index if _temperature > 0 else 42
                        perturbed_prompt = prompt

                    response = create_chat_completion(
                        self._client,
                        model=self._model_name,
                        messages=[
                            {"role": "system", "content": base_prompt},
                            {"role": "user", "content": perturbed_prompt},
                        ],
                        extra_body={
                            "enable_thinking": False,
                            "thinking": {"type": "disabled"},
                            "return_usage": True,
                        },
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed_value,
                    )
                    results.append(response.choices[0].message.content or "")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise RuntimeError(
                    f"Clarify planning query failed after retries: {type(last_error).__name__}: {last_error}"
                ) from last_error
        return results

    def _compare_commands(self, task1, task2) -> float:
        if len(task1) != len(task2):
            return 0.0

        modules1 = [item.get("module", "") for item in task1]
        modules2 = [item.get("module", "") for item in task2]
        if modules1 != modules2:
            return 0.0

        total_score = 0.0
        for item1, item2 in zip(task1, task2):
            cmd1 = item1.get("command", "").strip() or "[Empty command]"
            cmd2 = item2.get("command", "").strip() or "[Empty command]"
            embeddings = self._semantic_model.encode(
                [self.PROMPT + cmd1, self.PROMPT + cmd2],
                normalize_embeddings=True,
            )
            raw_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            total_score += raw_sim

        return total_score / len(task1) if task1 else 0.0

    def _extract_merged_tasks(self, task_text: str) -> List[Dict[str, Any]]:
        content = extract_task_steps(task_text)
        task_payload = _parse_json_response(content)
        return merge_module_tasks(task_payload) if task_payload else []

    def _run_semantic_model(self, task_lists: List[str]) -> bool:
        if len(task_lists) < 2:
            return True

        tasks = [self._extract_merged_tasks(task) for task in task_lists]
        if len(tasks) < 2 or any(not task for task in tasks):
            return False

        scores = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                scores.append(self._compare_commands(tasks[i], tasks[j]))

        min_score = min(scores) if scores else 1.0
        inconsistency_score = 1 - min_score
        return inconsistency_score <= self.threshold

    def _analyze_semantic_consistency(self, current_query: str, solutions: List[str]) -> Dict[str, Any]:
        if self._run_semantic_model(solutions):
            return {
                "consistent": True,
                "summary": "All solutions are highly consistent in task understanding.",
                "differences": [],
                "clarification_question": "",
            }

        analyze_prompt = f"""
Please analyze whether the following {len(solutions)} solutions are consistent in terms of user intent and task understanding.

Focus on:
- Are different assumptions being made about unspecified user requirements?
- Are there fundamental disagreements?

Requirements:
- If there are discrepancies in understanding, generate a concise, natural, user-oriented clarification question based on these discrepancies to help the user clarify ambiguous needs.
- The clarification question should focus on the one or two most critical ambiguous points and avoid being verbose.

Output strictly in JSON format with the following fields:
{{
    "consistent": true/false,
    "summary": "The core intent commonly expressed by all solutions (one sentence)",
    "differences": ["Understanding difference 1 caused by ambiguous user requirements", ...],
    "clarification_question": "[Specific question]?"
}}

User's original task: {current_query}

Proposed solutions:
""" + "\n\n".join(f"Solution {i + 1}:\n{s}" for i, s in enumerate(solutions)) + f"\n{'-' * 50}"

        response = self._query_llm(
            prompt=analyze_prompt,
            num_outputs=1,
            base_prompt="",
            _temperature=0,
            use_perturbation=False,
        )
        return self._parse_analysis_response(response[0] if response else "", "semantic_consistency")

    def _analyze_violations(self, current_query: str, solutions: List[str]) -> Dict[str, Any]:
        serialized_solutions: List[str] = []
        for solution in solutions:
            task = self._extract_merged_tasks(solution)
            if task:
                serialized_solutions.append(str(task))

        rules_context = """
Microscope Operation and System Module Notes:
    3D samples must have Z-stack scanning parameters set (e.g., organoids, cell clusters).
    There is no independent hardware-based autofocus; focusing must be implemented via software based on image feedback.
    If imaging parameters are already determined before a focusing operation, re-adjustment is unnecessary; exposure, halogen lamp brightness, and focusing can be executed independently and do not need to be performed consecutively or as a bundled operation.
    Parameter settings:
        Fluorescence: filter must match the channel, the halogen lamp (brightfield exclusive light source) brightness=0 (turn off completely), high exposure;
        Brightfield: filter=brightfield, low exposure, auto-brightness for the halogen lamp enabled.
    Focusing strategy:
        Switching fluorescence channels does not require refocusing;
        When switching between brightfield and fluorescence modes, refocusing must be performed.
        After changing objectives, repositioning, halogen lamp brightness adjustment, and refocusing are required.
"""

        analyze_prompt = f"""You are a microscope operation compliance analysis assistant.

Please review the high-level plan below for ambiguity, omissions, or non-compliant operations according to the following notes:
{rules_context}

Requirements:
Focus on the following:
- Whether the plan can fulfill the task requirements.
- If historical information exists, incorporate it. Any changes to objective magnification or fluorescence must be explicitly stated, specifying the exact fluorescence channel/color or objective magnification (e.g., 4x, 10x, 20x, 40x, 60x).
- If all proposed plans comply with the rules, return has_violation=false and an empty string for clarification_question.
- Analyze whether the parameter settings in the plan make unwarranted assumptions about the user's original request.
- Explicitly distinguish between the halogen lamp (brightfield exclusive light source) and the fluorescence excitation light source; confirm that any "brightness" parameter mentioned in the plan refers exclusively to the halogen lamp brightness (no ambiguity allowed).
- Generate a concise, natural, user-oriented clarification question.

Important: Your response must be a pure JSON object. Do not include any additional content, explanations, Markdown, line breaks, or extra text. Output only the following JSON format:
{{"has_violation": true/false, "clarification_question": "Your question"}}
User's original task: {current_query}
User-provided plans:
{'-' * 50}
""" + "\n\n".join(f"Plan {i + 1}:\n{s}" for i, s in enumerate(serialized_solutions)) + f"\n{'-' * 50}"

        response = self._query_llm(
            prompt=analyze_prompt,
            num_outputs=1,
            base_prompt="",
            _temperature=0,
            use_perturbation=False,
        )
        return self._parse_violation_response(response[0] if response else "")

    def _parse_analysis_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        try:
            result = _parse_json_response(response_text)
            required_fields = ["consistent", "summary", "differences", "clarification_question"]
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")
            if not isinstance(result["consistent"], bool):
                raise TypeError("consistent must be bool")
            if not isinstance(result["summary"], str):
                raise TypeError("summary must be str")
            if not isinstance(result["differences"], list):
                raise TypeError("differences must be list")
            if not isinstance(result["clarification_question"], str):
                raise TypeError("clarification_question must be str")
            return result
        except Exception:
            return self._get_default_analysis_response(analysis_type)

    def _parse_violation_response(self, response_text: str) -> Dict[str, Any]:
        try:
            result = json.loads(response_text.strip())
        except Exception:
            result = None
        if not isinstance(result, dict):
            return {
                "has_violation": True,
                "clarification_question": (
                    "To ensure the feasibility of the plan, please clarify: "
                    "Do you wish to perform any operations beyond the functionality of the standard microscope module? "
                    "For example, manual focusing, custom Z-scanning, etc.?"
                ),
            }
        has_violation = bool(result.get("has_violation", False))
        clarification_question = str(result.get("clarification_question", "")).strip() if has_violation else ""
        return {
            "has_violation": has_violation,
            "clarification_question": clarification_question,
        }

    def _get_default_analysis_response(self, analysis_type: str) -> Dict[str, Any]:
        if analysis_type == "semantic_consistency":
            return {
                "consistent": False,
                "summary": "Failed to parse consistency result",
                "differences": ["LLM returned invalid format or missing required fields"],
                "clarification_question": (
                    "To better assist you, please provide detailed information about your experimental requirements, "
                    "such as: sample type (2D/3D), fluorescence channels used, objective magnification, etc."
                ),
            }
        return {}

    def _fallback_consistency_question(self, consistency_result: Dict[str, Any]) -> str:
        question = str(consistency_result.get("clarification_question") or "").strip()
        if question:
            return question
        differences = consistency_result.get("differences") or []
        base_diff = differences[0] if differences else "Please further explain your experimental setup"
        return f"To ensure we accurately understand your request, please clarify: {base_diff}?"

    def _render_ask_user_response(self, question: str, reason: str) -> str:
        payload = {
            "status": "ask_user",
            "question": question,
            "selected_skills": [],
            "reason": reason,
        }
        return "<Planner State>\n" + json.dumps(payload, ensure_ascii=False) + "\n</Planner State>"

    def _render_final_plan_response(self, steps: List[Dict[str, Any]], reason: str) -> str:
        planner_state = {
            "status": "final_plan",
            "question": "",
            "selected_skills": [],
            "reason": reason,
        }
        return (
            "<Planner State>\n"
            + json.dumps(planner_state, ensure_ascii=False)
            + "\n</Planner State>\n"
            + "<Task Ready>\n"
            + json.dumps({"Status": "OK"}, ensure_ascii=False)
            + "\n</Task Ready>\n"
            + "<Task steps>\n"
            + json.dumps(steps, ensure_ascii=False)
            + "\n</Task steps>"
        )

    def _pick_final_solution(self, solutions: List[str]) -> ClarifyPlanningDecision:
        for solution in solutions:
            steps = self._extract_merged_tasks(solution)
            if steps:
                return ClarifyPlanningDecision(
                    status="final_plan",
                    steps=steps,
                    raw_response=solution,
                    reason="Clarify produced a consistent and executable final plan.",
                )
        return ClarifyPlanningDecision(
            status="error",
            error="Clarify could not extract executable task steps from the generated solutions.",
            raw_response=solutions[0] if solutions else "",
        )

    def generate_planning_decision(
        self,
        user_input: str,
        num_solutions: int = 3,
    ) -> ClarifyPlanningDecision:
        exploration_prompt = (
            f"{user_input}\n\n"
            "Note: The user's request may contain unspecified parameters.\n"
            "Please proactively make different reasonable assumptions about these ambiguous points to generate diverse plans.\n"
            "Maintain the required output format and do not add any content beyond the specified format.\n"
        )

        try:
            solutions = self._query_llm(
                prompt=exploration_prompt,
                num_outputs=num_solutions,
                base_prompt=self._base_prompt,
                _temperature=1.0,
                use_perturbation=True,
            )
        except Exception as exc:
            return ClarifyPlanningDecision(
                status="error",
                error=f"Clarify planning failed while generating candidate solutions: {type(exc).__name__}: {exc}",
            )

        if not solutions:
            return ClarifyPlanningDecision(
                status="error",
                error="Clarify planning failed because no candidate solutions were generated.",
            )

        consistency_result = self._analyze_semantic_consistency(user_input, solutions)
        violation_result = self._analyze_violations(user_input, solutions)

        if consistency_result["consistent"] and not violation_result["has_violation"]:
            return self._pick_final_solution(solutions)

        if not consistency_result["consistent"]:
            question = self._fallback_consistency_question(consistency_result)
            return ClarifyPlanningDecision(
                status="ask_user",
                question=question,
                raw_response=self._render_ask_user_response(
                    question,
                    "Clarify found inconsistent interpretations of the user request.",
                ),
                reason="Clarify found inconsistent interpretations of the user request.",
            )

        question = str(violation_result.get("clarification_question") or "").strip()
        if not question:
            question = (
                "To ensure the plan is feasible, please clarify the missing imaging constraint that should guide the microscope setup."
            )
        return ClarifyPlanningDecision(
            status="ask_user",
            question=question,
            raw_response=self._render_ask_user_response(
                question,
                "Clarify found a blocking compliance or feasibility issue in the candidate plans.",
            ),
            reason="Clarify found a blocking compliance or feasibility issue in the candidate plans.",
        )

    def generate_code_solutions(self, user_input: str, num_solutions: int = 3) -> str:
        decision = self.generate_planning_decision(user_input, num_solutions=num_solutions)
        if decision.status == "final_plan":
            if decision.raw_response:
                return decision.raw_response
            return self._render_final_plan_response(
                decision.steps,
                decision.reason or "Clarify produced a consistent and executable final plan.",
            )
        if decision.status == "ask_user":
            return decision.raw_response or self._render_ask_user_response(
                decision.question,
                decision.reason or "Clarify requires one blocking user clarification.",
            )
        raise RuntimeError(decision.error or "Clarify planning failed.")
