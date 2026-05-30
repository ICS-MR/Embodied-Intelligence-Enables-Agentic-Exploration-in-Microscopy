from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from adapters.llm_clients import create_chat_completion
from agent.utils import _parse_json_object_response, _parse_json_response, extract_task_steps, merge_module_tasks

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
    candidate_solutions: List[str] = field(default_factory=list)
    consistency_result: Dict[str, Any] = field(default_factory=dict)
    violation_result: Dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    reason: str = ""
    error: str = ""


class Clarify:
    def __init__(self, client: OpenAI, base_prompt: str, model_name: str, semantic_model_dir, threshold, historymanager=None):
        self._base_prompt = base_prompt
        self._model_name = model_name
        self.threshold = threshold
        self.PROMPT = "Represent this sentence for semantic similarity comparison: "
        self._client = client
        self._semantic_model = self._load_semantic_model(semantic_model_dir)
        self._historyManager = historymanager

        self._GENERIC_CLARIFY_PATTERNS = (
            "please clarify",
            "missing imaging constraint",
            "missing operational detail",
            "provide detailed information",
            "experimental requirements",
        )

    def _record_interaction(self, event_type: str, message: str, payload: Dict[str, Any]) -> None:
        if self._historyManager:
            self._historyManager.record_interaction(
                agent_name="Clarify",
                event_type=event_type,
                message=message,
                payload=payload,
            )

    def _microscope_rules_context(self) -> str:
        return """
Microscope Operation and System Module Notes:
    3D samples such as organoids or cell clusters require Z-stack scanning parameters.
    There is no independent hardware autofocus; all focusing must be done through software based on image feedback.

    Plans may rely on the current system state when it already matches the required objective, imaging mode, or parameter. Do not add redundant set-operations unless a previous step changes that state.
    If the user request already contains resolved clarification items or explicitly specified parameters, treat them as authoritative and do not ask to revisit them.

    Parameter settings:
        Brightfield imaging uses the brightfield filter, low exposure, and auto-brightness for the halogen lamp.
        Fluorescence imaging uses the matching fluorescence channel, high exposure, and the halogen lamp remains off.
        If the user explicitly requests fluorescence imaging with brightness set to 0, brightness=0 is already the correct configured state before autofocus and acquisition. Do not ask whether brightness should be re-optimized instead of being set to 0.
        If the task requires brightfield autofocus, adjusting brightness before focusing is a default operational requirement rather than a user preference. Do not ask whether to preserve the current brightfield brightness versus adjust it unless the user explicitly asks to keep a fixed brightness value.

    Focusing rules:
        Switching between brightfield and fluorescence generally requires refocusing.
        Changing objectives generally requires refocusing.
        Brightfield focusing requires valid halogen lamp illumination conditions appropriate for the current objective and sample.
        In fluorescence imaging, autofocus should be evaluated after the final fluorescence configuration is in place; a user-specified brightness=0 is compatible with that workflow and is not itself a reason to ask for clarification.
        Once focus has been established in one fluorescence channel, that focus should be reused for other fluorescence channels unless the task explicitly requires separate channel-specific focusing.
        Multiple fluorescence channels may be acquired either sequentially with the same preserved focus or in one multi-channel acquisition step.
        Switching between fluorescence channels alone should not be treated as requiring refocusing or clarification unless the task explicitly says so.

    Positioning rules:
        If the task is target-specific and a valid target location or centering check is available, reposition before the final focusing step.
        Objective switching and repositioning do not require a fixed order unless the task explicitly specifies one.
        If the task is not target-specific, repositioning is not required by default.
"""

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
        seed_offset: int = 0,
    ) -> List[str]:
        def _run_single_query(call_index: int) -> str:
            last_error: Optional[Exception] = None
            for _attempt in range(3):
                try:
                    if use_perturbation:
                        temperature = 1.5
                        top_p = 0.75
                        seed_value = 4242 + seed_offset + call_index
                        perturbed_prompt = prompt
                    else:
                        temperature = _temperature
                        top_p = 1.0
                        seed_value = (42 + seed_offset + call_index) if _temperature > 0 else 42
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
                    return response.choices[0].message.content or ""
                except Exception as exc:
                    last_error = exc
            raise RuntimeError(
                f"Clarify planning query failed after retries: {type(last_error).__name__}: {last_error}"
            ) from last_error

        if num_outputs <= 1:
            return [_run_single_query(0)]

        with ThreadPoolExecutor(max_workers=min(num_outputs, 4)) as executor:
            return list(executor.map(_run_single_query, range(num_outputs)))

    def _compare_commands(self, task1, task2) -> float:
        if len(task1) != len(task2):
            return 0.0

        modules1 = [item.get("module", "") for item in task1]
        modules2 = [item.get("module", "") for item in task2]
        if modules1 != modules2:
            return 0.0

        scores: List[float] = []
        for item1, item2 in zip(task1, task2):
            cmd1 = item1.get("command", "").strip() or "[Empty command]"
            cmd2 = item2.get("command", "").strip() or "[Empty command]"
            embeddings = self._semantic_model.encode(
                [self.PROMPT + cmd1, self.PROMPT + cmd2],
                normalize_embeddings=True,
            )
            raw_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            scores.append(float(raw_sim))

        return min(scores) if scores else 0.0

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

    def _normalize_text_for_match(self, text: str) -> str:
        lowered = str(text or "").strip().lower()
        return re.sub(r"\s+", " ", lowered)

    def _extract_resolved_questions(self, current_query: str) -> List[str]:
        normalized = str(current_query or "")
        pattern = r'Resolved clarification for request "(.*?)":'
        return [match.strip() for match in re.findall(pattern, normalized, flags=re.DOTALL) if match.strip()]

    def _is_repeated_or_low_value_question(self, current_query: str, question: str) -> bool:
        normalized_question = self._normalize_text_for_match(question)
        if not normalized_question:
            return True
        if any(pattern in normalized_question for pattern in self._GENERIC_CLARIFY_PATTERNS):
            return True

        resolved_questions = self._extract_resolved_questions(current_query)
        if not resolved_questions:
            return False

        tokens = set(re.findall(r"[a-z0-9]+", normalized_question))
        if not tokens:
            return False

        for resolved in resolved_questions:
            normalized_resolved = self._normalize_text_for_match(resolved)
            if normalized_question == normalized_resolved:
                return True
            resolved_tokens = set(re.findall(r"[a-z0-9]+", normalized_resolved))
            if not resolved_tokens:
                continue
            overlap = len(tokens & resolved_tokens) / max(1, min(len(tokens), len(resolved_tokens)))
            if overlap >= 0.75:
                return True
        return False

    def _analyze_semantic_consistency(
        self,
        current_query: str,
        solutions: List[str],
        *,
        use_semantic_score_gate: bool = True,
    ) -> Dict[str, Any]:
        if use_semantic_score_gate and self._run_semantic_model(solutions):
            return {
                "consistent": True,
                "summary": "All solutions are highly consistent in task understanding.",
                "differences": [],
                "clarification_question": "",
            }

        analyze_prompt = f"""
You are analyzing multiple candidate plans generated from the same user request.

Your goal is not to check whether the candidates are identical.
Your goal is to discover whether their differences reveal a genuinely valuable unresolved ambiguity that should be clarified with the user.

Please compare the following {len(solutions)} solutions in terms of user intent and task understanding.

Analysis procedure:
1. Infer the key task-level assumptions made by each candidate.
2. Identify where the candidates truly disagree in user-facing intent, execution constraints, acquisition strategy, analysis goal, or required output.
3. Separate high-value unresolved ambiguities from low-value differences.
4. Decide whether the user must answer a clarification question before replanning.

Definitions:
- A high-value unresolved ambiguity is a missing or unclear user constraint that causes different candidates to choose materially different workflows, outputs, imaging strategies, analysis strategies, or experimental goals.
- A low-value difference is any variation in wording, plan granularity, module boundaries, step ordering, harmless default choices, or implementation detail that does not require the user to make a meaningful choice.

Requirements:
- Do not require the candidate solutions to be textually similar, structurally identical, or step-by-step the same.
- Treat solutions as consistent if they are different surface realizations of the same underlying user intent.
- Treat solutions as consistent if one candidate is simply more explicit than another, as long as they do not imply different user-facing choices.
- Treat solutions as consistent if their differences come only from default operational details that the user does not need to decide.
- Only mark the solutions as inconsistent when their differences expose a genuinely unresolved ambiguity that would materially change execution, output, acquisition strategy, analysis strategy, or an important experimental constraint.
- If the current task already contains resolved clarification details or authoritative workflow constraints, do not ask again about those same points.
- Do not ask clarification questions merely because candidates choose different reasonable defaults, unless the default changes what the user is actually asking for.
- If all candidates are acceptable interpretations of the same user intent, return consistent=true even if they are not identical.
- If clarification is needed, ask only one concise, natural, user-oriented question.
- The clarification question must target the single most valuable unresolved ambiguity revealed by cross-candidate disagreement.
- If no high-value ambiguity remains, return consistent=true and an empty clarification_question.

Output strictly in JSON format with the following fields:
{{
    "consistent": true/false,
    "summary": "The shared core user intent or the main unresolved ambiguity (one sentence)",
    "differences": ["Only include genuinely valuable unresolved ambiguities; omit low-value surface differences", ...],
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
        raw_response = response[0] if response else ""
        parsed_response = self._parse_analysis_response(raw_response, "semantic_consistency")
        if (
            not parsed_response.get("consistent", False)
            and self._is_repeated_or_low_value_question(
                current_query,
                str(parsed_response.get("clarification_question") or "").strip(),
            )
        ):
            parsed_response = {
                "consistent": True,
                "summary": str(parsed_response.get("summary") or "The candidate solutions are practically aligned for execution."),
                "differences": [],
                "clarification_question": "",
            }
        self._record_interaction(
            event_type="clarify_semantic_consistency_analysis",
            message="Clarify analyzed whether candidate solutions disagree on user intent.",
            payload={
                "current_query": current_query,
                "solutions": solutions,
                "semantic_score_gate_used": use_semantic_score_gate,
                "analysis_prompt": analyze_prompt,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            },
        )
        return parsed_response

    def _analyze_violations(self, current_query: str, solutions: List[str]) -> Dict[str, Any]:
        serialized_solutions: List[str] = []
        for solution in solutions:
            task = self._extract_merged_tasks(solution)
            if task:
                serialized_solutions.append(str(task))

        rules_context = self._microscope_rules_context()

        analyze_prompt = f"""You are a microscope operation compliance analysis assistant.

Please review the high-level plan below for ambiguity, omissions, or non-compliant operations according to the following notes:
{rules_context}

Requirements:
Focus on the following:
- Whether the plan can fulfill the task requirements.
- If historical information exists, incorporate it. Any changes to objective magnification or imaging mode must be explicitly stated, specifying only the exact supported system state (brightfield, DAPI, FITC, or TRITC) or objective magnification (e.g., 4x, 10x, 20x, 40x, 60x).
- Treat only task-relevant missing constraints, unsupported assumptions, true operational conflicts, or compliance issues as violations.
- Do not treat minor wording differences, near-synonymous parameter descriptions, plan granularity differences, or implementation-detail phrasing differences as violations if they refer to the same operational choice.
- Do not treat default operational steps implied by the microscope rules as missing user constraints unless the user explicitly overrides them.
- Do not ask clarification questions that merely choose between "use the current hardware value" and "apply the default preparation rule" when the default rule already determines the correct action.
- Do not ask about whether autofocus should use software/image-based focusing; that is a fixed system fact, not a user choice.
- If multiple issues exist, ask about only the single most critical blocking issue first.
- The clarification question must be specific and directly answerable. It must name the exact missing or conflicting constraint.
- Do not combine multiple independent questions into one clarification question.
- Do not ask generic questions such as "please clarify the missing imaging constraint".
- If all proposed plans comply with the rules, return has_violation=false and an empty string for clarification_question.
- Analyze whether the plan makes unwarranted assumptions about the user's original request, but prioritize user-relevant experimental constraints over internal implementation details.
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
        raw_response = response[0] if response else ""
        parsed_response = self._parse_violation_response(raw_response)
        self._record_interaction(
            event_type="clarify_violation_analysis",
            message="Clarify analyzed candidate solutions for ambiguity, omissions, and compliance issues.",
            payload={
                "current_query": current_query,
                "serialized_solutions": serialized_solutions,
                "analysis_prompt": analyze_prompt,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            },
        )
        return parsed_response

    def _build_exploration_prompt(self, user_input: str, stance: str = "default") -> str:
        stance_instructions = {
            "conservative": (
                "Exploration stance for this run: conservative concretization.\n"
                "When unresolved task details require concretization, choose a restrained, minimally expansive, but still fully executable interpretation.\n"
                "Prefer the least assumption-heavy concrete choices that keep the workflow complete.\n"
            ),
            "default": (
                "Exploration stance for this run: default concretization.\n"
                "When unresolved task details require concretization, instantiate them using standard operational defaults and common laboratory workflow conventions.\n"
                "Prefer a conventional, routine, and fully executable interpretation.\n"
            ),
            "bold": (
                "Exploration stance for this run: bold concretization.\n"
                "Only when the request truly leaves a material decision unresolved, choose a stronger but still plausible and task-consistent concrete interpretation.\n"
                "If the task is already sufficiently specified, stay close to the explicit request and avoid introducing extra workflow divergence merely to create diversity.\n"
                "Prefer concrete choices that materially shape the workflow only when those choices are genuinely supported by the underspecified request.\n"
            ),
        }
        stance_prompt = stance_instructions.get(stance, stance_instructions["default"])
        return (
            f"{user_input}\n\n"
            "Note: This call should produce exactly one candidate plan for ambiguity exploration.\n"
            "Different calls may produce different reasonable interpretations, but this single response must contain only one complete plan.\n"
            "The user's request may contain unspecified or ambiguous requirements.\n"
            "This response must still be fully concrete and executable.\n"
            "Do not preserve key unresolved task parameters as open-ended ambiguity in the returned plan.\n"
            "If a key parameter is underspecified, choose one reasonable concrete value or workflow interpretation and commit to it.\n"
            "Differences across runs should come from how underspecified but important task details are concretized.\n"
            "For this one response, choose one plausible interpretation of any unresolved user constraint and commit to it concretely.\n"
            "Prioritize interpretation choices that could reveal genuinely different user intentions, execution constraints, acquisition strategies, analysis goals, or required outputs.\n"
            "High-value concretization dimensions include, when applicable: objective magnification, imaging mode or channel choice, single-field versus multi-position versus full-plate acquisition scope, whether Z-stack is required, whether downstream analysis is included, and whether the task is single-acquisition or time-series.\n"
            "Do not create diversity by merely changing wording, step ordering, module decomposition, or harmless implementation details.\n"
            "If the request is already sufficiently specified, do not introduce additional divergence merely to create diversity.\n"
            "Do not override explicit user constraints or fixed system constraints.\n"
            "If no meaningful interpretation fork exists for some part of the task, keep that part aligned instead of inventing artificial variation.\n"
            "Return exactly one candidate plan.\n"
            "Do not provide multiple alternatives in one response.\n"
            "Do not enumerate Candidate 1/2/3 or Plan 1/2/3 inside the same response.\n"
            "Do not include more than one <Planner State> block.\n"
            "Do not include more than one <Task steps> block.\n"
            "Maintain the required output format and do not add any extra candidate lists, menus, separators, or alternative plans.\n"
            + stance_prompt
        )

    def _normalize_single_candidate_solution(self, solution: str) -> str:
        text = str(solution or "").strip()
        if not text:
            return ""

        planner_state_count = text.count("<Planner State>")
        task_steps_count = text.count("<Task steps>")
        has_bundle_markers = bool(re.search(r"candidate\s+(plan\s+)?[0-9]+", text, flags=re.IGNORECASE))

        if planner_state_count <= 1 and task_steps_count <= 1 and not has_bundle_markers:
            return text

        first_plan_match = re.search(
            r"(<Planner State>[\s\S]*?</Planner State>[\s\S]*?<Task steps>[\s\S]*?</Task steps>)",
            text,
            flags=re.IGNORECASE,
        )
        if first_plan_match:
            normalized = first_plan_match.group(1).strip()
            self._record_interaction(
                event_type="clarify_candidate_solution_normalized",
                message="Clarify trimmed a multi-plan response down to the first complete candidate plan.",
                payload={
                    "original_response": text,
                    "normalized_response": normalized,
                    "planner_state_count": planner_state_count,
                    "task_steps_count": task_steps_count,
                    "has_bundle_markers": has_bundle_markers,
                },
            )
            return normalized

        task_steps_match = re.search(r"(<Task steps>[\s\S]*?</Task steps>)", text, flags=re.IGNORECASE)
        if task_steps_match:
            normalized = (
                "<Planner State>\n"
                + json.dumps({"status": "final_plan"}, ensure_ascii=False)
                + "\n</Planner State>\n"
                + task_steps_match.group(1).strip()
            )
            self._record_interaction(
                event_type="clarify_candidate_solution_normalized",
                message="Clarify reconstructed a single candidate plan from a malformed multi-plan response.",
                payload={
                    "original_response": text,
                    "normalized_response": normalized,
                    "planner_state_count": planner_state_count,
                    "task_steps_count": task_steps_count,
                    "has_bundle_markers": has_bundle_markers,
                },
            )
            return normalized

        return text

    def generate_candidate_solutions(
        self,
        user_input: str,
        num_solutions: int = 3,
    ) -> List[str]:
        stances = ["conservative", "default", "bold"]
        prompts = [
            self._build_exploration_prompt(user_input, stance=stances[index % len(stances)])
            for index in range(num_solutions)
        ]
        def _run_stance_query(call_index: int, exploration_prompt: str) -> str:
            response = self._query_llm(
                prompt=exploration_prompt,
                num_outputs=1,
                base_prompt=self._base_prompt,
                _temperature=1.0,
                use_perturbation=True,
                seed_offset=call_index,
            )
            return self._normalize_single_candidate_solution(response[0] if response else "")

        with ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as executor:
            solutions = list(executor.map(lambda item: _run_stance_query(*item), enumerate(prompts)))
        self._record_interaction(
            event_type="clarify_candidate_generation",
            message="Clarify generated candidate solutions for ambiguity analysis.",
            payload={
                "user_input": user_input,
                "exploration_prompts": prompts,
                "base_prompt": self._base_prompt,
                "num_solutions": num_solutions,
                "stances": [stances[index % len(stances)] for index in range(num_solutions)],
                "candidate_solutions": solutions,
            },
        )
        return solutions

    def _parse_analysis_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        try:
            result = _parse_json_object_response(response_text)
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
        result = _parse_json_object_response(response_text)
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
                    candidate_solutions=solutions,
                    raw_response=solution,
                    reason="Clarify produced a consistent and executable final plan.",
                )
        return ClarifyPlanningDecision(
            status="error",
            candidate_solutions=solutions,
            error="Clarify could not extract executable task steps from the generated solutions.",
            raw_response=solutions[0] if solutions else "",
        )

    def generate_planning_decision(
        self,
        user_input: str,
        num_solutions: int = 3,
        *,
        force_llm_consistency_analysis: bool = False,
    ) -> ClarifyPlanningDecision:
        try:
            solutions = self.generate_candidate_solutions(user_input, num_solutions=num_solutions)
        except Exception as exc:
            return ClarifyPlanningDecision(
                status="error",
                candidate_solutions=[],
                error=f"Clarify planning failed while generating candidate solutions: {type(exc).__name__}: {exc}",
            )

        if not solutions:
            return ClarifyPlanningDecision(
                status="error",
                candidate_solutions=[],
                error="Clarify planning failed because no candidate solutions were generated.",
            )

        consistency_result = self._analyze_semantic_consistency(
            user_input,
            solutions,
            use_semantic_score_gate=not force_llm_consistency_analysis,
        )
        violation_result = self._analyze_violations(user_input, solutions)

        if consistency_result["consistent"] and not violation_result["has_violation"]:
            return self._pick_final_solution(solutions)

        if not consistency_result["consistent"]:
            question = self._fallback_consistency_question(consistency_result)
            return ClarifyPlanningDecision(
                status="ask_user",
                question=question,
                candidate_solutions=solutions,
                consistency_result=consistency_result,
                violation_result=violation_result,
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
            candidate_solutions=solutions,
            consistency_result=consistency_result,
            violation_result=violation_result,
            raw_response=self._render_ask_user_response(
                question,
                "Clarify found a blocking compliance or feasibility issue in the candidate plans.",
            ),
            reason="Clarify found a blocking compliance or feasibility issue in the candidate plans.",
        )

    def generate_code_solutions(
        self,
        user_input: str,
        num_solutions: int = 3,
        *,
        force_llm_consistency_analysis: bool = False,
    ) -> str:
        decision = self.generate_planning_decision(
            user_input,
            num_solutions=num_solutions,
            force_llm_consistency_analysis=force_llm_consistency_analysis,
        )
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
