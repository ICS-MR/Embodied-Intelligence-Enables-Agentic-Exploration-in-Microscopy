import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from adapters.llm_clients import create_chat_completion
from utils.planning_skills import (
    PlanningSkill,
    build_active_template_metadata,
    find_skills_by_name,
    format_skills_for_routing_prompt,
    format_selected_skills_for_prompt,
    load_planning_skills,
)


@dataclass
class SkillResolutionRequest:
    user_request: str
    system_state: Any
    clarification_history: str = ""


@dataclass
class SkillResolutionResult:
    status: str
    question: str = ""
    resolved_task_instruction: str = ""
    selected_skills: List[str] = field(default_factory=list)
    reason: str = ""
    active_templates: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    raw_response: str = ""
    routing_raw_response: str = ""
    error: Optional[str] = None


class SkillResolver:
    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        seed: Optional[int] = None,
        history_manager: Any = None,
        skill_dirs: Optional[Sequence[str]] = None,
        skill_max_files: int = 20,
        skill_max_chars_per_file: int = 2000,
        skill_max_selected: int = 2,
        skill_route_max_tokens: int = 512,
        skill_route_temperature: float = 0.0,
        resolution_max_tokens: int = 4096,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._seed = seed
        self._history_manager = history_manager
        self._skill_dirs = list(skill_dirs or [])
        self._skill_max_files = int(skill_max_files)
        self._skill_max_chars_per_file = int(skill_max_chars_per_file)
        self._skill_max_selected = int(skill_max_selected)
        self._skill_route_max_tokens = int(skill_route_max_tokens)
        self._skill_route_temperature = float(skill_route_temperature)
        self._resolution_max_tokens = int(resolution_max_tokens)

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        prompt: str,
        temperature: float,
        max_tokens: Optional[int],
    ) -> tuple[str, Optional[Dict[str, int]]]:
        response = create_chat_completion(
            self._client,
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            seed=self._seed,
            stop=[],
            stream=False,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or "", {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

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

    def _serialize_state_text(self, state: Any) -> str:
        try:
            if isinstance(state, dict):
                return json.dumps(state, indent=2)
            if hasattr(state, "__dict__"):
                return json.dumps(state.__dict__, indent=2)
            return str(state)
        except Exception:
            return str(state)

    def _wrap_prompt_section(self, title: str, content: str) -> str:
        normalized = str(content or "").strip()
        if not normalized:
            return ""
        return f"<{title}>\n{normalized}\n</{title}>"

    def _merge_usage(self, *usages: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        merged: Dict[str, int] = {}
        for usage in usages:
            if not usage:
                continue
            for key, value in usage.items():
                merged[key] = merged.get(key, 0) + int(value)
        return merged or None

    def _load_skills(self) -> List[PlanningSkill]:
        return load_planning_skills(
            skill_dirs=self._skill_dirs,
            max_files=self._skill_max_files,
            max_chars_per_file=self._skill_max_chars_per_file,
        )

    def route_skills(self, user_request: str, system_state: Any, clarification_history: str = "") -> SkillResolutionResult:
        all_skills = self._load_skills()
        if not all_skills:
            return SkillResolutionResult(status="ready_for_planner", resolved_task_instruction=user_request)

        prompt_parts = []
        prompt_parts.append(self._wrap_prompt_section("Available Skill Packages", format_skills_for_routing_prompt(all_skills)))
        if clarification_history.strip():
            prompt_parts.append(self._wrap_prompt_section("Clarification History", clarification_history))
        prompt_parts.append(self._wrap_prompt_section("Current System State", self._serialize_state_text(system_state)))
        prompt_parts.append(
            self._wrap_prompt_section(
                "Routing Task",
                (
                    "Select the most relevant skill packages for this request. Read the available skill package summaries "
                    "and content excerpts, then decide whether each skill should actually be applied to the current request. "
                    "Use semantic understanding of the workflow intent instead of relying only on names, keywords, or triggers. "
                    "Only selected skills will be read in full during the later resolution stage. "
                    f"Select at most {self._skill_max_selected} skills. Return JSON only in the form "
                    "{\"need_skill\": true, \"selected_skills\": [\"skill-name\"], \"reason\": \"short reason\"}."
                ),
            )
        )
        prompt_parts.append(self._wrap_prompt_section("User Request", user_request))
        prompt = "\n\n".join(part for part in prompt_parts if part)
        routing_system_prompt = (
            "You are a skill router. Read the available skill package summaries and excerpts, understand their workflow "
            "semantics, and decide which reusable skills are truly relevant to the current request. Do not route by "
            "keyword matching alone. Use progressive disclosure: make a routing decision from the concise skill views, "
            "then let the later resolution stage read the selected skills in full. Return valid JSON only."
        )
        raw_response, usage = self._chat_completion(
            system_prompt=routing_system_prompt,
            prompt=prompt,
            temperature=self._skill_route_temperature,
            max_tokens=self._skill_route_max_tokens,
        )
        payload = self._parse_json_object(raw_response) or {}
        selected_skills = find_skills_by_name(
            all_skills,
            payload.get("selected_skills") or [],
            max_selected=self._skill_max_selected,
        )
        selected_skill_names = [skill.name for skill in selected_skills]
        need_skill = bool(payload.get("need_skill")) and bool(selected_skill_names)
        reason = str(payload.get("reason") or "").strip()
        active_templates = build_active_template_metadata(selected_skills if need_skill else [])

        if self._history_manager:
            self._history_manager.record_interaction(
                agent_name="Task_manager",
                event_type="skill_routing",
                message="Skill resolver routed skill packages for the current request.",
                payload={
                    "query": user_request,
                    "clarification_history": clarification_history,
                    "catalog_size": len(all_skills),
                    "selected_skills": selected_skill_names,
                    "need_skill": need_skill,
                    "reason": reason,
                    "active_templates": active_templates,
                    "system_prompt": routing_system_prompt,
                    "raw_response": raw_response,
                    "usage": usage or {},
                },
            )

        return SkillResolutionResult(
            status="ready_for_planner",
            resolved_task_instruction=user_request,
            selected_skills=selected_skill_names if need_skill else [],
            reason=reason,
            active_templates=active_templates,
            usage=usage,
            routing_raw_response=raw_response,
        )

    def resolve_with_selected_skills(
        self,
        user_request: str,
        system_state: Any,
        clarification_history: str,
        *,
        selected_skill_names: Sequence[str],
        reason: str,
        active_templates: Sequence[Dict[str, Any]],
    ) -> SkillResolutionResult:
        all_skills = self._load_skills()
        selected_skills = find_skills_by_name(
            all_skills,
            selected_skill_names,
            max_selected=self._skill_max_selected,
        )
        if not selected_skills:
            return SkillResolutionResult(
                status="ready_for_planner",
                resolved_task_instruction=user_request,
                selected_skills=[],
                reason=reason,
                active_templates=list(active_templates),
            )

        prompt_parts = []
        prompt_parts.append(self._wrap_prompt_section("Selected Skills", format_selected_skills_for_prompt(selected_skills)))
        prompt_parts.append(
            self._wrap_prompt_section(
                "Resolver Protocol",
                (
                    "Read the selected skill packages and decide whether the user request still lacks blocking information. "
                    "If blocking information is still missing, ask exactly one consolidated clarification question. "
                    "If the request is sufficiently specified, rewrite it into one complete natural-language task instruction "
                    "for the downstream planner. Do not output task steps. Do not output planner tags. "
                    "When a skill describes a formal experiment workflow, preserve that workflow style in the resolved task instruction. "
                    "For the resolved task instruction, preserve all user-confirmed parameters exactly and do not omit confirmed constraints. "
                    "If the selected skill defines specific required sections or workflow headings, include them explicitly in the resolved task instruction. "
                    "If the selected skill provides a resolved instruction example, treat its structure and section-heading style as the preferred output style. "
                    "Prefer a formal experiment-protocol style over shorthand notes. "
                    "Start the resolved task instruction with one introductory paragraph before any numbered section. "
                    "If the selected skill requires numbered workflow sections, preserve the numbered headings exactly and in the same order. "
                    "Do not replace numbered section headings with placeholder dividers such as '--- INITIALIZATION ---', '--- GLOBAL SCAN LOOP ---', or other ad hoc heading styles. "
                    "Return valid JSON only in one of these forms:\n"
                    "{\"status\":\"ask_user\",\"question\":\"...\",\"reason\":\"...\"}\n"
                    "{\"status\":\"ready_for_planner\",\"resolved_task_instruction\":\"...\",\"reason\":\"...\"}\n"
                    "{\"status\":\"error\",\"reason\":\"...\"}"
                ),
            )
        )
        if clarification_history.strip():
            prompt_parts.append(self._wrap_prompt_section("Clarification History", clarification_history))
        prompt_parts.append(self._wrap_prompt_section("Current System State", self._serialize_state_text(system_state)))
        prompt_parts.append(self._wrap_prompt_section("User Request", user_request))
        prompt = "\n\n".join(part for part in prompt_parts if part)
        resolution_system_prompt = (
            "You are a skill resolver. Use the selected skill packages as the source of truth, combine them with the "
            "user request and clarification history, and either ask one blocking clarification question or produce one "
            "complete task instruction for the downstream planner. When information is complete, the resolved task "
            "instruction must be detailed enough to hand directly to the downstream planner without requiring the "
            "planner to reinterpret missing workflow semantics. Match the selected skill's preferred workflow style as "
            "closely as possible, especially when it provides an explicit resolved instruction example. Return valid JSON only."
        )
        raw_response, usage = self._chat_completion(
            system_prompt=resolution_system_prompt,
            prompt=prompt,
            temperature=0,
            max_tokens=self._resolution_max_tokens,
        )
        payload = self._parse_json_object(raw_response) or {}
        status = str(payload.get("status") or "").strip().lower()
        question = str(payload.get("question") or "").strip()
        resolved_task_instruction = str(payload.get("resolved_task_instruction") or "").strip()
        resolved_reason = str(payload.get("reason") or "").strip() or reason

        if self._history_manager:
            self._history_manager.record_interaction(
                agent_name="Task_manager",
                event_type="skill_resolution",
                message="Skill resolver processed the selected skill packages before planning.",
                payload={
                    "query": user_request,
                    "clarification_history": clarification_history,
                    "selected_skills": [skill.name for skill in selected_skills],
                    "reason": resolved_reason,
                    "active_templates": list(active_templates),
                    "system_prompt": resolution_system_prompt,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "resolution_status": status or "error",
                    "question": question,
                    "resolved_task_instruction": resolved_task_instruction,
                    "usage": usage or {},
                },
            )

        if status == "ask_user" and question:
            return SkillResolutionResult(
                status="ask_user",
                question=question,
                selected_skills=[skill.name for skill in selected_skills],
                reason=resolved_reason,
                active_templates=list(active_templates),
                usage=usage,
                raw_response=raw_response,
            )

        if status == "ready_for_planner" and resolved_task_instruction:
            return SkillResolutionResult(
                status="ready_for_planner",
                resolved_task_instruction=resolved_task_instruction,
                selected_skills=[skill.name for skill in selected_skills],
                reason=resolved_reason,
                active_templates=list(active_templates),
                usage=usage,
                raw_response=raw_response,
            )

        return SkillResolutionResult(
            status="error",
            selected_skills=[skill.name for skill in selected_skills],
            reason=resolved_reason,
            active_templates=list(active_templates),
            usage=usage,
            raw_response=raw_response,
            error="Skill resolver did not return a valid question or resolved task instruction.",
        )

    def resolve(self, request: SkillResolutionRequest) -> SkillResolutionResult:
        routing_result = self.route_skills(request.user_request, request.system_state, request.clarification_history)
        if not routing_result.selected_skills:
            return routing_result
        resolution_result = self.resolve_with_selected_skills(
            request.user_request,
            request.system_state,
            request.clarification_history,
            selected_skill_names=routing_result.selected_skills,
            reason=routing_result.reason,
            active_templates=routing_result.active_templates,
        )
        resolution_result.usage = self._merge_usage(routing_result.usage, resolution_result.usage)
        resolution_result.routing_raw_response = routing_result.routing_raw_response
        if not resolution_result.selected_skills:
            resolution_result.selected_skills = list(routing_result.selected_skills)
        if not resolution_result.reason:
            resolution_result.reason = routing_result.reason
        if not resolution_result.active_templates:
            resolution_result.active_templates = list(routing_result.active_templates)
        return resolution_result
