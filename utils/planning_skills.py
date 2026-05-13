import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence


SUPPORTED_SKILL_SUFFIXES = {".md", ".txt", ".json"}
DEFAULT_SKILL_DIRS = [Path("user_skills") / "planning"]
SKILL_PACKAGE_FILENAME = "SKILL.md"
_FRONT_MATTER_DELIMITER = "---"


@dataclass
class PlanningSkill:
    name: str
    path: Path
    content: str
    tags: List[str]
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    priority: int = 0
    skill_type: str = ""
    template_goal: str = ""
    planning_stages: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    output_strategy: str = "direct_plan"
    source_kind: str = "file"
    score: int = 0
    selected_reasons: List[str] = field(default_factory=list)


@dataclass
class _SkillPayload:
    name: str
    content: str
    tags: List[str]
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    priority: int = 0
    skill_type: str = ""
    template_goal: str = ""
    planning_stages: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    output_strategy: str = "direct_plan"


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9\u4e00-\u9fff_+-]+", text or "") if len(token) >= 2}


def _truncate_text(text: str, max_chars: int) -> str:
    normalized = str(text or "").strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def _normalize_skill_dirs(skill_dirs: Optional[Sequence[str | Path]]) -> List[Path]:
    raw_dirs = skill_dirs or DEFAULT_SKILL_DIRS
    normalized: List[Path] = []
    for item in raw_dirs:
        path = Path(item)
        normalized.append(path if path.is_absolute() else Path.cwd() / path)
    return normalized


def _extract_markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _coerce_string_list(value: Any) -> List[str]:
    if isinstance(value, str):
        candidates = re.split(r"[,\n]", value)
        return [item.strip() for item in candidates if item.strip()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_skill_type(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return ""
    return normalized


def _normalize_output_strategy(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized == "single_question_then_plan":
        return "single_question_then_plan"
    return "direct_plan"


def _parse_front_matter(raw_text: str) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith(_FRONT_MATTER_DELIMITER):
        return {}, raw_text

    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        return {}, raw_text

    metadata: dict[str, Any] = {}
    body_start = None
    current_key: Optional[str] = None
    for index in range(1, len(lines)):
        stripped = lines[index].strip()
        if stripped == _FRONT_MATTER_DELIMITER:
            body_start = index + 1
            break
        if not stripped:
            current_key = None
            continue
        if stripped.startswith("- ") and current_key:
            metadata.setdefault(current_key, [])
            metadata[current_key].append(stripped[2:].strip())
            continue
        if ":" not in stripped:
            current_key = None
            continue
        key, value = stripped.split(":", 1)
        current_key = key.strip().lower()
        value = value.strip()
        if value:
            metadata[current_key] = value
        else:
            metadata[current_key] = []

    if body_start is None:
        return {}, raw_text
    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def _payload_from_json(raw_text: str, fallback_name: str) -> Optional[_SkillPayload]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _SkillPayload(
        name=str(payload.get("name") or payload.get("title") or fallback_name).strip() or fallback_name,
        content=str(payload.get("content") or payload.get("guidance") or raw_text).strip(),
        tags=_coerce_string_list(payload.get("tags")),
        description=str(payload.get("description") or "").strip(),
        triggers=_coerce_string_list(payload.get("triggers") or payload.get("trigger_keywords") or payload.get("when")),
        examples=_coerce_string_list(payload.get("examples") or payload.get("queries")),
        priority=_coerce_int(payload.get("priority"), 0),
        skill_type=_normalize_skill_type(payload.get("skill_type") or payload.get("type")),
        template_goal=str(payload.get("template_goal") or "").strip(),
        planning_stages=_coerce_string_list(payload.get("planning_stages")),
        required_inputs=_coerce_string_list(payload.get("required_inputs")),
        output_strategy=_normalize_output_strategy(payload.get("output_strategy")),
    )


def _payload_from_text(raw_text: str, fallback_name: str, *, preferred_name: Optional[str] = None) -> _SkillPayload:
    metadata, body = _parse_front_matter(raw_text)
    content = body or raw_text.strip()
    name = str(metadata.get("name") or metadata.get("title") or preferred_name or fallback_name).strip() or fallback_name
    if not metadata and fallback_name.lower().endswith(".md"):
        name = _extract_markdown_title(raw_text, name)
    return _SkillPayload(
        name=_extract_markdown_title(content, name) if fallback_name.lower().endswith(".md") and not metadata.get("name") and not metadata.get("title") else name,
        content=content.strip(),
        tags=_coerce_string_list(metadata.get("tags")),
        description=str(metadata.get("description") or "").strip(),
        triggers=_coerce_string_list(metadata.get("triggers") or metadata.get("when")),
        examples=_coerce_string_list(metadata.get("examples") or metadata.get("queries")),
        priority=_coerce_int(metadata.get("priority"), 0),
        skill_type=_normalize_skill_type(metadata.get("skill_type") or metadata.get("type")),
        template_goal=str(metadata.get("template_goal") or "").strip(),
        planning_stages=_coerce_string_list(metadata.get("planning_stages")),
        required_inputs=_coerce_string_list(metadata.get("required_inputs")),
        output_strategy=_normalize_output_strategy(metadata.get("output_strategy")),
    )


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            return None
    except OSError:
        return None


def _build_skill(path: Path, payload: _SkillPayload, *, max_chars: int, source_kind: str) -> Optional[PlanningSkill]:
    content = payload.content[:max_chars].strip()
    if not content:
        return None
    return PlanningSkill(
        name=payload.name,
        path=path,
        content=content,
        tags=payload.tags,
        description=payload.description,
        triggers=payload.triggers,
        examples=payload.examples,
        priority=payload.priority,
        skill_type=payload.skill_type,
        template_goal=payload.template_goal,
        planning_stages=payload.planning_stages,
        required_inputs=payload.required_inputs,
        output_strategy=payload.output_strategy,
        source_kind=source_kind,
    )


def _load_skill_file(path: Path, max_chars: int) -> Optional[PlanningSkill]:
    raw_text = _read_text(path)
    if not raw_text:
        return None

    if path.suffix.lower() == ".json":
        payload = _payload_from_json(raw_text, path.stem)
        if payload is None:
            payload = _SkillPayload(name=path.stem, content=raw_text, tags=[])
    else:
        payload = _payload_from_text(raw_text, path.name)
    return _build_skill(path, payload, max_chars=max_chars, source_kind="file")


def _load_skill_package(directory: Path, max_chars: int) -> Optional[PlanningSkill]:
    skill_file = directory / SKILL_PACKAGE_FILENAME
    raw_text = _read_text(skill_file)
    if not raw_text:
        return None
    payload = _payload_from_text(raw_text, skill_file.name, preferred_name=directory.name)
    if not payload.name:
        payload.name = directory.name
    return _build_skill(skill_file, payload, max_chars=max_chars, source_kind="package")


def load_planning_skills(
    *,
    skill_dirs: Optional[Sequence[str | Path]] = None,
    max_files: int = 20,
    max_chars_per_file: int = 2000,
) -> List[PlanningSkill]:
    loaded: List[PlanningSkill] = []
    seen_paths: set[Path] = set()
    for directory in _normalize_skill_dirs(skill_dirs):
        if not directory.exists() or not directory.is_dir():
            continue

        package_dirs = sorted(path for path in directory.rglob("*") if path.is_dir() and (path / SKILL_PACKAGE_FILENAME).is_file())
        for package_dir in package_dirs:
            skill = _load_skill_package(package_dir, max_chars=max_chars_per_file)
            if skill is not None and skill.path not in seen_paths:
                loaded.append(skill)
                seen_paths.add(skill.path)
            if len(loaded) >= max_files:
                return loaded

        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SKILL_SUFFIXES:
                continue
            if path.name == SKILL_PACKAGE_FILENAME:
                continue
            skill = _load_skill_file(path, max_chars=max_chars_per_file)
            if skill is not None and skill.path not in seen_paths:
                loaded.append(skill)
                seen_paths.add(skill.path)
            if len(loaded) >= max_files:
                return loaded
    return loaded


def _phrase_hits(query: str, phrases: Sequence[str]) -> List[str]:
    lowered_query = query.lower()
    hits: List[str] = []
    for phrase in phrases:
        normalized = phrase.strip().lower()
        if normalized and normalized in lowered_query:
            hits.append(phrase.strip())
    return hits


def select_relevant_planning_skills(
    query: str,
    skills: Iterable[PlanningSkill],
    *,
    top_k: int = 3,
) -> List[PlanningSkill]:
    query_tokens = _tokenize(query)
    lowered_query = query.lower()
    ranked: List[PlanningSkill] = []
    for skill in skills:
        name_tokens = _tokenize(skill.name)
        tag_tokens = _tokenize(" ".join(skill.tags))
        trigger_tokens = _tokenize(" ".join(skill.triggers))
        example_tokens = _tokenize(" ".join(skill.examples))
        content_tokens = _tokenize(skill.content)

        score = 0
        reasons: List[str] = []

        explicit_invocation = skill.name.lower() in lowered_query or any(token in lowered_query for token in name_tokens if len(token) >= 4)
        if explicit_invocation:
            score += 12
            reasons.append("name match")

        trigger_hits = _phrase_hits(query, skill.triggers)
        if trigger_hits:
            score += 8 * len(trigger_hits)
            reasons.append("trigger match")

        tag_overlap = len(query_tokens & tag_tokens)
        if tag_overlap:
            score += 4 * tag_overlap
            reasons.append("tag overlap")

        example_overlap = len(query_tokens & example_tokens)
        if example_overlap:
            score += 3 * example_overlap
            reasons.append("example overlap")

        content_overlap = len(query_tokens & (content_tokens | trigger_tokens | name_tokens))
        if content_overlap:
            score += content_overlap
            reasons.append("content overlap")

        filename_bonus = 2 if any(token in skill.path.stem.lower() for token in query_tokens) else 0
        if filename_bonus:
            score += filename_bonus
            reasons.append("filename bonus")

        if skill.priority:
            score += skill.priority
            reasons.append("priority")

        skill.score = score
        skill.selected_reasons = reasons
        ranked.append(skill)

    ranked.sort(key=lambda item: (item.score, item.priority, item.name.lower()), reverse=True)
    selected = [skill for skill in ranked if skill.score > 0][:top_k]
    if selected:
        return selected
    return ranked[:top_k]


def format_planning_skills_for_prompt(skills: Sequence[PlanningSkill]) -> str:
    if not skills:
        return ""

    lines = [
        "User-provided planning skills:",
        "Treat these as reusable planning skills. Prefer them when their scope matches the current request, but do not violate system constraints or the user's latest instruction.",
    ]
    for index, skill in enumerate(skills, start=1):
        lines.append(f"Skill {index}: {skill.name}")
        if skill.skill_type:
            lines.append(f"Skill type: {skill.skill_type}")
        if skill.description:
            lines.append(f"Description: {skill.description}")
        if skill.output_strategy:
            lines.append(f"Output strategy: {skill.output_strategy}")
        if skill.required_inputs:
            lines.append(f"Required inputs: {', '.join(skill.required_inputs)}")
        lines.append("Guidance:")
        lines.append(skill.content)
    return "\n".join(lines)


def build_skill_catalog(skills: Sequence[PlanningSkill]) -> List[dict[str, Any]]:
    catalog: List[dict[str, Any]] = []
    for skill in skills:
        catalog.append(
            {
                "name": skill.name,
                "description": skill.description,
                "skill_type": skill.skill_type,
            }
        )
    return catalog


def format_skills_for_routing_prompt(skills: Sequence[PlanningSkill], *, excerpt_chars: int = 900) -> str:
    if not skills:
        return "No planning skills are available."

    lines = [
        "Available planning skill packages:",
        "Read each skill package summary and content excerpt, then decide whether its actual workflow content should be applied.",
        "Use semantic understanding of the skill package. Do not rely only on the skill name or obvious keyword overlap.",
        "This is a progressive disclosure stage: routing sees a concise semantic profile first, and only selected skills are read in full later.",
    ]
    for index, skill in enumerate(skills, start=1):
        lines.append(f"Skill {index}: {skill.name}")
        if skill.description:
            lines.append(f"Description: {skill.description}")
        if skill.triggers:
            lines.append(f"Triggers: {', '.join(skill.triggers)}")
        if skill.examples:
            lines.append(f"Examples: {', '.join(skill.examples)}")
        if skill.output_strategy:
            lines.append(f"Output strategy: {skill.output_strategy}")
        if skill.required_inputs:
            lines.append(f"Required inputs: {', '.join(skill.required_inputs)}")
        if skill.template_goal:
            lines.append(f"Template goal: {skill.template_goal}")
        lines.append("Skill package content excerpt:")
        lines.append(_truncate_text(skill.content, excerpt_chars))
    return "\n".join(lines)


def find_skills_by_name(
    skills: Sequence[PlanningSkill],
    selected_names: Sequence[str],
    *,
    max_selected: int = 2,
) -> List[PlanningSkill]:
    if max_selected <= 0:
        return []

    selected: List[PlanningSkill] = []
    seen_names: set[str] = set()
    skill_by_name = {skill.name.strip().lower(): skill for skill in skills if skill.name.strip()}

    for raw_name in selected_names:
        normalized_name = str(raw_name).strip().lower()
        if not normalized_name or normalized_name in seen_names:
            continue
        skill = skill_by_name.get(normalized_name)
        if skill is None:
            continue
        selected.append(skill)
        seen_names.add(normalized_name)
        if len(selected) >= max_selected:
            break
    return selected


def format_skill_catalog_for_prompt(catalog: Sequence[dict[str, Any]]) -> str:
    if not catalog:
        return "No planning skills are available."

    lines = [
        "Available planning skill catalog:",
        "Use this catalog only to decide whether extra planning guidance is needed.",
    ]
    for index, skill in enumerate(catalog, start=1):
        lines.append(f"Skill {index}: {str(skill.get('name') or '').strip()}")
        skill_type = str(skill.get("skill_type") or "").strip()
        if skill_type:
            lines.append(f"Skill type: {skill_type}")
        description = str(skill.get("description") or "").strip()
        if description:
            lines.append(f"Description: {description}")
    return "\n".join(lines)


def format_selected_skills_for_prompt(skills: Sequence[PlanningSkill]) -> str:
    return format_planning_skills_for_prompt(skills)


def extract_active_planning_templates(skills: Sequence[PlanningSkill]) -> List[PlanningSkill]:
    active_templates: List[PlanningSkill] = []
    for skill in skills:
        if skill.output_strategy == "single_question_then_plan":
            active_templates.append(skill)
            continue
        if skill.required_inputs or skill.template_goal or skill.planning_stages:
            active_templates.append(skill)
    return active_templates


def build_active_template_metadata(skills: Sequence[PlanningSkill]) -> List[dict[str, Any]]:
    metadata: List[dict[str, Any]] = []
    for skill in extract_active_planning_templates(skills):
        metadata.append(
            {
                "name": skill.name,
                "skill_type": skill.skill_type,
                "template_goal": skill.template_goal,
                "required_inputs": list(skill.required_inputs),
                "planning_stages": list(skill.planning_stages),
                "output_strategy": skill.output_strategy,
            }
        )
    return metadata


def format_active_templates_for_prompt(skills: Sequence[PlanningSkill]) -> str:
    templates = extract_active_planning_templates(skills)
    if not templates:
        return ""

    lines = [
        "Active planning template skills for this round:",
        "Use these templates as planning workflows, not as executable tools.",
    ]
    for index, skill in enumerate(templates, start=1):
        lines.append(f"Template {index}: {skill.name}")
        if skill.description:
            lines.append(f"Description: {skill.description}")
        lines.append(f"Output strategy: {skill.output_strategy}")
        if skill.required_inputs:
            lines.append(f"Required inputs: {', '.join(skill.required_inputs)}")
        lines.append("Template guidance:")
        lines.append(skill.content)
    return "\n".join(lines)
