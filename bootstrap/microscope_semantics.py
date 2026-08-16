from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional


def normalized_semantic_key(value: Any) -> str:
    return str(value or "").strip().lower()


def objective_entries(system_config: Any) -> Dict[str, Dict[str, Any]]:
    return {
        normalized_semantic_key(key): dict(item)
        for key, item in getattr(system_config, "objectives", {}).items()
        if isinstance(item, Mapping)
    }


def channel_entries(system_config: Any) -> Dict[str, Dict[str, Any]]:
    return {
        normalized_semantic_key(key): dict(item)
        for key, item in getattr(system_config, "channels", {}).items()
        if isinstance(item, Mapping)
    }


def derived_objective_labels(system_config: Any) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    for item in objective_entries(system_config).values():
        label = str(item.get("label") or "").strip()
        magnification = item.get("magnification")
        if not label or magnification is None:
            continue
        try:
            labels[label] = int(magnification)
        except (TypeError, ValueError):
            continue
    return labels


def derived_dichroic_colors(system_config: Any) -> Dict[str, tuple[int, int, int]]:
    colors: Dict[str, tuple[int, int, int]] = {}
    for item in channel_entries(system_config).values():
        label = str(item.get("label") or "").strip()
        color = item.get("color")
        if label and isinstance(color, (list, tuple)) and len(color) == 3:
            colors[label] = (int(color[0]), int(color[1]), int(color[2]))
    return colors


def resolve_objective_input(value: str, system_config: Any) -> tuple[str, Optional[str], Dict[str, Any]]:
    requested = str(value or "").strip()
    semantic_key = normalized_semantic_key(requested)
    objectives = objective_entries(system_config)
    if semantic_key in objectives:
        entry = objectives[semantic_key]
        label = str(entry.get("label") or "").strip()
        if label:
            return label, semantic_key, entry

    for key, entry in objectives.items():
        label = str(entry.get("label") or "").strip()
        if requested == label:
            return label, key, entry

    return requested, None, {}


def resolve_channel_input(value: str, system_config: Any) -> tuple[str, Optional[str], Dict[str, Any]]:
    requested = str(value or "").strip()
    semantic_key = normalized_semantic_key(requested)
    channels = channel_entries(system_config)
    if semantic_key in channels:
        entry = channels[semantic_key]
        label = str(entry.get("label") or "").strip()
        if label:
            return label, semantic_key, entry

    for key, entry in channels.items():
        label = str(entry.get("label") or "").strip()
        if requested == label:
            return label, key, entry

    return requested, None, {}


def objective_semantic_for_label(label: str, system_config: Any) -> str:
    resolved_label, semantic_key, entry = resolve_objective_input(label, system_config)
    if semantic_key:
        return semantic_key
    magnification = getattr(system_config, "objective_labels", {}).get(resolved_label)
    return f"{magnification}x" if magnification is not None else ""


def channel_semantic_for_label(label: str, system_config: Any) -> str:
    _resolved_label, semantic_key, _entry = resolve_channel_input(label, system_config)
    return semantic_key or ""


def objective_display_name(label_or_semantic: str, system_config: Any) -> str:
    resolved_label, semantic_key, entry = resolve_objective_input(label_or_semantic, system_config)
    if entry.get("display_name"):
        return str(entry["display_name"])
    if semantic_key:
        return semantic_key
    magnification = getattr(system_config, "objective_labels", {}).get(resolved_label)
    return f"{magnification}x objective" if magnification is not None else "Unknown"


def channel_display_name(label_or_semantic: str, system_config: Any) -> str:
    _resolved_label, semantic_key, entry = resolve_channel_input(label_or_semantic, system_config)
    if entry.get("display_name"):
        return str(entry["display_name"])
    return semantic_key or "Unknown"


def is_brightfield_channel(label_or_semantic: str, system_config: Any) -> bool:
    _resolved_label, semantic_key, entry = resolve_channel_input(label_or_semantic, system_config)
    if semantic_key == "brightfield":
        return True
    illumination = str(entry.get("illumination") or "").strip().lower()
    return illumination == "transmitted"


def _entry_label(entries: Mapping[str, Mapping[str, Any]], semantic_key: str) -> str:
    entry = entries.get(normalized_semantic_key(semantic_key), {})
    label = str(entry.get("label") or "").strip()
    return label or str(semantic_key)


def render_microscope_prompt_template(prompt_text: str, system_config: Any) -> str:
    objectives = objective_entries(system_config)
    channels = channel_entries(system_config)
    transmitted_light = dict(getattr(system_config, "transmitted_light", {}) or {})

    def replace_placeholder(match: re.Match[str]) -> str:
        kind = match.group("kind")
        key = match.group("key")
        field = match.group("field")
        if field != "label":
            return match.group(0)
        if kind == "objective":
            return _entry_label(objectives, key)
        if kind == "channel":
            return _entry_label(channels, key)
        return match.group(0)

    rendered = re.sub(
        r"\{\{(?P<kind>objective|channel)\.(?P<key>[^.{}]+)\.(?P<field>[^{}]+)\}\}",
        replace_placeholder,
        prompt_text,
    )
    rendered = rendered.replace("{{transmitted_light.min}}", str(transmitted_light.get("min", "")))
    rendered = rendered.replace("{{transmitted_light.max}}", str(transmitted_light.get("max", "")))
    system_bounds = {
        "min_x": getattr(system_config, "Min_X_position", ""),
        "max_x": getattr(system_config, "Max_X_position", ""),
        "min_y": getattr(system_config, "Min_Y_position", ""),
        "max_y": getattr(system_config, "Max_Y_position", ""),
        "min_z": getattr(system_config, "Min_Z_position", ""),
        "max_z": getattr(system_config, "Max_Z_position", ""),
        "min_exposure": getattr(system_config, "Min_exposure", ""),
        "max_exposure": getattr(system_config, "Max_exposure", ""),
    }
    for bound_key, bound_value in system_bounds.items():
        rendered = rendered.replace("{{system.%s}}" % bound_key, str(bound_value))
    return rendered
