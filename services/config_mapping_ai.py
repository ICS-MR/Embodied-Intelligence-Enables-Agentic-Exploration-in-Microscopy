"""AI-assisted Micro-Manager cfg mapping drafts."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping

from adapters.llm_clients import create_chat_completion
from api.models import ConfigMappingAnalysis, ConfigMappingDraftField
from runtime.agent_factory import build_clients


logger = logging.getLogger(__name__)


class ConfigMappingAIError(RuntimeError):
    pass


_ROLE_FIELDS = (
    "camera_device",
    "xy_stage_device",
    "objective_device",
    "focus_drive",
    "Dichroic",
)
_CORE_ROLE_BY_FIELD = {
    "camera_device": "Camera",
    "xy_stage_device": "XYStage",
    "focus_drive": "Focus",
}
_FORM_OBJECTIVE_KEYS = ("4x", "10x", "20x", "30x", "40x", "60x", "100x")
_FORM_CHANNEL_KEYS = ("brightfield", "dapi", "fitc", "tritc")
_INTENSITY_TOKENS = ("brightness", "intensity", "power", "level", "percent")
_CONFIDENCE_VALUES = {"high", "medium", "low", "unknown"}
_DRAFT_FIELD_KEYS = {
    "value",
    "candidates",
    "source",
    "confidence",
    "reason",
    "needs_review",
    "rule_value",
    "ai_value",
    "current_value",
}


def _parse_json_response(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if not text:
        raise ConfigMappingAIError("The mapping model returned an empty response.")
    # Strip markdown code fences (``` or ```json).
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :].strip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            payload = _extract_json_object(text)
        except ConfigMappingAIError:
            raise ConfigMappingAIError("The mapping model did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ConfigMappingAIError("The mapping model returned a JSON value instead of an object.")
    return payload


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model output."""
    start = text.find("{")
    if start == -1:
        raise ConfigMappingAIError("The mapping model did not return valid JSON (no object found).")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ConfigMappingAIError(
                        "The mapping model did not return valid JSON."
                    ) from exc
                if isinstance(payload, dict):
                    return payload
                raise ConfigMappingAIError(
                    "The mapping model returned a JSON value instead of an object."
                )
    raise ConfigMappingAIError("The mapping model did not return valid JSON (unbalanced object).")


def _response_content(completion: Any) -> str:
    try:
        return str(completion.choices[0].message.content or "")
    except (AttributeError, IndexError, TypeError) as exc:
        raise ConfigMappingAIError("The mapping model returned an empty response.") from exc


def _validate_analysis_payload(payload: dict[str, Any]) -> ConfigMappingAnalysis:
    return ConfigMappingAnalysis.model_validate(payload)


def _model_dump(value: Any) -> dict[str, Any]:
    return value.model_dump()


def _current_value(current_system: Any, name: str) -> str:
    if current_system is None:
        return ""
    if isinstance(current_system, Mapping):
        return str(current_system.get(name) or "")
    return str(getattr(current_system, name, "") or "")


def _current_nested_entry(current_system: Any, group: str, key: str) -> dict[str, Any]:
    container = {}
    if isinstance(current_system, Mapping):
        container = current_system.get(group) or {}
    elif current_system is not None:
        container = getattr(current_system, group, {}) or {}
    return dict(container.get(key) or {}) if isinstance(container, Mapping) else {}


def _current_nested_label(current_system: Any, group: str, key: str) -> str:
    entry = _current_nested_entry(current_system, group, key)
    return str(entry.get("label") or "")


def _positive_float(value: Any) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value <= 0:
        return None
    return numeric_value


def _objective_pixel_size_extra(rule_entry: Mapping[str, Any], current_entry: Mapping[str, Any]) -> dict[str, float]:
    rule_pixel_size = _positive_float(rule_entry.get("pixel_size_um"))
    if rule_pixel_size is not None:
        return {"pixel_size_um": rule_pixel_size}
    current_pixel_size = _positive_float(current_entry.get("pixel_size_um"))
    if current_pixel_size is not None:
        return {"pixel_size_um": current_pixel_size}
    return {}


def _current_transmitted_light(current_system: Any, key: str) -> str:
    container = {}
    if isinstance(current_system, Mapping):
        container = current_system.get("transmitted_light") or {}
    elif current_system is not None:
        container = getattr(current_system, "transmitted_light", {}) or {}
    return str(dict(container).get(key) or "") if isinstance(container, Mapping) else ""


def _unique(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def _device_names(inventory: Mapping[str, Any]) -> list[str]:
    return _unique([str(device.get("name") or "") for device in inventory.get("devices", []) if isinstance(device, Mapping)])


def _state_labels(inventory: Mapping[str, Any], device_name: str) -> list[str]:
    for device in inventory.get("devices", []):
        if isinstance(device, Mapping) and str(device.get("name") or "") == device_name:
            return _unique([str(label) for label in device.get("state_labels") or []])
    return []


def _properties(inventory: Mapping[str, Any], device_name: str) -> list[str]:
    for device in inventory.get("devices", []):
        if isinstance(device, Mapping) and str(device.get("name") or "") == device_name:
            declared = [str(prop) for prop in device.get("properties") or []]
            return _unique(declared)
    return []


def _runtime_property_metadata(
    inventory: Mapping[str, Any],
    device_name: str,
) -> list[dict[str, Any]]:
    for device in inventory.get("devices", []):
        if isinstance(device, Mapping) and str(device.get("name") or "") == device_name:
            return [
                dict(item)
                for item in device.get("runtime_properties") or []
                if isinstance(item, Mapping) and str(item.get("name") or "").strip()
            ]
    return []


def _values_are_numeric(values: list[Any]) -> bool:
    if not values:
        return False
    try:
        for value in values:
            float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _is_runtime_numeric_control(metadata: Mapping[str, Any]) -> bool:
    if bool(metadata.get("read_only")) or bool(metadata.get("pre_init")):
        return False
    property_type = str(metadata.get("type") or "").lower()
    numeric_type = any(token in property_type for token in ("integer", "float", "double"))
    return bool(
        numeric_type
        or metadata.get("has_limits")
        or _values_are_numeric(list(metadata.get("allowed_values") or []))
    )


def _intensity_property_score(name: str) -> int:
    normalized = str(name or "").strip().lower()
    if not normalized or any(token in normalized for token in ("mode", "status", "enable", "description")):
        return 0
    for index, token in enumerate(_INTENSITY_TOKENS):
        if normalized == token:
            return 200 - index
    for index, token in enumerate(_INTENSITY_TOKENS):
        if token in normalized:
            return 100 - index
    return 0


def _intensity_properties(inventory: Mapping[str, Any], device_name: str) -> list[str]:
    runtime_metadata = _runtime_property_metadata(inventory, device_name)
    if runtime_metadata:
        candidates = [
            str(item.get("name") or "")
            for item in runtime_metadata
            if _intensity_property_score(str(item.get("name") or "")) > 0
            and _is_runtime_numeric_control(item)
        ]
        return sorted(
            _unique(candidates),
            key=lambda name: (-_intensity_property_score(name), name.lower()),
        )
    return [name for name in _properties(inventory, device_name) if _is_intensity_property(name)]


def _runtime_intensity_recommendation(
    inventory: Mapping[str, Any],
) -> tuple[str, str, str]:
    candidates: list[tuple[int, str, str]] = []
    device_tokens = ("transmitted", "illumination", "illuminator", "lamp", "light", "led")
    for device in inventory.get("devices", []):
        if not isinstance(device, Mapping):
            continue
        device_name = str(device.get("name") or "")
        if not _runtime_property_metadata(inventory, device_name):
            continue
        description = str(device.get("description") or "")
        device_text = f"{device_name} {description}".lower()
        device_score = 40 if any(token in device_text for token in device_tokens) else 0
        for property_name in _intensity_properties(inventory, device_name):
            score = _intensity_property_score(property_name) + device_score
            candidates.append((score, device_name, property_name))
    if not candidates:
        return "", "", ""
    candidates.sort(key=lambda item: (-item[0], item[1].lower(), item[2].lower()))
    top_score = candidates[0][0]
    top = [item for item in candidates if item[0] == top_score]
    if len(top) != 1:
        return "", "", "Multiple runtime intensity controls require AI or user selection."
    _score, device_name, property_name = top[0]
    return (
        device_name,
        property_name,
        "Detected as the strongest writable numeric intensity control in the pymmcore runtime inventory.",
    )


def _property_evidence(inventory: Mapping[str, Any], device_name: str, property_name: str) -> tuple[str, str]:
    for device in inventory.get("devices", []):
        if not isinstance(device, Mapping) or str(device.get("name") or "") != device_name:
            continue
        runtime_property = next(
            (
                item
                for item in device.get("runtime_properties") or []
                if isinstance(item, Mapping) and str(item.get("name") or "") == property_name
            ),
            None,
        )
        if runtime_property is not None:
            property_type = str(runtime_property.get("type") or "unknown")
            return (
                "runtime",
                f"pymmcore verified writable numeric property '{property_name}' with type {property_type}.",
            )
        if property_name in {str(prop) for prop in device.get("properties") or []}:
            return "cfg", f"cfg property '{property_name}' looks like an illumination intensity control."
    return "", ""


def _is_intensity_property(name: str) -> bool:
    return any(token in str(name or "").lower() for token in _INTENSITY_TOKENS)


def _draft_field(
    *,
    value: str = "",
    candidates: list[str] | tuple[str, ...] = (),
    source: str = "manual_required",
    confidence: str = "unknown",
    reason: str = "",
    needs_review: bool = True,
    rule_value: str = "",
    ai_value: str = "",
    current_value: str = "",
    extra: Mapping[str, Any] | None = None,
) -> ConfigMappingDraftField:
    confidence_value = confidence if confidence in _CONFIDENCE_VALUES else "unknown"
    payload: dict[str, Any] = {
        "value": str(value or ""),
        "candidates": _unique([str(candidate) for candidate in candidates]),
        "source": source,
        "confidence": confidence_value,
        "reason": reason,
        "needs_review": needs_review,
        "rule_value": str(rule_value or ""),
        "ai_value": str(ai_value or ""),
        "current_value": str(current_value or ""),
    }
    if extra:
        payload.update(dict(extra))
    return ConfigMappingDraftField(**payload)  # type: ignore[arg-type]


def _field_extra(field: ConfigMappingDraftField) -> dict[str, Any]:
    return {
        key: value
        for key, value in _model_dump(field).items()
        if key not in _DRAFT_FIELD_KEYS
    }


def _suggestion_value(inventory: Mapping[str, Any], field: str) -> str:
    suggestion = dict((inventory.get("suggestions") or {}).get(field) or {})
    return str(suggestion.get("value") or "")


def _suggestion_source(inventory: Mapping[str, Any], field: str) -> str:
    suggestion = dict((inventory.get("suggestions") or {}).get(field) or {})
    return str(suggestion.get("source") or "")


def _rule_mapping(inventory: Mapping[str, Any], group: str) -> dict[str, Any]:
    return dict((inventory.get("rule_mapping") or {}).get(group) or {})


def _build_role_fields(inventory: Mapping[str, Any], current_system: Any) -> dict[str, ConfigMappingDraftField]:
    device_candidates = _device_names(inventory)
    core_roles = dict(inventory.get("core_roles") or {})
    fields: dict[str, ConfigMappingDraftField] = {}
    for field in _ROLE_FIELDS:
        current = _current_value(current_system, field)
        if current not in set(device_candidates):
            current = ""
        core_role = _CORE_ROLE_BY_FIELD.get(field)
        core_value = str(core_roles.get(core_role) or "") if core_role else ""
        rule_value = core_value or _suggestion_value(inventory, field)
        if core_value:
            fields[field] = _draft_field(
                value=core_value,
                candidates=device_candidates,
                source="core",
                confidence="high",
                reason=f"Micro-Manager Core property '{core_role}' binds this role to '{core_value}'.",
                needs_review=False,
                rule_value=core_value,
                current_value=current,
            )
            continue

        if rule_value:
            fields[field] = _draft_field(
                value=rule_value,
                candidates=device_candidates,
                source="rule",
                confidence="medium",
                reason=_suggestion_source(inventory, field) or "Heuristic cfg match.",
                needs_review=True,
                rule_value=rule_value,
                current_value=current,
            )
            continue

        fields[field] = _draft_field(
            value=current,
            candidates=device_candidates,
            source="current_config" if current else "manual_required",
            confidence="unknown",
            reason="No clear cfg match was found; select a device from the cfg inventory.",
            needs_review=True,
            current_value=current,
        )
    return fields


def _build_objective_drafts(
    inventory: Mapping[str, Any],
    current_system: Any,
    objective_device: str,
) -> dict[str, ConfigMappingDraftField]:
    labels = _state_labels(inventory, objective_device)
    rule_objective_device = _suggestion_value(inventory, "objective_device")
    rule_objectives = _rule_mapping(inventory, "objectives") if objective_device == rule_objective_device else {}
    drafts: dict[str, ConfigMappingDraftField] = {}
    for key in _FORM_OBJECTIVE_KEYS:
        rule_entry = dict(rule_objectives.get(key) or {})
        rule_label = str(rule_entry.get("label") or "")
        current_entry = _current_nested_entry(current_system, "objectives", key)
        current = str(current_entry.get("label") or "")
        if current not in set(labels):
            current = ""
        extra = _objective_pixel_size_extra(rule_entry, current_entry)
        if rule_label and rule_label in labels:
            drafts[key] = _draft_field(
                value=rule_label,
                candidates=labels,
                source="rule",
                confidence="high",
                reason=f"cfg label '{rule_label}' contains clear magnification evidence for {key}.",
                needs_review=False,
                rule_value=rule_label,
                current_value=current,
                extra=extra,
            )
        else:
            value = current if current in labels else ""
            drafts[key] = _draft_field(
                value=value,
                candidates=labels,
                source="current_config" if value else "manual_required",
                confidence="unknown",
                reason=f"No clear cfg label maps to EIMS objective key '{key}'.",
                needs_review=True,
                current_value=current,
                extra=extra,
            )
    return drafts


def _build_channel_drafts(
    inventory: Mapping[str, Any],
    current_system: Any,
    channel_device: str,
) -> dict[str, ConfigMappingDraftField]:
    labels = _state_labels(inventory, channel_device)
    rule_channel_device = _suggestion_value(inventory, "Dichroic")
    rule_channels = _rule_mapping(inventory, "channels") if channel_device == rule_channel_device else {}
    drafts: dict[str, ConfigMappingDraftField] = {}
    for key in _FORM_CHANNEL_KEYS:
        rule_label = str(dict(rule_channels.get(key) or {}).get("label") or "")
        current = _current_nested_label(current_system, "channels", key)
        if current not in set(labels):
            current = ""
        if rule_label and rule_label in labels:
            drafts[key] = _draft_field(
                value=rule_label,
                candidates=labels,
                source="rule",
                confidence="high",
                reason=f"cfg label '{rule_label}' contains clear channel evidence for {key}.",
                needs_review=False,
                rule_value=rule_label,
                current_value=current,
            )
        else:
            value = current if current in labels else ""
            drafts[key] = _draft_field(
                value=value,
                candidates=labels,
                source="current_config" if value else "manual_required",
                confidence="unknown",
                reason=f"No clear cfg label maps to EIMS channel key '{key}'.",
                needs_review=True,
                current_value=current,
            )
    return drafts


def _build_transmitted_light_draft(
    inventory: Mapping[str, Any],
    current_system: Any,
) -> dict[str, Any]:
    rule_light = _rule_mapping(inventory, "transmitted_light")
    rule_device = str(rule_light.get("device") or "")
    runtime_device, runtime_property, runtime_reason = _runtime_intensity_recommendation(inventory)
    rule_device_has_intensity = bool(_intensity_properties(inventory, rule_device))
    current_device = _current_transmitted_light(current_system, "device")
    if current_device not in set(_device_names(inventory)):
        current_device = ""
    light_device = (
        rule_device
        if rule_device and rule_device_has_intensity
        else (runtime_device or rule_device or current_device)
    )
    rule_property = str(rule_light.get("intensity_property") or "") if light_device == rule_device else ""
    runtime_property = runtime_property if light_device == runtime_device else ""
    current_property = _current_transmitted_light(current_system, "intensity_property")
    properties = _intensity_properties(inventory, light_device)
    if current_property not in set(properties):
        current_property = ""
    property_value = (
        rule_property
        if rule_property in properties
        else (
            runtime_property
            if runtime_property in properties
            else (current_property if current_property in properties else "")
        )
    )
    source = (
        "rule"
        if property_value and property_value == rule_property
        else (
            "runtime"
            if property_value and property_value == runtime_property
            else ("current_config" if property_value else "manual_required")
        )
    )
    property_evidence, property_reason = _property_evidence(inventory, light_device, property_value)
    confidence = (
        "high"
        if property_evidence == "runtime"
        else (
            "high"
            if source == "rule" and property_evidence == "cfg" and _is_intensity_property(property_value)
            else ("medium" if source == "rule" and property_evidence else "unknown")
        )
    )
    return {
        "device": _draft_field(
            value=light_device,
            candidates=_device_names(inventory),
            source=(
                "runtime"
                if light_device == runtime_device and light_device
                else (
                "rule"
                if light_device == rule_device and light_device
                else ("current_config" if light_device else "manual_required")
                )
            ),
            confidence="high" if light_device == runtime_device and runtime_property else ("medium" if light_device else "unknown"),
            reason=(
                runtime_reason
                if light_device == runtime_device and runtime_reason
                else "Selected transmitted-light control device."
                if light_device
                else "Optional. Select a cfg device only if EIMS should control transmitted-light intensity."
            ),
            needs_review=not bool(light_device == runtime_device and runtime_property),
            rule_value=rule_device,
            current_value=current_device,
        ),
        "intensity_property": _draft_field(
            value=property_value,
            candidates=properties,
            source=source,
            confidence=confidence,
            reason=(
                property_reason
                if property_value and property_reason
                else "Optional. Leave empty to disable EIMS brightness control, or enter the Micro-Manager intensity property if this device exposes one."
            ),
            needs_review=source not in {"rule", "runtime"} or property_evidence not in {"cfg", "runtime"},
            rule_value=rule_property,
            current_value=current_property,
        ),
        "min": int(rule_light.get("min", 0) or 0),
        "max": int(rule_light.get("max", 250) or 250),
    }


def _build_rule_analysis(inventory: Mapping[str, Any], current_system: Any) -> ConfigMappingAnalysis:
    fields = _build_role_fields(inventory, current_system)
    objectives = _build_objective_drafts(inventory, current_system, fields["objective_device"].value)
    channels = _build_channel_drafts(inventory, current_system, fields["Dichroic"].value)
    transmitted_light = _build_transmitted_light_draft(inventory, current_system)
    warnings = [str(warning) for warning in inventory.get("warnings") or []]
    warnings.extend([
        f"The cfg does not identify a {field}; select it manually."
        for field in inventory.get("unresolved_fields") or []
    ])
    return ConfigMappingAnalysis(
        ai_status="not_configured",
        fields=fields,
        objectives=objectives,
        channels=channels,
        transmitted_light=transmitted_light,
        warnings=warnings,
    )


def _field_from_ai(value: Any) -> ConfigMappingDraftField:
    if isinstance(value, ConfigMappingDraftField):
        return value
    if isinstance(value, Mapping):
        return ConfigMappingDraftField.model_validate(dict(value))
    return ConfigMappingDraftField(value=str(value or ""), source="ai")


def _replace_with_ai(
    field: ConfigMappingDraftField,
    ai_field: ConfigMappingDraftField,
    *,
    candidates: list[str],
) -> ConfigMappingDraftField:
    confidence = ai_field.confidence if ai_field.confidence in _CONFIDENCE_VALUES else "unknown"
    return _draft_field(
        value=ai_field.value,
        candidates=candidates,
        source="ai",
        confidence=confidence,
        reason=ai_field.reason or "AI recommended this value from the parsed cfg inventory.",
        needs_review=True,
        rule_value=field.rule_value,
        ai_value=ai_field.value,
        current_value=field.current_value,
        extra=_field_extra(field),
    )


def _merge_valid_ai_value(
    field: ConfigMappingDraftField,
    ai_field: ConfigMappingDraftField,
    *,
    candidates: list[str],
) -> tuple[ConfigMappingDraftField, str]:
    """Merge an AI suggestion into a draft field.

    Verified facts (source core/runtime/rule with needs_review=False) are
    authoritative and cannot be replaced by AI; AI only fills empty or
    ambiguous fields. Returns (merged_field, warning_text).
    """
    ai_value = str(ai_field.value or "").strip()
    if field.source in {"core", "runtime", "rule"} and not field.needs_review:
        if ai_value and ai_value != field.value:
            return field, (
                f"AI suggestion '{ai_value}' was ignored because '{field.value}' "
                f"is already verified by {field.source}."
            )
        if ai_value == field.value:
            return (
                _draft_field(
                    value=field.value,
                    candidates=candidates,
                    source=field.source,
                    confidence=field.confidence,
                    reason=field.reason,
                    needs_review=False,
                    rule_value=field.rule_value,
                    ai_value=ai_value,
                    current_value=field.current_value,
                    extra=_field_extra(field),
                ),
                "",
            )
        return field, ""
    return _replace_with_ai(field, ai_field, candidates=candidates), ""


def _merge_ai_analysis(
    base: ConfigMappingAnalysis,
    ai: ConfigMappingAnalysis,
    inventory: Mapping[str, Any],
    current_system: Any,
) -> ConfigMappingAnalysis:
    warnings = list(base.warnings) + list(ai.warnings)
    device_names = set(_device_names(inventory))
    fields = {key: _field_from_ai(_model_dump(value)) for key, value in base.fields.items()}

    for key, ai_value in ai.fields.items():
        if key not in _ROLE_FIELDS:
            warnings.append(f"AI field suggestion '{key}' was ignored because it is not an EIMS mapping field.")
            continue
        ai_field = _field_from_ai(ai_value)
        if not ai_field.value:
            continue
        base_field = fields[key]
        if ai_field.value not in device_names:
            warnings.append(f"AI suggestion for '{key}' was ignored because '{ai_field.value}' is not a cfg device.")
            continue
        merged_field, merge_warning = _merge_valid_ai_value(base_field, ai_field, candidates=_device_names(inventory))
        fields[key] = merged_field
        if merge_warning:
            warnings.append(merge_warning)

    objectives = _build_objective_drafts(inventory, current_system, fields["objective_device"].value)
    objective_labels = set(_state_labels(inventory, fields["objective_device"].value))
    for key, ai_value in ai.objectives.items():
        if key not in _FORM_OBJECTIVE_KEYS:
            warnings.append(f"AI objective suggestion '{key}' was ignored because EIMS does not use that objective key.")
            continue
        ai_field = _field_from_ai(ai_value)
        if not ai_field.value:
            continue
        if ai_field.value not in objective_labels:
            warnings.append(f"AI objective suggestion '{key}' was ignored because '{ai_field.value}' is not a label on '{fields['objective_device'].value}'.")
            continue
        merged_field, merge_warning = _merge_valid_ai_value(
            objectives[key],
            ai_field,
            candidates=_state_labels(inventory, fields["objective_device"].value),
        )
        objectives[key] = merged_field
        if merge_warning:
            warnings.append(merge_warning)

    channels = _build_channel_drafts(inventory, current_system, fields["Dichroic"].value)
    channel_labels = set(_state_labels(inventory, fields["Dichroic"].value))
    for key, ai_value in ai.channels.items():
        if key not in _FORM_CHANNEL_KEYS:
            warnings.append(f"AI channel suggestion '{key}' was ignored because EIMS does not use that channel key.")
            continue
        ai_field = _field_from_ai(ai_value)
        if not ai_field.value:
            continue
        if ai_field.value not in channel_labels:
            warnings.append(f"AI channel suggestion '{key}' was ignored because '{ai_field.value}' is not a label on '{fields['Dichroic'].value}'.")
            continue
        merged_field, merge_warning = _merge_valid_ai_value(
            channels[key],
            ai_field,
            candidates=_state_labels(inventory, fields["Dichroic"].value),
        )
        channels[key] = merged_field
        if merge_warning:
            warnings.append(merge_warning)

    transmitted_light = _build_transmitted_light_draft(inventory, current_system)
    ai_light = dict(ai.transmitted_light or {})
    ai_light_device = ai_light.get("device")
    if ai_light_device is not None:
        ai_field = _field_from_ai(ai_light_device)
        if ai_field.value and ai_field.value in device_names:
            base_device_field = _field_from_ai(transmitted_light["device"])
            merged_device, device_warning = _merge_valid_ai_value(
                base_device_field,
                ai_field,
                candidates=_device_names(inventory),
            )
            if device_warning:
                warnings.append(device_warning)
            transmitted_light["device"] = merged_device
            selected_device = _field_from_ai(transmitted_light["device"]).value
            allowed_properties = _intensity_properties(inventory, selected_device)
            base_property_field = _field_from_ai(transmitted_light["intensity_property"])
            if selected_device != base_device_field.value or base_property_field.value not in set(allowed_properties):
                # Device changed (or the base property is not valid for it):
                # re-derive the intensity property for the selected device.
                current_property = _current_transmitted_light(current_system, "intensity_property")
                if current_property not in set(allowed_properties):
                    current_property = ""
                transmitted_light["intensity_property"] = _draft_field(
                    value=current_property if current_property in allowed_properties else "",
                    candidates=allowed_properties,
                    source="current_config" if current_property in allowed_properties else "manual_required",
                    confidence="unknown",
                    reason="Optional. Select a verified writable numeric intensity property to enable EIMS brightness control.",
                    needs_review=True,
                    current_value=current_property,
                )
            # Device unchanged and the rule/runtime property draft is still
            # valid for it: preserve the verified property instead of wiping it.
        elif ai_field.value:
            warnings.append(f"AI transmitted-light device was ignored because '{ai_field.value}' is not a cfg device.")

    ai_property = ai_light.get("intensity_property")
    if ai_property is not None:
        ai_field = _field_from_ai(ai_property)
        selected_device = _field_from_ai(transmitted_light.get("device", {})).value
        allowed_properties = _intensity_properties(inventory, selected_device)
        allowed = set(allowed_properties)
        if ai_field.value and ai_field.value in allowed:
            merged_property, property_warning = _merge_valid_ai_value(
                transmitted_light["intensity_property"],
                ai_field,
                candidates=allowed_properties,
            )
            transmitted_light["intensity_property"] = merged_property
            if property_warning:
                warnings.append(property_warning)
        elif ai_field.value:
            warnings.append(
                f"AI transmitted-light property was ignored because '{ai_field.value}' is not a verified writable numeric intensity property on '{selected_device}'."
            )

    return ConfigMappingAnalysis(
        ai_status="completed",
        fields=fields,
        objectives=objectives,
        channels=channels,
        transmitted_light=transmitted_light,
        warnings=warnings,
    )


def analyze_config_mapping(
    *,
    inventory: Mapping[str, Any],
    model_config: Any,
    current_system: Any = None,
) -> ConfigMappingAnalysis:
    """Build a cfg-backed mapping draft and let AI recommend ambiguous values."""
    base = _build_rule_analysis(inventory, current_system)
    api_key = str(getattr(model_config, "openai_api_key", "") or "").strip()
    model_name = str(getattr(model_config, "model_name", "") or "").strip()
    if not api_key or not model_name:
        return ConfigMappingAnalysis(
            ai_status="not_configured",
            fields=base.fields,
            objectives=base.objectives,
            channels=base.channels,
            transmitted_light=base.transmitted_light,
            warnings=base.warnings + ["AI mapping is unavailable until the main LLM API key and model name are configured."],
        )

    prompt_payload = {
        "cfg_inventory": dict(inventory),
        "current_eims_mapping": {
            field: base.fields[field].current_value for field in _ROLE_FIELDS
        },
        "parser_draft": _model_dump(base),
        "stable_semantic_keys": {
            "objectives": list(_FORM_OBJECTIVE_KEYS),
            "channels": list(_FORM_CHANNEL_KEYS),
        },
    }
    instructions = """
Analyze a structured Micro-Manager hardware inventory and recommend an editable EIMS mapping draft.
The inventory may combine cfg facts with properties inspected from the loaded pymmcore Device Adapters.
It is authoritative: do not invent device names, state labels, properties, paths, or comments.
Core role bindings from Micro-Manager are facts. For non-Core fields, you may recommend any device from the cfg inventory,
even if it is not a parser candidate, when that corrects a likely parser mistake.
For device properties, use runtime metadata when present. Prefer writable, non-PreInit numeric controls and reject modes,
status fields, read-only properties, and non-numeric values for intensity control.
Objective pixel_size_um values are calibration facts supplied by official Micro-Manager cfg/API data or current EIMS config; do not infer or invent them.
Return JSON only with fields, objectives, channels, transmitted_light, and warnings.
Each recommended item must include value, confidence, and reason. Use confidence high, medium, low, or unknown.
Generic labels may still be recommended when useful, but explain that the user must verify them.
""".strip()

    def request_ai_mapping() -> ConfigMappingAnalysis:
        llm_client, _ = build_clients(model_config)
        completion = create_chat_completion(
            llm_client,
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            seed=getattr(model_config, "llm_seed", None),
            max_tokens=1800,
            retries=1,
        )
        return _validate_analysis_payload(_parse_json_response(_response_content(completion)))

    def unavailable_analysis(reason: str) -> ConfigMappingAnalysis:
        return ConfigMappingAnalysis(
            ai_status="unavailable",
            fields=base.fields,
            objectives=base.objectives,
            channels=base.channels,
            transmitted_light=base.transmitted_light,
            warnings=base.warnings
            + [f"AI mapping was unavailable; parser candidates were used instead: {reason}"],
        )

    try:
        ai = request_ai_mapping()
    except ConfigMappingAIError as exc:
        logger.warning("AI cfg mapping returned malformed output; retrying once: %s", exc)
        try:
            ai = request_ai_mapping()
        except ConfigMappingAIError as retry_exc:
            logger.error("AI cfg mapping returned malformed output on retry: %s", retry_exc)
            return unavailable_analysis("the model returned an invalid response format after a retry.")
        except Exception as retry_exc:
            logger.error("AI cfg mapping retry failed: %s", retry_exc)
            return unavailable_analysis("the model request failed on retry.")
    except Exception as exc:
        logger.error("AI cfg mapping request failed: %s", exc)
        return unavailable_analysis("the model request failed.")
    return _merge_ai_analysis(base, ai, inventory, current_system)
