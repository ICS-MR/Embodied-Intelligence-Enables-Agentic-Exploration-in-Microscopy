import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from dotenv import dotenv_values

from bootstrap.microscope_semantics import (
    derived_dichroic_colors,
    derived_objective_labels,
    resolve_channel_input,
    resolve_objective_input,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = ROOT_DIR / "config" / "runtime_config.json"
ENV_PATH = ROOT_DIR / ".env"
DEMO_CONFIG_PATH = ROOT_DIR / "demo_cfg" / "MMConfig_demo.cfg"


DEFAULT_OBJECTIVE_LABELS: Dict[str, int] = {
    "6-UPLSAPO30XS": 30,
    "5-LUCPLFLN60X": 60,
    "4-LUCPLFLN40X": 40,
    "3-LUCPLFLN20XRC": 20,
    "2-SOB": 10,
    "1-UPLFLN4XPH": 4,
}

DEFAULT_DICHROIC_COLORS: Dict[str, Tuple[int, int, int]] = {
    "8-IX3-FDICT": (128, 128, 128),
    "7-NONE": (128, 128, 128),
    "6-NONE": (128, 128, 128),
    "5-NONE": (128, 128, 128),
    "4-U-FGNA": (255, 0, 0),
    "3-U-FBNA": (0, 255, 0),
    "2-U-FUNA": (0, 0, 255),
    "1-NONE": (128, 128, 128),
}

DEFAULT_OBJECTIVES: Dict[str, Dict[str, Any]] = {
    "4x": {"label": "1-UPLFLN4XPH", "magnification": 4, "display_name": "4x objective"},
    "10x": {"label": "2-SOB", "magnification": 10, "display_name": "10x objective"},
    "20x": {"label": "3-LUCPLFLN20XRC", "magnification": 20, "display_name": "20x objective"},
    "30x": {"label": "6-UPLSAPO30XS", "magnification": 30, "display_name": "30x objective"},
    "40x": {"label": "4-LUCPLFLN40X", "magnification": 40, "display_name": "40x objective"},
    "60x": {"label": "5-LUCPLFLN60X", "magnification": 60, "display_name": "60x objective"},
}

DEFAULT_CHANNELS: Dict[str, Dict[str, Any]] = {
    "brightfield": {
        "label": "1-NONE",
        "display_name": "Brightfield",
        "color": [128, 128, 128],
        "illumination": "transmitted",
    },
    "dapi": {
        "label": "2-U-FUNA",
        "display_name": "DAPI / 405 nm",
        "color": [0, 0, 255],
        "illumination": "fluorescence",
    },
    "fitc": {
        "label": "3-U-FBNA",
        "display_name": "FITC / 488 nm",
        "color": [0, 255, 0],
        "illumination": "fluorescence",
    },
    "tritc": {
        "label": "4-U-FGNA",
        "display_name": "TRITC / 640 nm",
        "color": [255, 0, 0],
        "illumination": "fluorescence",
    },
}

DEFAULT_TRANSMITTED_LIGHT: Dict[str, Any] = {
    "device": "",
    "intensity_property": "",
    "min": 0,
    "max": 250,
}
PUBLIC_TRANSMITTED_LIGHT_FIELDS = ("device", "intensity_property", "min", "max")

DEMO_OBJECTIVES: Dict[str, Dict[str, Any]] = {
    "4x": {"label": "1-UPLFLN4XPH", "magnification": 4, "display_name": "4x objective"},
    "10x": {"label": "2-SOB", "magnification": 10, "display_name": "10x objective"},
    "20x": {"label": "3-LUCPLFLN20XRC", "magnification": 20, "display_name": "20x objective"},
    "30x": {"label": "6-UPLSAPO30XS", "magnification": 30, "display_name": "30x objective"},
    "40x": {"label": "4-LUCPLFLN40X", "magnification": 40, "display_name": "40x objective"},
    "60x": {"label": "5-LUCPLFLN60X", "magnification": 60, "display_name": "60x objective"},
}

DEMO_CHANNELS: Dict[str, Dict[str, Any]] = {
    "brightfield": {
        "label": "1-NONE",
        "display_name": "Brightfield",
        "color": [128, 128, 128],
        "illumination": "transmitted",
    },
    "dapi": {
        "label": "2-U-FUNA",
        "display_name": "DAPI / 405 nm",
        "color": [0, 0, 255],
        "illumination": "fluorescence",
    },
    "fitc": {
        "label": "3-U-FBNA",
        "display_name": "FITC / 488 nm",
        "color": [0, 255, 0],
        "illumination": "fluorescence",
    },
    "tritc": {
        "label": "4-U-FGNA",
        "display_name": "TRITC / 640 nm",
        "color": [255, 0, 0],
        "illumination": "fluorescence",
    },
}

DEMO_TRANSMITTED_LIGHT: Dict[str, Any] = {
    "device": "DCam",
    "intensity_property": "BeadBrightness",
    "min": 0,
    "max": 250,
    "control_kind": "demo_camera_bead_brightness",
    "surrogate_min_property_value": 0.5,
    "surrogate_scale": 100.0,
}

DEFAULT_DETECTION_TARGETS: Dict[str, Dict[str, Any]] = {
    "2Dcell": {
        "target_class_id": 0,
        "target_class_name": "2Dcell",
        "score_thr": 0.2,
        "output_filename": "2Dcell_locations_list.json",
        "model_config": "detector_models/cell2d/config.py",
        "model_checkpoint": "detector_models/cell2d/weights.pth",
    },
    "organoid": {
        "target_class_id": 0,
        "target_class_name": "organoid",
        "score_thr": 0.2,
        "output_filename": "organoid_locations_list.json",
        "model_config": "detector_models/organoid/config.py",
        "model_checkpoint": "detector_models/organoid/weights.pth",
    },
    "mitosis": {
        "target_class_id": 0,
        "target_class_name": "mitosis",
        "score_thr": 0.2,
        "output_filename": "mitosis_locations_list.json",
        "model_config": "detector_models/mitosis/config.py",
        "model_checkpoint": "detector_models/mitosis/weights.pth",
    },
}


DEFAULT_KNOWLEDGE_BASE_PATH = "docs_public/c3_knowledge_base/knowledge_base_reviewed.json"


@dataclass
class StartupConfig:
    objective: str = "40x"
    channel: str = "brightfield"
    exposure: float = 10.0
    brightness: int = 100
    z_position: float = 4100.0
    x_position: float = 50000.0
    y_position: float = 50000.0
    start_preview: bool = True


@dataclass(frozen=True)
class TaskRuntimeConfig:
    HISTORY_DIR: str = "history"
    OUTPUT_DIR: str = "output"
    MAX_RETRY_TIMES: int = 3
    RETRY_INTERVAL: int = 3


@dataclass
class SystemConfig:
    MM_DIR: str = r""
    CONFIG_PATH: str = str(DEMO_CONFIG_PATH)
    FIJI_PATH: str = r""
    MAVEN_BIN: str = r""
    camera_device: str = ""
    xy_stage_device: str = ""
    objective_device: str = ""
    focus_drive: str = ""
    Dichroic: str = ""
    objective_labels: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE_LABELS))
    dichroic_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: dict(DEFAULT_DICHROIC_COLORS))
    objectives: Dict[str, Dict[str, Any]] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_OBJECTIVES)))
    channels: Dict[str, Dict[str, Any]] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CHANNELS)))
    transmitted_light: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TRANSMITTED_LIGHT))
    Max_X_position: float = 100000.0
    Min_X_position: float = 0.0
    Max_Y_position: float = 70000.0
    Min_Y_position: float = 0.0
    Max_Z_position: float = 10000.0
    Min_Z_position: float = 0.0
    Max_brightness: int = 250
    Min_brightness: int = 0
    Max_exposure: int = 1000
    Min_exposure: int = 0
    PSF_40X: str = "PSF/40x.tif"
    PSF_60X: str = "PSF/60x.tif"
    PSF_100X: str = "PSF/100x.tif"
    in_process_executor_timeout_seconds: float = 180.0
    fiji_executor_timeout_seconds: float = 300.0


@dataclass
class ModelConfig:
    microscope_mode: str = "demo"
    image_analysis_mode: str = "mock"
    segmentation_mode: str = "mock"
    clarify_enabled: bool = False
    checker_enabled: bool = False
    skill_mode: str = "disabled"
    llm_seed: int | None = 42
    openai_api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4.1"
    vlm_api_key: str = ""
    vlm_base_url: str = "https://api.openai.com/v1"
    vlm_model_name: str = "gpt-4.1"
    CROSS_ENCODER_MODEL_PATH: str = r"embedding_model\bge-m3"
    # C3 conformal threshold calibrated from docs_public/c3_calibration/calibration_overview.json.
    # Keep this value aligned with that calibration set's selected_threshold.
    task_similarity_threshold: float = 0.029
    knowledge_base_path: str = DEFAULT_KNOWLEDGE_BASE_PATH

@dataclass
class RuntimeSettings:
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    detection_targets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_DETECTION_TARGETS)))


def build_demo_system_overrides() -> Dict[str, Any]:
    return {
        "CONFIG_PATH": str(DEMO_CONFIG_PATH),
        "camera_device": "DCam",
        "xy_stage_device": "DXYStage",
        "objective_device": "DObjective",
        "focus_drive": "DStage",
        "Dichroic": "DStateDevice",
        "objectives": json.loads(json.dumps(DEMO_OBJECTIVES)),
        "channels": json.loads(json.dumps(DEMO_CHANNELS)),
        "transmitted_light": dict(DEMO_TRANSMITTED_LIGHT),
        "Min_X_position": 0.0,
        "Max_X_position": 100000.0,
        "Min_Y_position": 0.0,
        "Max_Y_position": 70000.0,
        "Min_Z_position": -300.0,
        "Max_Z_position": 300.0,
        "Min_brightness": 0,
        "Max_brightness": 250,
        "Min_exposure": 0,
        "Max_exposure": 1000,
    }


def build_demo_startup_overrides() -> Dict[str, Any]:
    return {
        "objective": "40x",
        "channel": "brightfield",
        "exposure": 10.0,
        "brightness": 100,
        "z_position": 0.0,
        "x_position": 50000.0,
        "y_position": 50000.0,
        "start_preview": True,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON config %s: %s; falling back to defaults.", path, exc)
        return {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_mode(value: Any, *, allowed: tuple[str, ...], default: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _coerce_color_map(value: Mapping[str, Any], fallback: Mapping[str, Tuple[int, int, int]]) -> Dict[str, Tuple[int, int, int]]:
    if not isinstance(value, Mapping):
        return dict(fallback)
    result: Dict[str, Tuple[int, int, int]] = {}
    for key, item in value.items():
        if isinstance(item, (list, tuple)) and len(item) == 3:
            result[str(key)] = (int(item[0]), int(item[1]), int(item[2]))
    return result or dict(fallback)


def _coerce_nested_mapping(value: Any, fallback: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {str(key): dict(item) for key, item in fallback.items()}
    result: Dict[str, Dict[str, Any]] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[str(key).strip().lower()] = dict(item)
    return result or {str(key): dict(item) for key, item in fallback.items()}


def _update_dataclass(instance: Any, updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if isinstance(current, bool):
            setattr(instance, key, _coerce_bool(value, current))
        elif key == "dichroic_colors":
            setattr(instance, key, _coerce_color_map(value, current))
        elif key == "objective_labels" and isinstance(value, Mapping):
            setattr(instance, key, {str(k): int(v) for k, v in value.items()})
        elif key in {"objectives", "channels"}:
            setattr(instance, key, _coerce_nested_mapping(value, current))
        elif key == "transmitted_light" and isinstance(value, Mapping):
            merged = dict(current)
            merged.update(dict(value))
            setattr(instance, key, merged)
        elif key == "microscope_mode":
            setattr(instance, key, _coerce_mode(value, allowed=("demo", "real"), default=current))
        elif key in {"image_analysis_mode", "segmentation_mode"}:
            setattr(instance, key, _coerce_mode(value, allowed=("real", "mock"), default=current))
        else:
            setattr(instance, key, value)


def _normalize_system_semantics(system_config: SystemConfig) -> None:
    if not system_config.objectives:
        system_config.objectives = _objectives_from_legacy_labels(system_config.objective_labels)
    if not system_config.channels:
        system_config.channels = _channels_from_legacy_colors(system_config.dichroic_colors)

    # Legacy maps are internal derived compatibility fields. The structured
    # semantic maps are the only mapping source users should maintain.
    system_config.objective_labels = derived_objective_labels(system_config)
    system_config.dichroic_colors = derived_dichroic_colors(system_config)


def _normalize_startup_semantics(settings: RuntimeSettings) -> None:
    _label, objective_key, _entry = resolve_objective_input(
        settings.startup.objective,
        settings.system,
    )
    if objective_key:
        settings.startup.objective = objective_key

    _label, channel_key, _entry = resolve_channel_input(
        settings.startup.channel,
        settings.system,
    )
    if channel_key:
        settings.startup.channel = channel_key


def _apply_demo_system_overrides(system_config: SystemConfig) -> None:
    _update_dataclass(system_config, build_demo_system_overrides())
    _normalize_system_semantics(system_config)


def _apply_demo_startup_overrides(startup_config: StartupConfig) -> None:
    _update_dataclass(startup_config, build_demo_startup_overrides())


def _objectives_from_legacy_labels(objective_labels: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    objectives: Dict[str, Dict[str, Any]] = {}
    for label, magnification in objective_labels.items():
        try:
            mag = int(magnification)
        except (TypeError, ValueError):
            continue
        key = f"{mag}x"
        objectives[key] = {
            "label": str(label),
            "magnification": mag,
            "display_name": f"{mag}x objective",
        }
    return objectives or json.loads(json.dumps(DEFAULT_OBJECTIVES))


def _channels_from_legacy_colors(dichroic_colors: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    channels = json.loads(json.dumps(DEFAULT_CHANNELS))
    for key, item in channels.items():
        label = item.get("label")
        if label in dichroic_colors:
            item["color"] = list(dichroic_colors[label])
    return channels


def _apply_legacy_mode_migration(settings: RuntimeSettings, model_payload: Mapping[str, Any]) -> None:
    if any(key in model_payload for key in ("microscope_mode", "image_analysis_mode", "segmentation_mode")):
        return
    if "Simulation_mode" not in model_payload:
        return

    simulation_mode = _coerce_bool(model_payload.get("Simulation_mode"), True)
    if simulation_mode:
        settings.model.microscope_mode = "demo"
        settings.model.image_analysis_mode = "mock"
        settings.model.segmentation_mode = "mock"
    else:
        settings.model.microscope_mode = "real"
        settings.model.image_analysis_mode = "real"
        settings.model.segmentation_mode = "real"


def _apply_file_overrides(settings: RuntimeSettings, payload: Mapping[str, Any]) -> None:
    system_payload = payload.get("system", {})
    model_payload = payload.get("model", {})
    startup_payload = payload.get("startup", {})
    detection_payload = payload.get("detection_targets", {})
    if isinstance(system_payload, Mapping):
        system_updates = dict(system_payload)
        legacy_light_device = str(system_updates.pop("transmittedIllumination", "") or "").strip()
        transmitted_light = system_updates.get("transmitted_light")
        transmitted_light_updates = (
            {
                key: transmitted_light[key]
                for key in PUBLIC_TRANSMITTED_LIGHT_FIELDS
                if key in transmitted_light
            }
            if isinstance(transmitted_light, Mapping)
            else {}
        )
        if legacy_light_device and not str(transmitted_light_updates.get("device") or "").strip():
            transmitted_light_updates["device"] = legacy_light_device
        if transmitted_light_updates:
            system_updates["transmitted_light"] = transmitted_light_updates
        _update_dataclass(settings.system, system_updates)
        if "objective_labels" in system_payload and "objectives" not in system_payload:
            settings.system.objectives = _objectives_from_legacy_labels(settings.system.objective_labels)
        if "dichroic_colors" in system_payload and "channels" not in system_payload:
            settings.system.channels = _channels_from_legacy_colors(settings.system.dichroic_colors)
    if isinstance(model_payload, Mapping):
        _apply_legacy_mode_migration(settings, model_payload)
        _update_dataclass(settings.model, model_payload)
    if isinstance(startup_payload, Mapping):
        _update_dataclass(settings.startup, startup_payload)
    if isinstance(detection_payload, Mapping):
        settings.detection_targets = _merge_detection_targets(settings.detection_targets, detection_payload)


def _merge_detection_targets(
    defaults: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {
        str(key): dict(value)
        for key, value in defaults.items()
        if isinstance(value, Mapping)
    }
    for key, value in overrides.items():
        if not isinstance(value, Mapping):
            continue
        target_key = str(key)
        base = dict(merged.get(target_key, {}))
        base.update(dict(value))
        merged[target_key] = base
    return merged


def _apply_env_overrides(settings: RuntimeSettings, env_values: Mapping[str, str]) -> None:
    model_env_map = {
        "llm_seed": ("EIMS_LLM_SEED",),
        "openai_api_key": ("EIMS_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "skill_mode": ("EIMS_SKILL_MODE",),
        "vlm_api_key": ("EIMS_VLM_API_KEY", "VLM_API_KEY"),
    }
    for field_name, env_names in model_env_map.items():
        for env_name in env_names:
            value = env_values.get(env_name)
            if value:
                setattr(settings.model, field_name, value)
                break

    checker_env = env_values.get("EIMS_CHECKER_ENABLED")
    if checker_env is not None:
        settings.model.checker_enabled = _coerce_bool(checker_env, settings.model.checker_enabled)


def is_demo_mode_settings(settings: RuntimeSettings) -> bool:
    return str(getattr(settings.model, "microscope_mode", "demo")).strip().lower() == "demo"


def is_demo_mode_snapshot(snapshot: Mapping[str, Any]) -> bool:
    agent_cfg = snapshot.get("agent", {}) if isinstance(snapshot, Mapping) else {}
    return str(agent_cfg.get("microscope_mode", "demo")).strip().lower() == "demo"


def _normalized_demo_mapping_payload() -> Dict[str, Any]:
    return {
        "CONFIG_PATH": str(DEMO_CONFIG_PATH),
        "camera_device": "DCam",
        "xy_stage_device": "DXYStage",
        "objective_device": "DObjective",
        "focus_drive": "DStage",
        "Dichroic": "DStateDevice",
        "objectives": json.loads(json.dumps(DEMO_OBJECTIVES)),
        "channels": json.loads(json.dumps(DEMO_CHANNELS)),
        "transmitted_light": dict(DEMO_TRANSMITTED_LIGHT),
    }


def is_demo_mapping_payload(
    *,
    config_path: str,
    camera_device: str,
    xy_stage_device: str,
    objective_device: str,
    focus_drive: str,
    dichroic: str,
    objectives: Mapping[str, Any],
    channels: Mapping[str, Any],
    transmitted_light: Mapping[str, Any],
) -> bool:
    demo = _normalized_demo_mapping_payload()
    return (
        str(config_path).strip() == str(demo["CONFIG_PATH"]).strip()
        and str(camera_device).strip() == str(demo["camera_device"]).strip()
        and str(xy_stage_device).strip() == str(demo["xy_stage_device"]).strip()
        and str(objective_device).strip() == str(demo["objective_device"]).strip()
        and str(focus_drive).strip() == str(demo["focus_drive"]).strip()
        and str(dichroic).strip() == str(demo["Dichroic"]).strip()
        and dict(objectives or {}) == dict(demo["objectives"])
        and dict(channels or {}) == dict(demo["channels"])
        and dict(transmitted_light or {}) == dict(demo["transmitted_light"])
    )


def _load_env_values(*, include_dotenv: bool) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if include_dotenv:
        for key, value in dotenv_values(ROOT_DIR / ".env").items():
            if value is not None:
                merged[key] = value
    for key, value in os.environ.items():
        merged[key] = value
    return merged


def save_env_secrets(*, openai_api_key: str | None = None, vlm_api_key: str | None = None, env_path: Optional[Path] = None) -> None:
    target_path = env_path or ENV_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    updates = {
        "EIMS_OPENAI_API_KEY": openai_api_key,
        "EIMS_VLM_API_KEY": vlm_api_key,
    }
    pending = {key: value for key, value in updates.items() if value is not None}
    if not pending:
        return

    existing_lines: list[str] = []
    if target_path.exists():
        existing_lines = target_path.read_text(encoding="utf-8").splitlines()

    remaining = dict(pending)
    rewritten_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        replaced = False
        if stripped and not stripped.startswith("#") and "=" in line:
            key, _sep, _value = line.partition("=")
            env_key = key.strip()
            if env_key in remaining:
                rewritten_lines.append(f"{env_key}={remaining.pop(env_key)}")
                replaced = True
        if not replaced:
            rewritten_lines.append(line)

    for env_key, env_value in remaining.items():
        rewritten_lines.append(f"{env_key}={env_value}")

    target_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def load_runtime_settings(
    config_path: Optional[Path] = None,
    *,
    apply_env: bool = True,
    apply_demo_overlay: bool = True,
) -> RuntimeSettings:
    settings = RuntimeSettings()
    target_path = config_path or RUNTIME_CONFIG_PATH
    payload = _read_json(target_path)
    _apply_file_overrides(settings, payload)
    _normalize_system_semantics(settings.system)
    _normalize_startup_semantics(settings)
    if apply_env:
        _apply_env_overrides(settings, _load_env_values(include_dotenv=target_path == RUNTIME_CONFIG_PATH))
        _normalize_system_semantics(settings.system)
        _normalize_startup_semantics(settings)
    if apply_demo_overlay and is_demo_mode_settings(settings):
        _apply_demo_system_overrides(settings.system)
        _apply_demo_startup_overrides(settings.startup)
    return settings


def _dataclass_dict(instance: Any) -> Dict[str, Any]:
    payload = asdict(instance)
    if "dichroic_colors" in payload:
        payload["dichroic_colors"] = {key: list(value) for key, value in payload["dichroic_colors"].items()}
    return payload


def _system_config_payload(system_config: SystemConfig) -> Dict[str, Any]:
    payload = _dataclass_dict(system_config)
    payload.pop("objective_labels", None)
    payload.pop("dichroic_colors", None)
    transmitted_light = payload.get("transmitted_light")
    if isinstance(transmitted_light, Mapping):
        payload["transmitted_light"] = {
            key: transmitted_light[key]
            for key in PUBLIC_TRANSMITTED_LIGHT_FIELDS
            if key in transmitted_light
        }
    return payload


def save_runtime_settings(
    system_updates: Optional[Mapping[str, Any]] = None,
    model_updates: Optional[Mapping[str, Any]] = None,
    startup_updates: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> RuntimeSettings:
    target_path = config_path or RUNTIME_CONFIG_PATH
    settings = load_runtime_settings(target_path, apply_env=False, apply_demo_overlay=False)
    if system_updates:
        _update_dataclass(settings.system, system_updates)
    if model_updates:
        _update_dataclass(settings.model, model_updates)
    if startup_updates:
        _update_dataclass(settings.startup, startup_updates)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _normalize_system_semantics(settings.system)
    _normalize_startup_semantics(settings)
    payload = {
        "system": _system_config_payload(settings.system),
        "model": _dataclass_dict(settings.model),
        "startup": _dataclass_dict(settings.startup),
        "detection_targets": settings.detection_targets,
    }
    payload["model"].pop("Simulation_mode", None)
    payload["model"].pop("openai_api_key", None)
    payload["model"].pop("vlm_api_key", None)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_path.parent,
            prefix=f".{target_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(payload, indent=2, ensure_ascii=False))
            temp_path = Path(handle.name)
        temp_path.replace(target_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return settings


def load_system_config(config_path: Optional[Path] = None) -> SystemConfig:
    return load_runtime_settings(config_path).system


def load_model_config(config_path: Optional[Path] = None) -> ModelConfig:
    return load_runtime_settings(config_path).model


def load_startup_config(config_path: Optional[Path] = None) -> StartupConfig:
    return load_runtime_settings(config_path).startup


def load_detection_targets(config_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    settings = load_runtime_settings(config_path)
    return {str(key): dict(value) for key, value in settings.detection_targets.items()}


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def _snapshot_payload(settings: RuntimeSettings, *, include_secrets: bool) -> Dict[str, Any]:
    agent_payload = {
        "microscope_mode": settings.model.microscope_mode,
        "image_analysis_mode": settings.model.image_analysis_mode,
        "segmentation_mode": settings.model.segmentation_mode,
        "clarify_enabled": settings.model.clarify_enabled,
        "checker_enabled": settings.model.checker_enabled,
        "skill_mode": settings.model.skill_mode,
        "knowledge_base_path": settings.model.knowledge_base_path,
        "base_url": settings.model.base_url,
        "model_name": settings.model.model_name,
        "vlm_base_url": settings.model.vlm_base_url,
        "vlm_model_name": settings.model.vlm_model_name,
        "masked": {
            "openai_api_key": mask_secret(settings.model.openai_api_key),
            "vlm_api_key": mask_secret(settings.model.vlm_api_key),
        },
    }
    if include_secrets:
        agent_payload["openai_api_key"] = settings.model.openai_api_key
        agent_payload["vlm_api_key"] = settings.model.vlm_api_key
    else:
        agent_payload["openai_api_key"] = ""
        agent_payload["vlm_api_key"] = ""

    return {
        "system": {
            "MM_DIR": settings.system.MM_DIR,
            "CONFIG_PATH": settings.system.CONFIG_PATH,
            "FIJI_PATH": settings.system.FIJI_PATH,
            "MAVEN_BIN": settings.system.MAVEN_BIN,
            "in_process_executor_timeout_seconds": settings.system.in_process_executor_timeout_seconds,
            "fiji_executor_timeout_seconds": settings.system.fiji_executor_timeout_seconds,
            "camera_device": settings.system.camera_device,
            "xy_stage_device": settings.system.xy_stage_device,
            "objective_device": settings.system.objective_device,
            "focus_drive": settings.system.focus_drive,
            "Dichroic": settings.system.Dichroic,
            "objectives": settings.system.objectives,
            "channels": settings.system.channels,
            "transmitted_light": settings.system.transmitted_light,
            "Max_X_position": settings.system.Max_X_position,
            "Min_X_position": settings.system.Min_X_position,
            "Max_Y_position": settings.system.Max_Y_position,
            "Min_Y_position": settings.system.Min_Y_position,
            "Max_Z_position": settings.system.Max_Z_position,
            "Min_Z_position": settings.system.Min_Z_position,
            "Max_brightness": settings.system.Max_brightness,
            "Min_brightness": settings.system.Min_brightness,
            "Max_exposure": settings.system.Max_exposure,
            "Min_exposure": settings.system.Min_exposure,
        },
        "agent": agent_payload,
        "startup": asdict(settings.startup),
        "detection_targets": settings.detection_targets,
    }


def read_config_snapshot(
    config_path: Optional[Path] = None,
    *,
    apply_env: bool = True,
    apply_demo_overlay: bool = True,
) -> Dict[str, Any]:
    settings = load_runtime_settings(config_path, apply_env=apply_env, apply_demo_overlay=apply_demo_overlay)
    return _snapshot_payload(settings, include_secrets=True)


def read_public_config_snapshot(
    config_path: Optional[Path] = None,
    *,
    apply_env: bool = True,
    apply_demo_overlay: bool = True,
) -> Dict[str, Any]:
    settings = load_runtime_settings(config_path, apply_env=apply_env, apply_demo_overlay=apply_demo_overlay)
    return _snapshot_payload(settings, include_secrets=False)



