import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = ROOT_DIR / "config" / "runtime_config.json"


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

DEFAULT_DETECTION_TARGETS: Dict[str, Dict[str, Any]] = {
    "tumor": {
        "target_class_id": 0,
        "target_class_name": "tumor",
        "score_thr": 0.2,
        "output_filename": "tumor_locations_list.json",
        "model_config": "",
        "model_checkpoint": "",
    },
    "lesion": {
        "target_class_id": 0,
        "target_class_name": "lesion",
        "score_thr": 0.2,
        "output_filename": "lesion_locations_list.json",
        "model_config": "",
        "model_checkpoint": "",
    },
    "bacteria": {
        "target_class_id": 0,
        "target_class_name": "bacteria",
        "score_thr": 0.2,
        "output_filename": "bacteria_locations_list.json",
        "model_config": "",
        "model_checkpoint": "",
    },
    "2Dcell": {
        "target_class_id": 0,
        "target_class_name": "2Dcell",
        "score_thr": 0.2,
        "output_filename": "2Dcell_locations_list.json",
        "model_config": "",
        "model_checkpoint": "",
    },
    "organoid": {
        "target_class_id": 0,
        "target_class_name": "organoid",
        "score_thr": 0.2,
        "output_filename": "organoid_locations_list.json",
        "model_config": "configs/organoid.py",
        "model_checkpoint": "weights/organoid.pth",
    },
}


@dataclass
class StartupConfig:
    objective: str = "4-LUCPLFLN40X"
    channel: str = "1-NONE"
    exposure: float = 10.0
    brightness: int = 100
    z_position: float = 4100.0
    x_position: float = 50000.0
    y_position: float = 50000.0
    start_preview: bool = True


@dataclass
class SystemConfig:
    MM_DIR: str = r""
    CONFIG_PATH: str = str(ROOT_DIR / "uploaded_cfg" / "MMConfig_demo2.cfg")
    FIJI_PATH: str = r""
    MAVEN_BIN: str = r""
    camera_device: str = ""
    xy_stage_device: str = ""
    objective_device: str = ""
    transmittedIllumination: str = ""
    focus_drive: str = ""
    Dichroic: str = ""
    objective_labels: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_OBJECTIVE_LABELS))
    dichroic_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: dict(DEFAULT_DICHROIC_COLORS))
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
    fiji_executor_timeout_seconds: float = 300.0


@dataclass
class ModelConfig:
    Simulation_mode: bool = True
    clarify_enabled: bool = False
    checker_enabled: bool = False
    openai_api_key: str = "sk-ngWYYJ3lt5pFB9YF5mTCuFRp7KQQtRATn0NdsV0X21rHRoSt"
    base_url: str = "https://jeniya.top/v1"
    model_name: str = "claude-sonnet-4-6"
    vlm_api_key: str = "sk-ngWYYJ3lt5pFB9YF5mTCuFRp7KQQtRATn0NdsV0X21rHRoSt"
    vlm_base_url: str = "https://jeniya.top/v1"
    vlm_model_name: str = "claude-sonnet-4-6"
    CROSS_ENCODER_MODEL_PATH: str = r"model\bge-m3"
    task_similarity_threshold: float = 0.17

@dataclass
class RuntimeSettings:
    system: SystemConfig = field(default_factory=SystemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    startup: StartupConfig = field(default_factory=StartupConfig)
    detection_targets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_DETECTION_TARGETS)))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
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


def _coerce_color_map(value: Mapping[str, Any], fallback: Mapping[str, Tuple[int, int, int]]) -> Dict[str, Tuple[int, int, int]]:
    if not isinstance(value, Mapping):
        return dict(fallback)
    result: Dict[str, Tuple[int, int, int]] = {}
    for key, item in value.items():
        if isinstance(item, (list, tuple)) and len(item) == 3:
            result[str(key)] = (int(item[0]), int(item[1]), int(item[2]))
    return result or dict(fallback)


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
        else:
            setattr(instance, key, value)


def _apply_file_overrides(settings: RuntimeSettings, payload: Mapping[str, Any]) -> None:
    system_payload = payload.get("system", {})
    model_payload = payload.get("model", {})
    startup_payload = payload.get("startup", {})
    detection_payload = payload.get("detection_targets", {})
    if isinstance(system_payload, Mapping):
        _update_dataclass(settings.system, system_payload)
    if isinstance(model_payload, Mapping):
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
    env_map = {
        "MM_DIR": "EIMS_MM_DIR",
        "CONFIG_PATH": "EIMS_CONFIG_PATH",
        "FIJI_PATH": "EIMS_FIJI_PATH",
        "MAVEN_BIN": "EIMS_MAVEN_BIN",
        "camera_device": "EIMS_CAMERA_DEVICE",
        "xy_stage_device": "EIMS_XY_STAGE_DEVICE",
        "objective_device": "EIMS_OBJECTIVE_DEVICE",
        "transmittedIllumination": "EIMS_TRANSMITTED_ILLUMINATION",
        "focus_drive": "EIMS_FOCUS_DRIVE",
        "Dichroic": "EIMS_DICHROIC",
    }
    for field_name, env_name in env_map.items():
        value = env_values.get(env_name)
        if value:
            setattr(settings.system, field_name, value)

    model_env_map = {
        "openai_api_key": ("EIMS_OPENAI_API_KEY", "OPENAI_API_KEY"),
        "base_url": ("EIMS_BASE_URL",),
        "model_name": ("EIMS_MODEL_NAME",),
        "vlm_api_key": ("EIMS_VLM_API_KEY", "VLM_API_KEY"),
        "vlm_base_url": ("EIMS_VLM_BASE_URL",),
        "vlm_model_name": ("EIMS_VLM_MODEL_NAME",),
    }
    for field_name, env_names in model_env_map.items():
        for env_name in env_names:
            value = env_values.get(env_name)
            if value:
                setattr(settings.model, field_name, value)
                break

    # Keep UI-managed simulation mode stable across saves: only a real process environment
    # override should win here, not a value persisted in .env.
    simulation_env = os.environ.get("EIMS_SIMULATION_MODE")
    if simulation_env is not None:
        settings.model.Simulation_mode = _coerce_bool(simulation_env, settings.model.Simulation_mode)

    checker_env = os.environ.get("EIMS_CHECKER_ENABLED")
    if checker_env is not None:
        settings.model.checker_enabled = _coerce_bool(checker_env, settings.model.checker_enabled)


def _load_env_values(*, include_dotenv: bool) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if include_dotenv:
        for key, value in dotenv_values(ROOT_DIR / ".env").items():
            if value is not None:
                merged[key] = value
    for key, value in os.environ.items():
        merged[key] = value
    return merged


def load_runtime_settings(config_path: Optional[Path] = None, *, apply_env: bool = True) -> RuntimeSettings:
    settings = RuntimeSettings()
    target_path = config_path or RUNTIME_CONFIG_PATH
    payload = _read_json(target_path)
    _apply_file_overrides(settings, payload)
    if apply_env:
        _apply_env_overrides(settings, _load_env_values(include_dotenv=target_path == RUNTIME_CONFIG_PATH))
    return settings


def _dataclass_dict(instance: Any) -> Dict[str, Any]:
    payload = asdict(instance)
    if "dichroic_colors" in payload:
        payload["dichroic_colors"] = {key: list(value) for key, value in payload["dichroic_colors"].items()}
    return payload


def save_runtime_settings(
    system_updates: Optional[Mapping[str, Any]] = None,
    model_updates: Optional[Mapping[str, Any]] = None,
    startup_updates: Optional[Mapping[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> RuntimeSettings:
    target_path = config_path or RUNTIME_CONFIG_PATH
    settings = load_runtime_settings(target_path, apply_env=False)
    if system_updates:
        _update_dataclass(settings.system, system_updates)
    if model_updates:
        _update_dataclass(settings.model, model_updates)
    if startup_updates:
        _update_dataclass(settings.startup, startup_updates)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": _dataclass_dict(settings.system),
        "model": _dataclass_dict(settings.model),
        "startup": _dataclass_dict(settings.startup),
        "detection_targets": settings.detection_targets,
    }
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
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
        "Simulation_mode": settings.model.Simulation_mode,
        "clarify_enabled": settings.model.clarify_enabled,
        "checker_enabled": settings.model.checker_enabled,
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
            "fiji_executor_timeout_seconds": settings.system.fiji_executor_timeout_seconds,
            "camera_device": settings.system.camera_device,
            "xy_stage_device": settings.system.xy_stage_device,
            "objective_device": settings.system.objective_device,
            "transmittedIllumination": settings.system.transmittedIllumination,
            "focus_drive": settings.system.focus_drive,
            "Dichroic": settings.system.Dichroic,
            "objective_labels": settings.system.objective_labels,
            "dichroic_colors": settings.system.dichroic_colors,
        },
        "agent": agent_payload,
        "startup": asdict(settings.startup),
        "detection_targets": settings.detection_targets,
    }


def read_config_snapshot(config_path: Optional[Path] = None, *, apply_env: bool = True) -> Dict[str, Any]:
    settings = load_runtime_settings(config_path, apply_env=apply_env)
    return _snapshot_payload(settings, include_secrets=True)


def read_public_config_snapshot(config_path: Optional[Path] = None, *, apply_env: bool = True) -> Dict[str, Any]:
    settings = load_runtime_settings(config_path, apply_env=apply_env)
    return _snapshot_payload(settings, include_secrets=False)


def config_is_complete(snapshot: Mapping[str, Any]) -> bool:
    system_cfg = snapshot["system"]
    agent_cfg = snapshot["agent"]
    simulation_mode = bool(agent_cfg.get("Simulation_mode", True))
    required_system = []
    if not simulation_mode:
        required_system = [
            "MM_DIR",
            "CONFIG_PATH",
            "camera_device",
            "xy_stage_device",
            "objective_device",
            "transmittedIllumination",
            "focus_drive",
            "Dichroic",
        ]
    required_agent = [
        "openai_api_key",
        "base_url",
        "model_name",
        "vlm_api_key",
        "vlm_base_url",
        "vlm_model_name",
    ]
    return all(system_cfg.get(field) for field in required_system) and all(agent_cfg.get(field) for field in required_agent)










