from typing import Any, Mapping, Optional

import threading

from bootstrap.microscope_semantics import resolve_channel_input, resolve_objective_input
from bootstrap.config import StartupConfig, is_demo_mapping_payload, load_startup_config, normalize_demo_environment
from runtime.config import _has_transmitted_light_brightness_control
from runtime.models import RuntimeContext


def _call_if_available(obj: Any, method_name: str, *args: Any) -> None:
    if hasattr(obj, method_name):
        getattr(obj, method_name)(*args)


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("hardware operation cancelled")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_managed_demo_microscope(env_olympus: Any) -> bool:
    system_config = getattr(env_olympus, "system_config", None)
    if system_config is None:
        return False
    return is_demo_mapping_payload(
        config_path=str(getattr(env_olympus, "config_path", getattr(system_config, "CONFIG_PATH", ""))),
        camera_device=str(getattr(env_olympus, "camera_device", getattr(system_config, "camera_device", ""))),
        xy_stage_device=str(getattr(env_olympus, "xy_stage_device", getattr(system_config, "xy_stage_device", ""))),
        objective_device=str(getattr(env_olympus, "objective_device", getattr(system_config, "objective_device", ""))),
        focus_drive=str(getattr(env_olympus, "focus_drive", getattr(system_config, "focus_drive", ""))),
        dichroic=str(getattr(env_olympus, "Dichroic", getattr(system_config, "Dichroic", ""))),
        objectives=_mapping(getattr(env_olympus, "objectives", getattr(system_config, "objectives", {}))),
        channels=_mapping(getattr(env_olympus, "channels", getattr(system_config, "channels", {}))),
        transmitted_light=_mapping(getattr(system_config, "transmitted_light", {})),
    )


def _apply_demo_environment(env_olympus: Any) -> None:
    if not _is_managed_demo_microscope(env_olympus):
        return
    system_config = getattr(env_olympus, "system_config", None)
    core = getattr(env_olympus, "core", None)
    camera_device = str(getattr(env_olympus, "camera_device", getattr(system_config, "camera_device", "")))
    if core is None or not camera_device:
        raise RuntimeError("Cannot apply demo environment: microscope core or camera device is unavailable.")

    env = normalize_demo_environment(getattr(system_config, "demo_environment", {}))
    properties = {
        "Mode": "Fluorescent Beads",
        "BeadDensity": int(env.get("sample_density", 100)),
        "BeadSize": float(env.get("sample_size", 2.0)),
        "BeadBlurRate": float(env.get("bead_blur_rate", 0.5)),
    }
    for prop, value in properties.items():
        try:
            core.setProperty(camera_device, prop, value)
            core.waitForDevice(camera_device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to apply demo environment {camera_device}.{prop}={value!r}."
            ) from exc


def _resolve_startup_label(value: str, system_config: Any, resolver: Any) -> str:
    if system_config is None:
        return value
    resolved_label, _semantic_key, _entry = resolver(value, system_config)
    return resolved_label


def initialize_microscope(env_olympus: Any, cancel_event: Optional[threading.Event] = None) -> None:
    _check_cancelled(cancel_event)
    env_olympus.initialize()
    _apply_demo_environment(env_olympus)
    _check_cancelled(cancel_event)


def apply_startup_state(
    env_olympus: Any,
    startup_config: Optional[StartupConfig] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    startup = startup_config or load_startup_config()
    system_config = getattr(env_olympus, "system_config", None)
    objective_label = _resolve_startup_label(startup.objective, system_config, resolve_objective_input)
    channel_label = _resolve_startup_label(startup.channel, system_config, resolve_channel_input)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_objective", objective_label)
    _check_cancelled(cancel_event)
    if hasattr(env_olympus, "_user_brightness"):
        env_olympus._user_brightness = int(startup.brightness)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_channel", channel_label)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_exposure", startup.exposure)
    _check_cancelled(cancel_event)
    supports_brightness = getattr(env_olympus, "_supports_transmitted_brightness", None)
    brightness_available = (
        bool(supports_brightness())
        if callable(supports_brightness)
        else _has_transmitted_light_brightness_control(system_config)
    )
    if brightness_available:
        _call_if_available(env_olympus, "set_brightness", startup.brightness)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_x_y_position", startup.x_position, startup.y_position)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_z_position", startup.z_position)
    _check_cancelled(cancel_event)


def setup_microscope(env_olympus: Any, startup_config: Optional[StartupConfig] = None) -> None:
    initialize_microscope(env_olympus)
    apply_startup_state(env_olympus, startup_config)


def release_resources(system_components: RuntimeContext) -> None:
    env_olympus = system_components.env_olympus
    env_imagej = system_components.env_imagej
    errors: list[str] = []

    if env_olympus and hasattr(env_olympus, "shutdown"):
        try:
            print("Shutting down microscope controller...")
            env_olympus.shutdown()
            print("Microscope controller shutdown complete.")
        except Exception as exc:
            errors.append(f"microscope shutdown: {exc}")

    if env_imagej and hasattr(env_imagej, "fiji_shutdown"):
        try:
            print("Shutting down Fiji/ImageJ...")
            env_imagej.fiji_shutdown()
            print("Fiji/ImageJ shutdown complete.")
        except Exception as exc:
            errors.append(f"Fiji shutdown: {exc}")

    try:
        system_components.storage_manager.clear_cache()
    except Exception as exc:
        errors.append(f"cache cleanup: {exc}")

    if errors:
        raise RuntimeError("; ".join(errors))
