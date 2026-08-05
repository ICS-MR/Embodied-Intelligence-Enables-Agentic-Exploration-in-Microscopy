from typing import Any, Optional

import threading

from bootstrap.config import StartupConfig, load_startup_config
from runtime.config import _has_transmitted_light_brightness_control
from runtime.models import RuntimeContext


def _call_if_available(obj: Any, method_name: str, *args: Any) -> None:
    if hasattr(obj, method_name):
        getattr(obj, method_name)(*args)


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("hardware operation cancelled")


def initialize_microscope(env_olympus: Any, cancel_event: Optional[threading.Event] = None) -> None:
    _check_cancelled(cancel_event)
    env_olympus.initialize()
    _check_cancelled(cancel_event)


def apply_startup_state(
    env_olympus: Any,
    startup_config: Optional[StartupConfig] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    startup = startup_config or load_startup_config()
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_objective", startup.objective)
    _check_cancelled(cancel_event)
    if hasattr(env_olympus, "_user_brightness"):
        env_olympus._user_brightness = int(startup.brightness)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_channel", startup.channel)
    _check_cancelled(cancel_event)
    _call_if_available(env_olympus, "set_exposure", startup.exposure)
    _check_cancelled(cancel_event)
    system_config = getattr(env_olympus, "system_config", None)
    if _has_transmitted_light_brightness_control(system_config):
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
