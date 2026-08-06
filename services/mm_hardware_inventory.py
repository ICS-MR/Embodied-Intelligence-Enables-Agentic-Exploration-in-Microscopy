from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, Callable, Mapping


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        return [str(item) for item in value]
    except TypeError:
        return []


def _enum_text(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value or "")
    return text.rsplit(".", 1)[-1]


def _safe_call(core: Any, method_name: str, *args: Any, default: Any = None) -> Any:
    method = getattr(core, method_name, None)
    if not callable(method):
        return default
    try:
        return method(*args)
    except Exception:
        return default


def _property_metadata(core: Any, device: str, property_name: str) -> dict[str, Any]:
    has_limits = bool(_safe_call(core, "hasPropertyLimits", device, property_name, default=False))
    metadata: dict[str, Any] = {
        "name": property_name,
        "type": _enum_text(_safe_call(core, "getPropertyType", device, property_name, default="")),
        "read_only": bool(
            _safe_call(core, "isPropertyReadOnly", device, property_name, default=False)
        ),
        "pre_init": bool(
            _safe_call(core, "isPropertyPreInit", device, property_name, default=False)
        ),
        "allowed_values": _string_list(
            _safe_call(core, "getAllowedPropertyValues", device, property_name, default=[])
        ),
        "has_limits": has_limits,
    }
    if has_limits:
        lower = _safe_call(core, "getPropertyLowerLimit", device, property_name)
        upper = _safe_call(core, "getPropertyUpperLimit", device, property_name)
        if lower is not None and upper is not None:
            metadata["min"] = float(lower)
            metadata["max"] = float(upper)
    return metadata


def _configure_probe_logging(core: Any) -> None:
    for method_name, args in (
        ("enableDebugLog", (False,)),
        ("enableStderrLog", (False,)),
    ):
        with suppress(Exception):
            getattr(core, method_name)(*args)


def _default_core_factory(mm_dir: str) -> Any:
    from pymmcore_plus import CMMCorePlus

    adapter_paths = [mm_dir] if mm_dir else ()
    return CMMCorePlus(mm_path=mm_dir or None, adapter_paths=adapter_paths)


def inspect_micro_manager_config(
    *,
    mm_dir: str,
    config_path: Path,
    core_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Load a cfg in an isolated MMCore instance and return structured capabilities.

    Loading a system configuration initializes its Device Adapters. Callers must make
    that hardware connection explicit to the user and must not run this alongside an
    active microscope runtime.
    """

    factory = core_factory or _default_core_factory
    core = factory(str(mm_dir or "").strip())
    _configure_probe_logging(core)
    _safe_call(core, "setTimeoutMs", 30000)
    try:
        core.loadSystemConfiguration(str(config_path))
        loaded_devices = [
            name
            for name in _string_list(core.getLoadedDevices())
            if name and name.lower() != "core"
        ]
        devices: list[dict[str, Any]] = []
        for device in loaded_devices:
            property_names = _string_list(
                _safe_call(core, "getDevicePropertyNames", device, default=[])
            )
            devices.append(
                {
                    "name": device,
                    "adapter": str(_safe_call(core, "getDeviceLibrary", device, default="") or ""),
                    "device_name": str(_safe_call(core, "getDeviceName", device, default="") or ""),
                    "device_type": _enum_text(
                        _safe_call(core, "getDeviceType", device, default="")
                    ),
                    "description": str(
                        _safe_call(core, "getDeviceDescription", device, default="") or ""
                    ),
                    "state_labels": _string_list(
                        _safe_call(core, "getStateLabels", device, default=[])
                    ),
                    "properties": [
                        _property_metadata(core, device, property_name)
                        for property_name in property_names
                    ],
                }
            )

        core_roles = {}
        for role, getter in (
            ("Camera", "getCameraDevice"),
            ("XYStage", "getXYStageDevice"),
            ("Focus", "getFocusDevice"),
            ("Shutter", "getShutterDevice"),
            ("AutoFocus", "getAutoFocusDevice"),
            ("Galvo", "getGalvoDevice"),
        ):
            value = str(_safe_call(core, getter, default="") or "").strip()
            if value:
                core_roles[role] = value

        return {
            "source": "pymmcore_runtime",
            "core_roles": core_roles,
            "devices": devices,
        }
    finally:
        with suppress(Exception):
            core.unloadAllDevices()
        with suppress(Exception):
            core.reset()


def merge_runtime_inventory(
    cfg_inventory: Mapping[str, Any],
    runtime_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    import copy

    merged = copy.deepcopy(dict(cfg_inventory))
    cfg_devices = {
        str(device.get("name") or ""): device
        for device in merged.get("devices", [])
        if isinstance(device, dict)
    }
    for runtime_device in runtime_inventory.get("devices", []):
        if not isinstance(runtime_device, Mapping):
            continue
        name = str(runtime_device.get("name") or "").strip()
        if not name:
            continue
        target = cfg_devices.get(name)
        if target is None:
            target = {
                "name": name,
                "adapter": str(runtime_device.get("adapter") or ""),
                "device_type": str(runtime_device.get("device_name") or ""),
                "state_labels": [],
                "properties": [],
            }
            merged.setdefault("devices", []).append(target)
            cfg_devices[name] = target

        runtime_properties = [
            dict(item)
            for item in runtime_device.get("properties", [])
            if isinstance(item, Mapping) and str(item.get("name") or "").strip()
        ]
        declared_properties = [str(item) for item in target.get("properties", [])]
        target["properties"] = list(
            dict.fromkeys(
                [*declared_properties, *[str(item["name"]) for item in runtime_properties]]
            )
        )
        target["runtime_properties"] = runtime_properties
        target["runtime_device_type"] = str(runtime_device.get("device_type") or "")
        target["description"] = str(runtime_device.get("description") or "")
        target["state_labels"] = list(
            dict.fromkeys(
                [
                    *[str(item) for item in target.get("state_labels", [])],
                    *[str(item) for item in runtime_device.get("state_labels", [])],
                ]
            )
        )

    merged["core_roles"] = {
        **dict(merged.get("core_roles") or {}),
        **dict(runtime_inventory.get("core_roles") or {}),
    }
    merged["runtime_inspection"] = {
        "source": "pymmcore_runtime",
        "device_count": len(runtime_inventory.get("devices", [])),
    }
    return merged
