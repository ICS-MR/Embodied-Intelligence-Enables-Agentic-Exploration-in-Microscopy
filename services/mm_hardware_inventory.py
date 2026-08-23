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


def _positive_float(
    value: Any,
    *,
    warnings: list[str] | None = None,
    context: str = "value",
) -> float | None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        if warnings is not None:
            warnings.append(f"{context} must be numeric, got {value!r}.")
        return None
    if numeric_value <= 0:
        if warnings is not None:
            warnings.append(f"{context} must be positive, got {numeric_value!r}.")
        return None
    return numeric_value


def _float_list(
    value: Any,
    *,
    warnings: list[str] | None = None,
    context: str = "value",
) -> list[float]:
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        if warnings is not None:
            warnings.append(f"{context} must be a numeric sequence, got {value!r}.")
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


def _pixel_size_api_call(core: Any, method_name: str, *args: Any, warnings: list[str]) -> Any:
    method = getattr(core, method_name, None)
    if not callable(method):
        warnings.append(f"MMCore pixel-size API '{method_name}' is unavailable.")
        return None
    try:
        return method(*args)
    except Exception as exc:
        arg_text = ", ".join(repr(arg) for arg in args)
        warnings.append(f"MMCore pixel-size API '{method_name}({arg_text})' failed: {type(exc).__name__}: {exc}")
        return None


def _setting_text(setting: Any, method_name: str, *, warnings: list[str] | None = None, context: str = "setting") -> str:
    method = getattr(setting, method_name, None)
    if callable(method):
        try:
            return str(method() or "").strip()
        except Exception as exc:
            if warnings is not None:
                warnings.append(f"Could not read {context}.{method_name}(): {type(exc).__name__}: {exc}")
            return ""
    return str(getattr(setting, method_name, "") or "").strip()


def _configuration_settings(config_data: Any, *, config_id: str, warnings: list[str]) -> list[dict[str, str]]:
    settings: list[dict[str, str]] = []
    size = _safe_call(config_data, "size")
    if isinstance(size, int):
        for index in range(size):
            setting = _safe_call(config_data, "getSetting", index)
            if setting is None:
                warnings.append(f"Pixel-size config '{config_id}' setting #{index} could not be read.")
                continue
            context = f"Pixel-size config '{config_id}' setting #{index}"
            device = _setting_text(setting, "getDeviceLabel", warnings=warnings, context=context)
            prop_name = _setting_text(setting, "getPropertyName", warnings=warnings, context=context)
            prop_value = _setting_text(setting, "getPropertyValue", warnings=warnings, context=context)
            if device and prop_name:
                settings.append({"device": device, "property": prop_name, "value": prop_value})
        return settings

    try:
        iterable = list(config_data)
    except TypeError as exc:
        warnings.append(f"Pixel-size config '{config_id}' settings are not iterable: {type(exc).__name__}: {exc}")
        return settings
    for index, setting in enumerate(iterable):
        context = f"Pixel-size config '{config_id}' setting #{index}"
        device = _setting_text(setting, "getDeviceLabel", warnings=warnings, context=context)
        prop_name = _setting_text(setting, "getPropertyName", warnings=warnings, context=context)
        prop_value = _setting_text(setting, "getPropertyValue", warnings=warnings, context=context)
        if device and prop_name:
            settings.append({"device": device, "property": prop_name, "value": prop_value})
    return settings


def _pixel_size_configs(core: Any, warnings: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    config_ids = _pixel_size_api_call(core, "getAvailablePixelSizeConfigs", warnings=warnings)
    for config_id in _string_list(config_ids):
        pixel_size_um = _positive_float(
            _pixel_size_api_call(core, "getPixelSizeUmByID", config_id, warnings=warnings),
            warnings=warnings,
            context=f"MMCore pixel size for config '{config_id}'",
        )
        config_data = _pixel_size_api_call(core, "getPixelSizeConfigData", config_id, warnings=warnings)
        entry: dict[str, Any] = {
            "id": config_id,
            "settings": _configuration_settings(config_data, config_id=config_id, warnings=warnings) if config_data is not None else [],
        }
        if pixel_size_um is not None:
            entry["pixel_size_um"] = pixel_size_um
        affine = _float_list(
            _pixel_size_api_call(core, "getPixelSizeAffineByID", config_id, warnings=warnings),
            warnings=warnings,
            context=f"MMCore pixel-size affine for config '{config_id}'",
        )
        if affine:
            entry["pixel_size_affine"] = affine
        configs.append(entry)
    return configs


def _pixel_sizes_by_objective_label(pixel_size_configs: list[dict[str, Any]], objective_device: str) -> dict[str, float]:
    pixel_sizes: dict[str, float] = {}
    if not objective_device:
        return pixel_sizes
    for config in pixel_size_configs:
        pixel_size_um = _positive_float(config.get("pixel_size_um"))
        if pixel_size_um is None:
            continue
        for setting in config.get("settings", []):
            if not isinstance(setting, Mapping):
                continue
            device = str(setting.get("device") or "").strip()
            prop_name = str(setting.get("property") or "").strip().lower()
            prop_value = str(setting.get("value") or "").strip()
            if device == objective_device and prop_name == "label" and prop_value:
                pixel_sizes[prop_value] = pixel_size_um
    return pixel_sizes


def _merge_objective_pixel_sizes(merged: dict[str, Any], pixel_size_configs: list[dict[str, Any]]) -> None:
    suggestions = dict(merged.get("suggestions") or {})
    objective_device = str(dict(suggestions.get("objective_device") or {}).get("value") or "").strip()
    pixel_sizes = _pixel_sizes_by_objective_label(pixel_size_configs, objective_device)
    if not pixel_sizes:
        return
    rule_mapping = merged.setdefault("rule_mapping", {})
    objectives = rule_mapping.setdefault("objectives", {})
    for entry in objectives.values():
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if label in pixel_sizes:
            entry["pixel_size_um"] = pixel_sizes[label]


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
    runtime_warnings: list[str] = []
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

        pixel_size_configs = _pixel_size_configs(core, runtime_warnings)
        current_pixel_size_um = _positive_float(
            _pixel_size_api_call(core, "getPixelSizeUm", warnings=runtime_warnings),
            warnings=runtime_warnings,
            context="MMCore current pixel size",
        )
        current_pixel_size_config = str(
            _pixel_size_api_call(core, "getCurrentPixelSizeConfig", warnings=runtime_warnings) or ""
        ).strip()

        result = {
            "source": "pymmcore_runtime",
            "core_roles": core_roles,
            "devices": devices,
            "pixel_size_configs": pixel_size_configs,
        }
        if current_pixel_size_um is not None:
            result["current_pixel_size_um"] = current_pixel_size_um
        if current_pixel_size_config:
            result["current_pixel_size_config"] = current_pixel_size_config
        if runtime_warnings:
            result["warnings"] = runtime_warnings
        return result
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
    pixel_size_configs = [
        dict(item)
        for item in runtime_inventory.get("pixel_size_configs", [])
        if isinstance(item, Mapping)
    ]
    if pixel_size_configs:
        merged["pixel_size_configs"] = pixel_size_configs
        _merge_objective_pixel_sizes(merged, pixel_size_configs)
    warnings = [str(warning) for warning in merged.get("warnings", [])]
    warnings.extend(str(warning) for warning in runtime_inventory.get("warnings", []))
    if warnings:
        merged["warnings"] = warnings
    merged["runtime_inspection"] = {
        "source": "pymmcore_runtime",
        "device_count": len(runtime_inventory.get("devices", [])),
    }
    return merged
