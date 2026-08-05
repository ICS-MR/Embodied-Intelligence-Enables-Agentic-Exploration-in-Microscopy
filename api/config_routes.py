import shutil
from os import path as os_path
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_runtime_manager
from api.models import ConfigSaveRequest, ConfigSaveResponse, ConfigStatusResponse, ConfigUploadResponse
from bootstrap.config import (
    build_demo_startup_overrides,
    build_demo_system_overrides,
    is_demo_mapping_payload,
    read_public_config_snapshot,
    save_env_secrets,
)
from runtime.asset_check import check_snapshot_assets
from services.runtime_manager import LifecycleConflictError
from system_config_wizard import (
    build_channels,
    build_objectives,
    build_transmitted_light_mapping,
    parse_mm_config,
    suggest_values,
)


router = APIRouter()
UPLOADED_CFG_DIR = Path(__file__).resolve().parents[1] / "uploaded_cfg"


def coalesce_text(new_value: str, current_value: str) -> str:
    value = new_value.strip()
    return value if value else current_value


def coalesce_number(new_value: Any, current_value: Any) -> Any:
    return current_value if new_value is None else new_value


def maybe_number_update(updates: dict[str, Any], key: str, new_value: Any) -> None:
    if new_value is not None:
        updates[key] = new_value


def normalize_config_path(new_value: str, current_value: str) -> str:
    value = new_value.strip()
    if not value:
        return current_value
    expanded = os_path.expandvars(os_path.expanduser(value))
    return str(Path(expanded))


@router.get("/api/config/status", response_model=ConfigStatusResponse)
async def get_config_status(runtime_manager=Depends(get_runtime_manager)) -> ConfigStatusResponse:
    snapshot = read_public_config_snapshot()
    persisted_snapshot = read_public_config_snapshot(apply_env=False, apply_demo_overlay=False)
    preview_phase = runtime_manager.get_preview_status().get("preview_phase", "idle")
    asset_check = check_snapshot_assets(runtime_manager.current_snapshot())
    return ConfigStatusResponse(
        configured=asset_check.ready,
        initialized=runtime_manager.system_status.initialized,
        initializing=runtime_manager.system_status.initializing,
        error=bool(runtime_manager.system_status.error),
        status_message=runtime_manager.system_status.message,
        system_phase=runtime_manager.system_status.system_phase,
        preview_phase=preview_phase,
        failure_step=runtime_manager.system_status.failure_step,
        system=snapshot["system"],
        real_system_draft=persisted_snapshot["system"],
        demo_system=build_demo_system_overrides(),
        demo_startup=build_demo_startup_overrides(),
        agent=snapshot["agent"],
        startup=snapshot["startup"],
    )


@router.post("/api/config/upload-cfg", response_model=ConfigUploadResponse)
async def upload_cfg(file: UploadFile = File(...), runtime_manager=Depends(get_runtime_manager)) -> ConfigUploadResponse:
    try:
        await runtime_manager.ensure_configuration_mutable()
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not file.filename.lower().endswith(".cfg"):
        raise HTTPException(status_code=400, detail="Please upload a .cfg file.")

    UPLOADED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOADED_CFG_DIR / Path(file.filename).name
    with saved_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    suggestions = suggest_values(saved_path)
    cfg_data = parse_mm_config(saved_path)
    objectives = build_objectives(cfg_data, suggestions["objective_device"]["value"], {})
    channels = build_channels(cfg_data, suggestions["Dichroic"]["value"], {})
    transmitted_light = build_transmitted_light_mapping(cfg_data, suggestions["transmittedIllumination"]["value"])

    return ConfigUploadResponse(
        config_path=str(saved_path),
        stored_config_path=str(saved_path),
        original_filename=Path(file.filename).name,
        suggestions=suggestions,
        objectives=objectives,
        channels=channels,
        transmitted_light=transmitted_light,
    )


@router.post("/api/config/save", response_model=ConfigSaveResponse)
async def save_config(req: ConfigSaveRequest, runtime_manager=Depends(get_runtime_manager)) -> ConfigSaveResponse:
    try:
        await runtime_manager.ensure_configuration_mutable()
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    snapshot = runtime_manager.current_snapshot(apply_env=False, apply_demo_overlay=False)
    agent_current = snapshot["agent"]
    system_current = snapshot["system"]
    startup_current = snapshot["startup"]
    microscope_mode = str(req.microscope_mode or "demo").strip().lower()
    preserve_persisted_hardware_fields = (
        microscope_mode != "demo"
        and is_demo_mapping_payload(
            config_path=req.config_path,
            camera_device=req.camera_device,
            xy_stage_device=req.xy_stage_device,
            objective_device=req.objective_device,
            transmitted_illumination=req.transmittedIllumination,
            focus_drive=req.focus_drive,
            dichroic=req.Dichroic,
            objectives=req.objectives,
            channels=req.channels,
            transmitted_light=req.transmitted_light,
        )
    )
    system_updates = {
        "MM_DIR": coalesce_text(req.mm_dir, system_current["MM_DIR"]),
        "CONFIG_PATH": (
            system_current["CONFIG_PATH"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else normalize_config_path(req.config_path, system_current["CONFIG_PATH"])
        ),
        "FIJI_PATH": coalesce_text(req.fiji_path, system_current["FIJI_PATH"]),
        "camera_device": (
            system_current["camera_device"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.camera_device, system_current["camera_device"])
        ),
        "xy_stage_device": (
            system_current["xy_stage_device"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.xy_stage_device, system_current["xy_stage_device"])
        ),
        "objective_device": (
            system_current["objective_device"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.objective_device, system_current["objective_device"])
        ),
        "transmittedIllumination": (
            system_current["transmittedIllumination"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.transmittedIllumination, system_current["transmittedIllumination"])
        ),
        "focus_drive": (
            system_current["focus_drive"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.focus_drive, system_current["focus_drive"])
        ),
        "Dichroic": (
            system_current["Dichroic"]
            if microscope_mode == "demo" or preserve_persisted_hardware_fields
            else coalesce_text(req.Dichroic, system_current["Dichroic"])
        ),
    }
    maybe_number_update(system_updates, "Max_X_position", req.Max_X_position)
    maybe_number_update(system_updates, "Min_X_position", req.Min_X_position)
    maybe_number_update(system_updates, "Max_Y_position", req.Max_Y_position)
    maybe_number_update(system_updates, "Min_Y_position", req.Min_Y_position)
    maybe_number_update(system_updates, "Max_Z_position", req.Max_Z_position)
    maybe_number_update(system_updates, "Min_Z_position", req.Min_Z_position)
    maybe_number_update(system_updates, "Max_brightness", req.Max_brightness)
    maybe_number_update(system_updates, "Min_brightness", req.Min_brightness)
    if req.objectives and not (microscope_mode == "demo" or preserve_persisted_hardware_fields):
        system_updates["objectives"] = req.objectives
    if req.channels and not (microscope_mode == "demo" or preserve_persisted_hardware_fields):
        system_updates["channels"] = req.channels
    if req.transmitted_light and not (microscope_mode == "demo" or preserve_persisted_hardware_fields):
        system_updates["transmitted_light"] = req.transmitted_light
    model_updates = {
        "microscope_mode": microscope_mode,
        "image_analysis_mode": req.image_analysis_mode,
        "segmentation_mode": req.segmentation_mode,
        "base_url": coalesce_text(req.base_url, agent_current["base_url"]),
        "model_name": coalesce_text(req.model_name, agent_current["model_name"]),
        "vlm_base_url": coalesce_text(req.vlm_base_url, agent_current["vlm_base_url"]),
        "vlm_model_name": coalesce_text(req.vlm_model_name, agent_current["vlm_model_name"]),
        "clarify_enabled": req.clarify_enabled,
        "checker_enabled": req.checker_enabled,
    }
    effective_system = (
        build_demo_system_overrides()
        if microscope_mode == "demo"
        else {**system_current, **system_updates}
    )
    startup_objective = coalesce_text(req.startup_objective, startup_current["objective"])
    startup_channel = coalesce_text(req.startup_channel, startup_current["channel"])
    if microscope_mode == "demo":
        demo_startup = build_demo_startup_overrides()
        startup_objective = str(demo_startup["objective"])
        startup_channel = str(demo_startup["channel"])
    objectives = effective_system.get("objectives", {})
    channels = effective_system.get("channels", {})
    objective_entry = objectives.get(startup_objective, {}) if isinstance(objectives, dict) else {}
    channel_entry = channels.get(startup_channel, {}) if isinstance(channels, dict) else {}
    if not isinstance(objective_entry, dict) or not str(objective_entry.get("label") or "").strip():
        raise HTTPException(
            status_code=422,
            detail=f"Startup objective '{startup_objective}' is not present in the current objective mapping.",
        )
    if not isinstance(channel_entry, dict) or not str(channel_entry.get("label") or "").strip():
        raise HTTPException(
            status_code=422,
            detail=f"Startup channel '{startup_channel}' is not present in the current channel mapping.",
        )
    startup_updates = {
        "objective": startup_objective,
        "channel": startup_channel,
        "exposure": coalesce_number(req.startup_exposure, startup_current["exposure"]),
        "brightness": coalesce_number(req.startup_brightness, startup_current["brightness"]),
        "z_position": coalesce_number(req.startup_z_position, startup_current["z_position"]),
        "x_position": coalesce_number(req.startup_x_position, startup_current["x_position"]),
        "y_position": coalesce_number(req.startup_y_position, startup_current["y_position"]),
        "start_preview": coalesce_number(req.startup_start_preview, startup_current["start_preview"]),
    }
    if microscope_mode == "demo":
        startup_updates.update(build_demo_startup_overrides())
    runtime_manager.update_settings(
        system_updates=system_updates,
        model_updates=model_updates,
        startup_updates=startup_updates,
    )
    save_env_secrets(
        openai_api_key=req.openai_api_key.strip() or None,
        vlm_api_key=req.vlm_api_key.strip() or None,
    )
    saved_snapshot = runtime_manager.current_snapshot()
    save_result = runtime_manager.refresh_status_after_config_save()
    preview_phase = runtime_manager.get_preview_status().get("preview_phase", "idle")
    return ConfigSaveResponse(
        saved=True,
        initialized=save_result["initialized"],
        initializing=save_result.get("initializing", False),
        message=save_result["message"],
        effective_config_path=saved_snapshot["system"]["CONFIG_PATH"],
        system_phase=save_result.get("system_phase", runtime_manager.system_status.system_phase),
        preview_phase=preview_phase,
        failure_step=save_result.get("failure_step", runtime_manager.system_status.failure_step),
    )

