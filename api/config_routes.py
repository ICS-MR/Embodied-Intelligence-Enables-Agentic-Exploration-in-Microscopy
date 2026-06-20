import shutil
from os import path as os_path
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.dependencies import get_runtime_manager
from api.models import ConfigSaveRequest, ConfigSaveResponse, ConfigStatusResponse, ConfigUploadResponse
from bootstrap.config import config_is_complete, read_public_config_snapshot, save_env_secrets
from system_config_wizard import build_dichroic_colors, build_objective_labels, parse_mm_config, suggest_values


router = APIRouter()
UPLOADED_CFG_DIR = Path(__file__).resolve().parents[1] / "uploaded_cfg"


def coalesce_text(new_value: str, current_value: str) -> str:
    value = new_value.strip()
    return value if value else current_value


def coalesce_number(new_value: Any, current_value: Any) -> Any:
    return current_value if new_value is None else new_value


def normalize_config_path(new_value: str, current_value: str) -> str:
    value = new_value.strip()
    if not value:
        return current_value
    expanded = os_path.expandvars(os_path.expanduser(value))
    return str(Path(expanded))


@router.get("/api/config/status", response_model=ConfigStatusResponse)
async def get_config_status(runtime_manager=Depends(get_runtime_manager)) -> ConfigStatusResponse:
    snapshot = read_public_config_snapshot()
    preview_phase = runtime_manager.get_preview_status().get("preview_phase", "idle")
    return ConfigStatusResponse(
        configured=config_is_complete(runtime_manager.current_snapshot()),
        initialized=runtime_manager.system_status.initialized,
        initializing=runtime_manager.system_status.initializing,
        error=bool(runtime_manager.system_status.error),
        status_message=runtime_manager.system_status.message,
        system_phase=runtime_manager.system_status.system_phase,
        preview_phase=preview_phase,
        failure_step=runtime_manager.system_status.failure_step,
        system=snapshot["system"],
        agent=snapshot["agent"],
        startup=snapshot["startup"],
    )


@router.post("/api/config/upload-cfg", response_model=ConfigUploadResponse)
async def upload_cfg(file: UploadFile = File(...), runtime_manager=Depends(get_runtime_manager)) -> ConfigUploadResponse:
    if not file.filename.lower().endswith(".cfg"):
        raise HTTPException(status_code=400, detail="Please upload a .cfg file.")

    UPLOADED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOADED_CFG_DIR / Path(file.filename).name
    with saved_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    suggestions = suggest_values(saved_path)
    cfg_data = parse_mm_config(saved_path)
    existing = runtime_manager.current_snapshot()["system"]
    objective_labels = build_objective_labels(cfg_data, suggestions["objective_device"]["value"], existing["objective_labels"])
    dichroic_colors = build_dichroic_colors(cfg_data, suggestions["Dichroic"]["value"], existing["dichroic_colors"])

    updates = {field: info["value"] for field, info in suggestions.items() if info["value"]}
    updates["CONFIG_PATH"] = str(saved_path)
    if objective_labels:
        updates["objective_labels"] = objective_labels
    if dichroic_colors:
        updates["dichroic_colors"] = dichroic_colors
    runtime_manager.update_settings(system_updates=updates)

    return ConfigUploadResponse(
        config_path=str(saved_path),
        stored_config_path=str(saved_path),
        original_filename=Path(file.filename).name,
        suggestions=suggestions,
        objective_labels=objective_labels,
        dichroic_colors=dichroic_colors,
    )


@router.post("/api/config/save", response_model=ConfigSaveResponse)
async def save_config(req: ConfigSaveRequest, runtime_manager=Depends(get_runtime_manager)) -> ConfigSaveResponse:
    snapshot = runtime_manager.current_snapshot(apply_env=False)
    agent_current = snapshot["agent"]
    system_current = snapshot["system"]
    startup_current = snapshot["startup"]
    system_updates = {
        "MM_DIR": coalesce_text(req.mm_dir, system_current["MM_DIR"]),
        "CONFIG_PATH": normalize_config_path(req.config_path, system_current["CONFIG_PATH"]),
        "FIJI_PATH": coalesce_text(req.fiji_path, system_current["FIJI_PATH"]),
        "camera_device": coalesce_text(req.camera_device, system_current["camera_device"]),
        "xy_stage_device": coalesce_text(req.xy_stage_device, system_current["xy_stage_device"]),
        "objective_device": coalesce_text(req.objective_device, system_current["objective_device"]),
        "transmittedIllumination": coalesce_text(req.transmittedIllumination, system_current["transmittedIllumination"]),
        "focus_drive": coalesce_text(req.focus_drive, system_current["focus_drive"]),
        "Dichroic": coalesce_text(req.Dichroic, system_current["Dichroic"]),
    }
    model_updates = {
        "Simulation_mode": req.simulation_mode,
        "base_url": coalesce_text(req.base_url, agent_current["base_url"]),
        "model_name": coalesce_text(req.model_name, agent_current["model_name"]),
        "vlm_base_url": coalesce_text(req.vlm_base_url, agent_current["vlm_base_url"]),
        "vlm_model_name": coalesce_text(req.vlm_model_name, agent_current["vlm_model_name"]),
        "clarify_enabled": req.clarify_enabled,
        "checker_enabled": req.checker_enabled,
    }
    startup_updates = {
        "objective": coalesce_text(req.startup_objective, startup_current["objective"]),
        "channel": coalesce_text(req.startup_channel, startup_current["channel"]),
        "exposure": coalesce_number(req.startup_exposure, startup_current["exposure"]),
        "brightness": coalesce_number(req.startup_brightness, startup_current["brightness"]),
        "start_preview": req.startup_start_preview,
    }
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

