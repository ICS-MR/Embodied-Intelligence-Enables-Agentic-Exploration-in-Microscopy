import shutil
from os import path as os_path
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import get_runtime_manager
from api.models import (
    ConfigSaveRequest,
    ConfigSaveResponse,
    ConfigStatusResponse,
    ConfigUploadResponse,
    LLMConnectionTestRequest,
    LLMConnectionTestResponse,
    VLMConnectionTestRequest,
    VLMConnectionTestResponse,
)
from bootstrap.config import (
    build_demo_startup_overrides,
    build_demo_system_overrides,
    build_mock_microscope_capabilities,
    is_demo_mapping_payload,
    read_config_snapshot,
    load_runtime_settings,
    read_public_config_snapshot,
    save_env_secrets,
)
from runtime.asset_check import check_snapshot_assets
from services.config_mapping_ai import analyze_config_mapping
from services.llm_health import (
    LLMConnectionConfig,
    VLMConnectionConfig,
    validate_llm_connection,
    validate_vlm_connection,
)
from services.mm_hardware_inventory import inspect_micro_manager_config, merge_runtime_inventory
from services.runtime_manager import LifecycleConflictError
from system_config_wizard import build_cfg_inventory


logger = logging.getLogger(__name__)


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


def _require_real_mapping_mode(microscope_mode: str) -> None:
    if microscope_mode != "real":
        raise HTTPException(
            status_code=409,
            detail="External Micro-Manager cfg import is available only in Real microscope mode.",
        )


def _format_llm_connection_error(exc: Exception) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _clean_form_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


@router.get("/api/config/status", response_model=ConfigStatusResponse)
async def get_config_status(runtime_manager=Depends(get_runtime_manager)) -> ConfigStatusResponse:
    snapshot = read_config_snapshot()
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
        restart_required=runtime_manager.system_status.restart_required,
        system=snapshot["system"],
        real_system_draft=persisted_snapshot["system"],
        real_startup_draft=persisted_snapshot["startup"],
        demo_system=build_demo_system_overrides(),
        demo_startup=build_demo_startup_overrides(),
        mock_capabilities=build_mock_microscope_capabilities(),
        transmitted_light_runtime=runtime_manager.get_transmitted_light_runtime_info(),
        agent=snapshot["agent"],
        startup=snapshot["startup"],
    )


@router.post("/api/config/upload-cfg", response_model=ConfigUploadResponse)
async def upload_cfg(
    file: UploadFile = File(...),
    microscope_mode: str = Form("real"),
    inspect_hardware: bool = Form(False),
    mm_dir: str = Form(""),
    openai_api_key: str = Form(""),
    base_url: str = Form(""),
    model_name: str = Form(""),
    runtime_manager=Depends(get_runtime_manager),
) -> ConfigUploadResponse:
    try:
        await runtime_manager.ensure_configuration_mutable()
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _require_real_mapping_mode((_clean_form_text(microscope_mode) or "real").lower())
    filename = str(file.filename or "")
    if not filename.lower().endswith(".cfg"):
        raise HTTPException(status_code=400, detail="Please upload a .cfg file.")

    UPLOADED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOADED_CFG_DIR / Path(filename).name
    with saved_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    runtime_settings = load_runtime_settings(apply_demo_overlay=False)
    request_api_key = _clean_form_text(openai_api_key)
    request_base_url = _clean_form_text(base_url)
    request_model_name = _clean_form_text(model_name)
    if request_api_key:
        runtime_settings.model.openai_api_key = request_api_key
    if request_base_url:
        runtime_settings.model.base_url = request_base_url
    if request_model_name:
        runtime_settings.model.model_name = request_model_name
    inventory = build_cfg_inventory(saved_path)
    inspection_status = "skipped"
    inspected_device_count = 0
    inspection_warning = ""
    if inspect_hardware:
        try:
            runtime_inventory = await runtime_manager.inspect_micro_manager_hardware(
                inspect_micro_manager_config,
                mm_dir=_clean_form_text(mm_dir) or str(runtime_settings.system.MM_DIR or ""),
                config_path=saved_path,
            )
            inventory = merge_runtime_inventory(inventory, runtime_inventory)
            inspection_status = "completed"
            inspected_device_count = len(runtime_inventory.get("devices", []))
        except LifecycleConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OSError as exc:
            inspection_status = "unavailable"
            logger.warning("Micro-Manager hardware inspection failed: %s", exc)
            inspection_warning = (
                "Micro-Manager hardware inspection failed: the cfg could not be loaded. "
                "The hardware may not be connected or a device adapter failed to initialize."
            )
        except Exception as exc:
            inspection_status = "unavailable"
            logger.warning("Micro-Manager hardware inspection failed: %s", exc)
            inspection_warning = "Micro-Manager hardware inspection failed."
    analysis = analyze_config_mapping(
        inventory=inventory,
        model_config=runtime_settings.model,
        current_system=runtime_settings.system,
    )
    analysis.hardware_inspection_status = inspection_status
    analysis.inspected_device_count = inspected_device_count
    if inspection_warning:
        analysis.warnings.append(inspection_warning)

    return ConfigUploadResponse(
        config_path=str(saved_path),
        mapping=analysis,
    )


@router.post("/api/config/test-llm", response_model=LLMConnectionTestResponse)
async def test_llm_connection(req: LLMConnectionTestRequest) -> LLMConnectionTestResponse:
    config = LLMConnectionConfig(
        openai_api_key=req.openai_api_key,
        base_url=req.base_url,
        model_name=req.model_name,
    )
    try:
        await asyncio.to_thread(validate_llm_connection, config)
    except Exception as exc:
        return LLMConnectionTestResponse(ok=False, detail=_format_llm_connection_error(exc))
    return LLMConnectionTestResponse(ok=True, detail="")


@router.post("/api/config/test-vlm", response_model=VLMConnectionTestResponse)
async def test_vlm_connection(req: VLMConnectionTestRequest) -> VLMConnectionTestResponse:
    config = VLMConnectionConfig(
        vlm_api_key=req.vlm_api_key,
        vlm_base_url=req.vlm_base_url,
        vlm_model_name=req.vlm_model_name,
    )
    try:
        await asyncio.to_thread(validate_vlm_connection, config)
    except Exception as exc:
        return VLMConnectionTestResponse(ok=False, detail=_format_llm_connection_error(exc))
    return VLMConnectionTestResponse(ok=True, detail="")


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
        microscope_mode in {"demo", "mock"}
        or is_demo_mapping_payload(
            config_path=req.config_path,
            camera_device=req.camera_device,
            xy_stage_device=req.xy_stage_device,
            objective_device=req.objective_device,
            focus_drive=req.focus_drive,
            dichroic=req.Dichroic,
            objectives=req.objectives,
            channels=req.channels,
            transmitted_light=req.transmitted_light,
        )
    )
    system_updates = {
        "MM_DIR": (
            system_current["MM_DIR"]
            if preserve_persisted_hardware_fields
            else coalesce_text(req.mm_dir, system_current["MM_DIR"])
        ),
        "CONFIG_PATH": (
            system_current["CONFIG_PATH"]
            if preserve_persisted_hardware_fields
            else normalize_config_path(req.config_path, system_current["CONFIG_PATH"])
        ),
        "FIJI_PATH": coalesce_text(req.fiji_path, system_current["FIJI_PATH"]),
        "camera_device": (
            system_current["camera_device"]
            if preserve_persisted_hardware_fields
            else coalesce_text(req.camera_device, system_current["camera_device"])
        ),
        "xy_stage_device": (
            system_current["xy_stage_device"]
            if preserve_persisted_hardware_fields
            else coalesce_text(req.xy_stage_device, system_current["xy_stage_device"])
        ),
        "objective_device": (
            system_current["objective_device"]
            if preserve_persisted_hardware_fields
            else coalesce_text(req.objective_device, system_current["objective_device"])
        ),
        "focus_drive": (
            system_current["focus_drive"]
            if preserve_persisted_hardware_fields
            else coalesce_text(req.focus_drive, system_current["focus_drive"])
        ),
        "Dichroic": (
            system_current["Dichroic"]
            if preserve_persisted_hardware_fields
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
    maybe_number_update(system_updates, "Max_exposure", req.Max_exposure)
    maybe_number_update(system_updates, "Min_exposure", req.Min_exposure)
    if req.objectives and not preserve_persisted_hardware_fields:
        system_updates["objectives"] = req.objectives
    if req.channels and not preserve_persisted_hardware_fields:
        system_updates["channels"] = req.channels
    if req.transmitted_light and not preserve_persisted_hardware_fields:
        system_updates["transmitted_light"] = req.transmitted_light
    if req.demo_environment and microscope_mode == "demo":
        system_updates["demo_environment"] = req.demo_environment
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
    startup_updates = None
    if microscope_mode in {"real", "mock"}:
        effective_system = {**system_current, **system_updates}
        startup_objective = coalesce_text(req.startup_objective, startup_current["objective"])
        startup_channel = coalesce_text(req.startup_channel, startup_current["channel"])
        if microscope_mode == "mock":
            mock_capabilities = build_mock_microscope_capabilities()
            objectives = mock_capabilities["objectives"]
            channels = mock_capabilities["channels"]
        else:
            objectives = effective_system.get("objectives", {})
            channels = effective_system.get("channels", {})
        objective_entry = objectives.get(startup_objective, {}) if isinstance(objectives, dict) else {}
        channel_entry = channels.get(startup_channel, {}) if isinstance(channels, dict) else {}
        objective_is_valid = isinstance(objective_entry, dict) and bool(objective_entry) and (
            microscope_mode == "mock" or bool(str(objective_entry.get("label") or "").strip())
        )
        channel_is_valid = isinstance(channel_entry, dict) and bool(channel_entry) and (
            microscope_mode == "mock" or bool(str(channel_entry.get("label") or "").strip())
        )
        if not objective_is_valid:
            raise HTTPException(
                status_code=422,
                detail=f"Startup objective '{startup_objective}' is not available in {microscope_mode} microscope mode.",
            )
        if not channel_is_valid:
            raise HTTPException(
                status_code=422,
                detail=f"Startup channel '{startup_channel}' is not available in {microscope_mode} microscope mode.",
            )
        startup_updates = {
            "objective": startup_objective,
            "channel": startup_channel,
            "exposure": coalesce_number(req.startup_exposure, startup_current["exposure"]),
            "brightness": coalesce_number(req.startup_brightness, startup_current["brightness"]),
            "z_position": coalesce_number(req.startup_z_position, startup_current["z_position"]),
            "x_position": coalesce_number(req.startup_x_position, startup_current["x_position"]),
            "y_position": coalesce_number(req.startup_y_position, startup_current["y_position"]),
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
        effective_config_path="" if microscope_mode == "mock" else saved_snapshot["system"]["CONFIG_PATH"],
        system_phase=save_result.get("system_phase", runtime_manager.system_status.system_phase),
        preview_phase=preview_phase,
        failure_step=save_result.get("failure_step", runtime_manager.system_status.failure_step),
    )

