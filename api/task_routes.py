import logging
import os
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from openai import APIStatusError

from api.dependencies import get_runtime_manager
from api.models import CommandRequest, PreviewStartResponse, RuntimeInitializationResponse, SystemShutdownResponse, SystemStatusResponse, TaskExecutionResponse
from runtime.asset_check import check_snapshot_assets
from services.runtime_manager import LifecycleConflictError


router = APIRouter()
logger = logging.getLogger(__name__)


def _upstream_http_status(exc: Exception) -> int | None:
    if isinstance(exc, APIStatusError):
        try:
            return int(exc.response.status_code)
        except Exception as status_exc:
            logger.debug("Failed to extract upstream API HTTP status from %s: %s", type(exc).__name__, status_exc, exc_info=True)
            return None
    return None


def _terminate_current_process() -> None:
    # Give the HTTP response a brief window to flush before terminating the server process.
    time.sleep(0.5)
    os._exit(0)


@router.post("/api/system/initialize", response_model=RuntimeInitializationResponse)
async def initialize_system_api(runtime_manager=Depends(get_runtime_manager)) -> RuntimeInitializationResponse:
    try:
        result = RuntimeInitializationResponse.model_validate(await runtime_manager.restart_runtime())
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result.initialized and not result.initializing:
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.post("/api/system/restart", response_model=RuntimeInitializationResponse)
async def restart_system_api(runtime_manager=Depends(get_runtime_manager)) -> RuntimeInitializationResponse:
    try:
        result = RuntimeInitializationResponse.model_validate(await runtime_manager.restart_runtime())
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result.initialized and not result.initializing and result.system_phase == "unconfigured":
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.post("/api/system/shutdown", response_model=SystemShutdownResponse)
async def shutdown_system_api(
    background_tasks: BackgroundTasks,
    runtime_manager=Depends(get_runtime_manager),
) -> SystemShutdownResponse:
    try:
        await runtime_manager.shutdown_runtime()
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    runtime_manager._set_system_status(
        phase="unconfigured",
        initialized=False,
        initializing=False,
        error=None,
        message="Backend shutdown requested.",
    )
    background_tasks.add_task(_terminate_current_process)
    return SystemShutdownResponse(
        shutting_down=True,
        message="Backend shutdown requested. This page will disconnect in a moment.",
    )


@router.get("/api/system/status", response_model=SystemStatusResponse)
async def get_system_status(runtime_manager=Depends(get_runtime_manager)) -> SystemStatusResponse:
    snapshot = runtime_manager.current_snapshot()
    preview_phase = runtime_manager.get_preview_status().get("preview_phase", "idle")
    asset_check = check_snapshot_assets(snapshot)
    return SystemStatusResponse(
        configured=asset_check.ready,
        initialized=runtime_manager.system_status.initialized,
        initializing=runtime_manager.system_status.initializing,
        error=bool(runtime_manager.system_status.error),
        message=runtime_manager.system_status.message,
        system_phase=runtime_manager.system_status.system_phase,
        preview_phase=preview_phase,
        failure_step=runtime_manager.system_status.failure_step,
    )


@router.post("/api/preview/start", response_model=PreviewStartResponse)
async def start_preview_api(runtime_manager=Depends(get_runtime_manager)) -> PreviewStartResponse:
    if runtime_manager.system_status.system_phase != "ready":
        raise HTTPException(status_code=409, detail=runtime_manager.system_status.message)
    return PreviewStartResponse.model_validate(await runtime_manager.start_preview())


@router.post("/api/execute", response_model=TaskExecutionResponse)
async def execute_command_api(req: CommandRequest, runtime_manager=Depends(get_runtime_manager)) -> TaskExecutionResponse:
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command cannot be empty.")

    runtime_manager.enqueue_output_message({"type": "robot_say", "text": f"Command received: {command}"})
    try:
        return TaskExecutionResponse.model_validate(await runtime_manager.execute_exclusive(command))
    except LifecycleConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Command execution failed")
        message = runtime_manager.humanize_exception_message(exc, context="execution")
        runtime_manager.enqueue_output_message({"type": "error", "text": message})
        raise HTTPException(status_code=_upstream_http_status(exc) or 500, detail=message) from exc
