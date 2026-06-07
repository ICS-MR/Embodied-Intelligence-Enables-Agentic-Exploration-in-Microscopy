from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_runtime_manager
from api.models import CommandRequest, PreviewStartResponse, RuntimeInitializationResponse, SystemStatusResponse, TaskExecutionResponse
from bootstrap.config import config_is_complete


router = APIRouter()


@router.post("/api/system/initialize", response_model=RuntimeInitializationResponse)
async def initialize_system_api(runtime_manager=Depends(get_runtime_manager)) -> RuntimeInitializationResponse:
    result = RuntimeInitializationResponse.model_validate(await runtime_manager.restart_runtime())
    if not result.initialized and not result.initializing:
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.post("/api/system/restart", response_model=RuntimeInitializationResponse)
async def restart_system_api(runtime_manager=Depends(get_runtime_manager)) -> RuntimeInitializationResponse:
    result = RuntimeInitializationResponse.model_validate(await runtime_manager.restart_runtime())
    if not result.initialized and not result.initializing and result.system_phase == "unconfigured":
        raise HTTPException(status_code=400, detail=result.message)
    return result


@router.get("/api/system/status", response_model=SystemStatusResponse)
async def get_system_status(runtime_manager=Depends(get_runtime_manager)) -> SystemStatusResponse:
    snapshot = runtime_manager.current_snapshot()
    preview_phase = runtime_manager.get_preview_status().get("preview_phase", "idle")
    return SystemStatusResponse(
        configured=config_is_complete(snapshot),
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
    if not runtime_manager.system_status.initialized:
        raise HTTPException(status_code=409, detail=runtime_manager.system_status.message)
    return PreviewStartResponse.model_validate(await runtime_manager.start_preview())


@router.post("/api/execute", response_model=TaskExecutionResponse)
async def execute_command_api(req: CommandRequest, runtime_manager=Depends(get_runtime_manager)) -> TaskExecutionResponse:
    if not runtime_manager.system_status.initialized:
        raise HTTPException(status_code=503, detail=runtime_manager.system_status.message)
    if runtime_manager.app_state.task.running:
        raise HTTPException(status_code=409, detail="Another task is already running. Please wait.")
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command cannot be empty.")

    runtime_manager.app_state.task.running = True
    runtime_manager.enqueue_output_message({"type": "robot_say", "text": f"Command received: {command}"})
    try:
        return TaskExecutionResponse.model_validate(await runtime_manager.execute_command(command))
    except Exception as exc:
        message = runtime_manager.humanize_exception_message(exc, context="execution")
        runtime_manager.enqueue_output_message({"type": "error", "text": message})
        raise HTTPException(status_code=500, detail=message) from exc
    finally:
        runtime_manager.app_state.task.running = False
