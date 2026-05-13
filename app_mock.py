import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from Empty_function import MicroscopeController
from api.models import ConfigSaveRequest
from bootstrap.config import config_is_complete, read_config_snapshot, read_public_config_snapshot, save_runtime_settings
from system_config_wizard import (
    build_dichroic_colors,
    build_objective_labels,
    parse_mm_config,
    suggest_values,
)


ROOT_DIR = Path(__file__).parent
UPLOADED_CFG_DIR = ROOT_DIR / "uploaded_cfg"
MOCK_OUTPUT_DIR = ROOT_DIR / "mock_output"
PREVIEW_START_COMMAND_TIMEOUT_SEC = 10.0
PREVIEW_START_REQUEST_GRACE_SEC = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app_mock")

app = FastAPI(
    title="AI Microscope Assistant (Mock)",
    description="Lightweight mock UI for config flow and simulated microscope runtime.",
    version="0.4.0",
)


class MockSession:
    def __init__(self) -> None:
        self.system_status: Dict[str, Any] = {
            "initialized": False,
            "initializing": False,
            "error": None,
            "message": "Please complete configuration first",
            "system_phase": "unconfigured",
            "failure_step": "",
        }
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.task_running: bool = False
        self.is_asking_user: bool = False
        self.first_connection_made: bool = False
        self.initialization_task: asyncio.Task | None = None
        self.initialization_lock = asyncio.Lock()
        self.preview_phase: str = "idle"
        self.preview_start_requested_at: float | None = None
        self.preview_starting: bool = False
        self.preview_started_once: bool = False


class CommandRequest(BaseModel):
    command: str


class UserInputRequest(BaseModel):
    text: str


session = MockSession()
mock_runtime: Dict[str, Any] = {"microscope": None}


def coalesce_text(new_value: str, current_value: str) -> str:
    value = new_value.strip()
    return value if value else current_value


def coalesce_number(new_value: Any, current_value: Any) -> Any:
    return current_value if new_value is None else new_value


def set_system_status(
    *,
    phase: str,
    message: str,
    initialized: bool | None = None,
    initializing: bool | None = None,
    error: str | None = None,
    failure_step: str = "",
) -> None:
    if initialized is None:
        initialized = phase == "ready"
    if initializing is None:
        initializing = phase in {"initializing", "resetting"}
    session.system_status.update(
        {
            "initialized": initialized,
            "initializing": initializing,
            "error": error,
            "message": message,
            "system_phase": phase,
            "failure_step": failure_step if error else "",
        }
    )


def reset_preview_state() -> None:
    session.preview_phase = "idle"
    session.preview_start_requested_at = None
    session.preview_starting = False
    session.preview_started_once = False


async def robot_say(message: str) -> None:
    await session.output_queue.put({"type": "robot_say", "text": message})
    await asyncio.sleep(0)


async def robot_action_log(message: str) -> None:
    logger.info("Robot action: %s", message)
    await session.output_queue.put({"type": "robot_action", "text": message})
    await asyncio.sleep(0)


def build_mock_runtime() -> Dict[str, Any]:
    snapshot = read_config_snapshot()
    system_cfg = snapshot["system"]
    startup_cfg = snapshot["startup"]

    microscope = MicroscopeController(
        config_path=system_cfg["CONFIG_PATH"],
        app_dir=system_cfg["MM_DIR"],
        output_path=str(MOCK_OUTPUT_DIR),
        storagemanger=None,
    )
    microscope.initialize()
    microscope.set_objective(startup_cfg["objective"])
    microscope.set_channel(startup_cfg["channel"])
    microscope.set_exposure(startup_cfg["exposure"])
    microscope.set_brightness(startup_cfg["brightness"])
    microscope.set_z_position(startup_cfg["z_position"])
    return {"microscope": microscope}


async def release_mock_runtime() -> None:
    microscope = mock_runtime.get("microscope")
    mock_runtime["microscope"] = None
    reset_preview_state()
    if microscope is None:
        return
    shutdown = getattr(microscope, "shutdown", None)
    if callable(shutdown):
        await asyncio.to_thread(shutdown)


def build_preview_status() -> Dict[str, Any]:
    microscope = mock_runtime.get("microscope")
    status: Dict[str, Any] = {
        "available": microscope is not None,
        "initialized": bool(session.system_status["initialized"]),
        "stream_state": "unavailable",
        "status_text": "Preview unavailable",
        "detail": "Runtime is not initialized yet.",
        "healthy": False,
        "preview_running": False,
        "acquisition_running": False,
        "auto_restart_enabled": False,
        "thread_alive": False,
        "has_frame": False,
        "fallback_active": True,
        "simulation_mode": True,
        "last_frame_age_sec": None,
        "time_since_preview_start_sec": None,
        "last_error": "",
        "preview_phase": session.preview_phase,
    }
    if microscope is None:
        if session.system_status["initialized"]:
            status.update(
                {
                    "stream_state": "stopped",
                    "status_text": "Preview idle",
                    "detail": "Start live preview from the runtime page.",
                    "preview_phase": session.preview_phase if session.preview_phase == "failed" else "idle",
                }
            )
        return status

    preview_running = bool(getattr(microscope, "preview_running", False))
    preview_error = str(getattr(microscope, "last_preview_error", "") or "").strip()
    preview_age = None
    if isinstance(getattr(microscope, "preview_started_at", None), (int, float)):
        preview_age = max(0.0, time.monotonic() - float(microscope.preview_started_at))
    elif isinstance(session.preview_start_requested_at, (int, float)):
        preview_age = max(0.0, time.monotonic() - float(session.preview_start_requested_at))

    frame = None
    if preview_running and hasattr(microscope, "get_live_preview_image"):
        try:
            frame = microscope.get_live_preview_image()
        except Exception:
            frame = None
    has_frame = frame is not None and np.asarray(frame).size > 0

    if preview_error:
        session.preview_phase = "failed"
    elif preview_running and has_frame:
        session.preview_phase = "live"
    elif preview_running:
        session.preview_phase = "starting"
    elif session.preview_phase == "starting" and preview_age is not None and preview_age > PREVIEW_START_REQUEST_GRACE_SEC:
        session.preview_phase = "stopped"
    elif session.preview_started_once:
        session.preview_phase = "stopped"
    else:
        session.preview_phase = "idle"

    status.update(
        {
            "preview_running": preview_running,
            "has_frame": has_frame,
            "healthy": session.preview_phase == "live",
            "time_since_preview_start_sec": preview_age,
            "last_error": preview_error,
            "preview_phase": session.preview_phase,
        }
    )

    if session.preview_phase == "live":
        status.update(
            {
                "stream_state": "live",
                "status_text": "Live preview",
                "detail": "Receiving mock microscope frames normally.",
            }
        )
    elif session.preview_phase == "starting":
        status.update(
            {
                "stream_state": "starting",
                "status_text": "Starting live preview",
                "detail": "Waiting for mock preview frames.",
            }
        )
    elif session.preview_phase == "failed":
        status.update(
            {
                "stream_state": "error",
                "status_text": "Preview start failed",
                "detail": preview_error or "Live preview could not be started.",
            }
        )
    elif session.preview_phase == "stopped":
        status.update(
            {
                "stream_state": "stopped",
                "status_text": "Preview stopped",
                "detail": "Live preview is not running. Use Restart Preview to try again.",
            }
        )
    else:
        status.update(
            {
                "stream_state": "stopped",
                "status_text": "Preview idle",
                "detail": "Live preview has not been started yet.",
            }
        )

    status["fallback_active"] = status["preview_phase"] != "live"
    return status


def generate_status_payload() -> Dict[str, Any]:
    snapshot = read_public_config_snapshot()
    preview_phase = build_preview_status()["preview_phase"]
    return {
        "configured": config_is_complete(read_config_snapshot()),
        "initialized": session.system_status["initialized"],
        "initializing": session.system_status["initializing"],
        "error": bool(session.system_status["error"]),
        "status_message": session.system_status["message"],
        "system_phase": session.system_status["system_phase"],
        "preview_phase": preview_phase,
        "failure_step": session.system_status["failure_step"],
        "system": snapshot["system"],
        "agent": snapshot["agent"],
        "startup": snapshot["startup"],
    }


def refresh_status_after_config_save() -> Dict[str, Any]:
    snapshot = read_config_snapshot()
    if not config_is_complete(snapshot):
        set_system_status(
            phase="unconfigured",
            initialized=False,
            initializing=False,
            error=None,
            message="Configuration saved. Please complete all required fields before starting the system.",
        )
    elif mock_runtime.get("microscope") is not None and session.system_status["initialized"]:
        set_system_status(
            phase="ready",
            initialized=True,
            initializing=False,
            error=None,
            message="Configuration saved. Reset and restart the system to apply changes.",
        )
    else:
        set_system_status(
            phase="ready_to_start",
            initialized=False,
            initializing=False,
            error=None,
            message="Configuration saved. Start the system when ready.",
        )
    return {
        "initialized": session.system_status["initialized"],
        "initializing": session.system_status["initializing"],
        "message": session.system_status["message"],
        "system_phase": session.system_status["system_phase"],
        "failure_step": session.system_status["failure_step"],
    }


async def initialize_runtime_once() -> Dict[str, Any]:
    snapshot = read_config_snapshot()
    if not config_is_complete(snapshot):
        set_system_status(
            phase="unconfigured",
            initialized=False,
            initializing=False,
            error=None,
            message="Please complete configuration first",
        )
        return {
            "initialized": False,
            "initializing": False,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    await release_mock_runtime()
    set_system_status(
        phase="initializing",
        initialized=False,
        initializing=True,
        error=None,
        message="Mock system initializing...",
    )
    await robot_action_log("Mock system initializing...")

    try:
        mock_runtime.update(build_mock_runtime())
    except Exception as exc:
        await release_mock_runtime()
        set_system_status(
            phase="init_failed",
            initialized=False,
            initializing=False,
            error=str(exc),
            failure_step="runtime_build",
            message=f"Initialization failed during runtime_build: {exc}",
        )
        return {
            "initialized": False,
            "initializing": False,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    simulation_mode = bool(snapshot["agent"].get("Simulation_mode", True))
    set_system_status(
        phase="ready",
        initialized=True,
        initializing=False,
        error=None,
        message="Mock system ready (simulated hardware)" if simulation_mode else "Mock system ready",
    )
    await robot_say(
        "Mock system initialization completed. Empty_function simulated hardware is active. Live preview will start automatically after entering the runtime page."
        if simulation_mode
        else "Mock system initialization completed. Live preview will start automatically after entering the runtime page."
    )
    return {
        "initialized": True,
        "initializing": False,
        "message": session.system_status["message"],
        "system_phase": session.system_status["system_phase"],
        "failure_step": session.system_status["failure_step"],
    }


async def initialize_runtime() -> Dict[str, Any]:
    async with session.initialization_lock:
        try:
            session.initialization_task = asyncio.current_task()
        except RuntimeError:
            session.initialization_task = None
        try:
            return await initialize_runtime_once()
        finally:
            current_task = None
            try:
                current_task = asyncio.current_task()
            except RuntimeError:
                current_task = None
            if session.initialization_task is current_task:
                session.initialization_task = None


def start_runtime_initialization() -> Dict[str, Any]:
    snapshot = read_config_snapshot()
    if not config_is_complete(snapshot):
        set_system_status(
            phase="unconfigured",
            initialized=False,
            initializing=False,
            error=None,
            message="Please complete configuration first",
        )
        return {
            "initialized": False,
            "initializing": False,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    if session.initialization_task is not None and not session.initialization_task.done():
        set_system_status(
            phase="initializing",
            initialized=False,
            initializing=True,
            error=None,
            message="Mock system initialization already in progress...",
        )
        return {
            "initialized": False,
            "initializing": True,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    set_system_status(
        phase="initializing",
        initialized=False,
        initializing=True,
        error=None,
        message="Mock system initializing...",
    )
    session.initialization_task = asyncio.create_task(initialize_runtime())
    return {
        "initialized": False,
        "initializing": True,
        "message": session.system_status["message"],
        "system_phase": session.system_status["system_phase"],
        "failure_step": session.system_status["failure_step"],
    }


async def restart_runtime() -> Dict[str, Any]:
    if session.initialization_task is not None and not session.initialization_task.done():
        set_system_status(
            phase="initializing",
            initialized=False,
            initializing=True,
            error=None,
            message="Mock system initialization already in progress...",
        )
        return {
            "initialized": False,
            "initializing": True,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    snapshot = read_config_snapshot()
    if not config_is_complete(snapshot):
        set_system_status(
            phase="unconfigured",
            initialized=False,
            initializing=False,
            error=None,
            message="Please complete configuration first",
        )
        return {
            "initialized": False,
            "initializing": False,
            "message": session.system_status["message"],
            "system_phase": session.system_status["system_phase"],
            "failure_step": session.system_status["failure_step"],
        }

    set_system_status(
        phase="resetting",
        initialized=False,
        initializing=True,
        error=None,
        message="Resetting mock runtime...",
    )
    await release_mock_runtime()
    return start_runtime_initialization()


async def start_preview() -> Dict[str, Any]:
    microscope = mock_runtime.get("microscope")
    if microscope is None or not session.system_status["initialized"]:
        return {"started": False, "message": "System is not ready yet.", "preview_phase": build_preview_status()["preview_phase"]}

    current_status = build_preview_status()
    if current_status["preview_phase"] == "live":
        return {"started": True, "message": "Preview already live.", "preview_phase": "live"}
    if session.preview_starting or current_status["preview_phase"] == "starting":
        return {"started": True, "message": "Preview start already in progress.", "preview_phase": "starting"}

    session.preview_phase = "starting"
    session.preview_start_requested_at = time.monotonic()
    session.preview_starting = True
    try:
        await asyncio.wait_for(asyncio.to_thread(microscope.start_preview), timeout=PREVIEW_START_COMMAND_TIMEOUT_SEC)
        session.preview_started_once = True
        message = "Preview start requested."
    except asyncio.TimeoutError:
        session.preview_phase = "failed"
        message = f"Preview start failed during preview_start: timed out after {PREVIEW_START_COMMAND_TIMEOUT_SEC:.0f}s"
    except Exception as exc:
        session.preview_phase = "failed"
        message = f"Preview start failed during preview_start: {exc}"
    finally:
        session.preview_starting = False

    status = build_preview_status()
    return {"started": status["preview_phase"] in {"starting", "live"}, "message": message, "preview_phase": status["preview_phase"]}


async def simulate_task(command: str) -> None:
    try:
        await robot_say(f"Command received: {command}")
        fake_steps = [
            "Parsing the task and generating a microscope operation plan...",
            "Moving the stage and performing autofocus...",
            "Capturing multi-channel images and applying denoising...",
            "Running Cellpose segmentation and exporting statistics...",
        ]
        for step in fake_steps:
            await robot_action_log(step)
            await asyncio.sleep(0.9)
        await session.output_queue.put({"type": "task_complete", "text": f"Task execution completed: {command}"})
    except Exception as exc:
        await session.output_queue.put({"type": "error", "text": f"Execution failed: {exc}"})
    finally:
        session.task_running = False


def _generate_mock_frame() -> np.ndarray:
    microscope = mock_runtime.get("microscope")
    if microscope is None or not getattr(microscope, "preview_running", False):
        return np.zeros((720, 720, 3), dtype=np.uint8)

    img = microscope._acquire_single_image()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    cv2.putText(
        img_color,
        "Mock Preview",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img_color


@app.on_event("startup")
async def startup_event() -> None:
    reset_preview_state()
    snapshot = read_config_snapshot()
    if config_is_complete(snapshot):
        set_system_status(
            phase="ready_to_start",
            initialized=False,
            initializing=False,
            error=None,
            message="Configuration loaded. Start the system when ready.",
        )
    else:
        set_system_status(
            phase="unconfigured",
            initialized=False,
            initializing=False,
            error=None,
            message="Please complete configuration first",
        )


@app.get("/api/config/status")
async def get_config_status() -> Dict[str, Any]:
    return generate_status_payload()


@app.get("/api/stream/preview_status")
async def preview_status() -> Dict[str, Any]:
    return build_preview_status()


@app.post("/api/config/upload-cfg")
async def upload_cfg(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename.lower().endswith(".cfg"):
        raise HTTPException(status_code=400, detail="Please upload a .cfg file.")

    UPLOADED_CFG_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = UPLOADED_CFG_DIR / Path(file.filename).name
    with saved_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    suggestions = suggest_values(saved_path)
    cfg_data = parse_mm_config(saved_path)
    existing = read_config_snapshot(apply_env=False)["system"]
    objective_labels = build_objective_labels(cfg_data, suggestions["objective_device"]["value"], existing["objective_labels"])
    dichroic_colors = build_dichroic_colors(cfg_data, suggestions["Dichroic"]["value"], existing["dichroic_colors"])

    system_updates = {field: info["value"] for field, info in suggestions.items() if info["value"]}
    system_updates["CONFIG_PATH"] = str(saved_path)
    if objective_labels:
        system_updates["objective_labels"] = objective_labels
    if dichroic_colors:
        system_updates["dichroic_colors"] = dichroic_colors
    save_runtime_settings(system_updates=system_updates)

    return {
        "config_path": str(saved_path),
        "suggestions": suggestions,
        "objective_labels": objective_labels,
        "dichroic_colors": dichroic_colors,
    }


@app.post("/api/config/save")
async def save_config(req: ConfigSaveRequest) -> Dict[str, Any]:
    snapshot = read_config_snapshot(apply_env=False)
    system_current = snapshot["system"]
    agent_current = snapshot["agent"]
    startup_current = snapshot["startup"]

    system_updates = {
        "MM_DIR": coalesce_text(req.mm_dir, system_current["MM_DIR"]),
        "CONFIG_PATH": coalesce_text(req.config_path, system_current["CONFIG_PATH"]),
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
        "openai_api_key": req.openai_api_key or agent_current["openai_api_key"],
        "base_url": coalesce_text(req.base_url, agent_current["base_url"]),
        "model_name": coalesce_text(req.model_name, agent_current["model_name"]),
        "vlm_api_key": req.vlm_api_key or agent_current["vlm_api_key"],
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

    save_runtime_settings(
        system_updates=system_updates,
        model_updates=model_updates,
        startup_updates=startup_updates,
    )
    result = refresh_status_after_config_save()
    return {
        "saved": True,
        "initialized": result["initialized"],
        "initializing": result["initializing"],
        "message": result["message"],
        "system_phase": result["system_phase"],
        "preview_phase": build_preview_status()["preview_phase"],
        "failure_step": result["failure_step"],
    }


@app.post("/api/system/initialize")
async def initialize_system_api() -> Dict[str, Any]:
    result = await restart_runtime()
    if not result["initialized"] and not result["initializing"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/system/restart")
async def restart_system_api() -> Dict[str, Any]:
    result = await restart_runtime()
    if not result["initialized"] and not result["initializing"] and result["system_phase"] == "unconfigured":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/preview/start")
async def start_preview_api() -> Dict[str, Any]:
    if not session.system_status["initialized"]:
        raise HTTPException(status_code=409, detail=session.system_status["message"])
    return await start_preview()


@app.get("/api/system/status")
async def get_system_status() -> Dict[str, Any]:
    snapshot = read_config_snapshot()
    return {
        "configured": config_is_complete(snapshot),
        "initialized": session.system_status["initialized"],
        "initializing": session.system_status["initializing"],
        "error": bool(session.system_status["error"]),
        "message": session.system_status["message"],
        "system_phase": session.system_status["system_phase"],
        "preview_phase": build_preview_status()["preview_phase"],
        "failure_step": session.system_status["failure_step"],
    }


@app.get("/api/stream/global")
async def global_message_stream() -> StreamingResponse:
    if not session.first_connection_made:
        session.first_connection_made = True
        if session.system_status["initialized"]:
            await robot_say("The mock microscope is ready. Enter a command to try the frontend workflow.")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            try:
                msg = await session.output_queue.get()
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("SSE error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/execute")
async def execute_command(req: CommandRequest) -> Dict[str, Any]:
    if not session.system_status["initialized"]:
        raise HTTPException(status_code=503, detail=session.system_status["message"])
    if session.task_running:
        raise HTTPException(status_code=409, detail="Another task is already running. Please wait.")
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command cannot be empty.")
    session.task_running = True
    asyncio.create_task(simulate_task(command))
    return {"status": "started", "command": command}


@app.post("/api/stream/user_input")
async def receive_user_input(_: UserInputRequest) -> Dict[str, str]:
    return {"status": "ignored", "message": "The mock environment is not waiting for user input."}


@app.get("/video/stream")
async def video_stream():
    async def frame_gen() -> AsyncGenerator[bytes, None]:
        while True:
            try:
                frame = _generate_mock_frame()
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    frame = np.zeros((720, 720, 3), dtype=np.uint8)
                    _, buffer = cv2.imencode(".jpg", frame)
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Video stream error: %s", exc)
                await asyncio.sleep(0.2)

    return StreamingResponse(frame_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/")
async def serve_frontend():
    index_path = ROOT_DIR / "front" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h2>front/index.html not found</h2>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))

