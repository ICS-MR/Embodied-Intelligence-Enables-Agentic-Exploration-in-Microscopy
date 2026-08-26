import asyncio
import logging
import threading
import time
from typing import Any, AsyncGenerator, Callable, Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)

PREVIEW_STALE_FRAME_SEC = 2.0
PREVIEW_START_REQUEST_GRACE_SEC = 5.0
PREVIEW_START_COMMAND_TIMEOUT_SEC = 10.0
PREVIEW_FALLBACK_LOG_INTERVAL_SEC = 5.0


def _runtime_modes(runtime_context: Any) -> tuple[str, str, str]:
    if runtime_context is None:
        return "demo", "mock", "mock"
    agent = runtime_context.runtime["agent"]
    return (
        str(getattr(agent, "microscope_mode", "demo")).strip().lower() or "demo",
        str(getattr(agent, "image_analysis_mode", "mock")).strip().lower() or "mock",
        str(getattr(agent, "segmentation_mode", "mock")).strip().lower() or "mock",
    )


def _mode_summary(microscope_mode: str, image_analysis_mode: str, segmentation_mode: str) -> str:
    return f"Microscope: {microscope_mode} | Fiji: {image_analysis_mode} | Cellpose: {segmentation_mode}"


def _normalize_stream_frame(frame: Any) -> Optional[np.ndarray]:
    if frame is None:
        return None

    array = np.asarray(frame)
    if array.size == 0:
        return None

    if array.ndim == 2:
        if array.dtype != np.uint8:
            array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)

    if array.ndim == 3:
        if array.shape[2] == 1:
            base = array[:, :, 0]
            if base.dtype != np.uint8:
                base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        color = array[:, :, :3]
        if color.dtype != np.uint8:
            color = cv2.normalize(color, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return np.ascontiguousarray(color)

    return None


class PreviewStreamService:
    def __init__(
        self,
        *,
        get_runtime_context: Callable[[], Any],
        is_system_initialized: Callable[[], bool],
        run_blocking_step: Callable[..., Any],
        emit_error: Callable[[str], None],
        preview_start_timeout_seconds: float = PREVIEW_START_COMMAND_TIMEOUT_SEC,
    ) -> None:
        self._get_runtime_context = get_runtime_context
        self._is_system_initialized = is_system_initialized
        self._run_blocking_step = run_blocking_step
        self._emit_error = emit_error
        self._preview_start_timeout_seconds = float(preview_start_timeout_seconds)
        self._last_preview_fallback_log_at = 0.0
        self._phase = "idle"
        self._start_requested_at: Optional[float] = None
        self._starting = False
        self._started_once = False

    def reset(self) -> None:
        self._phase = "idle"
        self._start_requested_at = None
        self._starting = False
        self._started_once = False

    async def start(self) -> dict[str, Any]:
        runtime_context = self._get_runtime_context()
        if not self._is_system_initialized() or runtime_context is None:
            return {
                "started": False,
                "message": "Preview controls are unavailable while the device is busy.",
                "preview_phase": self.get_status().get("preview_phase", "idle"),
            }

        env_olympus = getattr(runtime_context, "env_olympus", None)
        if env_olympus is None or not hasattr(env_olympus, "start_preview") or not hasattr(env_olympus, "get_live_preview_image"):
            self._phase = "failed"
            return {
                "started": False,
                "message": "Preview start failed during preview_start: preview methods are unavailable.",
                "preview_phase": self._phase,
            }

        current_status = self.get_status()
        if current_status["preview_phase"] == "live":
            return {"started": True, "message": "Preview already live.", "preview_phase": "live"}
        if self._starting or current_status["preview_phase"] == "starting":
            return {"started": True, "message": "Preview start already in progress.", "preview_phase": "starting"}

        self._phase = "starting"
        self._start_requested_at = time.monotonic()
        self._starting = True

        try:
            await self._run_blocking_step(
                lambda env: env.start_preview(),
                env_olympus,
                timeout=self._preview_start_timeout_seconds,
                cancel_event=threading.Event(),
            )
            self._started_once = True
            message = "Preview start requested."
        except asyncio.TimeoutError:
            self._phase = "failed"
            message = f"Preview start failed during preview_start: timed out after {self._preview_start_timeout_seconds:.0f}s"
        except Exception as exc:
            self._phase = "failed"
            message = f"Preview start failed during preview_start: {exc}"
        finally:
            self._starting = False

        status = self.get_status()
        if status["preview_phase"] not in {"live", "starting"}:
            self._emit_error(message)
            return {"started": False, "message": message, "preview_phase": status["preview_phase"]}

        return {"started": True, "message": message, "preview_phase": status["preview_phase"]}

    def get_status(self, env_olympus: Any | None = None) -> dict[str, Any]:
        runtime_context = self._get_runtime_context()
        env = env_olympus
        microscope_mode, image_analysis_mode, segmentation_mode = _runtime_modes(runtime_context)
        if runtime_context is not None:
            env = env or runtime_context.env_olympus

        status: dict[str, Any] = {
            "available": env is not None,
            "initialized": self._is_system_initialized(),
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
            "microscope_mode": microscope_mode,
            "image_analysis_mode": image_analysis_mode,
            "segmentation_mode": segmentation_mode,
            "mode_summary": _mode_summary(microscope_mode, image_analysis_mode, segmentation_mode),
            "last_frame_age_sec": None,
            "time_since_preview_start_sec": None,
            "last_error": "",
            "preview_phase": self._phase,
        }
        if env is None:
            if self._is_system_initialized():
                status.update(
                    {
                        "stream_state": "stopped",
                        "status_text": "Preview idle",
                        "detail": "Start live preview from the runtime page.",
                        "preview_phase": self._phase if self._phase == "failed" else "idle",
                    }
                )
            return status

        preview_running = bool(getattr(env, "preview_running", False))
        acquisition_running = bool(getattr(env, "acquisition_running", False))
        acquisition_thread = getattr(env, "acquisition_thread", None)
        thread_alive = bool(acquisition_thread and acquisition_thread.is_alive())
        preview_error = str(getattr(env, "last_preview_error", "") or "").strip()
        last_frame_at = getattr(env, "last_preview_frame_at", None)
        preview_started_at = getattr(env, "preview_started_at", None)

        last_frame_age = None
        if isinstance(last_frame_at, (int, float)):
            last_frame_age = max(0.0, time.monotonic() - float(last_frame_at))

        preview_age = None
        if isinstance(preview_started_at, (int, float)):
            preview_age = max(0.0, time.monotonic() - float(preview_started_at))
        elif isinstance(self._start_requested_at, (int, float)):
            preview_age = max(0.0, time.monotonic() - float(self._start_requested_at))

        has_frame = False
        latest_frame = getattr(env, "latest_display_frame", None)
        if latest_frame is not None:
            try:
                has_frame = np.asarray(latest_frame).size > 0
            except Exception as exc:
                logger.warning("Preview cached frame is unreadable; treating preview as not ready: %s", exc, exc_info=True)
                has_frame = False
        if not has_frame and hasattr(env, "get_live_preview_image") and not acquisition_running:
            try:
                sampled_frame = env.get_live_preview_image()
                has_frame = sampled_frame is not None and np.asarray(sampled_frame).size > 0
                if has_frame and last_frame_age is None:
                    last_frame_age = 0.0
            except Exception as exc:
                preview_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Preview frame probe failed; preview status will report failure: %s", exc, exc_info=True)
                has_frame = False

        healthy = bool(
            not acquisition_running
            and has_frame
            and (last_frame_age is None or last_frame_age <= PREVIEW_STALE_FRAME_SEC)
            and (preview_running or thread_alive)
            and not preview_error
        )

        preview_phase = self._phase
        if preview_error:
            preview_phase = "failed"
        elif healthy:
            preview_phase = "live"
        elif preview_running:
            preview_phase = "starting"
        elif preview_phase == "starting":
            if preview_age is not None and preview_age > PREVIEW_START_REQUEST_GRACE_SEC:
                preview_phase = "stopped"
        elif self._started_once:
            preview_phase = "stopped"
        else:
            preview_phase = "idle"
        self._phase = preview_phase

        status.update(
            {
                "preview_running": preview_running,
                "acquisition_running": acquisition_running,
                "thread_alive": thread_alive,
                "has_frame": has_frame,
                "healthy": healthy,
                "last_frame_age_sec": last_frame_age,
                "time_since_preview_start_sec": preview_age,
                "last_error": preview_error,
                "preview_phase": preview_phase,
            }
        )

        if acquisition_running:
            status.update(
                {
                    "stream_state": "busy",
                    "status_text": "Preview paused during acquisition",
                    "detail": "The camera is busy with an acquisition task. Live preview will resume afterward.",
                }
            )
        elif preview_phase == "live":
            status.update(
                {
                    "stream_state": "live",
                    "status_text": "Live preview",
                    "detail": "Receiving microscope frames normally.",
                }
            )
        elif preview_phase == "failed":
            status.update(
                {
                    "stream_state": "error",
                    "status_text": "Preview start failed",
                    "detail": preview_error or "Live preview could not be started.",
                }
            )
        elif preview_phase == "starting":
            status.update(
                {
                    "stream_state": "starting",
                    "status_text": "Starting live preview",
                    "detail": "Waiting for the microscope to deliver live preview frames.",
                }
            )
        elif preview_phase == "stopped":
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

    def _build_placeholder_frame(self, status: dict[str, Any]) -> np.ndarray:
        frame = np.full((720, 720, 3), 24, dtype=np.uint8)
        accent_map = {
            "starting": (0, 180, 255),
            "error": (0, 96, 255),
            "busy": (255, 191, 0),
            "stopped": (128, 128, 128),
            "unavailable": (128, 128, 128),
        }
        accent = accent_map.get(status.get("stream_state", "unavailable"), (128, 128, 128))
        cv2.rectangle(frame, (24, 24), (696, 696), accent, 2)
        cv2.rectangle(frame, (24, 24), (696, 120), accent, -1)
        cv2.putText(frame, "Microscope Preview Status", (48, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)

        lines = [str(status.get("status_text") or "Preview unavailable")]
        detail = str(status.get("detail") or "").strip()
        if detail:
            while detail and len(lines) < 5:
                lines.append(detail[:54])
                detail = detail[54:]

        y = 180
        for line in lines[:5]:
            cv2.putText(frame, line, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
            y += 56

        meta = []
        if status.get("last_frame_age_sec") is not None:
            meta.append(f"Last frame age: {status['last_frame_age_sec']:.1f}s")
        microscope_mode = status.get("microscope_mode")
        hardware = {
            "demo": "Micro-Manager demo hardware",
            "mock": "mock microscope",
            "real": "real hardware",
        }.get(microscope_mode, f"unknown microscope mode ({microscope_mode})")
        meta.append(f"Preview source: {hardware}")
        if status.get("mode_summary"):
            meta.append(status["mode_summary"])
        if status.get("auto_restart_enabled") is not None:
            meta.append(f"Auto restart: {'on' if status['auto_restart_enabled'] else 'off'}")

        y = 520
        for line in meta:
            cv2.putText(frame, line, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
            y += 42
        return frame

    async def generate_mjpeg_frames(self) -> AsyncGenerator[bytes, None]:
        while True:
            try:
                frame: Optional[np.ndarray] = None
                env_olympus = None
                runtime_context = self._get_runtime_context()
                if runtime_context is not None and self._is_system_initialized():
                    env_olympus = runtime_context.env_olympus

                preview_status = self.get_status(env_olympus)
                if env_olympus is not None and hasattr(env_olympus, "get_live_preview_image"):
                    try:
                        frame = _normalize_stream_frame(env_olympus.get_live_preview_image())
                    except Exception as exc:
                        logger.warning("Preview frame retrieval failed; streaming placeholder frame: %s", exc, exc_info=True)
                        frame = None

                if frame is None:
                    if (time.monotonic() - self._last_preview_fallback_log_at) >= PREVIEW_FALLBACK_LOG_INTERVAL_SEC:
                        logger.debug(
                            "Streaming preview placeholder frame. state=%s detail=%s",
                            preview_status.get("stream_state"),
                            preview_status.get("detail"),
                        )
                        self._last_preview_fallback_log_at = time.monotonic()
                    frame = self._build_placeholder_frame(preview_status)

                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Video stream error: %s", exc, exc_info=True)
                await asyncio.sleep(0.2)
