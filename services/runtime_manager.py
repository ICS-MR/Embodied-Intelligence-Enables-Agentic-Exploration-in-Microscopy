import asyncio
import json
import logging
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from urllib.parse import quote

import cv2
import numpy as np

from api.models import RuntimeInitializationResponse, TaskExecutionResponse, UserInputResponse
from api.state import AppState
from bootstrap.config import (
    config_is_complete,
    load_runtime_settings,
    missing_required_fields,
    read_config_snapshot,
    save_runtime_settings,
)
from services.runtime_state import SystemStatus
from services.task_orchestrator import TaskRequest
from utils.interaction_flow import interpret_plan_feedback, is_debug_plan_request, pick_text, prefers_chinese
from utils.runtime_core import (
    RuntimeContext,
    apply_startup_state,
    initialize_microscope,
    initialize_system_components,
    release_resources,
)
from utils.runtime_text import format_raw_planner_debug


logger = logging.getLogger(__name__)


PREVIEW_STALE_FRAME_SEC = 2.0
PREVIEW_STARTUP_GRACE_SEC = 3.0
PREVIEW_START_REQUEST_GRACE_SEC = 5.0
PREVIEW_START_COMMAND_TIMEOUT_SEC = 10.0
PREVIEW_FALLBACK_LOG_INTERVAL_SEC = 5.0
INIT_COMPONENT_TIMEOUT_SEC = 90.0
MICROSCOPE_SETUP_TIMEOUT_SEC = 30.0


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


class RuntimeManager:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.app_state = AppState()
        self.runtime_context: Optional[RuntimeContext] = None
        self.orchestrator = None
        self.server_loop: Optional[asyncio.AbstractEventLoop] = None
        self.system_status = SystemStatus()
        self._last_preview_fallback_log_at = 0.0
        self._initialization_lock = asyncio.Lock()
        self._initialization_task: Optional[asyncio.Task[dict[str, Any]]] = None
        self._preview_phase = "idle"
        self._preview_start_requested_at: Optional[float] = None
        self._preview_starting = False
        self._preview_started_once = False
        self._artifact_preview_window_name = "Fiji Detection Result"

    def _show_local_artifact_preview(self, artifact_path: str, *, title: str, display_seconds: float) -> None:
        display_seconds = max(0.0, float(display_seconds))
        if display_seconds <= 0:
            return
        preview_script = (self.root_dir / "scripts" / "show_timed_image_preview.py").resolve()
        if not preview_script.exists():
            logger.warning("Timed preview script not found: %s", preview_script)
            return

        try:
            subprocess.Popen(
                [
                    sys.executable,
                    str(preview_script),
                    "--image",
                    str(Path(artifact_path).expanduser().resolve()),
                    "--title",
                    str(title or self._artifact_preview_window_name),
                    "--seconds",
                    str(display_seconds),
                ],
                cwd=str(self.root_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.exception("Failed to launch timed local artifact preview for %s", artifact_path)

    def bind_event_loop(self) -> None:
        try:
            self.server_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.server_loop = None

    def current_snapshot(self, *, apply_env: bool = True) -> dict[str, Any]:
        return read_config_snapshot(apply_env=apply_env)

    def update_settings(
        self,
        *,
        system_updates: Optional[dict[str, Any]] = None,
        model_updates: Optional[dict[str, Any]] = None,
        startup_updates: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        save_runtime_settings(
            system_updates=system_updates,
            model_updates=model_updates,
            startup_updates=startup_updates,
        )
        return self.current_snapshot()

    def enqueue_output_message(self, message: dict[str, Any]) -> None:
        if not self.server_loop:
            return
        self.server_loop.call_soon_threadsafe(
            self.app_state.session.output_queue.put_nowait,
            message,
        )

    def _send_message(self, message_type: str, text: str, **extra: Any) -> None:
        payload = {"type": message_type, "text": text}
        payload.update(extra)
        self.enqueue_output_message(payload)

    def _record_user_input(
        self,
        text: str,
        *,
        input_kind: str,
        prompt_text: str = "",
        prompt_mode: str = "",
        command_snapshot: str = "",
    ) -> None:
        if self.runtime_context is None:
            return
        self.runtime_context.history_manager.record_user_input(
            str(text),
            source="web",
            input_kind=input_kind,
            prompt_text=prompt_text,
            prompt_mode=prompt_mode,
            command_snapshot=command_snapshot,
        )

    def _bind_interaction_artifact_listener(self) -> None:
        if self.runtime_context is None:
            return
        env_imagej = getattr(self.runtime_context, "env_imagej", None)
        if env_imagej is not None and hasattr(env_imagej, "set_interaction_artifact_listener"):
            env_imagej.set_interaction_artifact_listener(self.emit_interaction_artifact)

    def _clear_interaction_artifact_listener(self) -> None:
        if self.runtime_context is None:
            return
        env_imagej = getattr(self.runtime_context, "env_imagej", None)
        if env_imagej is not None and hasattr(env_imagej, "set_interaction_artifact_listener"):
            env_imagej.set_interaction_artifact_listener(None)

    def resolve_runtime_artifact_path(self, artifact_path: str) -> Path:
        if self.runtime_context is None:
            raise FileNotFoundError("Runtime is not initialized")

        output_dir = Path(self.runtime_context.output_dir).expanduser().resolve()
        candidate = (output_dir / artifact_path).expanduser().resolve()
        try:
            candidate.relative_to(output_dir)
        except ValueError as exc:
            raise ValueError("Artifact path is outside the current runtime output directory") from exc

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        return candidate

    def emit_interaction_artifact(self, artifact: dict[str, Any]) -> None:
        if self.runtime_context is None:
            return

        artifact_path = str(artifact.get("path") or "").strip()
        if not artifact_path:
            return
        title = str(artifact.get("title") or "")
        display_seconds = max(0.0, float(artifact.get("display_seconds") or 0.0))

        try:
            output_dir = Path(self.runtime_context.output_dir).expanduser().resolve()
            resolved_path = Path(artifact_path).expanduser().resolve()
            relative_path = resolved_path.relative_to(output_dir).as_posix()
        except Exception:
            logger.warning("Ignoring interaction artifact outside runtime output directory: %s", artifact_path)
            return

        self._show_local_artifact_preview(
            str(resolved_path),
            title=title or "Fiji Detection Result",
            display_seconds=display_seconds,
        )

    def _set_system_status(
        self,
        *,
        phase: str,
        message: str,
        initialized: Optional[bool] = None,
        initializing: Optional[bool] = None,
        error: Optional[str] = None,
        failure_step: str = "",
    ) -> None:
        if initialized is None:
            initialized = phase == "ready"
        if initializing is None:
            initializing = phase in {"initializing", "resetting"}
        self.system_status.initialized = initialized
        self.system_status.initializing = initializing
        self.system_status.error = error
        self.system_status.message = message
        self.system_status.system_phase = phase
        self.system_status.failure_step = failure_step if error else ""

    def _reset_preview_state(self) -> None:
        self._preview_phase = "idle"
        self._preview_start_requested_at = None
        self._preview_starting = False
        self._preview_started_once = False

    def refresh_status_after_config_save(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if not config_is_complete(snapshot):
            missing = missing_required_fields(snapshot)
            missing_fields = [*missing["agent"], *missing["system"]]
            missing_text = ", ".join(missing_fields)
            simulation_mode = bool(snapshot["agent"].get("Simulation_mode", True))
            if simulation_mode:
                message = (
                    "Configuration saved. Before starting the system, complete these required fields "
                    f"for simulation mode: {missing_text}."
                )
            else:
                message = (
                    "Configuration saved. Before starting the system, complete these required fields "
                    f"for real hardware mode: {missing_text}."
                )
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message=message,
            )
        elif self.runtime_context is not None and self.orchestrator is not None and self.system_status.initialized:
            self._set_system_status(
                phase="ready",
                initialized=True,
                initializing=False,
                error=None,
                message="Configuration saved. Reset and restart the system to apply changes.",
            )
        else:
            self._set_system_status(
                phase="ready_to_start",
                initialized=False,
                initializing=False,
                error=None,
                message="Configuration saved. Start the system when ready.",
            )
        return self._make_init_response().model_dump()

    def _make_init_response(self) -> RuntimeInitializationResponse:
        return RuntimeInitializationResponse(
            initialized=self.system_status.initialized,
            initializing=self.system_status.initializing,
            message=self.system_status.message,
            system_phase=self.system_status.system_phase,
            failure_step=self.system_status.failure_step,
        )

    def _make_task_response(
        self,
        *,
        status: str,
        retry_times: int,
        summary: str,
        task_id: str,
        model_name: str,
    ) -> TaskExecutionResponse:
        response = TaskExecutionResponse(
            status=status,
            retry_times=retry_times,
            summary=summary,
            task_id=task_id,
            model_name=model_name,
        )
        self.app_state.task.last_result = response
        self.app_state.task.current_task_id = task_id
        return response

    def _start_llm_stream(self, *, role: str, final_type: str) -> str:
        stream_id = uuid.uuid4().hex
        self.enqueue_output_message(
            {
                "type": "llm_stream_start",
                "stream_id": stream_id,
                "role": role,
                "final_type": final_type,
            }
        )
        return stream_id

    def _push_llm_stream_delta(self, stream_id: str, delta: str) -> None:
        if not delta:
            return
        self.enqueue_output_message(
            {
                "type": "llm_stream_delta",
                "stream_id": stream_id,
                "delta": delta,
            }
        )

    def _finish_llm_stream(self, stream_id: str, *, final_type: str) -> None:
        self.enqueue_output_message(
            {
                "type": "llm_stream_end",
                "stream_id": stream_id,
                "final_type": final_type,
            }
        )

    async def _stream_scopebot_message(self, producer, *, final_type: str = "robot_say") -> str:
        stream_id = self._start_llm_stream(role="robot", final_type=final_type)
        emitted_chunks: list[str] = []

        def on_delta(delta: str) -> None:
            if not delta:
                return
            emitted_chunks.append(delta)
            self._push_llm_stream_delta(stream_id, delta)

        try:
            text = await asyncio.to_thread(producer, on_delta)
            if text and not emitted_chunks:
                self._push_llm_stream_delta(stream_id, text)
                emitted_chunks.append(text)
            return text or "".join(emitted_chunks)
        finally:
            self._finish_llm_stream(stream_id, final_type=final_type)

    def _clear_pending_user_inputs(self) -> None:
        while True:
            try:
                self.app_state.session.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _build_cancel_payload(self, task_id: str, model_name: str, *, prefers_zh: bool) -> TaskExecutionResponse:
        cancellation_text = pick_text(
            prefers_zh,
            "Okay, I will not execute this plan for now. You can revise the request and send it again.",
            "Okay, I will not execute this plan for now. You can revise the request and send it again.",
        )
        self._send_message("robot_say", cancellation_text)
        self._send_message("task_complete", "")
        return self._make_task_response(
            status="cancelled",
            retry_times=0,
            summary=cancellation_text,
            task_id=task_id,
            model_name=model_name,
        )

    def _build_task_failure_summary(self, command: str, result: Any, *, prefers_zh: bool) -> str:
        detail = str(getattr(result, "summary", "") or getattr(result, "error", "") or "").strip()
        retry_times = int(getattr(result, "retry_times", 0) or 0)
        checker_summary = str(getattr(result, "checker_summary", "") or "").strip()

        if prefers_zh:
            retry_text = f" after {retry_times} attempt(s)" if retry_times else ""
            message = f"This task did not complete successfully{retry_text}, so I have stopped execution. "
            message += f"Reason: {detail}" if detail else "No more specific failure reason was returned."
            if checker_summary:
                message += f" Checker feedback: {checker_summary}"
            return message

        retry_text = f" after {retry_times} attempt(s)" if retry_times else ""
        message = f"This task did not complete successfully{retry_text}, so I have stopped execution. "
        message += f"Reason: {detail}" if detail else "No more specific failure reason was returned."
        if checker_summary:
            message += f" Checker feedback: {checker_summary}"
        return message

    def _emit_plan_outline(self, plan: Any) -> None:
        if self.orchestrator is None:
            return
        try:
            outline = str(self.orchestrator.present_plan(plan) or "").strip()
        except Exception:
            logger.exception("Failed to build plan outline for frontend")
            return
        if outline:
            self._send_message("robot_say", outline)

    def _emit_skill_summary(self, plan: Any, *, prefers_zh: bool) -> None:
        task_manager = getattr(self.runtime_context, "task_manager", None)
        if task_manager is None or not getattr(task_manager, "_skill_enabled", False):
            return

        selected_skills = [str(item).strip() for item in getattr(plan, "selected_skills", []) if str(item).strip()]
        if not selected_skills:
            return

        reason = str(getattr(plan, "skill_reason", "") or "").strip()
        skill_text = ", ".join(selected_skills)
        if prefers_zh:
            message = f"This planning round will use these skills: {skill_text}."
            if reason:
                message += f" Reason: {reason}"
        else:
            message = f"This planning round will use these skills: {skill_text}."
            if reason:
                message += f" Reason: {reason}"
        self._send_message("robot_say", message)

    def _emit_raw_planner_debug(self, plan: Any, *, prefers_zh: bool) -> None:
        message = format_raw_planner_debug(plan, prefers_zh=prefers_zh)
        if message:
            self._send_message("robot_say", message)

    async def _prompt_for_plan_feedback(self, prompt_text: str, *, command_snapshot: str = "") -> str:
        self.app_state.session.is_asking_user = True
        self._send_message("ask_user", prompt_text, mode="plan_confirmation")
        try:
            user_reply = await self.app_state.session.input_queue.get()
            self._record_user_input(
                user_reply,
                input_kind="plan_feedback",
                prompt_text=prompt_text,
                prompt_mode="plan_confirmation",
                command_snapshot=command_snapshot,
            )
            return user_reply
        finally:
            self.app_state.session.is_asking_user = False

    async def _prompt_for_plan_feedback_with_debug(
        self,
        plan: Any,
        prompt_text: str,
        *,
        command_snapshot: str = "",
        prefers_zh: bool,
    ) -> str:
        while True:
            user_reply = await self._prompt_for_plan_feedback(
                prompt_text,
                command_snapshot=command_snapshot,
            )
            if is_debug_plan_request(user_reply):
                self._emit_raw_planner_debug(plan, prefers_zh=prefers_zh)
                continue
            return user_reply

    async def release_system(self) -> None:
        current_context = self.runtime_context
        self.runtime_context = None
        self.orchestrator = None
        self._reset_preview_state()
        if current_context is not None:
            try:
                env_imagej = getattr(current_context, "env_imagej", None)
                if env_imagej is not None and hasattr(env_imagej, "set_interaction_artifact_listener"):
                    env_imagej.set_interaction_artifact_listener(None)
                await asyncio.to_thread(release_resources, current_context)
            except Exception:
                logger.exception("Failed to release system resources cleanly")

    def _build_failure_message(self, step: str, exc: Exception) -> str:
        detail = str(exc).strip() or type(exc).__name__
        if step == "startup_state_apply":
            normalized_detail = detail.lower()
            if normalized_detail in {"xy position out of range", "z position out of range"}:
                try:
                    settings = load_runtime_settings()
                    startup = settings.startup
                    system = settings.system
                    return (
                        "Initialization failed while applying the startup stage position. "
                        f"Saved startup position: X={startup.x_position}, Y={startup.y_position}, Z={startup.z_position}. "
                        f"Allowed range: X {system.Min_X_position} to {system.Max_X_position}, "
                        f"Y {system.Min_Y_position} to {system.Max_Y_position}, "
                        f"Z {system.Min_Z_position} to {system.Max_Z_position}. "
                        "The stage origin may not be aligned with the saved startup coordinates, so the requested startup position falls outside the configured travel range."
                    )
                except Exception:
                    return (
                        "Initialization failed while applying the startup stage position because it is outside the configured travel range. "
                        "The stage origin may not be aligned with the saved startup coordinates."
                    )
            return (
                f"Initialization failed during {step}: {detail}. "
                "XY initial movement was not executed during startup."
            )
        return f"Initialization failed during {step}: {detail}"

    def humanize_exception_message(self, exc: Exception, *, context: str = "runtime") -> str:
        detail = str(exc).strip() or type(exc).__name__
        normalized = detail.lower()

        if normalized == "xy position out of range":
            return "The requested XY stage position is outside the configured travel range."
        if normalized == "z position out of range":
            return "The requested Z stage position is outside the configured travel range."
        if normalized == "stitching area out of range":
            return "The requested stitching area extends outside the configured stage travel range."
        if "validation error for taskexecutionresponse" in normalized:
            return "The backend produced an invalid internal task response."
        if re.search(r"timed out after \d+s", normalized):
            if context == "initialization":
                return "System initialization timed out while waiting for a runtime or hardware step to finish."
            if context == "execution":
                return "Task execution timed out while waiting for a runtime or hardware step to finish."
            return "An internal operation timed out while waiting for a runtime or hardware step to finish."

        if context == "execution":
            return "Task execution failed because of an internal runtime error."
        if context == "initialization":
            return "System initialization failed because of an internal runtime error."
        return "The system encountered an internal runtime error."

    async def _finalize_init_failure(self, step: str, exc: Exception, runtime_context: RuntimeContext | None = None) -> dict[str, Any]:
        if runtime_context is not None:
            try:
                await asyncio.to_thread(release_resources, runtime_context)
            except Exception:
                logger.exception("Failed to release partially initialized runtime after %s", step)

        self.runtime_context = None
        self.orchestrator = None
        self._reset_preview_state()

        detail = str(exc).strip() or type(exc).__name__
        message = self._build_failure_message(step, exc)
        self._set_system_status(
            phase="init_failed",
            initialized=False,
            initializing=False,
            error=detail,
            message=message,
            failure_step=step,
        )
        self._send_message("error", message)
        return self._make_init_response().model_dump()

    def _validate_runtime_context(self, runtime_context: RuntimeContext) -> None:
        if runtime_context is None:
            raise RuntimeError("runtime context is missing")

        env_olympus = getattr(runtime_context, "env_olympus", None)
        if env_olympus is None:
            raise RuntimeError("microscope environment is missing")

        required_methods = ("start_preview", "get_live_preview_image")
        for method_name in required_methods:
            if not hasattr(env_olympus, method_name):
                raise RuntimeError(f"microscope environment is missing '{method_name}'")

        task_orchestrator = getattr(runtime_context, "task_orchestrator", None)
        if task_orchestrator is None:
            raise RuntimeError("task orchestrator is missing")

    async def _initialize_runtime_once(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if not config_is_complete(snapshot):
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message="Please complete configuration first",
            )
            return self._make_init_response().model_dump()

        await self.release_system()
        self._set_system_status(
            phase="initializing",
            initialized=False,
            initializing=True,
            error=None,
            message="System initializing...",
        )
        self._send_message("robot_say", "System initializing...")

        runtime_context = None
        settings = load_runtime_settings()

        try:
            runtime_context = await asyncio.wait_for(
                asyncio.to_thread(initialize_system_components, settings.model.Simulation_mode),
                timeout=INIT_COMPONENT_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return await self._finalize_init_failure(
                "runtime_build",
                TimeoutError(f"timed out after {INIT_COMPONENT_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except Exception as exc:
            logger.exception("System initialization failed during runtime build")
            return await self._finalize_init_failure("runtime_build", exc, runtime_context)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(initialize_microscope, runtime_context.env_olympus),
                timeout=MICROSCOPE_SETUP_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return await self._finalize_init_failure(
                "microscope_initialize",
                TimeoutError(f"timed out after {MICROSCOPE_SETUP_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except Exception as exc:
            logger.exception("System initialization failed during microscope initialization")
            return await self._finalize_init_failure("microscope_initialize", exc, runtime_context)

        try:
            await asyncio.wait_for(
                asyncio.to_thread(apply_startup_state, runtime_context.env_olympus, settings.startup),
                timeout=MICROSCOPE_SETUP_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            return await self._finalize_init_failure(
                "startup_state_apply",
                TimeoutError(f"timed out after {MICROSCOPE_SETUP_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except Exception as exc:
            logger.exception("System initialization failed during startup state apply")
            return await self._finalize_init_failure("startup_state_apply", exc, runtime_context)

        try:
            self._validate_runtime_context(runtime_context)
        except Exception as exc:
            logger.exception("System initialization failed during post-init validation")
            return await self._finalize_init_failure("post_init_validation", exc, runtime_context)

        self.runtime_context = runtime_context
        self.orchestrator = runtime_context.task_orchestrator
        self._bind_interaction_artifact_listener()
        self._reset_preview_state()

        simulation_mode = bool(runtime_context.runtime["agent"].Simulation_mode)
        ready_message = (
            "System initialization completed. Simulation mode is active. Open the runtime page to start live preview."
            if simulation_mode
            else "System initialization completed. Open the runtime page to start live preview."
        )
        self._set_system_status(
            phase="ready",
            initialized=True,
            initializing=False,
            error=None,
            message="System ready (simulated hardware)" if simulation_mode else "System ready",
        )
        self._send_message("robot_say", ready_message)
        return self._make_init_response().model_dump()

    async def initialize_runtime(self) -> dict[str, Any]:
        async with self._initialization_lock:
            try:
                self._initialization_task = asyncio.current_task()
            except RuntimeError:
                self._initialization_task = None
            try:
                return await self._initialize_runtime_once()
            finally:
                current_task = None
                try:
                    current_task = asyncio.current_task()
                except RuntimeError:
                    current_task = None
                if self._initialization_task is current_task:
                    self._initialization_task = None

    def start_runtime_initialization(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        if not config_is_complete(snapshot):
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message="Please complete configuration first",
            )
            return self._make_init_response().model_dump()

        if self._initialization_task is not None and not self._initialization_task.done():
            self._set_system_status(
                phase="initializing",
                initialized=False,
                initializing=True,
                error=None,
                message="System initialization already in progress...",
            )
            return self._make_init_response().model_dump()

        self._set_system_status(
            phase="initializing",
            initialized=False,
            initializing=True,
            error=None,
            message="System initializing...",
        )

        loop = self.server_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

        if loop is None:
            logger.warning("No running event loop available, falling back to awaited runtime initialization")
            return self._make_init_response().model_dump()

        self._initialization_task = loop.create_task(self.initialize_runtime())
        return self._make_init_response().model_dump()

    async def restart_runtime(self) -> dict[str, Any]:
        if self._initialization_task is not None and not self._initialization_task.done():
            self._set_system_status(
                phase="initializing",
                initialized=False,
                initializing=True,
                error=None,
                message="System initialization already in progress...",
            )
            return self._make_init_response().model_dump()

        snapshot = self.current_snapshot()
        if not config_is_complete(snapshot):
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message="Please complete configuration first",
            )
            return self._make_init_response().model_dump()

        self._set_system_status(
            phase="resetting",
            initialized=False,
            initializing=True,
            error=None,
            message="Resetting system resources...",
        )
        await self.release_system()
        return self.start_runtime_initialization()

    async def start_preview(self) -> dict[str, Any]:
        if not self.system_status.initialized or self.runtime_context is None:
            return {
                "started": False,
                "message": "System is not ready yet.",
                "preview_phase": self.get_preview_status().get("preview_phase", "idle"),
            }

        env_olympus = getattr(self.runtime_context, "env_olympus", None)
        if env_olympus is None or not hasattr(env_olympus, "start_preview") or not hasattr(env_olympus, "get_live_preview_image"):
            self._preview_phase = "failed"
            return {
                "started": False,
                "message": "Preview start failed during preview_start: preview methods are unavailable.",
                "preview_phase": self._preview_phase,
            }

        current_status = self.get_preview_status()
        if current_status["preview_phase"] == "live":
            return {"started": True, "message": "Preview already live.", "preview_phase": "live"}
        if self._preview_starting or current_status["preview_phase"] == "starting":
            return {"started": True, "message": "Preview start already in progress.", "preview_phase": "starting"}

        self._preview_phase = "starting"
        self._preview_start_requested_at = time.monotonic()
        self._preview_starting = True

        try:
            await asyncio.wait_for(
                asyncio.to_thread(env_olympus.start_preview),
                timeout=PREVIEW_START_COMMAND_TIMEOUT_SEC,
            )
            self._preview_started_once = True
            message = "Preview start requested."
        except asyncio.TimeoutError:
            self._preview_phase = "failed"
            message = f"Preview start failed during preview_start: timed out after {PREVIEW_START_COMMAND_TIMEOUT_SEC:.0f}s"
        except Exception as exc:
            self._preview_phase = "failed"
            message = f"Preview start failed during preview_start: {exc}"
        finally:
            self._preview_starting = False

        status = self.get_preview_status()
        if status["preview_phase"] not in {"live", "starting"}:
            self._send_message("error", message)
            return {"started": False, "message": message, "preview_phase": status["preview_phase"]}

        return {"started": True, "message": message, "preview_phase": status["preview_phase"]}

    async def startup(self) -> None:
        self.bind_event_loop()
        self._reset_preview_state()
        snapshot = self.current_snapshot()
        if config_is_complete(snapshot):
            self._set_system_status(
                phase="ready_to_start",
                initialized=False,
                initializing=False,
                error=None,
                message="Configuration loaded. Start the system when ready.",
            )
        else:
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message="Please complete configuration first",
            )

    async def execute_command(self, command: str) -> dict[str, Any]:
        if self.runtime_context is None or self.orchestrator is None:
            raise RuntimeError("System not yet initialized")

        runtime_agent = self.runtime_context.runtime["agent"]
        original_command = command.strip()
        self._record_user_input(original_command, input_kind="initial_command", command_snapshot=original_command)
        prefers_zh = prefers_chinese(original_command)
        current_command = original_command
        revisions: list[str] = []
        plan = None

        self._clear_pending_user_inputs()
        try:
            while True:
                request = TaskRequest(user_command=current_command, session_id="default", human_mode=True)
                # Planner and its LLM calls are synchronous; keep them off the event loop so
                # MJPEG preview and SSE updates can continue while the model is thinking.
                plan = await asyncio.to_thread(self.orchestrator.plan, request)
                self.app_state.task.current_task_id = plan.task_id
                self._emit_skill_summary(plan, prefers_zh=prefers_zh)

                if plan.ready and plan.steps:
                    await self._stream_scopebot_message(
                        lambda on_delta: self.orchestrator.stream_plan_preview(plan, on_delta),
                        final_type="robot_say",
                    )
                    user_reply = await self._prompt_for_plan_feedback_with_debug(
                        plan,
                        pick_text(
                            prefers_zh,
                            "If this plan looks good, reply with 'confirm' or 'continue'. If you want changes, send the revision directly. If you want to inspect the raw planner output, reply with 'debug_plan'. If you want to stop, reply with 'cancel'.",
                            "If this plan looks good, reply with 'confirm' or 'continue'. If you want changes, send the revision directly. If you want to inspect the raw planner output, reply with 'debug_plan'. If you want to stop, reply with 'cancel'.",
                        ),
                        command_snapshot=current_command,
                        prefers_zh=prefers_zh,
                    )
                    decision = interpret_plan_feedback(
                        user_reply,
                        plan_ready=True,
                        original_command=original_command,
                        revisions=revisions,
                    )
                    if decision.action == "confirm":
                        break
                    if decision.action == "cancel":
                        return self._build_cancel_payload(plan.task_id, runtime_agent.model_name, prefers_zh=prefers_zh).model_dump()
                    if decision.action == "empty":
                        self._send_message(
                            "robot_say",
                            pick_text(
                                prefers_zh,
                                "I have not received any revision yet. You can reply with 'confirm' or 'continue', send an edit directly, or reply with 'cancel'.",
                                "I have not received any revision yet. You can reply with 'confirm' or 'continue', send an edit directly, or reply with 'cancel'.",
                            ),
                        )
                        continue

                    revisions = decision.revisions
                    current_command = decision.current_command
                    self._send_message(
                        "robot_say",
                        pick_text(
                            prefers_zh,
                            "Received. I will reorganize the plan based on your update.",
                            "Received. I will reorganize the plan based on your update.",
                        ),
                    )
                    continue

                if getattr(plan, "status", "") == "ask_user":
                    prompt_text = str(plan.question or "").strip() or pick_text(
                        prefers_zh,
                        "I need one key detail before I can continue planning.",
                        "I need one key detail before I can continue planning.",
                    )
                    self._send_message("robot_say", prompt_text)
                    user_reply = await self._prompt_for_plan_feedback_with_debug(
                        plan,
                        pick_text(
                            prefers_zh,
                            f"{prompt_text}\nYou can also reply with 'debug_plan' to inspect the raw planner output, or reply with 'cancel'.",
                            f"{prompt_text}\nYou can also reply with 'debug_plan' to inspect the raw planner output, or reply with 'cancel'.",
                        ),
                        command_snapshot=current_command,
                        prefers_zh=prefers_zh,
                    )
                    decision = interpret_plan_feedback(
                        user_reply,
                        plan_ready=False,
                        original_command=original_command,
                        revisions=revisions,
                        planner_question=prompt_text,
                    )
                    if decision.action == "cancel":
                        task_id = plan.task_id if plan is not None else uuid.uuid4().hex
                        return self._build_cancel_payload(task_id, runtime_agent.model_name, prefers_zh=prefers_zh).model_dump()
                    if decision.action == "confirm_without_plan":
                        self._send_message(
                            "robot_say",
                            pick_text(
                                prefers_zh,
                                "I still do not have an executable plan yet, so I cannot start. Please answer the question first or reply with 'cancel'.",
                                "I still do not have an executable plan yet, so I cannot start. Please answer the question first or reply with 'cancel'.",
                            ),
                        )
                        continue
                    if decision.action == "empty":
                        self._send_message(
                            "robot_say",
                            pick_text(
                                prefers_zh,
                                "I have not received any new detail yet. You can answer the question or reply with 'cancel'.",
                                "I have not received any new detail yet. You can answer the question or reply with 'cancel'.",
                            ),
                        )
                        continue

                    revisions = decision.revisions
                    current_command = decision.current_command
                    self._send_message(
                        "robot_say",
                        pick_text(
                            prefers_zh,
                            "Received. I will replan with that new detail.",
                            "Received. I will replan with that new detail.",
                        ),
                    )
                    continue

                if getattr(plan, "status", "") == "unsupported":
                    unsupported_text = pick_text(
                        prefers_zh,
                        "The current system cannot execute this request. Here is the original planner output:",
                        "The current system cannot execute this request. Here is the original planner output:",
                    )
                    self._send_message("robot_say", unsupported_text)
                    self._send_message(
                        "robot_say",
                        str(getattr(plan, "planner_raw_response", "") or getattr(plan, "error", "") or "Unsupported request."),
                    )
                    self._send_message("task_complete", "")
                    return self._make_task_response(
                        status="failed",
                        retry_times=0,
                        summary=str(getattr(plan, "planner_raw_response", "") or getattr(plan, "error", "") or unsupported_text),
                        task_id=plan.task_id if plan is not None else uuid.uuid4().hex,
                        model_name=runtime_agent.model_name,
                    ).model_dump()

                self._send_message(
                    "robot_say",
                    pick_text(
                        prefers_zh,
                        "I still cannot turn the current request into an executable plan. You can add more detail or reply with 'cancel'.",
                        "I still cannot turn the current request into an executable plan. You can add more detail or reply with 'cancel'.",
                    ),
                )
                user_reply = await self._prompt_for_plan_feedback_with_debug(
                    plan,
                    pick_text(
                        prefers_zh,
                        "Please add more revisions. If you want to inspect the raw planner output, reply with 'debug_plan'. If you want to stop this task, reply with 'cancel'.",
                        "Please add more revisions. If you want to inspect the raw planner output, reply with 'debug_plan'. If you want to stop this task, reply with 'cancel'.",
                    ),
                    command_snapshot=current_command,
                    prefers_zh=prefers_zh,
                )
                decision = interpret_plan_feedback(
                    user_reply,
                    plan_ready=False,
                    original_command=original_command,
                    revisions=revisions,
                )
                if decision.action == "cancel":
                    task_id = plan.task_id if plan is not None else uuid.uuid4().hex
                    return self._build_cancel_payload(task_id, runtime_agent.model_name, prefers_zh=prefers_zh).model_dump()
                if decision.action == "confirm_without_plan":
                    self._send_message(
                        "robot_say",
                        pick_text(
                            prefers_zh,
                            "I still do not have an executable updated plan, so I cannot start yet. Please add more revisions or reply with 'cancel'.",
                            "I still do not have an executable updated plan, so I cannot start yet. Please add more revisions or reply with 'cancel'.",
                        ),
                    )
                    continue
                if decision.action == "empty":
                    self._send_message(
                        "robot_say",
                        pick_text(
                            prefers_zh,
                            "I have not received any new revision yet. You can keep refining the request or reply with 'cancel'.",
                            "I have not received any new revision yet. You can keep refining the request or reply with 'cancel'.",
                        ),
                    )
                    continue

                revisions = decision.revisions
                current_command = decision.current_command
                self._send_message(
                    "robot_say",
                    pick_text(
                        prefers_zh,
                        "Received. I will continue replanning based on your update.",
                        "Received. I will continue replanning based on your update.",
                    ),
                )

            self._send_message(
                "robot_say",
                pick_text(
                    prefers_zh,
                    "Confirmation received. I am starting execution now.",
                    "Confirmation received. I am starting execution now.",
                ),
            )

            result = await asyncio.to_thread(
                self.orchestrator.execute,
                plan,
                self.emit_robot_action,
                self.emit_step_summary,
                self.emit_checker_warning,
                False,
            )
            if not result.success:
                failure_summary = self._build_task_failure_summary(
                    original_command,
                    result,
                    prefers_zh=prefers_zh,
                )
                self._send_message("robot_say", failure_summary)
                self._send_message("task_complete", "")
                response = self._make_task_response(
                    status="failed",
                    retry_times=result.retry_times,
                    summary=failure_summary,
                    task_id=result.task_id,
                    model_name=runtime_agent.model_name,
                )
                return response.model_dump()

            summary_text = await self._stream_scopebot_message(
                lambda on_delta: self.orchestrator.stream_task_summary(plan, on_delta, steps=result.steps),
                final_type="robot_say",
            )
            self._send_message("task_complete", "")
            response = self._make_task_response(
                status="executed",
                retry_times=result.retry_times,
                summary=summary_text,
                task_id=result.task_id,
                model_name=runtime_agent.model_name,
            )
            return response.model_dump()
        finally:
            self.app_state.session.is_asking_user = False
    def emit_robot_action(self, summary: str) -> None:
        if summary:
            self._send_message("robot_action", summary)

    def emit_step_summary(self, summary: str) -> None:
        if summary:
            self._send_message("robot_action", summary)

    def emit_checker_warning(self, summary: str) -> None:
        if summary:
            self._send_message("robot_say", summary)

    async def global_message_stream(self) -> AsyncGenerator[str, None]:
        if not self.app_state.session.first_connection_made:
            self.app_state.session.first_connection_made = True
            if self.system_status.initialized:
                await self.app_state.session.output_queue.put(
                    {"type": "robot_say", "text": "Microscope is ready! Please enter commands."}
                )

        while True:
            try:
                msg = await self.app_state.session.output_queue.get()
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("SSE generator error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': self.humanize_exception_message(exc)})}\n\n"

    async def receive_user_input(self, text: str) -> dict[str, str]:
        if not self.app_state.session.is_asking_user:
            return UserInputResponse(status="ignored", message="No user input is being waited for currently").model_dump()
        await self.app_state.session.input_queue.put(text)
        return UserInputResponse(status="ok", message="Input received").model_dump()

    def _build_preview_status(self, env_olympus: Any | None = None) -> dict[str, Any]:
        env = env_olympus
        simulation_mode = True
        if self.runtime_context is not None:
            env = env or self.runtime_context.env_olympus
            simulation_mode = bool(self.runtime_context.runtime["agent"].Simulation_mode)

        status: dict[str, Any] = {
            "available": env is not None,
            "initialized": self.system_status.initialized,
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
            "simulation_mode": simulation_mode,
            "last_frame_age_sec": None,
            "time_since_preview_start_sec": None,
            "last_error": "",
            "preview_phase": self._preview_phase,
        }
        if env is None:
            if self.system_status.initialized:
                status.update(
                    {
                        "stream_state": "stopped",
                        "status_text": "Preview idle",
                        "detail": "Start live preview from the runtime page.",
                        "preview_phase": self._preview_phase if self._preview_phase == "failed" else "idle",
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
        elif isinstance(self._preview_start_requested_at, (int, float)):
            preview_age = max(0.0, time.monotonic() - float(self._preview_start_requested_at))

        has_frame = False
        latest_frame = getattr(env, "latest_display_frame", None)
        if latest_frame is not None:
            try:
                has_frame = np.asarray(latest_frame).size > 0
            except Exception:
                has_frame = False
        if not has_frame and hasattr(env, "get_live_preview_image") and not acquisition_running:
            try:
                sampled_frame = env.get_live_preview_image()
                has_frame = sampled_frame is not None and np.asarray(sampled_frame).size > 0
                if has_frame and last_frame_age is None:
                    last_frame_age = 0.0
            except Exception:
                has_frame = False

        healthy = bool(
            not acquisition_running
            and has_frame
            and (last_frame_age is None or last_frame_age <= PREVIEW_STALE_FRAME_SEC)
            and (preview_running or thread_alive)
            and not preview_error
        )

        preview_phase = self._preview_phase
        if preview_error:
            preview_phase = "failed"
        elif healthy:
            preview_phase = "live"
        elif preview_running:
            preview_phase = "starting"
        elif preview_phase == "starting":
            if preview_age is not None and preview_age > PREVIEW_START_REQUEST_GRACE_SEC:
                preview_phase = "stopped"
        elif self._preview_started_once:
            preview_phase = "stopped"
        else:
            preview_phase = "idle"
        self._preview_phase = preview_phase

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

    def get_preview_status(self) -> dict[str, Any]:
        return self._build_preview_status()

    def _build_preview_placeholder_frame(self, status: dict[str, Any]) -> np.ndarray:
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
        if status.get("simulation_mode") is not None:
            mode = "Simulation" if status["simulation_mode"] else "Real hardware"
            meta.append(f"Mode: {mode}")
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
                if self.runtime_context is not None and self.system_status.initialized:
                    env_olympus = self.runtime_context.env_olympus

                preview_status = self._build_preview_status(env_olympus)
                if env_olympus is not None and hasattr(env_olympus, "get_live_preview_image"):
                    frame = _normalize_stream_frame(env_olympus.get_live_preview_image())

                if frame is None:
                    if (time.monotonic() - self._last_preview_fallback_log_at) >= PREVIEW_FALLBACK_LOG_INTERVAL_SEC:
                        logger.warning(
                            "Streaming preview placeholder frame. state=%s detail=%s",
                            preview_status.get("stream_state"),
                            preview_status.get("detail"),
                        )
                        self._last_preview_fallback_log_at = time.monotonic()
                    frame = self._build_preview_placeholder_frame(preview_status)

                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ret:
                    await asyncio.sleep(0.1)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Video stream error: %s", exc)
                await asyncio.sleep(0.2)


