import asyncio
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from urllib.parse import quote

from openai import APIStatusError

from api.models import RuntimeInitializationResponse, TaskExecutionResponse, UserInputResponse
from api.state import AppState
from bootstrap.config import (
    load_runtime_settings,
    read_config_snapshot,
    save_runtime_settings,
)
from runtime.asset_check import AssetCheckResult, check_snapshot_assets
from services.runtime_state import SystemStatus
from services.preview_stream import PreviewStreamService
from services.runtime_lifecycle import RuntimeLifecyclePorts, RuntimeLifecycleService
from services.task_interaction import InteractionOutcome, TaskInteractionPorts, TaskInteractionSession
from interfaces.interaction_flow import pick_text, prefers_chinese
from runtime.models import RuntimeContext


logger = logging.getLogger(__name__)


class LifecycleConflictError(RuntimeError):
    pass


def _summarize_api_status_error(exc: APIStatusError) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            code = str(error.get("code") or "").strip()
            extra = f" code={code}" if code else ""
            if message:
                return f"Upstream model API returned HTTP {status_code}.{extra} {message}".strip()
        message = str(body.get("message") or "").strip()
        if message:
            return f"Upstream model API returned HTTP {status_code}. {message}"
    if isinstance(body, str) and body.strip():
        return f"Upstream model API returned HTTP {status_code}. {body.strip()}"
    return f"Upstream model API returned HTTP {status_code}."


class RuntimeManager:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.app_state = AppState()
        self.runtime_context: Optional[RuntimeContext] = None
        self.orchestrator = None
        self.server_loop: Optional[asyncio.AbstractEventLoop] = None
        self.system_status = SystemStatus()
        self._initialization_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._initialization_task: Optional[asyncio.Task[dict[str, Any]]] = None
        self._execution_task: Optional[asyncio.Task[dict[str, Any]]] = None
        self._preview_task: Optional[asyncio.Task[dict[str, Any]]] = None
        self.runtime_lifecycle = RuntimeLifecycleService(
            RuntimeLifecyclePorts(
                current_snapshot=self.current_snapshot,
                make_init_response=self._make_init_response,
                set_system_status=self._set_system_status,
                send_message=self._send_message,
                clear_runtime_context=self._clear_runtime_context,
                set_runtime_context=self._set_runtime_context,
                bind_interaction_artifact_listener=self._bind_interaction_artifact_listener,
                reset_preview_state=self._reset_preview_state,
                build_failure_message=self._build_failure_message,
            )
        )
        self.preview_stream = PreviewStreamService(
            get_runtime_context=lambda: self.runtime_context,
            is_system_initialized=lambda: self.system_status.initialized,
            run_blocking_step=self.runtime_lifecycle.run_initialization_step,
            emit_error=lambda message: self._send_message("error", message),
        )
        self._latest_task_progress: Optional[dict[str, Any]] = None

    def bind_event_loop(self) -> None:
        try:
            self.server_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.server_loop = None

    def current_snapshot(self, *, apply_env: bool = True, apply_demo_overlay: bool = True) -> dict[str, Any]:
        return read_config_snapshot(apply_env=apply_env, apply_demo_overlay=apply_demo_overlay)

    def get_transmitted_light_runtime_info(self) -> dict[str, Any]:
        if not self.system_status.initialized or self.runtime_context is None:
            return {}
        microscope = getattr(self.runtime_context, "env_olympus", None)
        get_info = getattr(microscope, "get_transmitted_light_runtime_info", None)
        if not callable(get_info):
            return {}
        return dict(get_info() or {})

    def _current_mode_summary(self) -> str:
        if self.runtime_context is not None:
            agent_cfg = self.runtime_context.runtime["agent"]
            return (
                f"Microscope: {getattr(agent_cfg, 'microscope_mode', 'demo')} | "
                f"Fiji: {getattr(agent_cfg, 'image_analysis_mode', 'mock')} | "
                f"Cellpose: {getattr(agent_cfg, 'segmentation_mode', 'mock')}"
            )
        snapshot = self.current_snapshot()
        agent_cfg = snapshot["agent"]
        return (
            f"Microscope: {agent_cfg.get('microscope_mode', 'demo')} | "
            f"Fiji: {agent_cfg.get('image_analysis_mode', 'mock')} | "
            f"Cellpose: {agent_cfg.get('segmentation_mode', 'mock')}"
        )

    def _asset_check(self) -> AssetCheckResult:
        return check_snapshot_assets(self.current_snapshot())

    @staticmethod
    def _asset_blocking_message(result: AssetCheckResult, prefix: str = "Please complete configuration first") -> str:
        detail = result.blocking_summary()
        if not detail:
            return prefix
        return f"{prefix} for {result.mode_summary}: {detail}"

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
            self._publish_output_message,
            message,
        )

    def _publish_output_message(self, message: dict[str, Any]) -> None:
        session = self.app_state.session
        if session.output_subscribers:
            for subscriber in tuple(session.output_subscribers):
                subscriber.put_nowait(dict(message))
            return
        # Keep messages produced before the first SSE connection for the next client.
        session.output_queue.put_nowait(message)

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

    def _bind_task_progress_listener(self) -> None:
        if self.runtime_context is None:
            return
        env_olympus = getattr(self.runtime_context, "env_olympus", None)
        if env_olympus is not None and hasattr(env_olympus, "set_task_progress_listener"):
            env_olympus.set_task_progress_listener(self.emit_task_progress)

    def _clear_interaction_artifact_listener(self) -> None:
        if self.runtime_context is None:
            return
        env_imagej = getattr(self.runtime_context, "env_imagej", None)
        if env_imagej is not None and hasattr(env_imagej, "set_interaction_artifact_listener"):
            env_imagej.set_interaction_artifact_listener(None)

    def _clear_task_progress_listener(self) -> None:
        if self.runtime_context is None:
            self._latest_task_progress = None
            return
        env_olympus = getattr(self.runtime_context, "env_olympus", None)
        if env_olympus is not None and hasattr(env_olympus, "set_task_progress_listener"):
            env_olympus.set_task_progress_listener(None)
        self._latest_task_progress = None

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

        self._send_message(
            "artifact",
            title or "Fiji Detection Result",
            kind="image",
            title=title or "Fiji Detection Result",
            path=relative_path,
            url=f"/api/artifacts/{quote(relative_path, safe='/')}",
            display_seconds=display_seconds,
        )

    def emit_task_progress(self, progress: dict[str, Any]) -> None:
        payload = dict(progress)
        if not payload.get("timestamp"):
            payload["timestamp"] = datetime.now().isoformat()
        task_id = str(self.app_state.task.current_task_id or "")
        payload["task_id"] = task_id
        current = int(payload.get("progress_current") or 0)
        total = int(payload.get("progress_total") or 0)
        if total > 0 and payload.get("progress_percent") in (None, ""):
            payload["progress_percent"] = int(max(0.0, min(100.0, (float(current) / float(total)) * 100.0)))
        self._latest_task_progress = dict(payload)
        self.enqueue_output_message({"type": "task_progress", **payload})
        if self.runtime_context is not None:
            self.runtime_context.history_manager.record_interaction(
                agent_name="Runtime",
                event_type="task_progress",
                message="Runtime emitted long-running task progress.",
                payload=payload,
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
            initializing = phase in {"initializing", "releasing"}
        self.system_status.initialized = initialized
        self.system_status.initializing = initializing
        self.system_status.error = error
        self.system_status.message = message
        self.system_status.system_phase = phase
        self.system_status.failure_step = failure_step if error else ""

    def _reset_preview_state(self) -> None:
        self.preview_stream.reset()

    def _clear_runtime_context(self) -> None:
        self._clear_task_progress_listener()
        self.runtime_context = None
        self.orchestrator = None

    def _set_runtime_context(self, runtime_context: RuntimeContext) -> None:
        self.runtime_context = runtime_context
        self.orchestrator = runtime_context.task_orchestrator
        self._bind_task_progress_listener()

    def refresh_status_after_config_save(self) -> dict[str, Any]:
        snapshot = self.current_snapshot()
        asset_check = check_snapshot_assets(snapshot)
        mode_summary = asset_check.mode_summary
        if not asset_check.ready:
            message = (
                "Configuration saved. Before starting the system, resolve these blocking asset issues "
                f"for {mode_summary}: {asset_check.blocking_summary()}."
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
                message=f"Configuration saved. Reset and restart the system to apply changes. ({mode_summary})",
            )
        else:
            self._set_system_status(
                phase="ready_to_start",
                initialized=False,
                initializing=False,
                error=None,
                message=f"Configuration saved. Start the system when ready. ({mode_summary})",
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

    def _finish_llm_stream(self, stream_id: str, *, final_type: str, text: str = "") -> None:
        self.enqueue_output_message(
            {
                "type": "llm_stream_end",
                "stream_id": stream_id,
                "final_type": final_type,
                "text": text,
            }
        )

    async def _stream_scopebot_message(self, producer, *, final_type: str = "robot_say") -> str:
        stream_id = self._start_llm_stream(role="robot", final_type=final_type)
        emitted_chunks: list[str] = []
        final_text = ""

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
            final_text = text or "".join(emitted_chunks)
            return final_text
        finally:
            self._finish_llm_stream(stream_id, final_type=final_type, text=final_text)

    def _clear_pending_user_inputs(self) -> None:
        while True:
            try:
                self.app_state.session.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _build_cancel_payload(self, task_id: str, model_name: str, summary: str) -> TaskExecutionResponse:
        self._send_message("task_complete", "")
        return self._make_task_response(
            status="cancelled",
            retry_times=0,
            summary=summary,
            task_id=task_id,
            model_name=model_name,
        )

    def _build_task_failure_summary(self, command: str, result: Any, *, prefers_zh: bool) -> str:
        detail = str(getattr(result, "error", "") or getattr(result, "summary", "") or "").strip()
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

    async def _prompt_for_plan_feedback(self, prompt_text: str, mode: str = "plan_confirmation") -> str:
        session = self.app_state.session
        session.is_asking_user = True
        session.pending_user_prompt = {"type": "ask_user", "text": prompt_text, "mode": mode}
        self._send_message("ask_user", prompt_text, mode=mode)
        try:
            return await session.input_queue.get()
        finally:
            session.is_asking_user = False
            session.pending_user_prompt = None

    async def release_system(self) -> None:
        await self.runtime_lifecycle.release_system(self.runtime_context)

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
                except Exception as snapshot_exc:
                    logger.debug(
                        "Failed to load saved startup/system ranges while formatting startup position failure: %s",
                        snapshot_exc,
                        exc_info=True,
                    )
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
        exception_label = type(exc).__name__

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
        if isinstance(exc, APIStatusError) and hasattr(exc, "response"):
            return _summarize_api_status_error(exc)

        if context == "execution":
            return f"Task execution failed: {exception_label}: {detail}"
        if context == "initialization":
            return f"System initialization failed: {exception_label}: {detail}"
        return f"The system encountered an internal runtime error: {exception_label}: {detail}"

    async def initialize_runtime(self) -> dict[str, Any]:
        async with self._initialization_lock:
            try:
                self._initialization_task = asyncio.current_task()
            except RuntimeError:
                self._initialization_task = None
            try:
                return await self.runtime_lifecycle.initialize_runtime_once(self.runtime_context)
            finally:
                current_task = None
                try:
                    current_task = asyncio.current_task()
                except RuntimeError:
                    current_task = None
                if self._initialization_task is current_task:
                    self._initialization_task = None

    def start_runtime_initialization(self) -> dict[str, Any]:
        asset_check = self._asset_check()
        if not asset_check.ready:
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message=self._asset_blocking_message(asset_check),
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
        async with self._lifecycle_lock:
            phase = self.system_status.system_phase
            if phase == "executing":
                raise LifecycleConflictError("A task is executing; the system cannot be restarted.")
            if phase in {"initializing", "releasing"}:
                raise LifecycleConflictError("A lifecycle operation is already in progress.")

            asset_check = self._asset_check()
            if not asset_check.ready:
                self._set_system_status(
                    phase="unconfigured",
                    initialized=False,
                    initializing=False,
                    error=None,
                    message=self._asset_blocking_message(asset_check),
                )
                return self._make_init_response().model_dump()

            self._set_system_status(
                phase="releasing",
                initialized=False,
                initializing=True,
                error=None,
                message="Releasing system resources...",
            )
        try:
            await self.release_system()
        except Exception as exc:
            self._set_system_status(
                phase="failed",
                initialized=False,
                initializing=False,
                error=str(exc) or type(exc).__name__,
                message="Safe resource release failed; initialization was not started.",
                failure_step="resource_release",
            )
            raise
        return self.start_runtime_initialization()

    async def shutdown_runtime(self) -> None:
        async with self._lifecycle_lock:
            phase = self.system_status.system_phase
            if phase == "executing":
                raise LifecycleConflictError("A task is executing; the system cannot be shut down.")
            if phase in {"initializing", "releasing"}:
                raise LifecycleConflictError("A lifecycle operation is already in progress.")
            self._set_system_status(
                phase="releasing",
                initialized=False,
                initializing=True,
                error=None,
                message="Safely releasing microscope resources...",
            )
        try:
            await self.release_system()
        except Exception as exc:
            self._set_system_status(
                phase="failed",
                initialized=False,
                initializing=False,
                error=str(exc) or type(exc).__name__,
                message="Safe microscope shutdown failed.",
                failure_step="resource_release",
            )
            raise

    async def stop_for_application_shutdown(self) -> None:
        initialization_task = self._initialization_task
        if initialization_task is not None and not initialization_task.done():
            initialization_task.cancel()
            try:
                await initialization_task
            except asyncio.CancelledError:
                logger.debug("Runtime initialization task cancelled during application shutdown.")
        execution_task = self._execution_task
        if execution_task is not None and not execution_task.done():
            # Do not release hardware while a task is still driving it. Waiting is
            # deliberate; cancellation cannot safely stop vendor SDK calls.
            try:
                await asyncio.shield(execution_task)
            except asyncio.CancelledError:
                logger.debug("Application shutdown was cancelled while waiting for task execution to finish.")
        preview_task = self._preview_task
        if preview_task is not None and not preview_task.done():
            try:
                await asyncio.shield(preview_task)
            except asyncio.CancelledError:
                logger.debug("Application shutdown was cancelled while waiting for preview operation to finish.")
        await self.release_system()

    async def ensure_configuration_mutable(self) -> None:
        async with self._lifecycle_lock:
            if self.system_status.system_phase in {"initializing", "executing", "releasing"}:
                raise LifecycleConflictError("Configuration cannot be changed while the device is busy.")

    async def inspect_micro_manager_hardware(self, inspector: Any, **kwargs: Any) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self.system_status.system_phase in {"initializing", "executing", "releasing"}:
                raise LifecycleConflictError("Hardware cannot be inspected while the device is busy.")
            if self.system_status.initialized:
                self._set_system_status(
                    phase="releasing",
                    initialized=False,
                    initializing=False,
                    error=None,
                    message="Releasing the active runtime for Micro-Manager hardware inspection...",
                )
                await self.release_system()
            self._set_system_status(
                phase="initializing",
                initialized=False,
                initializing=True,
                error=None,
                message="Inspecting Micro-Manager Device Adapter capabilities...",
            )
            try:
                return await asyncio.to_thread(inspector, **kwargs)
            finally:
                self._set_system_status(
                    phase="ready_to_start",
                    initialized=False,
                    initializing=False,
                    error=None,
                    message="Hardware inspection finished. Review and save the mapping draft.",
                )

    async def execute_exclusive(self, command: str) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self.system_status.system_phase != "ready":
                raise LifecycleConflictError(self.system_status.message or "System is not ready.")
            self.app_state.task.running = True
            self._set_system_status(
                phase="executing",
                initialized=True,
                initializing=False,
                error=None,
                message="Task executing...",
            )
        try:
            self._execution_task = asyncio.current_task()
            return await self.execute_command(command)
        finally:
            async with self._lifecycle_lock:
                if self._execution_task is asyncio.current_task():
                    self._execution_task = None
                self.app_state.task.running = False
                if self.system_status.system_phase == "executing":
                    self._set_system_status(
                        phase="ready",
                        initialized=True,
                        initializing=False,
                        error=None,
                        message=f"System ready ({self._current_mode_summary()})",
                    )

    async def start_preview(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            self._preview_task = asyncio.current_task()
            try:
                return await self._start_preview_locked()
            finally:
                if self._preview_task is asyncio.current_task():
                    self._preview_task = None

    async def _start_preview_locked(self) -> dict[str, Any]:
        if self.system_status.system_phase != "ready":
            return {
                "started": False,
                "message": "Preview controls are unavailable while the device is busy.",
                "preview_phase": self.get_preview_status().get("preview_phase", "idle"),
            }
        return await self.preview_stream.start()

    async def startup(self) -> None:
        self.bind_event_loop()
        self._reset_preview_state()
        asset_check = self._asset_check()
        if asset_check.ready:
            self._set_system_status(
                phase="ready_to_start",
                initialized=False,
                initializing=False,
                error=None,
                message=f"Configuration loaded. Start the system when ready. ({asset_check.mode_summary})",
            )
        else:
            self._set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message=self._asset_blocking_message(asset_check),
            )

    async def _run_plan_interaction(self, original_command: str) -> InteractionOutcome:
        if self.orchestrator is None:
            raise RuntimeError("Task orchestrator is not initialized")

        async def plan_request(request):
            # Planner and its LLM calls are synchronous; keep them off the event loop so
            # MJPEG preview and SSE updates can continue while the model is thinking.
            plan = await asyncio.to_thread(self.orchestrator.plan, request)
            self.app_state.task.current_task_id = plan.task_id
            return plan

        async def stream_preview(plan) -> str:
            return await self._stream_scopebot_message(
                lambda on_delta: self.orchestrator.stream_plan_preview(plan, on_delta),
                final_type="robot_say",
            )

        async def prompt_user(
            prompt_text: str,
            command_snapshot: str,
            prompt_mode: str = "plan_confirmation",
        ) -> str:
            del command_snapshot
            return await self._prompt_for_plan_feedback(prompt_text, mode=prompt_mode)

        def record_user_input(
            text: str,
            input_kind: str,
            prompt_text: str,
            command_snapshot: str,
            prompt_mode: str = "plan_confirmation",
        ) -> None:
            self._record_user_input(
                text,
                input_kind=input_kind,
                prompt_text=prompt_text,
                prompt_mode=prompt_mode,
                command_snapshot=command_snapshot,
            )

        ports = TaskInteractionPorts(
            plan=plan_request,
            stream_plan_preview=stream_preview,
            prompt_user=prompt_user,
            send_robot_message=lambda text: self._send_message("robot_say", text),
            emit_skill_summary=lambda plan, prefers_zh: self._emit_skill_summary(plan, prefers_zh=prefers_zh),
            record_user_input=record_user_input,
            log_planner_tokens=lambda tokens: logger.info("Planner tokens: %s", tokens),
        )
        return await TaskInteractionSession(ports).request_plan_confirmation(original_command)

    async def execute_command(self, command: str) -> dict[str, Any]:
        if self.runtime_context is None or self.orchestrator is None:
            raise RuntimeError("System not yet initialized")

        runtime_agent = self.runtime_context.runtime["agent"]
        original_command = command.strip()
        self._record_user_input(original_command, input_kind="initial_command", command_snapshot=original_command)
        prefers_zh = prefers_chinese(original_command)

        self._clear_pending_user_inputs()
        try:
            outcome = await self._run_plan_interaction(original_command)
            plan = outcome.plan
            if not outcome.confirmed:
                if outcome.status == "unsupported":
                    self._send_message("task_complete", "")
                    return self._make_task_response(
                        status="failed",
                        retry_times=0,
                        summary=outcome.summary,
                        task_id=plan.task_id if plan is not None else "",
                        model_name=runtime_agent.model_name,
                    ).model_dump()
                return self._build_cancel_payload(
                    plan.task_id if plan is not None else "",
                    runtime_agent.model_name,
                    outcome.summary,
                ).model_dump()

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
        except asyncio.CancelledError as exc:
            latest_progress = dict(self._latest_task_progress or {})
            task_id = str(self.app_state.task.current_task_id or "")
            detail = str(latest_progress.get("detail") or "The task was cancelled before completion.")
            stage_label = str(latest_progress.get("stage_label") or latest_progress.get("task_kind") or "task execution")
            message = f"Task execution was cancelled while running {stage_label}. {detail}".strip()
            logger.warning(
                "Task execution coroutine cancelled. task_id=%s latest_progress=%s",
                task_id,
                latest_progress,
            )
            if latest_progress:
                cancelled_progress = dict(latest_progress)
                cancelled_progress["status"] = "cancelled"
                cancelled_progress["detail"] = message
                self.emit_task_progress(cancelled_progress)
            else:
                self.emit_task_progress(
                    {
                        "task_kind": "execution",
                        "status": "cancelled",
                        "title": "Task Execution",
                        "detail": message,
                        "progress_current": 0,
                        "progress_total": 0,
                        "progress_percent": 0,
                        "stage_key": "cancelled",
                        "stage_label": "Execution cancelled",
                        "timestamp": "",
                    }
                )
            if self.runtime_context is not None:
                self.runtime_context.history_manager.record_interaction(
                    agent_name="Runtime",
                    event_type="executor_execution_failed",
                    message="Executor was interrupted because task execution was cancelled.",
                    payload={
                        "task_id": task_id,
                        "command": original_command,
                        "latest_progress": latest_progress,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "cancelled": True,
                    },
                )
            self._send_message("error", message)
            self._send_message("task_complete", "")
            raise
        finally:
            self.app_state.session.is_asking_user = False
    def emit_robot_action(self, summary: str) -> None:
        if summary:
            self._send_message("robot_action", summary)

    def emit_step_summary(self, summary: str) -> None:
        if summary:
            self._send_message("step_summary", summary)

    def emit_checker_warning(self, summary: str) -> None:
        if summary:
            self._send_message("robot_say", summary)

    async def global_message_stream(self) -> AsyncGenerator[str, None]:
        session = self.app_state.session
        subscriber: asyncio.Queue = asyncio.Queue()
        while True:
            try:
                subscriber.put_nowait(session.output_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        session.output_subscribers.add(subscriber)
        if not session.first_connection_made:
            session.first_connection_made = True
            if self.system_status.initialized:
                self._publish_output_message(
                    {"type": "robot_say", "text": "Microscope is ready! Please enter commands."}
                )
        if session.pending_user_prompt is not None:
            subscriber.put_nowait(dict(session.pending_user_prompt))

        while True:
            try:
                msg = await subscriber.get()
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                session.output_subscribers.discard(subscriber)
                break
            except Exception as exc:
                logger.error("SSE generator error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'text': self.humanize_exception_message(exc)})}\n\n"

    async def receive_user_input(self, text: str) -> dict[str, str]:
        if not self.app_state.session.is_asking_user:
            return UserInputResponse(status="ignored", message="No user input is being waited for currently").model_dump()
        self.app_state.session.pending_user_prompt = None
        await self.app_state.session.input_queue.put(text)
        return UserInputResponse(status="ok", message="Input received").model_dump()

    def get_preview_status(self) -> dict[str, Any]:
        return self.preview_stream.get_status()

    async def generate_mjpeg_frames(self) -> AsyncGenerator[bytes, None]:
        async for frame in self.preview_stream.generate_mjpeg_frames():
            yield frame


