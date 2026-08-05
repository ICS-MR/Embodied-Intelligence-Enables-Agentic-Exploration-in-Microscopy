import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable

from bootstrap.config import load_runtime_settings
from runtime.asset_check import check_snapshot_assets
from runtime.factory import initialize_system_components
from runtime.hardware_lifecycle import apply_startup_state, initialize_microscope, release_resources
from runtime.models import RuntimeContext


logger = logging.getLogger(__name__)

INIT_COMPONENT_TIMEOUT_SEC = 90.0
MICROSCOPE_SETUP_TIMEOUT_SEC = 30.0


@dataclass(frozen=True)
class RuntimeLifecyclePorts:
    current_snapshot: Callable[[], dict[str, Any]]
    make_init_response: Callable[[], Any]
    set_system_status: Callable[..., None]
    send_message: Callable[[str, str], None]
    clear_runtime_context: Callable[[], None]
    set_runtime_context: Callable[[RuntimeContext], None]
    bind_interaction_artifact_listener: Callable[[], None]
    reset_preview_state: Callable[[], None]
    build_failure_message: Callable[[str, Exception], str]


class RuntimeLifecycleService:
    def __init__(self, ports: RuntimeLifecyclePorts) -> None:
        self.ports = ports

    async def release_system(self, current_context: RuntimeContext | None) -> None:
        if current_context is not None:
            try:
                env_imagej = getattr(current_context, "env_imagej", None)
                if env_imagej is not None and hasattr(env_imagej, "set_interaction_artifact_listener"):
                    env_imagej.set_interaction_artifact_listener(None)
                await asyncio.to_thread(release_resources, current_context)
            except Exception:
                logger.exception("Failed to release system resources cleanly")
                raise
        self.ports.clear_runtime_context()
        self.ports.reset_preview_state()

    async def finalize_init_failure(
        self,
        step: str,
        exc: Exception,
        runtime_context: RuntimeContext | None = None,
    ) -> dict[str, Any]:
        cleanup_error: Exception | None = None
        if runtime_context is not None:
            try:
                await asyncio.to_thread(release_resources, runtime_context)
            except Exception as release_exc:
                logger.exception("Failed to release partially initialized runtime after %s", step)
                cleanup_error = release_exc

        if cleanup_error is not None and runtime_context is not None:
            self.ports.set_runtime_context(runtime_context)
        else:
            self.ports.clear_runtime_context()
        self.ports.reset_preview_state()

        detail = str(exc).strip() or type(exc).__name__
        message = self.ports.build_failure_message(step, exc)
        if cleanup_error is not None:
            cleanup_detail = str(cleanup_error).strip() or type(cleanup_error).__name__
            detail = f"{detail}; cleanup failed: {cleanup_detail}"
            message = f"{message}. Safe hardware cleanup failed; retry shutdown before reinitializing."
        self.ports.set_system_status(
            phase="failed",
            initialized=False,
            initializing=False,
            error=detail,
            message=message,
            failure_step=step,
        )
        self.ports.send_message("error", message)
        return self.ports.make_init_response().model_dump()

    async def run_initialization_step(
        self,
        func: Any,
        *args: Any,
        timeout: float,
        cancel_event: threading.Event,
    ) -> Any:
        task = asyncio.create_task(asyncio.to_thread(func, *args))

        def request_cancel() -> None:
            cancel_event.set()
            if args:
                shutdown_event = getattr(args[0], "shutdown_event", None)
                if shutdown_event is not None and hasattr(shutdown_event, "set"):
                    shutdown_event.set()

        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            request_cancel()
            try:
                await task
            except Exception as exc:
                logger.exception("Initialization worker stopped with an error after cancellation")
                raise exc
            raise
        except asyncio.CancelledError:
            request_cancel()
            try:
                await asyncio.shield(task)
            except Exception:
                logger.exception("Initialization worker stopped with an error during runtime cancellation")
            raise

    def validate_runtime_context(self, runtime_context: RuntimeContext) -> None:
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

    @staticmethod
    def _mode_summary(settings: Any) -> str:
        return (
            f"Microscope: {settings.model.microscope_mode} | "
            f"Fiji: {settings.model.image_analysis_mode} | "
            f"Cellpose: {settings.model.segmentation_mode}"
        )

    async def initialize_runtime_once(self, current_context: RuntimeContext | None) -> dict[str, Any]:
        snapshot = self.ports.current_snapshot()
        asset_check = check_snapshot_assets(snapshot)
        if not asset_check.ready:
            detail = asset_check.blocking_summary()
            message = (
                f"Please complete configuration first for {asset_check.mode_summary}: {detail}"
                if detail
                else "Please complete configuration first"
            )
            self.ports.set_system_status(
                phase="unconfigured",
                initialized=False,
                initializing=False,
                error=None,
                message=message,
            )
            return self.ports.make_init_response().model_dump()

        await self.release_system(current_context)
        self.ports.set_system_status(
            phase="initializing",
            initialized=False,
            initializing=True,
            error=None,
            message="System initializing...",
        )
        self.ports.send_message("robot_say", "System initializing...")

        runtime_context = None
        settings = load_runtime_settings()
        cancel_event = threading.Event()

        try:
            runtime_context = await self.run_initialization_step(
                initialize_system_components,
                cancel_event,
                timeout=INIT_COMPONENT_TIMEOUT_SEC,
                cancel_event=cancel_event,
            )
        except asyncio.TimeoutError:
            return await self.finalize_init_failure(
                "runtime_build",
                TimeoutError(f"timed out after {INIT_COMPONENT_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except asyncio.CancelledError:
            await self.finalize_init_failure(
                "runtime_build",
                RuntimeError("runtime initialization cancelled during application shutdown"),
                runtime_context,
            )
            raise
        except Exception as exc:
            logger.exception("System initialization failed during runtime build")
            return await self.finalize_init_failure(
                "runtime_build",
                exc,
                getattr(exc, "runtime_context", runtime_context),
            )

        try:
            await self.run_initialization_step(
                initialize_microscope,
                runtime_context.env_olympus,
                cancel_event,
                timeout=MICROSCOPE_SETUP_TIMEOUT_SEC,
                cancel_event=cancel_event,
            )
        except asyncio.TimeoutError:
            return await self.finalize_init_failure(
                "microscope_initialize",
                TimeoutError(f"timed out after {MICROSCOPE_SETUP_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except asyncio.CancelledError:
            await self.finalize_init_failure(
                "microscope_initialize",
                RuntimeError("microscope initialization cancelled during application shutdown"),
                runtime_context,
            )
            raise
        except Exception as exc:
            logger.exception("System initialization failed during microscope initialization")
            return await self.finalize_init_failure("microscope_initialize", exc, runtime_context)

        try:
            await self.run_initialization_step(
                apply_startup_state,
                runtime_context.env_olympus,
                settings.startup,
                cancel_event,
                timeout=MICROSCOPE_SETUP_TIMEOUT_SEC,
                cancel_event=cancel_event,
            )
        except asyncio.TimeoutError:
            return await self.finalize_init_failure(
                "startup_state_apply",
                TimeoutError(f"timed out after {MICROSCOPE_SETUP_TIMEOUT_SEC:.0f}s"),
                runtime_context,
            )
        except asyncio.CancelledError:
            await self.finalize_init_failure(
                "startup_state_apply",
                RuntimeError("startup state cancelled during application shutdown"),
                runtime_context,
            )
            raise
        except Exception as exc:
            logger.exception("System initialization failed during startup state apply")
            return await self.finalize_init_failure("startup_state_apply", exc, runtime_context)

        try:
            self.validate_runtime_context(runtime_context)
        except Exception as exc:
            logger.exception("System initialization failed during post-init validation")
            return await self.finalize_init_failure("post_init_validation", exc, runtime_context)

        self.ports.set_runtime_context(runtime_context)
        self.ports.bind_interaction_artifact_listener()
        self.ports.reset_preview_state()

        microscope_mode = str(runtime_context.runtime["agent"].microscope_mode).strip().lower()
        mode_summary = self._mode_summary(settings)
        ready_message = (
            "System initialization completed. Micro-Manager demo hardware is active. "
            f"{mode_summary}. Open the runtime page to start live preview."
            if microscope_mode == "demo"
            else f"System initialization completed. {mode_summary}. Open the runtime page to start live preview."
        )
        self.ports.set_system_status(
            phase="ready",
            initialized=True,
            initializing=False,
            error=None,
            message=f"System ready ({mode_summary})",
        )
        self.ports.send_message("robot_say", ready_message)
        return self.ports.make_init_response().model_dump()
