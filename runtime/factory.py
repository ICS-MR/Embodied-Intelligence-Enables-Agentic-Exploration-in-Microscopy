from typing import Any

from runtime.agent_factory import build_clients
from runtime.config_loader import load_runtime_config
from runtime.context_builder import build_runtime_context
from runtime.models import RuntimeContext


class RuntimeInitializationCleanupError(RuntimeError):
    def __init__(self, message: str, runtime_context: RuntimeContext) -> None:
        super().__init__(message)
        self.runtime_context = runtime_context


def _cleanup_cancelled_runtime(context: RuntimeContext) -> None:
    cleanup_errors: list[str] = []
    try:
        microscope_shutdown = getattr(context.env_olympus, "shutdown", None)
        if callable(microscope_shutdown):
            microscope_shutdown()
    except Exception as exc:
        cleanup_errors.append(f"microscope shutdown: {exc}")
    try:
        fiji_shutdown = getattr(context.env_imagej, "fiji_shutdown", None)
        if callable(fiji_shutdown):
            fiji_shutdown()
    except Exception as exc:
        cleanup_errors.append(f"Fiji shutdown: {exc}")
    try:
        context.storage_manager.clear_cache()
    except Exception as exc:
        cleanup_errors.append(f"cache cleanup: {exc}")

    if cleanup_errors:
        raise RuntimeInitializationCleanupError("; ".join(cleanup_errors), context)


def initialize_system_components(
    cancel_event: Any = None,
) -> RuntimeContext:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("runtime initialization cancelled")

    runtime = load_runtime_config()
    agent_config = runtime["agent"]

    llm_client, vlm_client = build_clients(agent_config)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("runtime initialization cancelled")

    context = build_runtime_context(runtime, llm_client, vlm_client)
    if cancel_event is not None and cancel_event.is_set():
        _cleanup_cancelled_runtime(context)
    return context
