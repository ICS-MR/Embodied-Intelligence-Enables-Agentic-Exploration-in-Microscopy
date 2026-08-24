from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HardwareOwnerHandoffSession:
    tool_id: str
    microscope_state: Dict[str, Any]


class HardwareOwnerHandoffManager:
    """Coordinate exclusive hardware ownership between MMCore and external tools."""

    _EXTERNAL_HARDWARE_TOOLS = {"frap": "frap"}

    def __init__(self, runtime_context: Any) -> None:
        self.runtime_context = runtime_context
        self._active_session: Optional[HardwareOwnerHandoffSession] = None

    def external_tool_for_module(self, module_name: Any) -> Optional[str]:
        normalized = str(module_name or "").strip().lower()
        return self._EXTERNAL_HARDWARE_TOOLS.get(normalized)

    def normalize_module_name(self, module_name: Any) -> str:
        external_tool = self.external_tool_for_module(module_name)
        return external_tool if external_tool is not None else str(module_name or "")

    def begin(self, tool_id: str) -> HardwareOwnerHandoffSession:
        normalized_tool_id = str(tool_id or "").strip().lower()
        if not normalized_tool_id:
            raise ValueError("External hardware tool id is required for hardware owner handoff.")
        if self._active_session is not None:
            if self._active_session.tool_id == normalized_tool_id:
                return self._active_session
            raise RuntimeError(
                "Cannot begin hardware owner handoff for "
                f"'{normalized_tool_id}' while '{self._active_session.tool_id}' is active."
            )

        microscope = getattr(self.runtime_context, "env_olympus", None)
        if microscope is None:
            raise RuntimeError("Cannot hand off hardware ownership: microscope controller is unavailable.")

        capture_state = getattr(microscope, "capture_hardware_owner_state", None)
        release_owner = getattr(microscope, "release_hardware_owner", None)
        if not callable(capture_state) or not callable(release_owner):
            raise RuntimeError(
                "Microscope controller does not implement the hardware-owner release protocol."
            )

        microscope_state = capture_state()
        if not isinstance(microscope_state, dict):
            raise RuntimeError(
                "Microscope hardware-owner state capture must return a dict, "
                f"got {type(microscope_state).__name__}."
            )
        try:
            release_owner()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to release microscope hardware ownership for '{normalized_tool_id}'."
            ) from exc

        self._active_session = HardwareOwnerHandoffSession(
            tool_id=normalized_tool_id,
            microscope_state=dict(microscope_state),
        )
        return self._active_session

    def end(self, session: HardwareOwnerHandoffSession) -> None:
        if session is None:
            raise ValueError("Hardware owner handoff session is required for restore.")
        errors: list[str] = []

        tool_binding = self.runtime_context.tool_registry.get_tool(session.tool_id)
        external_owner = getattr(tool_binding, "env", None) if tool_binding is not None else None
        release_external_owner = getattr(external_owner, "release_session", None)
        if not callable(release_external_owner):
            errors.append(
                f"External hardware tool '{session.tool_id}' does not implement release_session()."
            )
        else:
            try:
                release_external_owner()
            except Exception as exc:
                errors.append(f"External hardware release ({session.tool_id}): {exc}")

        microscope = getattr(self.runtime_context, "env_olympus", None)
        restore_owner = getattr(microscope, "restore_hardware_owner", None)
        if not callable(restore_owner):
            errors.append("Microscope controller does not implement the hardware-owner restore protocol.")
        else:
            try:
                restore_owner(dict(session.microscope_state))
            except Exception as exc:
                errors.append(f"Microscope hardware restore: {exc}")

        if self._active_session is session:
            self._active_session = None

        if errors:
            raise RuntimeError("; ".join(errors))

    def end_active(self) -> None:
        if self._active_session is None:
            raise RuntimeError("No active hardware owner handoff session to end.")
        self.end(self._active_session)
