from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


class PortValidationError(TypeError):
    """Raised when a tool implementation does not satisfy the expected port."""


@runtime_checkable
class ToolPort(Protocol):
    @classmethod
    def get_public_methods(cls) -> List[str]:
        ...

    @classmethod
    def get_tool_descriptors(cls) -> List[Dict[str, Any]]:
        ...


@runtime_checkable
class MicroscopePort(ToolPort, Protocol):
    def initialize(self) -> None:
        ...

    def set_x_y_position(self, x: float, y: float) -> None:
        ...

    def get_x_y_position(self) -> tuple[float, float]:
        ...

    def set_z_position(self, z: float) -> None:
        ...

    def get_z_position(self) -> float:
        ...

    def set_exposure(self, exposure_time: float) -> None:
        ...

    def get_exposure(self) -> float:
        ...

    def set_brightness(self, brightness: int) -> None:
        ...

    def get_brightness(self) -> int:
        ...

    def set_objective(self, objective_label: str) -> None:
        ...

    def get_objective(self) -> str:
        ...

    def set_channel(self, channel: str) -> None:
        ...

    def get_channel(self) -> str:
        ...

    def start_preview(self) -> None:
        ...

    def get_live_preview_image(self) -> Any:
        ...


@runtime_checkable
class ImageAnalysisPort(ToolPort, Protocol):
    def fiji_initialize(self, *args: Any, **kwargs: Any) -> None:
        ...

    def load_image(self, file_name: str) -> Any:
        ...

    def save_image(self, image_meta: Any, filename: str, description: str) -> Any:
        ...


@runtime_checkable
class SegmentationPort(ToolPort, Protocol):
    def cellpose_initialize(self, *args: Any, **kwargs: Any) -> None:
        ...

    def segment(self, image: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def save_masks(self, masks: Any, filename: str, description: str) -> Any:
        ...


PORT_REQUIREMENTS = {
    "microscope": (
        "initialize",
        "set_x_y_position",
        "get_x_y_position",
        "set_z_position",
        "get_z_position",
        "set_exposure",
        "get_exposure",
        "set_brightness",
        "get_brightness",
        "set_objective",
        "get_objective",
        "set_channel",
        "get_channel",
        "start_preview",
        "get_live_preview_image",
        "get_public_methods",
        "get_tool_descriptors",
    ),
    "image_analysis": (
        "fiji_initialize",
        "load_image",
        "save_image",
        "get_public_methods",
        "get_tool_descriptors",
    ),
    "segmentation": (
        "cellpose_initialize",
        "segment",
        "save_masks",
        "get_public_methods",
        "get_tool_descriptors",
    ),
}


def validate_port_implementation(obj: Any, port_kind: str) -> None:
    required = PORT_REQUIREMENTS.get(port_kind, ())
    missing = [name for name in required if not hasattr(obj, name)]
    if missing:
        raise PortValidationError(
            f"{type(obj).__name__} does not satisfy the '{port_kind}' port. Missing: {', '.join(missing)}"
        )
