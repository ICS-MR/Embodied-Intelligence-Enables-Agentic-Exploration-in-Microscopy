from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from bootstrap.config import RuntimeSettings, build_demo_system_overrides


REAL_RUNTIME_REQUIREMENTS: dict[str, list[tuple[str, str | None]]] = {
    "core_tool.microscope": [
        ("cv2", None),
        ("numpy", None),
        ("aicsimageio.types", "PhysicalPixelSizes"),
        ("aicsimageio.writers", "OmeTiffWriter"),
        ("pymmcore_plus", "CMMCorePlus"),
        ("torch", None),
        ("mmdet.apis", "init_detector"),
        ("mmdet.apis", "inference_detector"),
    ],
    "core_tool.fiji": [
        ("torch", None),
        ("imagej", None),
        ("scyjava", "jimport"),
        ("tifffile", None),
        ("cv2", None),
        ("mmdet.apis", "init_detector"),
        ("mmdet.apis", "inference_detector"),
        ("aicsimageio", "AICSImage"),
    ],
    "core_tool.cellpose_tool": [
        ("numpy", None),
        ("pandas", None),
        ("matplotlib.pyplot", None),
        ("skimage", None),
        ("tifffile", None),
        ("cellpose.models", None),
        ("cellpose.core", None),
    ],
}

MOCK_RUNTIME_MODULES: dict[str, str] = {
    "image_analysis": "simulation.imagej:ImageJProcessor",
    "segmentation": "simulation.cellpose:Cellpose2D",
}


@dataclass(frozen=True)
class AssetIssue:
    category: str
    name: str
    message: str
    path: str = ""
    mode: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AssetCheckResult:
    mode_summary: str
    blocking: list[AssetIssue]
    warnings: list[AssetIssue]

    @property
    def ready(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mode_summary": self.mode_summary,
            "blocking": [issue.to_dict() for issue in self.blocking],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }

    def blocking_summary(self) -> str:
        return "; ".join(issue.message for issue in self.blocking)


def check_runtime_assets(settings: RuntimeSettings) -> AssetCheckResult:
    snapshot = {
        "system": asdict(settings.system),
        "agent": asdict(settings.model),
        "startup": asdict(settings.startup),
        "detection_targets": settings.detection_targets,
    }
    return check_snapshot_assets(snapshot)


def check_snapshot_assets(snapshot: Mapping[str, Any]) -> AssetCheckResult:
    system = _mapping(snapshot.get("system"))
    agent = _mapping(snapshot.get("agent"))
    detection_targets = _mapping(snapshot.get("detection_targets"))
    microscope_mode = _mode(agent.get("microscope_mode"), default="demo", allowed={"demo", "real"})
    image_analysis_mode = _mode(agent.get("image_analysis_mode"), default="mock", allowed={"mock", "real"})
    segmentation_mode = _mode(agent.get("segmentation_mode"), default="mock", allowed={"mock", "real"})
    mode_summary = (
        f"Microscope: {microscope_mode} | "
        f"Fiji: {image_analysis_mode} | "
        f"Cellpose: {segmentation_mode}"
    )

    blocking: list[AssetIssue] = []
    warnings: list[AssetIssue] = []
    _check_agent_fields(agent, blocking)
    _check_microscope_assets(system, microscope_mode, blocking)
    _check_image_analysis_assets(system, image_analysis_mode, detection_targets, blocking, warnings)
    _check_segmentation_assets(segmentation_mode, blocking)
    return AssetCheckResult(mode_summary=mode_summary, blocking=blocking, warnings=warnings)


def check_standalone_evaluation_assets(snapshot: Mapping[str, Any]) -> AssetCheckResult:
    del snapshot
    return AssetCheckResult(
        mode_summary="Standalone evaluation",
        blocking=[],
        warnings=[
            AssetIssue(
                category="standalone_evaluation",
                name="not_connected_to_runtime_startup",
                message="Standalone evaluation asset checks are intentionally not used to block Web/CLI runtime startup.",
                mode="standalone",
            )
        ],
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mode(value: Any, *, default: str, allowed: set[str]) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _add_missing(
    issues: list[AssetIssue],
    *,
    category: str,
    name: str,
    mode: str,
    path: str = "",
) -> None:
    target = f" at {path}" if path else ""
    issues.append(
        AssetIssue(
            category=category,
            name=name,
            message=f"{mode} requires {name}{target}.",
            path=path,
            mode=mode,
        )
    )


def _require_text(
    source: Mapping[str, Any],
    field_name: str,
    issues: list[AssetIssue],
    *,
    category: str,
    mode: str,
) -> None:
    if not _text(source.get(field_name)):
        _add_missing(issues, category=category, name=field_name, mode=mode)


def _check_agent_fields(agent: Mapping[str, Any], issues: list[AssetIssue]) -> None:
    for field_name in (
        "openai_api_key",
        "base_url",
        "model_name",
        "vlm_api_key",
        "vlm_base_url",
        "vlm_model_name",
    ):
        _require_text(agent, field_name, issues, category="agent_config", mode="agent")


def _check_microscope_assets(system: Mapping[str, Any], mode: str, issues: list[AssetIssue]) -> None:
    _require_text(system, "MM_DIR", issues, category="microscope_config", mode=f"microscope:{mode}")
    _require_text(system, "CONFIG_PATH", issues, category="microscope_config", mode=f"microscope:{mode}")
    _check_dependency_stack("core_tool.microscope", issues, mode=f"microscope:{mode}")
    if mode == "demo":
        demo_overrides = build_demo_system_overrides()
        expected_config_path = _text(demo_overrides.get("CONFIG_PATH"))
        if expected_config_path and not _same_path_text(_text(system.get("CONFIG_PATH")), expected_config_path):
            issues.append(
                AssetIssue(
                    category="demo_microscope_mapping",
                    name="CONFIG_PATH",
                    message=f"microscope:demo requires CONFIG_PATH={expected_config_path}.",
                    path=_text(system.get("CONFIG_PATH")),
                    mode="microscope:demo",
                )
            )
        expected = {
            "camera_device": "DCam",
            "xy_stage_device": "DXYStage",
            "objective_device": "DObjective",
            "focus_drive": "DStage",
            "Dichroic": "DStateDevice",
        }
        for field_name, expected_value in expected.items():
            if _text(system.get(field_name)) != expected_value:
                issues.append(
                    AssetIssue(
                        category="demo_microscope_mapping",
                        name=field_name,
                        message=f"microscope:demo requires {field_name}={expected_value}.",
                        mode="microscope:demo",
                    )
                )
        transmitted_light = _mapping(system.get("transmitted_light"))
        if _text(transmitted_light.get("device")) != "DCam":
            issues.append(
                AssetIssue(
                    category="demo_microscope_mapping",
                    name="transmitted_light.device",
                    message="microscope:demo requires transmitted_light.device=DCam.",
                    mode="microscope:demo",
                )
            )
        if _text(transmitted_light.get("intensity_property")) != "BeadBrightness":
            issues.append(
                AssetIssue(
                    category="demo_microscope_mapping",
                    name="transmitted_light.intensity_property",
                    message="microscope:demo requires transmitted_light.intensity_property=BeadBrightness.",
                    mode="microscope:demo",
                )
            )
        return

    for field_name in (
        "camera_device",
        "xy_stage_device",
        "objective_device",
        "transmittedIllumination",
        "focus_drive",
        "Dichroic",
    ):
        _require_text(system, field_name, issues, category="microscope_config", mode="microscope:real")
    if not _mapping(system.get("objectives")):
        _add_missing(issues, category="microscope_mapping", name="objectives", mode="microscope:real")
    if not _mapping(system.get("channels")):
        _add_missing(issues, category="microscope_mapping", name="channels", mode="microscope:real")


def _check_image_analysis_assets(
    system: Mapping[str, Any],
    mode: str,
    detection_targets: Mapping[str, Any],
    issues: list[AssetIssue],
    warnings: list[AssetIssue],
) -> None:
    if mode == "mock":
        _check_import_path(MOCK_RUNTIME_MODULES["image_analysis"], issues, mode="image_analysis:mock")
        return

    _require_text(system, "FIJI_PATH", issues, category="image_analysis_config", mode="image_analysis:real")
    _check_dependency_stack("core_tool.fiji", issues, mode="image_analysis:real")
    if not detection_targets:
        warnings.append(
            AssetIssue(
                category="detector_assets",
                name="detection_targets",
                message="image_analysis:real has no detection_targets mapping in this snapshot.",
                mode="image_analysis:real",
            )
        )
        return
    for target_name, target_spec in detection_targets.items():
        spec = _mapping(target_spec)
        if spec.get("enabled") is False:
            continue
        _check_local_file(
            spec.get("model_config"),
            issues,
            category="detector_assets",
            name=f"detection_targets.{target_name}.model_config",
            mode="image_analysis:real",
        )
        _check_local_file(
            spec.get("model_checkpoint"),
            issues,
            category="detector_assets",
            name=f"detection_targets.{target_name}.model_checkpoint",
            mode="image_analysis:real",
        )


def _check_segmentation_assets(mode: str, issues: list[AssetIssue]) -> None:
    if mode == "mock":
        _check_import_path(MOCK_RUNTIME_MODULES["segmentation"], issues, mode="segmentation:mock")
        return
    _check_dependency_stack("core_tool.cellpose_tool", issues, mode="segmentation:real")


def _check_local_file(
    value: Any,
    issues: list[AssetIssue],
    *,
    category: str,
    name: str,
    mode: str,
) -> None:
    path_text = _text(value)
    if not path_text:
        _add_missing(issues, category=category, name=name, mode=mode)
        return
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists() or not path.is_file():
        issues.append(
            AssetIssue(
                category=category,
                name=name,
                message=f"{mode} requires existing file for {name}: {path_text}.",
                path=path_text,
                mode=mode,
            )
        )


def _check_import_path(class_path: str, issues: list[AssetIssue], *, mode: str) -> None:
    module_name, _sep, attr_name = class_path.partition(":")
    problem = _check_dependency(module_name, attr_name or None)
    if problem is not None:
        issues.append(
            AssetIssue(
                category="runtime_dependency",
                name=class_path,
                message=f"{mode} requires importable {class_path}: {problem}.",
                mode=mode,
            )
        )


def _check_dependency_stack(module_name: str, issues: list[AssetIssue], *, mode: str) -> None:
    for dependency_name, attr_name in REAL_RUNTIME_REQUIREMENTS.get(module_name, []):
        problem = _check_dependency(dependency_name, attr_name)
        if problem is not None:
            issues.append(
                AssetIssue(
                    category="runtime_dependency",
                    name=dependency_name,
                    message=f"{mode} requires {dependency_name}: {problem}.",
                    mode=mode,
                )
            )
    problem = _check_dependency(module_name, None)
    if problem is not None:
        issues.append(
            AssetIssue(
                category="runtime_dependency",
                name=module_name,
                message=f"{mode} requires importable {module_name}: {problem}.",
                mode=mode,
            )
        )


def _check_dependency(module_name: str, attr_name: str | None) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    if attr_name and not hasattr(module, attr_name):
        return f"missing required attribute '{attr_name}'"
    return None


def _same_path_text(left: str, right: str) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(strict=False)
    except Exception:
        return left.strip() == right.strip()
