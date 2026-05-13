from __future__ import annotations

import ast
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOOL_MANIFEST_PATH = ROOT_DIR / "config" / "tool_manifest.json"
TOOL_SOURCE_DIR = ROOT_DIR / "tool"
SUPPORTED_CONSTRUCTOR_KINDS = {"microscope", "storage_output", "no_args"}
SYSTEM_TOOL_ROLES = ("microscope", "image_analysis", "segmentation")
SYSTEM_TOOL_METADATA = {
    "microscope": {
        "tool_id": "microscope_operation",
        "platform_name": "Microscope Operation Platform",
        "prompt_source": "prompts.micro_control_prompt_full:prompt_olympus",
        "port_kind": "microscope",
    },
    "image_analysis": {
        "tool_id": "image_analysis",
        "platform_name": "Image Analysis Platform",
        "prompt_source": "prompts.ansis_platform_prompt_full:prompt_imagej",
        "port_kind": "image_analysis",
    },
    "segmentation": {
        "tool_id": "cell_segmentation",
        "platform_name": "Cell Segmentation Platform",
        "prompt_source": "prompts.cell_seg_prompt_full:prompt_cellpose",
        "port_kind": "segmentation",
    },
}


class ToolManifestError(ValueError):
    """Raised when the tool manifest is invalid or incomplete."""


def resolve_default_tool_manifest_path() -> Path:
    override = os.environ.get("EIMS_TOOL_MANIFEST_PATH", "").strip()
    if override:
        return Path(override)
    return DEFAULT_TOOL_MANIFEST_PATH


@dataclass(frozen=True)
class SystemToolEntry:
    role: str
    tool_id: str
    platform_name: str
    prompt_source: str
    port_kind: str
    real_class_path: str
    simulation_class_path: str
    constructor_kind: str
    validate_real_stack: bool = False

    def class_path_for_mode(self, simulation_mode: bool) -> str:
        return self.simulation_class_path if simulation_mode else self.real_class_path


@dataclass(frozen=True)
class UserToolManifestEntry:
    tool_id: str
    class_path: str
    enabled: bool = True
    planning_hint: str = ""
    execution_hint: str = ""


@dataclass(frozen=True)
class ToolManifest:
    system_tools: Dict[str, SystemToolEntry] = field(default_factory=dict)
    user_tools: List[UserToolManifestEntry] = field(default_factory=list)


@dataclass(frozen=True)
class DiscoveredToolCandidate:
    class_name: str
    file_path: str
    class_path: str


def import_string(path: str) -> Any:
    if ":" not in path:
        raise ToolManifestError(f"Import path '{path}' must use the format 'module.submodule:Attribute'")
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ToolManifestError(f"Module '{module_name}' does not expose attribute '{attr_name}'") from exc


def default_tool_manifest_payload() -> Dict[str, Any]:
    return {
        "system_tools": {
            "microscope": {
                "real_class_path": "core_tool.microscope:MicroscopeController",
                "simulation_class_path": "Empty_function:MicroscopeController",
                "prompt_source": SYSTEM_TOOL_METADATA["microscope"]["prompt_source"],
                "constructor_kind": "microscope",
                "validate_real_stack": True,
            },
            "image_analysis": {
                "real_class_path": "core_tool.fiji:ImageJProcessor",
                "simulation_class_path": "Empty_function:ImageJProcessor",
                "prompt_source": SYSTEM_TOOL_METADATA["image_analysis"]["prompt_source"],
                "constructor_kind": "storage_output",
                "validate_real_stack": True,
            },
            "segmentation": {
                "real_class_path": "core_tool.cellpose_tool:Cellpose2D",
                "simulation_class_path": "Empty_function:Cellpose2D",
                "prompt_source": SYSTEM_TOOL_METADATA["segmentation"]["prompt_source"],
                "constructor_kind": "storage_output",
                "validate_real_stack": True,
            },
        },
        "user_tools": [],
    }


def read_tool_manifest_payload(manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    target_path = manifest_path or resolve_default_tool_manifest_path()
    if not target_path.exists():
        return default_tool_manifest_payload()
    try:
        return json.loads(target_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ToolManifestError(f"Tool manifest is not valid JSON: {target_path}") from exc


def write_tool_manifest_payload(payload: Dict[str, Any], manifest_path: Optional[Path] = None) -> None:
    target_path = manifest_path or resolve_default_tool_manifest_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_tool_manifest(manifest_path: Optional[Path] = None) -> ToolManifest:
    payload = read_tool_manifest_payload(manifest_path)
    if "tools" in payload and "system_tools" not in payload:
        raise ToolManifestError(
            "Legacy tool manifest format is no longer supported. "
            "Use top-level 'system_tools' and 'user_tools'."
        )

    raw_system_tools = payload.get("system_tools")
    if not isinstance(raw_system_tools, dict):
        raise ToolManifestError("Tool manifest must contain a top-level 'system_tools' object")

    raw_user_tools = payload.get("user_tools", [])
    if not isinstance(raw_user_tools, list):
        raise ToolManifestError("Tool manifest must contain a top-level 'user_tools' list")

    system_tools = {
        role: _parse_system_tool_entry(role, raw_system_tools.get(role))
        for role in SYSTEM_TOOL_ROLES
    }
    user_tools = [_parse_user_tool_entry(item) for item in raw_user_tools]
    manifest = ToolManifest(system_tools=system_tools, user_tools=user_tools)
    _validate_manifest_entries(manifest)
    return manifest


def load_enabled_user_tool_entries(manifest_path: Optional[Path] = None) -> List[UserToolManifestEntry]:
    return [entry for entry in load_tool_manifest(manifest_path).user_tools if entry.enabled]


def discover_tool_candidates(tool_dir: Optional[Path] = None) -> List[DiscoveredToolCandidate]:
    root = tool_dir or TOOL_SOURCE_DIR
    if not root.exists():
        return []

    candidates: List[DiscoveredToolCandidate] = []
    for file_path in root.rglob("*.py"):
        if file_path.name == "__init__.py":
            continue
        relative = file_path.relative_to(ROOT_DIR)
        module_name = ".".join(relative.with_suffix("").parts)
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _inherits_from_base_tool(node):
                candidates.append(
                    DiscoveredToolCandidate(
                        class_name=node.name,
                        file_path=str(file_path),
                        class_path=f"{module_name}:{node.name}",
                    )
                )
    candidates.sort(key=lambda item: item.class_path)
    return candidates


def find_unconfigured_tool_candidates(manifest: ToolManifest) -> List[DiscoveredToolCandidate]:
    configured_class_paths = {entry.class_path for entry in manifest.user_tools}
    return [
        candidate
        for candidate in discover_tool_candidates()
        if candidate.class_path not in configured_class_paths
    ]


def append_tool_manifest_entry(raw_entry: Dict[str, Any], manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    payload = read_tool_manifest_payload(manifest_path)
    system_tools = payload.setdefault("system_tools", default_tool_manifest_payload()["system_tools"])
    if not isinstance(system_tools, dict):
        raise ToolManifestError("Tool manifest must contain a top-level 'system_tools' object")
    user_tools = payload.setdefault("user_tools", [])
    if not isinstance(user_tools, list):
        raise ToolManifestError("Tool manifest must contain a top-level 'user_tools' list")
    user_tools.append(raw_entry)
    load_tool_manifest_from_payload(payload)
    write_tool_manifest_payload(payload, manifest_path)
    return payload


def load_tool_manifest_from_payload(payload: Dict[str, Any]) -> ToolManifest:
    if "system_tools" not in payload or "user_tools" not in payload:
        raise ToolManifestError("Manifest payload must contain 'system_tools' and 'user_tools'")
    raw_system_tools = payload.get("system_tools")
    raw_user_tools = payload.get("user_tools")
    if not isinstance(raw_system_tools, dict) or not isinstance(raw_user_tools, list):
        raise ToolManifestError("Manifest payload uses invalid root types")
    system_tools = {
        role: _parse_system_tool_entry(role, raw_system_tools.get(role))
        for role in SYSTEM_TOOL_ROLES
    }
    user_tools = [_parse_user_tool_entry(item) for item in raw_user_tools]
    manifest = ToolManifest(system_tools=system_tools, user_tools=user_tools)
    _validate_manifest_entries(manifest)
    return manifest


def _parse_system_tool_entry(role: str, raw_item: Any) -> SystemToolEntry:
    if role not in SYSTEM_TOOL_METADATA:
        raise ToolManifestError(f"Unsupported system tool role '{role}'")
    if not isinstance(raw_item, dict):
        raise ToolManifestError(f"System tool '{role}' must be configured as an object")

    metadata = SYSTEM_TOOL_METADATA[role]
    real_class_path = str(raw_item.get("real_class_path", "")).strip()
    simulation_class_path = str(raw_item.get("simulation_class_path", "")).strip()
    prompt_source = str(raw_item.get("prompt_source", metadata["prompt_source"])).strip()
    constructor_kind = str(raw_item.get("constructor_kind", "")).strip()

    if not real_class_path:
        raise ToolManifestError(f"System tool '{role}' is missing 'real_class_path'")
    if not simulation_class_path:
        raise ToolManifestError(f"System tool '{role}' is missing 'simulation_class_path'")
    if not prompt_source:
        raise ToolManifestError(f"System tool '{role}' is missing 'prompt_source'")
    if constructor_kind not in SUPPORTED_CONSTRUCTOR_KINDS:
        raise ToolManifestError(
            f"System tool '{role}' uses unsupported constructor kind '{constructor_kind}'"
        )

    return SystemToolEntry(
        role=role,
        tool_id=metadata["tool_id"],
        platform_name=metadata["platform_name"],
        prompt_source=prompt_source,
        port_kind=metadata["port_kind"],
        real_class_path=real_class_path,
        simulation_class_path=simulation_class_path,
        constructor_kind=constructor_kind,
        validate_real_stack=bool(raw_item.get("validate_real_stack", False)),
    )


def _parse_user_tool_entry(raw_item: Any) -> UserToolManifestEntry:
    if not isinstance(raw_item, dict):
        raise ToolManifestError("Each user tool manifest entry must be an object")
    tool_id = str(raw_item.get("tool_id", "")).strip()
    class_path = str(raw_item.get("class_path", "")).strip()
    if not tool_id:
        raise ToolManifestError("User tool manifest entry is missing 'tool_id'")
    if not class_path:
        raise ToolManifestError(f"User tool '{tool_id}' is missing 'class_path'")
    return UserToolManifestEntry(
        tool_id=tool_id,
        class_path=class_path,
        enabled=bool(raw_item.get("enabled", True)),
        planning_hint=str(raw_item.get("planning_hint", "") or "").strip(),
        execution_hint=str(raw_item.get("execution_hint", "") or "").strip(),
    )


def _validate_manifest_entries(manifest: ToolManifest) -> None:
    seen_user_tool_ids: Dict[str, str] = {}
    seen_user_class_paths: Dict[str, str] = {}

    for role in SYSTEM_TOOL_ROLES:
        if role not in manifest.system_tools:
            raise ToolManifestError(f"Missing system tool configuration for role '{role}'")

    for entry in manifest.user_tools:
        if entry.tool_id in seen_user_tool_ids:
            raise ToolManifestError(
                f"Duplicate user tool_id '{entry.tool_id}' found in manifest "
                f"('{seen_user_tool_ids[entry.tool_id]}' and '{entry.class_path}')"
            )
        seen_user_tool_ids[entry.tool_id] = entry.class_path

        if entry.class_path in seen_user_class_paths:
            raise ToolManifestError(
                f"Duplicate user tool class_path '{entry.class_path}' found in manifest"
            )
        seen_user_class_paths[entry.class_path] = entry.tool_id


def _inherits_from_base_tool(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTool":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseTool":
            return True
    return False
