import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen

from bootstrap.config import ROOT_DIR, RUNTIME_CONFIG_PATH, load_system_config, save_runtime_settings


@dataclass
class DeviceRecord:
    label: str
    adapter: str
    device_name: str


@dataclass
class MMConfigData:
    devices: List[DeviceRecord] = field(default_factory=list)
    core_props: Dict[str, str] = field(default_factory=dict)
    device_props: Dict[str, Dict[str, str]] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    focus_directions: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, List[str]] = field(default_factory=dict)


DEFAULT_GRAY = (128, 128, 128)
FILTER_COLOR_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[int, int, int]], ...] = (
    (("u-fg", "tritc", "txred", "red", "rhod"), (255, 0, 0)),
    (("u-fb", "fitc", "gfp", "green"), (0, 255, 0)),
    (("u-fu", "dapi", "hoechst", "blue"), (0, 0, 255)),
)
CHANNEL_SEMANTIC_HINTS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("brightfield", ("none", "bright", "bf", "transmitted"), "Brightfield"),
    ("dapi", ("u-fu", "dapi", "hoechst", "blue", "405"), "DAPI / 405 nm"),
    ("fitc", ("u-fb", "fitc", "gfp", "green", "488"), "FITC / 488 nm"),
    ("tritc", ("u-fg", "tritc", "txred", "red", "640", "561"), "TRITC / 640 nm"),
)
INTENSITY_PROPERTY_HINTS = ("brightness", "intensity", "power", "level", "percent")

DEFAULT_MMCORE_DEST = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "EIMS" / "Micro-Manager"
DEFAULT_FIJI_DEST = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "EIMS" / "Fiji"
FIJI_MANUAL_DOWNLOAD_URL = "https://imagej.net/software/fiji/"
FIJI_DOWNLOAD_BASE_URL = os.environ.get("EIMS_FIJI_DOWNLOAD_BASE_URL", "https://downloads.imagej.net/fiji/stable")


@dataclass
class FieldRule:
    name: str
    description: str
    core_property: Optional[str] = None
    label_tokens: Tuple[str, ...] = ()
    adapter_tokens: Tuple[str, ...] = ()
    require_labels: bool = False
    numeric_suffix_preference: str = "none"


FIELD_RULES: List[FieldRule] = [
    FieldRule(
        name="camera_device",
        description="Primary imaging camera",
        core_property="Camera",
        label_tokens=("camera",),
        adapter_tokens=("camera", "pvcam", "demo"),
    ),
    FieldRule(
        name="xy_stage_device",
        description="XY stage",
        core_property="XYStage",
        label_tokens=("xy", "stage"),
        adapter_tokens=("prior", "stage"),
    ),
    FieldRule(
        name="focus_drive",
        description="Z drive / focus drive",
        core_property="Focus",
        label_tokens=("focus", "z"),
    ),
    FieldRule(
        name="objective_device",
        description="Objective changer",
        label_tokens=("objective", "nosepiece", "turret"),
        require_labels=True,
    ),
    FieldRule(
        name="transmitted_light_device",
        description="Optional transmitted-light intensity-control device",
        label_tokens=("transmittedillumination", "transmitted", "illumination", "lamp", "led", "white light", "shutter"),
    ),
    FieldRule(
        name="Dichroic",
        description="Dichroic / filter-block switcher",
        label_tokens=("dichroic", "filter", "mirror"),
        require_labels=True,
    ),
]


def parse_mm_config(cfg_path: Path) -> MMConfigData:
    data = MMConfigData()
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    for raw_parts in csv.reader(text.splitlines()):
        if not raw_parts:
            continue
        parts = [part.strip() for part in raw_parts]
        if not parts or not parts[0] or parts[0].startswith("#"):
            continue
        keyword = parts[0].lower()

        if keyword == "device" and len(parts) >= 4 and parts[1]:
            data.devices.append(DeviceRecord(parts[1], parts[2], parts[3]))
        elif keyword == "property" and len(parts) >= 4:
            scope = parts[1]
            prop_name = parts[2]
            prop_value = ",".join(parts[3:]).strip()
            if scope.lower() == "core" and prop_name and prop_value:
                data.core_props[prop_name] = prop_value
            elif scope and prop_name:
                data.device_props.setdefault(scope, {})[prop_name] = prop_value
        elif keyword == "parent" and len(parts) >= 3:
            data.parents[parts[1]] = parts[2]
        elif keyword == "focusdirection" and len(parts) >= 3:
            data.focus_directions[parts[1]] = parts[2]
        elif keyword == "label" and len(parts) >= 4:
            data.labels.setdefault(parts[1], []).append(parts[3])
    return data


def numeric_suffix(label: str) -> int:
    match = re.search(r"(\d+)\s*$", label)
    return int(match.group(1)) if match else -1


def score_device(record: DeviceRecord, rule: FieldRule, data: MMConfigData) -> int:
    label_lower = record.label.lower()
    adapter_lower = record.adapter.lower()
    device_name_lower = record.device_name.lower()
    score = 0

    for token in rule.label_tokens:
        token_lower = token.lower()
        if token_lower in label_lower:
            score += 80
        if token_lower in device_name_lower:
            score += 20

    for token in rule.adapter_tokens:
        if token.lower() in adapter_lower:
            score += 40

    if rule.require_labels and record.label in data.labels:
        score += 120
    if record.label in data.parents:
        score += 10
    if record.label in data.focus_directions:
        score += 120
    if rule.name == "transmitted_light_device":
        props = data.device_props.get(record.label, {})
        has_declared_intensity_property = any(
            any(token in str(prop_name).lower() for token in INTENSITY_PROPERTY_HINTS)
            for prop_name in props
        )
        if has_declared_intensity_property:
            # Prefer devices that actually expose an intensity-like control property
            # over shutters or path selectors whose names happen to match light tokens.
            score += 160

    return score


def choose_candidate(candidates: List[Tuple[int, DeviceRecord]], rule: FieldRule) -> Optional[DeviceRecord]:
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[0],
            -numeric_suffix(item[1].label) if rule.numeric_suffix_preference == "highest" else numeric_suffix(item[1].label),
            item[1].label,
        )
    )
    return candidates[0][1]


def infer_device(rule: FieldRule, data: MMConfigData) -> Tuple[Optional[str], str, List[str]]:
    if rule.core_property and rule.core_property in data.core_props:
        return data.core_props[rule.core_property], f"Core property '{rule.core_property}'", []

    scored: List[Tuple[int, DeviceRecord]] = []
    for device in data.devices:
        score = score_device(device, rule, data)
        if score > 0:
            scored.append((score, device))

    best = choose_candidate(scored, rule)
    candidates = [f"{record.label} (score={score}, adapter={record.adapter})" for score, record in sorted(scored, key=lambda item: -item[0])]

    if not best:
        return None, "No confident match", candidates

    top_score = max(score for score, _record in scored)
    if sum(1 for score, _record in scored if score == top_score) > 1:
        return None, "Ambiguous heuristic match", candidates

    if len(scored) > 1 and rule.numeric_suffix_preference == "highest":
        return best.label, "Heuristic match with highest numeric suffix preference", candidates

    return best.label, "Heuristic match", candidates


def suggest_values(cfg_path: Path) -> Dict[str, Dict[str, Any]]:
    data = parse_mm_config(cfg_path)
    suggestions: Dict[str, Dict[str, Any]] = {}
    for rule in FIELD_RULES:
        value, source, candidates = infer_device(rule, data)
        suggestions[rule.name] = {
            "value": value,
            "source": source,
            "description": rule.description,
            "candidates": candidates,
        }
    return suggestions


def parse_objective_magnification(label: str, existing_map: Dict[str, int]) -> Optional[int]:
    if label in existing_map:
        return existing_map[label]

    patterns = (
        r"(\d+)\s*[Xx](?:$|[A-Z])",
        r"(\d+)[Xx]",
        r"(\d+)(?=XS?$)",
    )
    for pattern in patterns:
        match = re.search(pattern, label)
        if match:
            return int(match.group(1))
    return None


def _build_objective_labels(data: MMConfigData, objective_device: Optional[str], existing_map: Dict[str, int]) -> Dict[str, int]:
    if not objective_device:
        return {}
    labels = data.labels.get(objective_device, [])
    objective_labels: Dict[str, int] = {}
    for label in labels:
        magnification = parse_objective_magnification(label, existing_map)
        if magnification is not None:
            objective_labels[label] = magnification
    return objective_labels


def infer_filter_color(label: str, existing_map: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if label in existing_map:
        return existing_map[label]

    label_lower = label.lower()
    if "none" in label_lower or "bright" in label_lower or "bf" in label_lower:
        return DEFAULT_GRAY

    for tokens, color in FILTER_COLOR_HINTS:
        if any(token in label_lower for token in tokens):
            return color
    return DEFAULT_GRAY


def build_objectives(data: MMConfigData, objective_device: Optional[str], existing_map: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    objective_labels = _build_objective_labels(data, objective_device, existing_map)
    objectives: Dict[str, Dict[str, Any]] = {}
    for label, magnification in sorted(objective_labels.items(), key=lambda item: item[1]):
        semantic_key = f"{int(magnification)}x"
        objectives[semantic_key] = {
            "label": label,
            "magnification": int(magnification),
            "display_name": f"{int(magnification)}x objective",
        }
    return objectives


def infer_channel_semantic(label: str, used_keys: set[str]) -> Tuple[str, str]:
    label_lower = label.lower()
    for semantic_key, tokens, display_name in CHANNEL_SEMANTIC_HINTS:
        if semantic_key in used_keys:
            continue
        if any(token in label_lower for token in tokens):
            return semantic_key, display_name

    base_key = "channel"
    index = 1
    while f"{base_key}_{index}" in used_keys:
        index += 1
    return f"{base_key}_{index}", label


def _leading_number(label: str) -> Optional[int]:
    match = re.match(r"\s*(\d+)\s*[-_]", str(label or ""))
    return int(match.group(1)) if match else None


def select_brightfield_label(labels: List[str]) -> Optional[str]:
    if not labels:
        return None

    exact_priority = ("1-NONE", "BF", "Brightfield")
    by_lower = {label.lower(): label for label in labels}
    for preferred in exact_priority:
        match = by_lower.get(preferred.lower())
        if match:
            return match

    named_candidates = [
        label
        for label in labels
        if any(token in label.lower() for token in ("brightfield", "bright", "transmitted"))
    ]
    if named_candidates:
        return named_candidates[0]

    none_candidates = [label for label in labels if "none" in label.lower()]
    if none_candidates:
        return min(none_candidates, key=lambda label: (_leading_number(label) is None, _leading_number(label) or 9999, label))

    bf_candidates = [
        label
        for label in labels
        if re.search(r"(^|[-_\s])bf($|[-_\s])", label, flags=re.IGNORECASE)
    ]
    if bf_candidates:
        return bf_candidates[0]

    return None


def build_channels(
    data: MMConfigData,
    dichroic_device: Optional[str],
    existing_map: Dict[str, Tuple[int, int, int]],
) -> Dict[str, Dict[str, Any]]:
    if not dichroic_device:
        return {}
    labels = data.labels.get(dichroic_device, [])
    channels: Dict[str, Dict[str, Any]] = {}
    used_keys: set[str] = set()
    brightfield_label = select_brightfield_label(labels)
    for label in labels:
        if brightfield_label is not None and label == brightfield_label:
            semantic_key, display_name = "brightfield", "Brightfield"
        else:
            blocked_keys = set(used_keys)
            if brightfield_label is not None:
                blocked_keys.add("brightfield")
            semantic_key, display_name = infer_channel_semantic(label, blocked_keys)
        used_keys.add(semantic_key)
        color = infer_filter_color(label, existing_map)
        illumination = "transmitted" if semantic_key == "brightfield" else "fluorescence"
        channels[semantic_key] = {
            "label": label,
            "display_name": display_name,
            "color": list(color),
            "illumination": illumination,
        }
    return channels


def build_transmitted_light_mapping(data: MMConfigData, transmitted_device: Optional[str]) -> Dict[str, Any]:
    if not transmitted_device:
        return {
            "device": "",
            "intensity_property": "",
            "min": 0,
            "max": 250,
        }
    props = data.device_props.get(transmitted_device, {})
    selected_property = ""
    for prop_name in props:
        lowered = prop_name.lower()
        if any(token in lowered for token in INTENSITY_PROPERTY_HINTS):
            selected_property = prop_name
            break
    if not selected_property:
        return {
            "device": transmitted_device,
            "intensity_property": "",
            "min": 0,
            "max": 250,
        }
    return {
        "device": transmitted_device,
        "intensity_property": selected_property,
        "min": 0,
        "max": 250,
    }


def build_cfg_inventory(cfg_path: Path) -> Dict[str, Any]:
    """Return the cfg-derived identifiers that may be shown to an external mapper.

    The inventory intentionally excludes the original cfg text, comments, and filesystem
    path. Callers can safely send this structure to an AI service after user consent.
    """
    data = parse_mm_config(cfg_path)
    suggestions = suggest_values(cfg_path)
    objective_device = suggestions.get("objective_device", {}).get("value")
    dichroic_device = suggestions.get("Dichroic", {}).get("value")
    transmitted_device = suggestions.get("transmitted_light_device", {}).get("value")
    devices = []
    for device in data.devices:
        devices.append(
            {
                "name": device.label,
                "adapter": device.adapter,
                "device_type": device.device_name,
                "state_labels": list(data.labels.get(device.label, [])),
                "properties": sorted(data.device_props.get(device.label, {}).keys()),
            }
        )

    return {
        "core_roles": dict(data.core_props),
        "devices": devices,
        "suggestions": suggestions,
        "rule_mapping": {
            "objectives": build_objectives(data, objective_device, {}),
            "channels": build_channels(data, dichroic_device, {}),
            "transmitted_light": build_transmitted_light_mapping(data, transmitted_device),
        },
        "unresolved_fields": [
            name for name, suggestion in suggestions.items() if not suggestion.get("value")
        ],
    }


def format_python_dict(mapping: Dict[str, Any]) -> str:
    if not mapping:
        return "{}"
    lines = ["{"]
    for key, value in mapping.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    return "\n".join(lines)


def apply_updates(system_config_path: Path, updates: Dict[str, str]) -> List[str]:
    del system_config_path
    save_runtime_settings(system_updates=updates)
    return [field for field, value in updates.items() if value]


def apply_dict_update(system_config_path: Path, field_name: str, mapping: Dict[str, Any]) -> bool:
    del system_config_path
    save_runtime_settings(system_updates={field_name: mapping})
    return True


def load_system_config_path(system_config_path: Path, explicit_cfg_path: Optional[Path]) -> Path:
    del system_config_path
    if explicit_cfg_path is not None:
        return explicit_cfg_path

    cfg_literal = load_system_config().CONFIG_PATH
    if not cfg_literal:
        raise ValueError("CONFIG_PATH is empty. Please provide --mm-config.")
    return Path(cfg_literal)


def print_suggestions(suggestions: Dict[str, Dict[str, Any]]) -> None:
    print("\nDetected device suggestions:\n")
    for rule in FIELD_RULES:
        info = suggestions[rule.name]
        value = info["value"] or "(No candidate detected; please confirm manually)"
        print(f"- {rule.name:<24} -> {value}    [{info['description']}; source: {info['source']}]")
        if info["candidates"]:
            preview = "; ".join(info["candidates"][:3])
            print(f"  candidates: {preview}")


def print_mapping_preview(title: str, mapping: Dict[str, Any]) -> None:
    print(f"\n{title}:")
    if not mapping:
        print("  (No syncable content detected)")
        return
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def resolve_mmcore_executable() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / "mmcore.exe",
        scripts_dir / "mmcore",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    discovered = shutil.which("mmcore")
    if discovered:
        return discovered

    raise FileNotFoundError(
        "Could not find the `mmcore` CLI in the current environment. "
        "Run `uv sync` first so `pymmcore-plus[cli]` is installed."
    )


def discover_mm_install_root(dest: Path) -> Path:
    installs = [path for path in dest.glob("Micro-Manager*") if path.is_dir()]
    if installs:
        return max(installs, key=lambda path: path.stat().st_mtime)
    if dest.exists():
        return dest
    raise FileNotFoundError(f"No Micro-Manager installation directory found under {dest}")


def list_mm_install_dirs(dest: Path) -> List[Path]:
    installs = [path for path in dest.glob("Micro-Manager*") if path.is_dir()]
    installs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return installs


def remove_mm_install_dirs(installs: List[Path]) -> None:
    for install_path in installs:
        shutil.rmtree(install_path)


def resolve_mmstudio_executable(mm_dir: Path) -> Path:
    install_root = mm_dir.expanduser().resolve()
    if not install_root.exists():
        raise FileNotFoundError(
            f"Micro-Manager directory not found: {install_root}\n"
            "Update MM_DIR first, or run `uv run python system_config_wizard.py --install-mmcore`."
        )

    candidates = [
        install_root / "ImageJ.exe",
        install_root / "ImageJ-win64.exe",
        install_root / "ImageJ-win32.exe",
        install_root / "MMStudio.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in install_root.glob("ImageJ*"):
        if candidate.is_file() and candidate.suffix.lower() == ".exe":
            return candidate

    raise FileNotFoundError(
        f"Could not find a Micro-Manager GUI executable under {install_root}.\n"
        "Expected something like ImageJ.exe or ImageJ-win64.exe."
    )


def open_mmstudio(mm_dir: Optional[Path] = None) -> Path:
    configured_dir = mm_dir or Path(load_system_config().MM_DIR)
    executable = resolve_mmstudio_executable(configured_dir)
    subprocess.Popen([str(executable)], cwd=str(executable.parent))
    print(f"Started MMStudio from {executable}")
    return executable


def is_fiji_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    launcher_names = (
        "ImageJ-win64.exe",
        "ImageJ-win32.exe",
        "ImageJ.exe",
        "Fiji.exe",
        "fiji-windows-x64.exe",
        "fiji-windows.exe",
        "fiji-linux64",
        "fiji-linux-x64",
        "fiji-macosx",
        "ImageJ-linux64",
        "ImageJ-macosx",
        "fiji",
        "fiji.bat",
    )
    has_launcher = any((path / name).is_file() for name in launcher_names)
    has_fiji_layout = (path / "jars").is_dir() and (path / "plugins").is_dir()
    has_fiji_metadata = (path / "db.xml.gz").is_file()
    return has_launcher and has_fiji_layout and has_fiji_metadata


def resolve_fiji_root(fiji_dir: Path) -> Path:
    candidate = fiji_dir.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    candidates = [candidate]
    if candidate.name.lower() != "fiji.app":
        candidates.append(candidate / "Fiji.app")
    for item in candidates:
        if is_fiji_root(item):
            return item
    raise FileNotFoundError(
        f"Fiji installation not found or incomplete: {candidate}\n"
        "Expected a Fiji.app directory containing an ImageJ launcher and jars/ or plugins/.\n"
        f"Manual download: {FIJI_MANUAL_DOWNLOAD_URL}"
    )


def _iter_fiji_root_candidates(
    root: Path,
    *,
    max_depth: int = 3,
    include_extract_tmp: bool = False,
) -> List[Path]:
    base = root.expanduser().resolve()
    if not base.exists():
        return []

    seen: set[Path] = set()
    candidates: List[Path] = []

    def add(candidate: Path) -> None:
        resolved = candidate.resolve()
        if not include_extract_tmp and any(part.lower() == "_fiji_extract_tmp" for part in resolved.parts):
            return
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    add(base)
    if base.is_file():
        add(base.parent)
        return candidates

    stack: List[Tuple[Path, int]] = [(base, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = list(current.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            add(child)
            if child.name.lower() != "fiji.app":
                add(child / "Fiji.app")
            stack.append((child, depth + 1))

    return candidates


def list_fiji_install_dirs(dest: Path) -> List[Path]:
    installs: List[Path] = []
    if not dest.exists():
        return installs
    for candidate in _iter_fiji_root_candidates(dest):
        try:
            root = resolve_fiji_root(candidate)
        except FileNotFoundError:
            continue
        if root not in installs:
            installs.append(root)
    installs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return installs


def discover_fiji_roots() -> List[Path]:
    configured_path = load_system_config().FIJI_PATH
    search_dirs = [
        Path(configured_path) if configured_path else None,
        DEFAULT_FIJI_DEST,
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path(os.environ.get("LOCALAPPDATA", "")),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
        Path.home() / "Downloads",
        Path.home() / "Desktop",
        ROOT_DIR.parent,
    ]
    roots: List[Path] = []
    for raw_dir in search_dirs:
        if raw_dir is None or not str(raw_dir):
            continue
        for candidate in _iter_fiji_root_candidates(raw_dir):
            try:
                root = resolve_fiji_root(candidate)
            except FileNotFoundError:
                continue
            if root not in roots:
                roots.append(root)
    roots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return roots


def _fiji_download_filename() -> str:
    system = sys.platform.lower()
    machine = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower() or os.environ.get("PROCESSOR_IDENTIFIER", "").lower()
    is_arm64 = "arm64" in machine or "aarch64" in machine
    if system.startswith("win"):
        return "fiji-stable-win-arm64-jdk.zip" if is_arm64 else "fiji-stable-win64-jdk.zip"
    if system == "darwin":
        return "fiji-stable-macos-arm64-jdk.zip" if is_arm64 else "fiji-stable-macos64-jdk.zip"
    if system.startswith("linux"):
        return "fiji-stable-linux-arm64-jdk.zip" if is_arm64 else "fiji-stable-linux64-jdk.zip"
    raise RuntimeError(
        f"Automatic Fiji download is not configured for platform '{sys.platform}'. "
        f"Please download Fiji manually from {FIJI_MANUAL_DOWNLOAD_URL}"
    )


def _normalize_fiji_download_dest(dest: Path) -> Path:
    candidate = dest.expanduser()
    if candidate.suffix.lower() == ".exe":
        return candidate.parent.resolve()
    if candidate.name.lower() == "fiji.app":
        return candidate.parent.resolve()
    return candidate.resolve()


def _download_file_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output:
        total_size = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        next_report_bytes = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = downloaded / total_size
                if downloaded >= next_report_bytes or downloaded == total_size:
                    print(f"Downloaded {downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB ({progress:.0%})")
                    next_report_bytes = downloaded + (25 * 1024 * 1024)
            elif downloaded >= next_report_bytes:
                print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB ...")
                next_report_bytes = downloaded + (25 * 1024 * 1024)
    final_size = destination.stat().st_size if destination.exists() else 0
    if total_size > 0 and final_size != total_size:
        raise RuntimeError(
            f"Downloaded file size mismatch for {url}: expected {total_size} bytes, got {final_size} bytes."
        )


def _safe_remove_tree(path: Path, expected_parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = expected_parent.resolve()
    if resolved_parent not in resolved_path.parents:
        raise RuntimeError(f"Refusing to remove unexpected path outside install destination: {resolved_path}")
    shutil.rmtree(resolved_path)


def install_fiji(dest: Path, *, update_runtime_config: bool) -> Path:
    target_dest = _normalize_fiji_download_dest(dest)
    target_dest.mkdir(parents=True, exist_ok=True)

    archive_name = _fiji_download_filename()
    archive_url = f"{FIJI_DOWNLOAD_BASE_URL}/{archive_name}"
    archive_path = target_dest / archive_name
    extract_root = target_dest / "_fiji_extract_tmp"

    if extract_root.exists():
        _safe_remove_tree(extract_root, target_dest)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading stable Fiji from {archive_url}")
    print(f"Destination: {target_dest}")
    try:
        _download_file_with_progress(archive_url, archive_path)
        if not zipfile.is_zipfile(archive_path):
            raise RuntimeError(
                f"Downloaded file is not a valid ZIP archive: {archive_path}"
            )
    except Exception as exc:
        raise RuntimeError(
            "Automatic Fiji download failed.\n"
            f"- URL: {archive_url}\n"
            f"- Destination: {archive_path}\n"
            f"- You can retry later or download manually from {FIJI_MANUAL_DOWNLOAD_URL}"
        ) from exc

    print("Extracting Fiji archive ...")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_root)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract Fiji archive: {archive_path}") from exc
    finally:
        if archive_path.exists():
            archive_path.unlink()

    extracted_root: Optional[Path] = None
    for candidate in _iter_fiji_root_candidates(extract_root, include_extract_tmp=True):
        try:
            extracted_root = resolve_fiji_root(candidate)
            break
        except FileNotFoundError:
            continue

    if extracted_root is None:
        raise RuntimeError(
            f"Downloaded Fiji archive did not contain a recognizable Fiji installation under {extract_root}"
        )

    final_root = target_dest / extracted_root.name
    if final_root.exists():
        if is_fiji_root(final_root):
            print(f"Replacing existing Fiji install under {final_root}")
        _safe_remove_tree(final_root, target_dest)
    shutil.move(str(extracted_root), str(final_root))
    if extract_root.exists():
        _safe_remove_tree(extract_root, target_dest)

    print(f"Installed Fiji to {final_root}")
    if update_runtime_config:
        save_runtime_settings(system_updates={"FIJI_PATH": str(final_root)})
        print(f"Updated FIJI_PATH in {RUNTIME_CONFIG_PATH}")
    return final_root


def resolve_fiji_executable(fiji_root: Path) -> Path:
    candidates = [
        fiji_root / "fiji-windows-x64.exe",
        fiji_root / "fiji-windows.exe",
        fiji_root / "ImageJ-win64.exe",
        fiji_root / "ImageJ-win32.exe",
        fiji_root / "ImageJ.exe",
        fiji_root / "Fiji.exe",
        fiji_root / "fiji.bat",
        fiji_root / "fiji",
        fiji_root / "fiji-linux-x64",
        fiji_root / "fiji-linux64",
        fiji_root / "fiji-macosx",
        fiji_root / "ImageJ-linux64",
        fiji_root / "ImageJ-macosx",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a Fiji/ImageJ launcher under {fiji_root}")


def configured_fiji_path() -> Path:
    configured_path = load_system_config().FIJI_PATH
    if not configured_path:
        raise FileNotFoundError(
            "FIJI_PATH is empty. Configure Fiji first:\n"
            "  uv run python system_config_wizard.py --setup-fiji\n"
            "Or point to an existing install:\n"
            '  uv run python system_config_wizard.py --detect-fiji --fiji-dir "C:\\Path\\To\\Fiji.app"'
        )
    return Path(configured_path)


def detect_fiji(fiji_dir: Optional[Path], *, update_runtime_config: bool) -> Path:
    if fiji_dir is not None:
        fiji_root = resolve_fiji_root(fiji_dir)
    else:
        roots = discover_fiji_roots()
        if not roots:
            raise FileNotFoundError(
                "Could not auto-detect a Fiji installation.\n"
                f"Download and install Fiji manually from: {FIJI_MANUAL_DOWNLOAD_URL}\n"
                "Then point to the installed Fiji.app:\n"
                '  uv run python system_config_wizard.py --detect-fiji --fiji-dir "C:\\Path\\To\\Fiji.app"'
            )
        fiji_root = roots[0]
        if len(roots) > 1:
            print("Detected multiple Fiji installs; using the most recently modified one:")
            for root in roots[:5]:
                print(f"  - {root}")

    print(f"Detected Fiji root: {fiji_root}")
    if update_runtime_config:
        save_runtime_settings(system_updates={"FIJI_PATH": str(fiji_root)})
        print(f"Updated FIJI_PATH in {RUNTIME_CONFIG_PATH}")
    return fiji_root


def open_fiji(fiji_dir: Optional[Path] = None) -> Path:
    configured_dir = fiji_dir or configured_fiji_path()
    fiji_root = resolve_fiji_root(configured_dir)
    executable = resolve_fiji_executable(fiji_root)
    subprocess.Popen([str(executable)], cwd=str(executable.parent))
    print(f"Started Fiji from {executable}")
    return executable


def find_bundled_fiji_java_home(fiji_root: Optional[Path]) -> Optional[Path]:
    if fiji_root is None:
        return None
    java_root = fiji_root / "java"
    if not java_root.is_dir():
        return None

    java_names = {"java.exe", "java"}
    candidates: List[Path] = []
    for candidate in java_root.rglob("*"):
        if candidate.is_file() and candidate.name.lower() in java_names and candidate.parent.name.lower() == "bin":
            candidates.append(candidate.parent.parent)

    if not candidates:
        return None
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0]


def prefer_java_home(java_home: Path) -> None:
    java_home = java_home.expanduser().resolve()
    bin_dir = java_home / "bin"
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"Java bin directory not found under {java_home}")

    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    normalized_bin = str(bin_dir)
    filtered_entries: List[str] = []
    for entry in path_entries:
        try:
            if Path(entry).expanduser().resolve() == bin_dir:
                continue
        except OSError:
            pass
        filtered_entries.append(entry)
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = os.pathsep.join([normalized_bin, *filtered_entries]) if filtered_entries else normalized_bin


def check_java(fiji_root: Optional[Path] = None) -> bool:
    print("Checking Java for pyimagej ...")
    bundled_java_home = find_bundled_fiji_java_home(fiji_root)
    if bundled_java_home is not None:
        prefer_java_home(bundled_java_home)
        print(f"Using bundled Fiji JDK: {bundled_java_home}")
    try:
        java_result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    except FileNotFoundError:
        print("Java check failed: `java` was not found on PATH.")
        print("Install a Java/JDK runtime and ensure `java -version` works in this same terminal.")
        return False
    java_output = (java_result.stderr or java_result.stdout).strip()
    if java_result.returncode != 0:
        print("Java check failed: `java -version` did not run successfully.")
        if java_output:
            print(java_output)
        print("Install a Java/JDK runtime and ensure `java -version` works in this same terminal.")
        return False
    print(java_output.splitlines()[0] if java_output else "`java -version` succeeded.")

    try:
        import jpype

        jvm_path = jpype.getDefaultJVMPath()
    except Exception as exc:
        print("JPype could not locate a JVM for pyimagej.")
        print(f"Error: {exc}")
        print("Install a Java/JDK runtime that JPype can find before using Fiji-dependent features.")
        return False

    print(f"JPype JVM path: {jvm_path}")
    return True


def ensure_project_maven() -> bool:
    try:
        from scyjava._cjdk_fetch import cjdk_fetch_maven

        print("Preparing project-managed Maven through scyjava/cjdk ...")
        cjdk_fetch_maven()
        print("Project-managed Maven is ready for pyimagej.")
        return True
    except Exception as exc:
        print("Project-managed Maven setup failed.")
        print(f"Error: {exc}")
        print("Retry with network access, or configure a valid EIMS_MAVEN_BIN / system.MAVEN_BIN.")
        return False


def _cleanup_fiji_bootstrap_caches() -> bool:
    runtime_root = ROOT_DIR / ".runtime"
    jgo_cache_dir = runtime_root / "jgo"
    m2_repo_dir = Path.home() / ".m2" / "repository"
    cleanup_targets = (
        m2_repo_dir / "cisd" / "jhdf5",
        m2_repo_dir / "net" / "imglib2-BOOTSTRAPPER",
        jgo_cache_dir,
    )
    removed_any = False
    for target in cleanup_targets:
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
            print(f"Removed stale Fiji bootstrap cache: {target}")
            removed_any = True
        except OSError as exc:
            print(f"Could not remove stale Fiji bootstrap cache: {target}")
            print(f"Error: {exc}")
    return removed_any


def _should_retry_fiji_bootstrap(exc: Exception) -> bool:
    message = str(exc or "").lower()
    retry_markers = (
        "failed to bootstrap the artifact",
        "dependencyresolutionexception",
        "premature end of content-length",
        "could not transfer artifact",
        "lastupdated",
        "failed to create a jvm with the requested environment",
    )
    return any(marker in message for marker in retry_markers)


def check_fiji(fiji_dir: Optional[Path], *, interactive: bool) -> bool:
    fiji_root = resolve_fiji_root(fiji_dir or configured_fiji_path())
    print(f"Checking Fiji root: {fiji_root}")
    resolve_fiji_executable(fiji_root)

    if not check_java(fiji_root):
        print("Skipping pyimagej initialization because Java/JVM is not ready.")
        return False

    try:
        import imagej
        import scyjava.config as sjconf

        runtime_root = ROOT_DIR / ".runtime"
        jgo_cache_dir = runtime_root / "jgo"
        m2_repo_dir = Path.home() / ".m2" / "repository"
        jgo_cache_dir.mkdir(parents=True, exist_ok=True)
        m2_repo_dir.mkdir(parents=True, exist_ok=True)
        sjconf.set_java_constraints(fetch="auto")
        sjconf.set_cache_dir(jgo_cache_dir)
        sjconf.set_m2_repo(m2_repo_dir)
        if not ensure_project_maven():
            return False
        mode = imagej.Mode.INTERACTIVE if interactive else imagej.Mode.HEADLESS
        print(f"Initializing pyimagej in {mode} mode ...")
        try:
            ij = imagej.init(str(fiji_root), mode=mode)
        except Exception as exc:
            if not _should_retry_fiji_bootstrap(exc):
                raise
            print("pyimagej bootstrap failed in a way that often leaves stale Maven/jgo cache entries.")
            print("Cleaning targeted Fiji bootstrap caches and retrying once ...")
            cleaned = _cleanup_fiji_bootstrap_caches()
            if not cleaned:
                print("No targeted stale Fiji bootstrap caches were found to clean.")
            if not ensure_project_maven():
                return False
            ij = imagej.init(str(fiji_root), mode=mode)
        print(f"ImageJ version: {ij.getVersion()}")
        try:
            from core_tool.fiji import check_declared_fiji_capabilities

            capability_results = check_declared_fiji_capabilities(ij)
            if capability_results:
                print("\nChecking declared Fiji capabilities ...")
                missing_results = []
                for result in capability_results:
                    status = "OK" if result["available"] else "MISSING"
                    print(f"- {result['label']}: {status} ({result['required_for']})")
                    if not result["available"]:
                        missing_results.append(result)
                        if result.get("install_hint"):
                            print(f"  hint: {result['install_hint']}")
                        if result.get("detail"):
                            print(f"  detail: {result['detail']}")
                if missing_results:
                    print(
                        "\nFiji initialized successfully, but some plugin-dependent "
                        "EIMS features may fail until the missing capabilities are installed."
                    )
        except Exception as exc:
            print("Fiji capability precheck could not be completed.")
            print(f"Error: {exc}")
        return True
    except Exception as exc:
        print("pyimagej initialization failed.")
        print(f"Error: {exc}")
        print("Fiji-dependent features require Fiji plus a Java/JDK environment usable by pyimagej.")
        return False


def setup_fiji(
    fiji_dir: Optional[Path],
    *,
    update_runtime_config: bool,
    interactive: bool,
) -> Path:
    print("Setting up Fiji. EIMS will first try to reuse an existing Fiji installation.")
    try:
        fiji_root = detect_fiji(fiji_dir, update_runtime_config=update_runtime_config)
    except FileNotFoundError:
        download_dest = fiji_dir or DEFAULT_FIJI_DEST
        print("No existing Fiji installation was detected.")
        print("Falling back to automatic download of the stable Fiji build with bundled JDK ...")
        fiji_root = install_fiji(download_dest, update_runtime_config=update_runtime_config)

    print("\nJava reminder:")
    print("Fiji image-processing runtime uses pyimagej, which requires a working Java/JDK environment.")
    print("This wizard does not install Java automatically. Ensure `java -version` works in this terminal.")
    print("")
    check_fiji(fiji_root, interactive=interactive)
    return fiji_root


def build_mmcore_install_command(
    mmcore_executable: str,
    dest: Path,
    release: str,
    *,
    test_adapters: bool,
) -> List[str]:
    command = [
        mmcore_executable,
        "install",
        "--plain-output",
        "--dest",
        str(dest),
        "--release",
        release,
    ]
    if test_adapters:
        command.append("--test-adapters")
    return command


def install_mmcore(
    dest: Path,
    release: str,
    *,
    test_adapters: bool,
    update_runtime_config: bool,
    clean_dest: bool,
    reuse_existing: bool,
) -> Path:
    target_dest = dest.expanduser().resolve()
    target_dest.mkdir(parents=True, exist_ok=True)

    existing_installs = list_mm_install_dirs(target_dest)
    if existing_installs:
        if reuse_existing:
            install_root = existing_installs[0]
            print(f"Reusing existing Micro-Manager install: {install_root}")
            if update_runtime_config:
                save_runtime_settings(system_updates={"MM_DIR": str(install_root)})
                print(f"Updated MM_DIR in {RUNTIME_CONFIG_PATH}")
            print("Next step: make sure CONFIG_PATH points to your real Micro-Manager .cfg file.")
            return install_root

        discovered = "\n".join(f"  - {path}" for path in existing_installs)
        print("Detected existing Micro-Manager installs in destination:")
        print(discovered)
        if clean_dest:
            print("Cleaning existing Micro-Manager installs from destination ...")
        else:
            print("Defaulting to overwrite: cleaning existing Micro-Manager installs before reinstalling ...")
        remove_mm_install_dirs(existing_installs)

    mmcore_executable = resolve_mmcore_executable()
    command = build_mmcore_install_command(
        mmcore_executable,
        target_dest,
        release,
        test_adapters=test_adapters,
    )

    print(f"Installing Micro-Manager into {target_dest} ...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Micro-Manager silent install failed.\n"
            f"- Command: {command}\n"
            f"- Exit status: {exc.returncode}\n"
            f"- Requested install parent: {target_dest}\n"
            f"- Windows long-path / permission issues are common here, especially in deep repo folders.\n"
            f"- Try a shorter destination, for example:\n"
            f"  uv run python system_config_wizard.py --install-mmcore --mmcore-dest \"{DEFAULT_MMCORE_DEST}\"\n"
            f"- If you already have a working Micro-Manager install, skip installation and point MM_DIR at it directly.\n"
            f"- After install, CONFIG_PATH still needs to point to your real .cfg file."
        ) from exc

    install_root = discover_mm_install_root(target_dest)
    print(f"Detected install root: {install_root}")

    if update_runtime_config:
        save_runtime_settings(system_updates={"MM_DIR": str(install_root)})
        print(f"Updated MM_DIR in {RUNTIME_CONFIG_PATH}")

    print("Next step: make sure CONFIG_PATH points to your real Micro-Manager .cfg file.")
    return install_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure EIMS external runtimes and suggest device names from a real Micro-Manager cfg."
    )
    parser.add_argument(
        "--mm-config",
        type=Path,
        default=None,
        help="Path to the Micro-Manager system configuration (.cfg). Defaults to system.CONFIG_PATH in config/runtime_config.json.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply detected device and semantic mapping updates to config/runtime_config.json.",
    )
    parser.add_argument(
        "--install-mmcore",
        action="store_true",
        help="Install a compatible Micro-Manager build with `mmcore install` and update MM_DIR.",
    )
    parser.add_argument(
        "--open-mmstudio",
        action="store_true",
        help="Open the Micro-Manager GUI from the configured MM_DIR.",
    )
    parser.add_argument(
        "--setup-fiji",
        action="store_true",
        help="Reuse an existing Fiji install when found; otherwise auto-download Fiji, update FIJI_PATH, and check Java/pyimagej.",
    )
    parser.add_argument(
        "--detect-fiji",
        action="store_true",
        help="Detect an existing Fiji installation and update FIJI_PATH.",
    )
    parser.add_argument(
        "--open-fiji",
        action="store_true",
        help="Open Fiji from the configured FIJI_PATH.",
    )
    parser.add_argument(
        "--check-java",
        action="store_true",
        help="Check whether Java/JVM is available for pyimagej. Does not install Java.",
    )
    parser.add_argument(
        "--check-fiji",
        action="store_true",
        help="Validate FIJI_PATH and try pyimagej initialization.",
    )
    parser.add_argument(
        "--mm-dir",
        type=Path,
        default=None,
        help="Override MM_DIR when opening the Micro-Manager GUI.",
    )
    parser.add_argument(
        "--fiji-dir",
        type=Path,
        default=None,
        help="Override or provide a Fiji.app directory for Fiji commands.",
    )
    parser.add_argument(
        "--mmcore-dest",
        type=Path,
        default=DEFAULT_MMCORE_DEST,
        help=f"Parent directory used by `mmcore install`. Default: {DEFAULT_MMCORE_DEST}",
    )
    parser.add_argument(
        "--mmcore-release",
        default="latest-compatible",
        help="Release passed to `mmcore install` (for example: latest-compatible, latest, or YYYYMMDD).",
    )
    parser.add_argument(
        "--test-adapters",
        action="store_true",
        help="Install only the Micro-Manager test adapters bundle.",
    )
    parser.add_argument(
        "--skip-config-update",
        action="store_true",
        help="Do not write detected Micro-Manager/Fiji paths back into config/runtime_config.json.",
    )
    parser.add_argument(
        "--clean-dest",
        action="store_true",
        help="Remove existing Micro-Manager* directories in --mmcore-dest before reinstalling. This is now the default unless --reuse-existing is used.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the latest existing Micro-Manager install in --mmcore-dest and skip reinstall.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mode for --check-fiji or --setup-fiji instead of headless mode.",
    )
    args = parser.parse_args()

    if args.install_mmcore:
        if args.clean_dest and args.reuse_existing:
            parser.error("--clean-dest and --reuse-existing cannot be used together.")
        install_mmcore(
            args.mmcore_dest,
            args.mmcore_release,
            test_adapters=args.test_adapters,
            update_runtime_config=not args.skip_config_update,
            clean_dest=args.clean_dest,
            reuse_existing=args.reuse_existing,
        )
        return

    if args.open_mmstudio:
        open_mmstudio(args.mm_dir)
        return

    if args.setup_fiji:
        if args.clean_dest and args.reuse_existing:
            parser.error("--clean-dest and --reuse-existing cannot be used together.")
        try:
            setup_fiji(
                args.fiji_dir,
                update_runtime_config=not args.skip_config_update,
                interactive=args.interactive,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            raise SystemExit(1) from exc
        return

    if args.detect_fiji:
        try:
            detect_fiji(args.fiji_dir, update_runtime_config=not args.skip_config_update)
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            raise SystemExit(1) from exc
        return

    if args.open_fiji:
        try:
            open_fiji(args.fiji_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            raise SystemExit(1) from exc
        return

    if args.check_java:
        if not check_java():
            raise SystemExit(1)
        return

    if args.check_fiji:
        try:
            if not check_fiji(args.fiji_dir, interactive=args.interactive):
                raise SystemExit(1)
        except (FileNotFoundError, RuntimeError) as exc:
            print(exc)
            raise SystemExit(1) from exc
        return

    system_config = load_system_config()
    cfg_path = load_system_config_path(Path("config/runtime_config.json"), args.mm_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Micro-Manager config not found: {cfg_path}")

    cfg_data = parse_mm_config(cfg_path)
    suggestions = suggest_values(cfg_path)
    print_suggestions(suggestions)

    objective_device = suggestions["objective_device"]["value"]
    dichroic_device = suggestions["Dichroic"]["value"]

    objectives = build_objectives(cfg_data, objective_device, {})
    channels = build_channels(cfg_data, dichroic_device, {})
    transmitted_light = build_transmitted_light_mapping(cfg_data, suggestions["transmitted_light_device"]["value"])

    print_mapping_preview("objectives semantic mapping preview", objectives)
    print_mapping_preview("channels semantic mapping preview", channels)
    print_mapping_preview("transmitted_light preview", transmitted_light)

    if args.apply:
        updates = {
            field: info["value"]
            for field, info in suggestions.items()
            if info["value"]
        }
        runtime_config_path = Path("config/runtime_config.json")
        applied = apply_updates(runtime_config_path, updates)
        dict_updates: List[str] = []
        if objectives and apply_dict_update(runtime_config_path, "objectives", objectives):
            dict_updates.append("objectives")
        if channels and apply_dict_update(runtime_config_path, "channels", channels):
            dict_updates.append("channels")
        if transmitted_light and apply_dict_update(runtime_config_path, "transmitted_light", transmitted_light):
            dict_updates.append("transmitted_light")

        all_updates = applied + dict_updates
        if all_updates:
            print(f"\nUpdated {runtime_config_path}: {', '.join(all_updates)}")
        else:
            print("\nNo changes were applied. Please check the candidates manually.")
    else:
        print("\nDry-run only. Re-run with:")
        print("    uv run python system_config_wizard.py --mm-config <your.cfg> --apply")


if __name__ == "__main__":
    main()
