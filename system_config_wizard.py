import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    parents: Dict[str, str] = field(default_factory=dict)
    focus_directions: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, List[str]] = field(default_factory=dict)


DEFAULT_GRAY = (128, 128, 128)
FILTER_COLOR_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[int, int, int]], ...] = (
    (("u-fg", "tritc", "txred", "red", "rhod"), (255, 0, 0)),
    (("u-fb", "fitc", "gfp", "green"), (0, 255, 0)),
    (("u-fu", "dapi", "hoechst", "blue"), (0, 0, 255)),
)

DEFAULT_MMCORE_DEST = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "EIMS" / "Micro-Manager"
DEFAULT_FIJI_DEST = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "EIMS" / "Fiji"
FIJI_MANUAL_DOWNLOAD_URL = "https://imagej.net/software/fiji/"


@dataclass
class FieldRule:
    name: str
    description: str
    core_property: Optional[str] = None
    exact_labels: Tuple[str, ...] = ()
    label_tokens: Tuple[str, ...] = ()
    adapter_tokens: Tuple[str, ...] = ()
    require_labels: bool = False
    numeric_suffix_preference: str = "none"


FIELD_RULES: List[FieldRule] = [
    FieldRule(
        name="camera_device",
        description="Primary imaging camera",
        core_property="Camera",
        exact_labels=("Camera-1",),
        label_tokens=("camera",),
        adapter_tokens=("camera", "pvcam", "demo"),
    ),
    FieldRule(
        name="xy_stage_device",
        description="XY stage",
        core_property="XYStage",
        exact_labels=("XYStage",),
        label_tokens=("xy", "stage"),
        adapter_tokens=("prior", "stage"),
    ),
    FieldRule(
        name="focus_drive",
        description="Z drive / focus drive",
        core_property="Focus",
        exact_labels=("FocusDrive",),
        label_tokens=("focus", "z"),
    ),
    FieldRule(
        name="objective_device",
        description="Objective changer",
        exact_labels=("Objective",),
        label_tokens=("objective", "nosepiece", "turret"),
        require_labels=True,
    ),
    FieldRule(
        name="transmittedIllumination",
        description="Transmitted-light illuminator",
        exact_labels=("TransmittedIllumination 2", "TransmittedIllumination 1"),
        label_tokens=("transmittedillumination", "transmitted", "illumination", "lamp", "led"),
        numeric_suffix_preference="highest",
    ),
    FieldRule(
        name="Dichroic",
        description="Dichroic / filter-block switcher",
        exact_labels=("Dichroic 2", "Dichroic"),
        label_tokens=("dichroic", "filter", "mirror"),
        require_labels=True,
    ),
]


def parse_mm_config(cfg_path: Path) -> MMConfigData:
    data = MMConfigData()
    for raw_line in cfg_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        keyword = parts[0].lower()

        if keyword == "device" and len(parts) >= 4 and parts[1]:
            data.devices.append(DeviceRecord(parts[1], parts[2], parts[3]))
        elif keyword == "property" and len(parts) >= 4:
            scope = parts[1]
            prop_name = parts[2]
            prop_value = ",".join(parts[3:]).strip()
            if scope.lower() == "core" and prop_name and prop_value:
                data.core_props[prop_name] = prop_value
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

    for idx, exact in enumerate(rule.exact_labels):
        if record.label == exact:
            score += 1000 - idx

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


def build_objective_labels(data: MMConfigData, objective_device: Optional[str], existing_map: Dict[str, int]) -> Dict[str, int]:
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


def build_dichroic_colors(
    data: MMConfigData,
    dichroic_device: Optional[str],
    existing_map: Dict[str, Tuple[int, int, int]],
) -> Dict[str, Tuple[int, int, int]]:
    if not dichroic_device:
        return {}
    labels = data.labels.get(dichroic_device, [])
    return {label: infer_filter_color(label, existing_map) for label in labels}


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
    has_launcher = any(
        (path / name).exists()
        for name in (
            "ImageJ-win64.exe",
            "ImageJ-win32.exe",
            "ImageJ.exe",
            "Fiji.exe",
            "ImageJ-linux64",
            "ImageJ-macosx",
        )
    ) or any(candidate.is_file() for candidate in path.glob("ImageJ*"))
    has_fiji_layout = (path / "jars").is_dir() or (path / "plugins").is_dir()
    return has_launcher and has_fiji_layout


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


def list_fiji_install_dirs(dest: Path) -> List[Path]:
    installs: List[Path] = []
    if not dest.exists():
        return installs
    candidates = [dest / "Fiji.app"]
    candidates.extend(path for path in dest.glob("Fiji*") if path.is_dir())
    candidates.extend(path / "Fiji.app" for path in dest.glob("*") if path.is_dir())
    for candidate in candidates:
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
        candidates = [raw_dir]
        if raw_dir.exists() and raw_dir.is_dir():
            candidates.extend(path for path in raw_dir.glob("Fiji*") if path.is_dir())
            candidates.extend(path / "Fiji.app" for path in raw_dir.glob("*") if path.is_dir())
        for candidate in candidates:
            try:
                root = resolve_fiji_root(candidate)
            except FileNotFoundError:
                continue
            if root not in roots:
                roots.append(root)
    roots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return roots


def resolve_fiji_executable(fiji_root: Path) -> Path:
    candidates = [
        fiji_root / "ImageJ-win64.exe",
        fiji_root / "ImageJ-win32.exe",
        fiji_root / "ImageJ.exe",
        fiji_root / "Fiji.exe",
        fiji_root / "ImageJ-linux64",
        fiji_root / "ImageJ-macosx",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    for candidate in fiji_root.glob("ImageJ*"):
        if candidate.is_file():
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


def check_java() -> bool:
    print("Checking Java for pyimagej ...")
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


def check_fiji(fiji_dir: Optional[Path], *, interactive: bool) -> bool:
    fiji_root = resolve_fiji_root(fiji_dir or configured_fiji_path())
    print(f"Checking Fiji root: {fiji_root}")
    resolve_fiji_executable(fiji_root)

    if not check_java():
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
        ij = imagej.init(str(fiji_root), mode=mode)
        print(f"ImageJ version: {ij.getVersion()}")
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
    print("Setting up Fiji from an existing installation. Java will be checked but not installed or modified.")
    fiji_root = detect_fiji(fiji_dir, update_runtime_config=update_runtime_config)

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
        help="Path to the Micro-Manager system configuration (.cfg). Defaults to CONFIG_PATH in system_config.py.",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        default=Path("config/system_config.py"),
        help="Path to config/system_config.py.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply detected device-name updates to system_config.py.",
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
        help="Detect an existing Fiji install, update FIJI_PATH, and check Java/pyimagej.",
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

    system_config_path = args.system_config
    if not system_config_path.exists():
        raise FileNotFoundError(f"system_config.py not found at {system_config_path}")

    system_config = load_system_config()
    cfg_path = load_system_config_path(system_config_path, args.mm_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Micro-Manager config not found: {cfg_path}")

    cfg_data = parse_mm_config(cfg_path)
    suggestions = suggest_values(cfg_path)
    print_suggestions(suggestions)

    existing_objective_labels = dict(system_config.objective_labels)
    existing_dichroic_colors = dict(system_config.dichroic_colors)
    objective_device = suggestions["objective_device"]["value"]
    dichroic_device = suggestions["Dichroic"]["value"]

    objective_labels = build_objective_labels(cfg_data, objective_device, existing_objective_labels)
    dichroic_colors = build_dichroic_colors(cfg_data, dichroic_device, existing_dichroic_colors)

    print_mapping_preview("objective_labels preview", objective_labels)
    print_mapping_preview("dichroic_colors preview", dichroic_colors)

    if args.apply:
        updates = {
            field: info["value"]
            for field, info in suggestions.items()
            if info["value"]
        }
        applied = apply_updates(system_config_path, updates)
        dict_updates: List[str] = []
        if objective_labels and apply_dict_update(system_config_path, "objective_labels", objective_labels):
            dict_updates.append("objective_labels")
        if dichroic_colors and apply_dict_update(system_config_path, "dichroic_colors", dichroic_colors):
            dict_updates.append("dichroic_colors")

        all_updates = applied + dict_updates
        if all_updates:
            print(f"\nUpdated {system_config_path}: {', '.join(all_updates)}")
        else:
            print("\nNo changes were applied. Please check the candidates manually.")
    else:
        print("\nDry-run only. Re-run with:")
        print("    python system_config_wizard.py --mm-config <your.cfg> --apply")


if __name__ == "__main__":
    main()
