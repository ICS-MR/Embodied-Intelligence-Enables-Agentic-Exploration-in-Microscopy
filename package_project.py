from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

from utils.tool_manifest import default_tool_manifest_payload


ROOT_DIR = Path(__file__).resolve().parent

INCLUDE_DIRS = [
    "adapters",
    "agent",
    "api",
    "assistant",
    "bootstrap",
    "config",
    "configs",
    "core_tool",
    "docs",
    "evaluation",
    "experiments",
    "front",
    "PSF",
    "prompts",
    "scripts",
    "services",
    "tests",
    "tool",
    "user_skills",
    "utils",
]

INCLUDE_FILES = [
    "app.py",
    "app_mock.py",
    "main.py",
    "system_config_wizard.py",
    "Empty_function.py",
    "create_tool.py",
    "tool_generation.py",
    "package_project.py",
    "README.md",
    "Hardware-Free-Demo.ipynb",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    ".python-version",
    ".gitignore",
    ".env.example",
    "LICENSE",
]

DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".venv",
    ".uv-cache",
    "__pycache__",
    "te",
    "model",
    "weights",
    "build",
    "dist",
    "wheels",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".micro-manager",
    "uploaded_cfg",
]

DEFAULT_EXCLUDE_FILES = [
    ".env",
    "config/runtime_config.json",
    "config/tool_manifest.json",
]

DEFAULT_EXCLUDE_GLOBS = [
    "*.log",
    "*.pyc",
    "*.pyo",
    "*.tmp.py",
    "*.pth",
    "*.pt",
    "*.onnx",
    "*.ckpt",
]

RUNTIME_ONLY_EXCLUDES = [
    "tests",
    "evaluation",
    "Hardware-Free-Demo.ipynb",
    "scripts/test_cellpose_api.py",
    "scripts/test_cellpose_real_workflow.py",
    "scripts/run_real_cellpose_targeting.py",
    "scripts/smoke_test_real_fiji_tool.py",
]

TEMPLATE_RUNTIME_CONFIG = "config/runtime_config.json"
TEMPLATE_TOOL_MANIFEST = "config/tool_manifest.json"
EXAMPLE_RUNTIME_CONFIG = "config/runtime_config.example.json"
TRANSFER_README = "TRANSFER_README.md"


@dataclass(frozen=True)
class PackagingStats:
    included_files: int
    included_bytes: int
    excluded_entries: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the project into an open-source-friendly ZIP without third-party dependencies."
    )
    parser.add_argument(
        "--output-dir",
        default="dist",
        help="Directory where the ZIP archive will be written. Default: dist/",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Archive filename prefix. Default: project folder name plus timestamp.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be packaged without writing a ZIP archive.",
    )
    parser.add_argument(
        "--extra-exclude",
        action="append",
        default=[],
        help="Additional exclude pattern. Can be passed multiple times.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print included and excluded paths while scanning.",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="Exclude validation, evaluation, and feasibility-test files to produce a lean runtime package.",
    )
    return parser.parse_args()


def to_posix(path: Path) -> str:
    return path.as_posix()


def path_matches_pattern(relative_path: PurePosixPath, pattern: str) -> bool:
    relative_str = relative_path.as_posix()
    trimmed = pattern.rstrip("/")
    if not trimmed:
        return False

    path_parts = relative_path.parts
    pattern_path = PurePosixPath(trimmed)
    pattern_parts = pattern_path.parts

    if pattern_parts and not any(char in trimmed for char in "*?[]"):
        if relative_str == trimmed:
            return True
        if any(part == trimmed for part in path_parts):
            return True
        if len(pattern_parts) <= len(path_parts):
            for idx in range(len(path_parts) - len(pattern_parts) + 1):
                if path_parts[idx : idx + len(pattern_parts)] == pattern_parts:
                    return True

    return fnmatch.fnmatch(relative_str, trimmed)


def should_exclude(relative_path: PurePosixPath, extra_patterns: Iterable[str]) -> bool:
    for pattern in DEFAULT_EXCLUDE_FILES:
        if path_matches_pattern(relative_path, pattern):
            return True

    for pattern in DEFAULT_EXCLUDE_DIRS:
        if path_matches_pattern(relative_path, pattern):
            return True

    for pattern in DEFAULT_EXCLUDE_GLOBS:
        if path_matches_pattern(relative_path, pattern):
            return True

    for pattern in extra_patterns:
        if path_matches_pattern(relative_path, pattern):
            return True

    return False


def build_runtime_excludes(output_dir: Path, extra_patterns: Iterable[str]) -> list[str]:
    runtime_patterns = list(extra_patterns)
    try:
        output_relative = output_dir.relative_to(ROOT_DIR)
    except ValueError:
        return runtime_patterns

    output_relative_str = output_relative.as_posix().strip()
    if output_relative_str and output_relative_str != ".":
        runtime_patterns.append(output_relative_str)
    return runtime_patterns


def build_effective_excludes(
    output_dir: Path,
    extra_patterns: Iterable[str],
    *,
    runtime_only: bool,
) -> list[str]:
    patterns = build_runtime_excludes(output_dir, extra_patterns)
    if runtime_only:
        patterns.extend(RUNTIME_ONLY_EXCLUDES)
    return patterns


def iter_project_files(extra_patterns: Iterable[str], verbose: bool) -> tuple[list[tuple[Path, PurePosixPath]], list[str]]:
    included: list[tuple[Path, PurePosixPath]] = []
    excluded: list[str] = []

    for dir_name in INCLUDE_DIRS:
        directory = ROOT_DIR / dir_name
        if not directory.exists():
            continue

        relative_directory = PurePosixPath(dir_name)
        if should_exclude(relative_directory, extra_patterns):
            excluded.append(f"{dir_name}/")
            if verbose:
                print(f"exclude dir : {dir_name}/")
            continue

        for source in directory.rglob("*"):
            relative_path = PurePosixPath(to_posix(source.relative_to(ROOT_DIR)))
            if source.is_dir():
                if should_exclude(relative_path, extra_patterns):
                    excluded.append(f"{relative_path.as_posix()}/")
                    if verbose:
                        print(f"exclude dir : {relative_path.as_posix()}/")
                continue

            if should_exclude(relative_path, extra_patterns):
                excluded.append(relative_path.as_posix())
                if verbose:
                    print(f"exclude file: {relative_path.as_posix()}")
                continue

            included.append((source, relative_path))
            if verbose:
                print(f"include file: {relative_path.as_posix()}")

    for file_name in INCLUDE_FILES:
        source = ROOT_DIR / file_name
        if not source.exists() or not source.is_file():
            continue

        relative_path = PurePosixPath(file_name)
        if should_exclude(relative_path, extra_patterns):
            excluded.append(relative_path.as_posix())
            if verbose:
                print(f"exclude file: {relative_path.as_posix()}")
            continue

        included.append((source, relative_path))
        if verbose:
            print(f"include file: {relative_path.as_posix()}")

    return deduplicate_included(included), excluded


def deduplicate_included(
    included: list[tuple[Path, PurePosixPath]]
) -> list[tuple[Path, PurePosixPath]]:
    deduped: dict[str, tuple[Path, PurePosixPath]] = {}
    for source, relative_path in included:
        deduped[relative_path.as_posix()] = (source, relative_path)
    return [deduped[key] for key in sorted(deduped)]


def load_template_runtime_config() -> str:
    example_path = ROOT_DIR / EXAMPLE_RUNTIME_CONFIG
    if not example_path.exists():
        raise FileNotFoundError(f"Missing template config: {example_path}")

    payload = json.loads(example_path.read_text(encoding="utf-8"))
    payload.setdefault("system", {})
    payload.setdefault("model", {})
    payload.setdefault("startup", {})

    system_cfg = payload["system"]
    system_cfg["CONFIG_PATH"] = "C:/Path/To/Your_Microscope_Config.cfg"
    system_cfg.setdefault("PSF_40X", "PSF/40x.tif")
    system_cfg.setdefault("PSF_60X", "PSF/60x.tif")
    system_cfg.setdefault("PSF_100X", "PSF/100x.tif")

    model_cfg = payload["model"]
    model_cfg["openai_api_key"] = ""
    model_cfg["vlm_api_key"] = ""

    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def load_template_tool_manifest() -> str:
    payload = default_tool_manifest_payload()
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def build_transfer_readme(package_name: str) -> str:
    lines = [
        f"# Open Source Package for {package_name}",
        "",
        "This archive contains the project source code and the minimum project files intended for open-source sharing.",
        "It does not include third-party Python packages, model weights, Micro-Manager, or Fiji.",
        "",
        "## What Is Included",
        "",
        "- Core project source code, prompts, frontend assets, docs, tests, and runtime startup scripts",
        "- User planning skills under `user_skills/`, model configuration files under `configs/`, and lightweight evaluation/demo resources",
        "- Packaged PSF resources under `PSF/` for supported deconvolution workflows",
        "- Tool onboarding scripts such as `create_tool.py` and the `tool_generation.py` compatibility wrapper",
        "- The hardware-free walkthrough notebook `Hardware-Free-Demo.ipynb`",
        "- Dependency manifests: `pyproject.toml`, `requirements.txt`, and `uv.lock`",
        "- A template `config/runtime_config.json` generated from `config/runtime_config.example.json`",
        "- A default `config/tool_manifest.json` with built-in system tools and no enabled user tools",
        "",
        "## What Was Intentionally Excluded",
        "",
        "- Virtual environments and package caches such as `.venv/` and `.uv-cache/`",
        "- Git metadata such as `.git/`",
        "- Runtime outputs and caches such as `te/`, `__pycache__/`, `.pytest_cache/`, and log files",
        "- Large local assets such as `weights/`, `model/`, and binary model artifacts (`*.pth`, `*.pt`, `*.onnx`, `*.ckpt`)",
        "- Sensitive local config files such as `.env`, the original `config/runtime_config.json`, and local `config/tool_manifest.json` overrides",
        "- Local microscope configuration folders such as `uploaded_cfg/`",
        "",
        "## How To Recreate Dependencies",
        "",
        "Recommended with `uv`:",
        "",
        "```bash",
        "uv venv --python 3.10",
        "uv sync",
        "```",
        "",
        "Fallback with `pip`:",
        "",
        "```bash",
        "python -m venv .venv",
        ".venv\\Scripts\\activate",
        "pip install -r requirements.txt",
        "```",
        "",
        "## Runtime Modes In This Package",
        "",
        "The packaged source supports both runtime chains:",
        "",
        "- Real hardware chain: `utils/runtime_factory.py` loads `core_tool.microscope`, `core_tool.fiji`, and `core_tool.cellpose_tool` when `Simulation_mode` is disabled.",
        "- Virtual hardware chain: `utils/runtime_factory.py` loads `Empty_function.py` when `Simulation_mode` is enabled.",
        "- Tool registration is driven by `config/tool_manifest.json`; only enabled manifest entries are loaded into the runtime.",
        "- The checker in `agent/experiment_checker.py` is mock-aware: deterministic placeholder files that begin with `mock microscope acquisition` are treated as mock outputs rather than parsed as real TIFF images.",
        "",
        "## Required Manual Configuration",
        "",
        "Before running the project on the target machine, update `config/runtime_config.json`:",
        "",
        "- `model.Simulation_mode`: set `true` for the virtual hardware chain, `false` for the real hardware chain",
        "- `system.MM_DIR`: local Micro-Manager install path",
        "- `system.CONFIG_PATH`: local microscope `.cfg` path",
        "- `system.FIJI_PATH`: local Fiji install path",
        "- `system.PSF_40X`, `system.PSF_60X`, and `system.PSF_100X`: packaged PSF files, if you move or replace them",
        "- `model.openai_api_key` and `model.vlm_api_key`: target-machine credentials",
        "- `model.base_url`, `model.model_name`, `model.vlm_base_url`, and `model.vlm_model_name` as needed",
        "- `config/tool_manifest.json` if you want to enable additional user tools or adjust manifest overrides beyond the default package template",
        "",
        "If you need real microscope control or image analysis, you must also install and configure Micro-Manager, Fiji, MMDetection-compatible dependencies, and any required model weights separately.",
        "If you only need the virtual hardware chain, you can keep `Simulation_mode` enabled and skip Micro-Manager / Fiji installation, subject to the Python packages needed by your workflow.",
        "The included `app_mock.py` entrypoint is a lightweight mock demo UI that shares the same runtime config but does not run the full planner/executor/checker stack.",
        "This package intentionally omits local experiment outputs, private environment files, and heavyweight local assets so it stays suitable for open-source sharing.",
        "",
    ]
    return "\n".join(lines)


def build_transfer_readme_runtime_only(package_name: str) -> str:
    lines = build_transfer_readme(package_name).splitlines()
    insert_at = lines.index("## What Was Intentionally Excluded")
    extra_lines = [
        "## Runtime-Only Packaging Notes",
        "",
        "This archive was created with `--runtime-only`.",
        "It intentionally excludes feasibility and validation assets such as `tests/`, `evaluation/`,",
        "the notebook demo, and dedicated smoke-test / exploratory scripts.",
        "",
    ]
    return "\n".join(lines[:insert_at] + extra_lines + lines[insert_at:])


def build_archive_name(name_override: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name_override.strip() if name_override else ROOT_DIR.name
    safe_prefix = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix)
    return f"{safe_prefix}_{timestamp}.zip"


def print_summary(
    archive_path: Path | None,
    stats: PackagingStats,
    excluded: list[str],
    verbose: bool,
    dry_run: bool,
    runtime_only: bool,
) -> None:
    mode_text = "Dry run completed" if dry_run else "Package created"
    print(mode_text)
    if archive_path is not None:
        print(f"Archive path : {archive_path}")
    print(f"Included files: {stats.included_files}")
    print(f"Included size : {stats.included_bytes / (1024 * 1024):.2f} MiB")
    print(f"Excluded items: {stats.excluded_entries}")
    print(f"Injected file : {TEMPLATE_RUNTIME_CONFIG}")
    print(f"Injected file : {TEMPLATE_TOOL_MANIFEST}")
    print(f"Injected file : {TRANSFER_README}")
    print(f"Runtime only : {runtime_only}")

    if not verbose and excluded:
        preview_count = min(15, len(excluded))
        print("Excluded preview:")
        for item in excluded[:preview_count]:
            print(f"  - {item}")
        if len(excluded) > preview_count:
            print(f"  - ... {len(excluded) - preview_count} more")


def package_project(args: argparse.Namespace) -> int:
    archive_name = build_archive_name(args.name)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    output_dir = output_dir.resolve()
    archive_path = output_dir / archive_name
    extra_patterns = [pattern.strip() for pattern in args.extra_exclude if pattern.strip()]
    effective_excludes = build_effective_excludes(
        output_dir,
        extra_patterns,
        runtime_only=bool(args.runtime_only),
    )
    included, excluded = iter_project_files(effective_excludes, args.verbose)
    template_runtime_config = load_template_runtime_config()
    template_tool_manifest = load_template_tool_manifest()
    transfer_readme = (
        build_transfer_readme_runtime_only(archive_name)
        if args.runtime_only
        else build_transfer_readme(archive_name)
    )

    stats = PackagingStats(
        included_files=len(included) + 3,
        included_bytes=(
            sum(source.stat().st_size for source, _ in included)
            + len(template_runtime_config.encode("utf-8"))
            + len(template_tool_manifest.encode("utf-8"))
            + len(transfer_readme.encode("utf-8"))
        ),
        excluded_entries=len(excluded),
    )

    if args.dry_run:
        print_summary(None, stats, excluded, args.verbose, dry_run=True, runtime_only=bool(args.runtime_only))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for source, relative_path in included:
            archive.write(source, arcname=relative_path.as_posix())
        archive.writestr(TEMPLATE_RUNTIME_CONFIG, template_runtime_config)
        archive.writestr(TEMPLATE_TOOL_MANIFEST, template_tool_manifest)
        archive.writestr(TRANSFER_README, transfer_readme)

    print_summary(archive_path, stats, excluded, args.verbose, dry_run=False, runtime_only=bool(args.runtime_only))
    return 0


def main() -> int:
    try:
        return package_project(parse_args())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


