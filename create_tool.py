from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Set

from config.agent_config import base_url, model_name, openai_api_key
from tool.base import BaseTool
from utils.tool_doc_paths import DEFAULT_USER_TOOL_DOCS_DIR, DEFAULT_USER_TOOL_DOCS_RELATIVE
from utils.tool_manifest import (
    DEFAULT_TOOL_MANIFEST_PATH,
    ToolManifestError,
    default_tool_manifest_payload,
    discover_tool_candidates,
    import_string,
    load_tool_manifest,
    load_tool_manifest_from_payload,
    read_tool_manifest_payload,
    write_tool_manifest_payload,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DOC_OUTPUT_DIR = str(DEFAULT_USER_TOOL_DOCS_RELATIVE)
DOC_ARTIFACT_SUFFIXES = ("planner_summary.txt", "executor_prompt.txt")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="User tool CLI helper")
    parser.add_argument("--manifest", default=str(DEFAULT_TOOL_MANIFEST_PATH), help="Tool manifest path")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOC_OUTPUT_DIR,
        help="Directory for generated tool prompt artifacts",
    )
    parser.set_defaults(
        command="register",
        class_path="",
        class_name="",
        file_path="",
        tool_id="",
        planning_hint="",
        execution_hint="",
        disable=None,
        create_template=False,
        dry_run=False,
        target_tool_id="",
        new_tool_id=None,
        enable_state=None,
        force=False,
        selected_tool_ids=[],
    )

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--manifest", default=str(DEFAULT_TOOL_MANIFEST_PATH), help="Tool manifest path")

    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument(
        "--output-dir",
        default=DEFAULT_DOC_OUTPUT_DIR,
        help="Directory for generated tool prompt artifacts",
    )

    subparsers = parser.add_subparsers(dest="command")

    register_parser = subparsers.add_parser(
        "register",
        parents=[common_parent, output_parent],
        help="Register a user tool via CLI wizard or explicit arguments",
    )
    register_parser.add_argument("--class-path", default="", help="Existing class path using module:Class format")
    register_parser.add_argument("--class-name", default="", help="Class name when creating a new tool template")
    register_parser.add_argument("--file-path", default="", help="Target file path when creating a new tool template")
    register_parser.add_argument("--tool-id", default="", help="Planner-visible user tool id")
    register_parser.add_argument("--planning-hint", default="", help="Optional planning hint stored in the manifest")
    register_parser.add_argument("--execution-hint", default="", help="Optional execution hint stored in the manifest")
    register_parser.add_argument("--disable", action="store_true", default=None, help="Write the manifest entry as disabled")
    register_parser.add_argument(
        "--create-template",
        action="store_true",
        help="Generate a new tool template at --file-path before registration",
    )
    register_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated manifest entry without writing files",
    )

    subparsers.add_parser(
        "list",
        parents=[common_parent],
        help="List user tools currently configured in the manifest",
    )

    update_parser = subparsers.add_parser(
        "update",
        parents=[common_parent],
        help="Update an existing user tool manifest entry",
    )
    update_parser.add_argument("target_tool_id", help="Existing tool_id to update")
    update_parser.add_argument("--new-tool-id", default=None, help="Replace the tool_id with a new value")
    update_parser.add_argument("--planning-hint", default=None, help="Replace the planning hint")
    update_parser.add_argument("--execution-hint", default=None, help="Replace the execution hint")
    toggle_group = update_parser.add_mutually_exclusive_group()
    toggle_group.add_argument("--enable", dest="enable_state", action="store_true", help="Enable the tool")
    toggle_group.add_argument("--disable", dest="enable_state", action="store_false", help="Disable the tool")
    update_parser.set_defaults(enable_state=None)

    enable_parser = subparsers.add_parser(
        "enable",
        parents=[common_parent],
        help="Enable a user tool in the manifest",
    )
    enable_parser.add_argument("target_tool_id", help="Tool id to enable")

    disable_parser = subparsers.add_parser(
        "disable",
        parents=[common_parent],
        help="Disable a user tool in the manifest",
    )
    disable_parser.add_argument("target_tool_id", help="Tool id to disable")

    remove_parser = subparsers.add_parser(
        "remove",
        parents=[common_parent],
        help="Remove a user tool from the manifest",
    )
    remove_parser.add_argument("target_tool_id", help="Tool id to remove")
    remove_parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt")

    uninstall_parser = subparsers.add_parser(
        "uninstall",
        parents=[common_parent],
        help="Uninstall a user tool from the manifest and clean its prompt artifacts",
    )
    uninstall_parser.add_argument("target_tool_id", help="Tool id to uninstall")
    uninstall_parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt")

    generate_docs_parser = subparsers.add_parser(
        "generate-docs",
        parents=[common_parent, output_parent],
        help="Generate prompt artifacts for enabled or explicitly selected user tools",
    )
    generate_docs_parser.add_argument(
        "--tool-id",
        dest="selected_tool_ids",
        action="append",
        default=[],
        help="Specific tool_id to generate docs for. Can be passed multiple times.",
    )

    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        args.command = "register"
    return args


def _slugify(value: str) -> str:
    lowered = value.strip().replace("-", "_").replace(" ", "_")
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in lowered)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_").lower()


def _humanize_identifier(value: str) -> str:
    text = value.split(":")[-1].strip()
    if not text:
        return ""
    chars = []
    for index, char in enumerate(text):
        if index > 0 and char.isupper() and text[index - 1].islower():
            chars.append(" ")
        chars.append(char)
    return " ".join("".join(chars).replace("_", " ").replace("-", " ").split())


def _normalize_manifest_path(manifest: str) -> Path:
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT_DIR / manifest_path
    return manifest_path


def _normalize_output_dir(output_dir: str) -> Path:
    target_dir = Path(output_dir)
    if not target_dir.is_absolute():
        target_dir = ROOT_DIR / target_dir
    return target_dir


def _derive_class_path(file_path: Path, class_name: str) -> str:
    relative = file_path.resolve().relative_to(ROOT_DIR.resolve())
    module_name = ".".join(relative.with_suffix("").parts)
    return f"{module_name}:{class_name}"


def _build_tool_template(class_name: str) -> str:
    return f'''from tool.base import BaseTool, tool_func


class {class_name}(BaseTool):
    """New extension tool generated by the CLI wizard."""

    planning_hint = ""
    execution_hint = ""

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir

    @tool_func
    def run(self) -> bool:
        """Placeholder tool entrypoint."""
        return True
'''


def _prompt_text(
    prompt: str,
    *,
    default: str = "",
    allow_empty: bool = False,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        response = input_func(f"{prompt}{suffix}: ").strip()
        if response:
            return response
        if default:
            return default
        if allow_empty:
            return ""
        print_func("[WARN] This value is required.")


def _prompt_yes_no(
    prompt: str,
    *,
    default: bool = True,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        response = input_func(f"{prompt} [{default_label}]: ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print_func("[WARN] Please answer yes or no.")


def _load_mutable_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    payload = read_tool_manifest_payload(manifest_path)
    system_tools = payload.setdefault("system_tools", default_tool_manifest_payload()["system_tools"])
    if not isinstance(system_tools, dict):
        raise ToolManifestError("Tool manifest must contain a top-level 'system_tools' object")
    user_tools = payload.setdefault("user_tools", [])
    if not isinstance(user_tools, list):
        raise ToolManifestError("Tool manifest must contain a top-level 'user_tools' list")
    return payload


def _manifest_user_tools(payload: dict[str, Any]) -> list[dict[str, Any]]:
    user_tools = payload.setdefault("user_tools", [])
    if not isinstance(user_tools, list):
        raise ToolManifestError("Tool manifest must contain a top-level 'user_tools' list")
    return user_tools


def _find_user_tool_index(payload: dict[str, Any], tool_id: str) -> int:
    normalized = tool_id.strip()
    if not normalized:
        raise ToolManifestError("Tool id is required")
    for index, entry in enumerate(_manifest_user_tools(payload)):
        if str(entry.get("tool_id", "")).strip() == normalized:
            return index
    raise ToolManifestError(f"User tool '{normalized}' was not found in the manifest")


def _write_validated_manifest_payload(payload: dict[str, Any], manifest_path: Path) -> None:
    load_tool_manifest_from_payload(payload)
    write_tool_manifest_payload(payload, manifest_path)


def _resolve_or_create_class_path(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> str:
    class_path = str(getattr(args, "class_path", "") or "").strip()
    create_template = bool(getattr(args, "create_template", False))
    class_name = str(getattr(args, "class_name", "") or "").strip()
    file_path_value = str(getattr(args, "file_path", "") or "").strip()
    if class_path:
        return class_path

    candidates = discover_tool_candidates()
    needs_interaction = not create_template and not class_path

    if needs_interaction:
        print_func("=" * 50)
        print_func("[WIZARD] Interactive user-tool registration")
        print_func("=" * 50)
        print_func(f"Manifest: {_normalize_manifest_path(args.manifest)}")
        if candidates:
            print_func("Discovered tool candidates under tool/:")
            for idx, candidate in enumerate(candidates, start=1):
                print_func(f"  {idx}. {candidate.class_path}")
        else:
            print_func("No discovered user-tool candidates were found under tool/.")

        create_template = _prompt_yes_no(
            "Create a new tool template instead of registering an existing tool?",
            default=False,
            input_func=input_func,
            print_func=print_func,
        )
        args.create_template = create_template

    if create_template:
        args.file_path = _prompt_text(
            "Tool file path",
            default=file_path_value or "tool/my_tool.py",
            input_func=input_func,
            print_func=print_func,
        )
        args.class_name = _prompt_text(
            "Tool class name",
            default=class_name or "MyTool",
            input_func=input_func,
            print_func=print_func,
        )
        file_path = Path(str(args.file_path).strip())
        if not file_path.is_absolute():
            file_path = ROOT_DIR / file_path
        if not bool(getattr(args, "dry_run", False)):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if file_path.exists():
                raise ToolManifestError(f"Template target already exists: {file_path}")
            file_path.write_text(_build_tool_template(str(args.class_name).strip()), encoding="utf-8")
        return _derive_class_path(file_path, str(args.class_name).strip())

    if candidates:
        choice = _prompt_text(
            "Select a candidate number or enter a class path manually",
            input_func=input_func,
            print_func=print_func,
        )
        if choice.isdigit():
            index = int(choice)
            if not (1 <= index <= len(candidates)):
                raise ToolManifestError(f"Candidate index out of range: {choice}")
            return candidates[index - 1].class_path
        return choice

    return _prompt_text(
        "Existing class path (module:Class)",
        input_func=input_func,
        print_func=print_func,
    )


def _validate_tool_class(class_path: str) -> type[BaseTool]:
    tool_obj = import_string(class_path)
    if not isinstance(tool_obj, type) or not issubclass(tool_obj, BaseTool):
        raise ToolManifestError(f"{class_path} must resolve to a BaseTool subclass")
    if not tool_obj.get_public_methods():
        raise ToolManifestError(f"{class_path} must expose at least one @tool_func method")
    return tool_obj


def _build_manifest_entry_payload(args: argparse.Namespace, class_path: str, tool_cls: type[BaseTool]) -> dict[str, Any]:
    class_name = str(getattr(args, "class_name", "") or "").strip() or class_path.split(":", 1)[1]
    tool_id = str(getattr(args, "tool_id", "") or "").strip() or tool_cls.get_default_tool_id() or _slugify(class_name)
    return {
        "tool_id": tool_id,
        "class_path": class_path,
        "enabled": not bool(getattr(args, "disable", False)),
        "planning_hint": str(getattr(args, "planning_hint", "") or "").strip(),
        "execution_hint": str(getattr(args, "execution_hint", "") or "").strip(),
    }


def _maybe_prompt_register_metadata(
    args: argparse.Namespace,
    class_path: str,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> None:
    class_name = str(getattr(args, "class_name", "") or "").strip() or class_path.split(":")[-1]
    suggested_tool_id = _slugify(class_name)
    prompted_any = False
    if not str(getattr(args, "tool_id", "") or "").strip():
        prompted_any = True
        args.tool_id = _prompt_text(
            "Tool ID (used directly in planner task steps)",
            default=_humanize_identifier(class_name) or suggested_tool_id,
            input_func=input_func,
            print_func=print_func,
        )
    if not str(getattr(args, "planning_hint", "") or "").strip():
        prompted_any = True
        args.planning_hint = _prompt_text(
            "Planning hint",
            allow_empty=True,
            input_func=input_func,
            print_func=print_func,
        )
    if not str(getattr(args, "execution_hint", "") or "").strip():
        prompted_any = True
        args.execution_hint = _prompt_text(
            "Execution hint",
            allow_empty=True,
            input_func=input_func,
            print_func=print_func,
        )
    if getattr(args, "disable", None) is None and prompted_any:
        enabled = _prompt_yes_no(
            "Enable this user tool?",
            default=True,
            input_func=input_func,
            print_func=print_func,
        )
        args.disable = not enabled


def _print_manifest_entry(entry: dict[str, Any], print_func: Callable[..., None] = print) -> None:
    print_func(json.dumps(entry, indent=2, ensure_ascii=False))


def _user_tool_doc_paths(tool_id: str, output_dir: str) -> list[Path]:
    target_dir = _normalize_output_dir(output_dir)
    return [target_dir / f"{tool_id}.{suffix}" for suffix in DOC_ARTIFACT_SUFFIXES]


def _remove_user_tool_doc_artifacts(tool_id: str, output_dir: str, *, print_func: Callable[..., None] = print) -> list[Path]:
    removed_paths: list[Path] = []
    for artifact_path in _user_tool_doc_paths(tool_id, output_dir):
        try:
            if artifact_path.exists():
                artifact_path.unlink()
                removed_paths.append(artifact_path)
        except OSError as exc:
            raise ToolManifestError(f"Failed to delete tool prompt artifact '{artifact_path}': {exc}") from exc
    if removed_paths:
        for path in removed_paths:
            print_func(f"[INFO] Removed prompt artifact: {path}")
    return removed_paths


def run_register_command(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> int:
    try:
        class_path = _resolve_or_create_class_path(args, input_func=input_func, print_func=print_func)
        _maybe_prompt_register_metadata(args, class_path, input_func=input_func, print_func=print_func)
        tool_cls = _validate_tool_class(class_path)
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    entry_payload = _build_manifest_entry_payload(args, class_path, tool_cls)

    print_func("=" * 50)
    print_func("[REGISTER] Generated user tool manifest entry")
    print_func("=" * 50)
    _print_manifest_entry(entry_payload, print_func=print_func)

    if bool(getattr(args, "dry_run", False)):
        print_func("[INFO] Dry run enabled; manifest was not modified.")
        return 0

    manifest_path = _normalize_manifest_path(args.manifest)
    try:
        payload = _load_mutable_manifest_payload(manifest_path)
        _manifest_user_tools(payload).append(entry_payload)
        _write_validated_manifest_payload(payload, manifest_path)
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    print_func(f"[SUCCESS] Added user tool entry to manifest: {manifest_path}")
    print_func("[INFO] Generating prompt artifacts now...")
    docs_args = argparse.Namespace(
        manifest=args.manifest,
        output_dir=getattr(args, "output_dir", DEFAULT_DOC_OUTPUT_DIR),
        selected_tool_ids=[entry_payload["tool_id"]],
    )
    return run_generate_docs_command(docs_args, print_func=print_func)


def run_list_command(args: argparse.Namespace, *, print_func: Callable[..., None] = print) -> int:
    manifest_path = _normalize_manifest_path(args.manifest)
    manifest = load_tool_manifest(manifest_path)
    print_func("=" * 50)
    print_func("[LIST] User tools")
    print_func("=" * 50)
    print_func(f"Manifest: {manifest_path}")
    if not manifest.user_tools:
        print_func("[INFO] No user tools are configured.")
        return 0

    for entry in manifest.user_tools:
        status = "enabled" if entry.enabled else "disabled"
        print_func(f"- {entry.tool_id} | {status} | {entry.class_path}")
    return 0


def run_update_command(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> int:
    manifest_path = _normalize_manifest_path(args.manifest)
    try:
        payload = _load_mutable_manifest_payload(manifest_path)
        index = _find_user_tool_index(payload, args.target_tool_id)
        entry = dict(_manifest_user_tools(payload)[index])
        old_tool_id = str(entry.get("tool_id", "")).strip()
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    if args.new_tool_id is None and args.planning_hint is None and args.execution_hint is None and args.enable_state is None:
        args.new_tool_id = _prompt_text(
            "Tool ID",
            default=str(entry.get("tool_id", "")).strip(),
            input_func=input_func,
            print_func=print_func,
        )
        args.planning_hint = _prompt_text(
            "Planning hint",
            default=str(entry.get("planning_hint", "") or ""),
            allow_empty=True,
            input_func=input_func,
            print_func=print_func,
        )
        args.execution_hint = _prompt_text(
            "Execution hint",
            default=str(entry.get("execution_hint", "") or ""),
            allow_empty=True,
            input_func=input_func,
            print_func=print_func,
        )
        args.enable_state = _prompt_yes_no(
            "Enable this user tool?",
            default=bool(entry.get("enabled", True)),
            input_func=input_func,
            print_func=print_func,
        )

    if args.new_tool_id is not None:
        entry["tool_id"] = str(args.new_tool_id).strip()
    if args.planning_hint is not None:
        entry["planning_hint"] = str(args.planning_hint).strip()
    if args.execution_hint is not None:
        entry["execution_hint"] = str(args.execution_hint).strip()
    if args.enable_state is not None:
        entry["enabled"] = bool(args.enable_state)

    try:
        _manifest_user_tools(payload)[index] = entry
        _write_validated_manifest_payload(payload, manifest_path)
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    new_tool_id = str(entry.get("tool_id", "")).strip()
    output_dir = getattr(args, "output_dir", DEFAULT_DOC_OUTPUT_DIR)
    if old_tool_id and new_tool_id and old_tool_id != new_tool_id:
        try:
            _remove_user_tool_doc_artifacts(old_tool_id, output_dir, print_func=print_func)
        except ToolManifestError as exc:
            print_func(f"[ERROR] {exc}")
            return 1

    print_func(f"[SUCCESS] Updated user tool '{args.target_tool_id}' in manifest: {manifest_path}")
    docs_args = argparse.Namespace(
        manifest=args.manifest,
        output_dir=output_dir,
        selected_tool_ids=[new_tool_id] if new_tool_id else [],
    )
    return run_generate_docs_command(docs_args, print_func=print_func)


def _set_enabled_state(args: argparse.Namespace, enabled: bool, *, print_func: Callable[..., None] = print) -> int:
    manifest_path = _normalize_manifest_path(args.manifest)
    try:
        payload = _load_mutable_manifest_payload(manifest_path)
        index = _find_user_tool_index(payload, args.target_tool_id)
        entry = dict(_manifest_user_tools(payload)[index])
        entry["enabled"] = enabled
        _manifest_user_tools(payload)[index] = entry
        _write_validated_manifest_payload(payload, manifest_path)
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    state_label = "enabled" if enabled else "disabled"
    print_func(f"[SUCCESS] {state_label.capitalize()} user tool '{args.target_tool_id}' in manifest: {manifest_path}")
    return 0


def run_remove_command(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> int:
    manifest_path = _normalize_manifest_path(args.manifest)
    try:
        payload = _load_mutable_manifest_payload(manifest_path)
        index = _find_user_tool_index(payload, args.target_tool_id)
        entry = dict(_manifest_user_tools(payload)[index])
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    if not bool(getattr(args, "force", False)):
        confirmed = _prompt_yes_no(
            f"Remove user tool '{args.target_tool_id}' from the manifest?",
            default=False,
            input_func=input_func,
            print_func=print_func,
        )
        if not confirmed:
            print_func("[INFO] Removal cancelled.")
            return 0

    removed_tool_id = str(entry.get("tool_id", "")).strip()
    del _manifest_user_tools(payload)[index]
    try:
        _write_validated_manifest_payload(payload, manifest_path)
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    try:
        _remove_user_tool_doc_artifacts(
            removed_tool_id,
            getattr(args, "output_dir", DEFAULT_DOC_OUTPUT_DIR),
            print_func=print_func,
        )
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    print_func(f"[SUCCESS] Removed user tool '{args.target_tool_id}' from manifest: {manifest_path}")
    return 0


def _selected_class_paths_from_manifest(manifest_path: Path, selected_tool_ids: Iterable[str]) -> Set[str]:
    manifest = load_tool_manifest(manifest_path)
    selected = {tool_id.strip() for tool_id in selected_tool_ids if str(tool_id).strip()}
    if selected:
        matches = {
            entry.class_path
            for entry in manifest.user_tools
            if entry.tool_id in selected
        }
        missing = sorted(selected - {entry.tool_id for entry in manifest.user_tools})
        if missing:
            raise ToolManifestError(f"Unknown tool_id for doc generation: {', '.join(missing)}")
        return matches
    return {
        entry.class_path
        for entry in manifest.user_tools
        if entry.enabled
    }


def run_generate_docs_command(args: argparse.Namespace, *, print_func: Callable[..., None] = print) -> int:
    manifest_path = _normalize_manifest_path(args.manifest)
    try:
        allowed_class_paths = _selected_class_paths_from_manifest(
            manifest_path,
            getattr(args, "selected_tool_ids", []) or [],
        )
    except ToolManifestError as exc:
        print_func(f"[ERROR] {exc}")
        return 1

    if not allowed_class_paths:
        print_func("[INFO] No user tools were selected for prompt artifact generation.")
        return 0

    from utils.tool_generation import ToolProcessingPipeline

    if not openai_api_key or openai_api_key == "your-openai-api-key":
        print_func(
            "[ERROR] OPENAI_API_KEY is not configured. Prompt artifact generation now requires a real LLM and will not use fallback text."
        )
        return 1
    pipeline = ToolProcessingPipeline(openai_api_key, base_url, model_name)
    artifacts = pipeline.run_pipeline(allowed_class_paths=allowed_class_paths, output_dir=args.output_dir)
    print_func("=" * 50)
    print_func("[GENERATED] Tool prompt artifacts")
    print_func("=" * 50)
    for artifact in artifacts:
        print_func(f"- {artifact.tool_id}: {artifact.execution_prompt_path} | {artifact.planning_summary_path}")
    return 0


def run_cli(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    print_func: Callable[..., None] = print,
) -> int:
    command = getattr(args, "command", "register") or "register"
    if command == "register":
        return run_register_command(args, input_func=input_func, print_func=print_func)
    if command == "list":
        return run_list_command(args, print_func=print_func)
    if command == "update":
        return run_update_command(args, input_func=input_func, print_func=print_func)
    if command == "enable":
        return _set_enabled_state(args, True, print_func=print_func)
    if command == "disable":
        return _set_enabled_state(args, False, print_func=print_func)
    if command in {"remove", "uninstall"}:
        return run_remove_command(args, input_func=input_func, print_func=print_func)
    if command == "generate-docs":
        return run_generate_docs_command(args, print_func=print_func)
    print_func(f"[ERROR] Unsupported command: {command}")
    return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 50)
    print("[START] User tool CLI")
    print("=" * 50)

    try:
        return run_cli(args)
    except ImportError:
        print("[ERROR] Could not import required tool configuration modules")
        return 1
    except EnvironmentError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:
        print(f"[FATAL ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
