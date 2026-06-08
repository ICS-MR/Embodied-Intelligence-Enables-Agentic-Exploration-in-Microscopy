from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.utils import SAFE_BUILTIN_CALLS, exec_safe
from utils.memory_manager import StorageManager


_SAY_MESSAGES: list[str] = []


def _say(message: str) -> None:
    msg = str(message)
    _SAY_MESSAGES.append(msg)
    print(f"robot says: {msg}")


def _path_arg(value: str) -> Path:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return Path(cleaned).expanduser().resolve()


def _load_code_from_history(
    history_path: Path,
    *,
    agent_name: str,
    event_type: str,
    match_index: int,
) -> tuple[str, dict[str, Any]]:
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"History file must contain a JSON list: {history_path}")

    matches = [
        entry
        for entry in payload
        if entry.get("agent_name") == agent_name and entry.get("event_type") == event_type
    ]
    if not matches:
        raise ValueError(
            f"No matching event found in {history_path} "
            f"for agent={agent_name!r}, event_type={event_type!r}"
        )

    entry = matches[match_index]
    event_payload = entry.get("payload") or {}
    code = event_payload.get("generated_code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Matched event does not contain a non-empty payload.generated_code")
    return code, entry


def _load_code(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.code_file:
        code_path = _path_arg(args.code_file)
        return code_path.read_text(encoding="utf-8"), {"source": str(code_path)}

    if not args.history:
        raise ValueError("Provide either --history or --code-file")

    history_path = _path_arg(args.history)
    return _load_code_from_history(
        history_path,
        agent_name=args.agent_name,
        event_type=args.event_type,
        match_index=args.match_index,
    )


def _build_imagej_env(*, real: bool, output_dir: Path, session_dir: Path):
    storage_manager = StorageManager(str(session_dir), str(output_dir))
    if real:
        from core_tool.fiji import ImageJProcessor
    else:
        from Empty_function import ImageJProcessor

    return ImageJProcessor(storage_manager, str(output_dir))


def _build_var_map(env: Any) -> dict[str, Any]:
    methods = env.get_public_methods()
    return {name: getattr(env, name) for name in methods if hasattr(env, name)}


def _run_code(code: str, env: Any) -> None:
    gvars = _build_var_map(env)
    gvars["say"] = _say

    allowed_call_names = {name for name, value in gvars.items() if callable(value)} | SAFE_BUILTIN_CALLS
    allowed_attribute_roots = set(gvars.keys())
    exec_safe(
        code,
        gvars,
        {},
        allowed_call_names=allowed_call_names,
        allowed_attribute_roots=allowed_attribute_roots,
    )


def _storage_snapshot(env: Any) -> dict[str, Any]:
    storage_manager = getattr(env, "_storagemanger", None)
    if storage_manager is None:
        return {"storage": {}, "cache": {}}

    try:
        storage = storage_manager.read_log(False)
    except Exception:
        storage = {}

    try:
        cache = storage_manager.read_cache()
    except Exception:
        cache = {}

    return {"storage": storage, "cache": cache}


def _write_result_json(
    result_path: Path | None,
    *,
    exit_code: int,
    status: str,
    env: Any = None,
    error: str = "",
) -> None:
    if result_path is None:
        return

    payload: dict[str, Any] = {
        "exit_code": exit_code,
        "status": status,
        "error": error,
        "say_messages": list(_SAY_MESSAGES),
        "metadata": _storage_snapshot(env) if env is not None else {"storage": {}, "cache": {}},
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _default_base_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return _path_arg(args.output_dir)
    if args.history:
        return _path_arg(args.history).parent
    return Path.cwd().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run generated Image Analysis Platform code directly, without starting "
            "the full application or calling the LLM."
        )
    )
    parser.add_argument("--history", help="Path to agent_interactions.json")
    parser.add_argument("--code-file", help="Path to a Python file containing generated code")
    parser.add_argument("--agent-name", default="Image Analysis Platform")
    parser.add_argument("--event-type", default="executor_execution_failed")
    parser.add_argument(
        "--match-index",
        type=int,
        default=-1,
        help="Index within matching history events. Defaults to the latest matching event.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use core_tool.fiji.ImageJProcessor. Without this flag the mock ImageJProcessor is used.",
    )
    parser.add_argument("--output-dir", help="Directory containing input images and receiving outputs")
    parser.add_argument("--session-dir", help="Directory for metadata files. Defaults to output-dir")
    parser.add_argument(
        "--workdir",
        help="Working directory for generated code. Defaults to output-dir so relative out_dir='.' is local.",
    )
    parser.add_argument(
        "--print-code",
        action="store_true",
        help="Print the selected generated code before execution.",
    )
    parser.add_argument(
        "--result-json",
        help="Optional path where execution status, say() messages, and metadata snapshot are written.",
    )
    parser.add_argument(
        "--force-exit",
        action="store_true",
        help="Call os._exit(exit_code) after cleanup. Useful for real Fiji/JVM child processes.",
    )
    args = parser.parse_args()

    result_json = _path_arg(args.result_json) if args.result_json else None
    env = None
    exit_code = 1
    status = "failed"
    error_text = ""

    try:
        code, entry = _load_code(args)
        output_dir = _default_base_dir(args)
        session_dir = _path_arg(args.session_dir) if args.session_dir else output_dir
        workdir = _path_arg(args.workdir) if args.workdir else output_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)

        print(f"[runner] backend={'real Fiji' if args.real else 'mock'}")
        print(f"[runner] output_dir={output_dir}")
        print(f"[runner] session_dir={session_dir}")
        print(f"[runner] workdir={workdir}")
        if entry.get("timestamp"):
            print(f"[runner] selected_event_timestamp={entry['timestamp']}")
        if args.print_code:
            print("[runner] selected code:")
            print(code)

        env = _build_imagej_env(real=args.real, output_dir=output_dir, session_dir=session_dir)
        previous_cwd = Path.cwd()
        os.chdir(workdir)
        _run_code(code, env)
        print("[runner] generated code completed successfully")
        exit_code = 0
        status = "succeeded"
    except Exception:
        print("[runner] generated code failed")
        traceback.print_exc()
        error_text = traceback.format_exc()
        exit_code = 1
        status = "failed"
    finally:
        if "previous_cwd" in locals():
            os.chdir(previous_cwd)
        if env is not None and getattr(env, "ij", None) is not None and hasattr(env, "fiji_shutdown"):
            try:
                env.fiji_shutdown()
            except Exception:
                print("[runner] cleanup fiji_shutdown failed")
                traceback.print_exc()
                if not error_text:
                    error_text = traceback.format_exc()
        _write_result_json(result_json, exit_code=exit_code, status=status, env=env, error=error_text)
        sys.stdout.flush()
        sys.stderr.flush()
        if args.force_exit:
            os._exit(exit_code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
