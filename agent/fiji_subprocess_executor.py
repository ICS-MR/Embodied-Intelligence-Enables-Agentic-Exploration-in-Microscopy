from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
ROBOT_SAYS_PREFIX = "robot says:"


@dataclass
class FijiSubprocessResult:
    command: list[str]
    code_path: Path
    result_path: Path
    log_path: Path
    returncode: int
    duration_seconds: float
    stdout: str = ""
    say_messages: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 1

    def payload(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "code_path": str(self.code_path),
            "result_path": str(self.result_path),
            "log_path": str(self.log_path),
            "returncode": self.returncode,
            "duration_seconds": self.duration_seconds,
            "stdout_tail": self.stdout[-4000:],
            "say_messages": list(self.say_messages),
            "attempts": self.attempts,
        }


class FijiSubprocessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.payload = payload or {}


class FijiSubprocessTimeout(FijiSubprocessError):
    pass


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            return
        except Exception:
            pass

    try:
        proc.kill()
    except Exception:
        pass


def _read_result_json(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]


def _is_probably_transient_startup_failure(
    *,
    returncode: int,
    duration_seconds: float,
    stdout_tail: str,
    say_messages: list[str],
    child_payload: dict[str, Any],
) -> bool:
    child_status = str(child_payload.get("status") or "").strip().lower()
    child_error = str(child_payload.get("error") or "").strip()
    has_child_result = bool(child_payload)
    has_output = bool(str(stdout_tail or "").strip())
    has_say_messages = bool(say_messages)

    # On Windows, 0xC000013A commonly indicates an interrupted child process.
    interrupted_exit_codes = {3221225786, -1073741510}
    if returncode not in interrupted_exit_codes:
        return False

    # We only want to retry cases that died before user code meaningfully started.
    if duration_seconds > 2.0:
        return False
    if has_output or has_say_messages:
        return False
    if has_child_result and (child_status or child_error):
        return False

    return True


def _tail_from_lines(lines: deque[str], max_chars: int = 4000) -> str:
    return "".join(lines)[-max_chars:]


def _merge_child_metadata(storage_manager: Any, metadata: dict[str, Any]) -> None:
    if storage_manager is None or not metadata:
        return
    if not hasattr(storage_manager, "merge_external_metadata"):
        return
    storage_manager.merge_external_metadata(
        storage=metadata.get("storage") or {},
        cache=metadata.get("cache") or {},
    )


def _cleanup_attempt_artifacts(paths: list[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def _build_temp_executor_dir(session_path: Path) -> Path:
    temp_root = Path(tempfile.gettempdir()).resolve() / "EIMS" / "fiji_executor"
    session_label = session_path.name or "default_session"
    temp_dir = temp_root / session_label
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _replay_say_messages(say_capture: Any, messages: list[str], *, skip: int = 0) -> None:
    if say_capture is None or not hasattr(say_capture, "say"):
        return
    for message in messages[max(skip, 0):]:
        say_capture.say(message)


def _replay_artifacts(artifact_emitter: Any, artifacts: list[dict[str, Any]]) -> None:
    if artifact_emitter is None or not callable(artifact_emitter):
        return
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        try:
            artifact_emitter(dict(artifact))
        except Exception:
            pass


def _start_output_forwarder(
    *,
    proc: subprocess.Popen[str],
    log_handle: Any,
    say_capture: Any,
    live_say_messages: list[str],
    tail_lines: deque[str],
) -> threading.Thread:
    def forward_output() -> None:
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            log_handle.write(line)
            log_handle.flush()
            tail_lines.append(line)
            stripped = line.rstrip("\r\n")
            if not stripped.startswith(ROBOT_SAYS_PREFIX):
                continue
            message = stripped[len(ROBOT_SAYS_PREFIX):].strip()
            live_say_messages.append(message)
            _replay_say_messages(say_capture, [message])

    thread = threading.Thread(target=forward_output, name="fiji-output-forwarder", daemon=True)
    thread.start()
    return thread


def run_generated_fiji_code_in_subprocess(
    code_str: str,
    *,
    session_dir: str | Path,
    output_dir: str | Path,
    timeout_seconds: float,
    storage_manager: Any = None,
    say_capture: Any = None,
    artifact_emitter: Any = None,
    workdir: str | Path | None = None,
    max_startup_retries: int = 2,
    startup_retry_backoff_seconds: float = 2.0,
) -> FijiSubprocessResult:
    session_path = Path(session_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    workdir_path = Path(workdir).expanduser().resolve() if workdir else output_path
    session_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    workdir_path.mkdir(parents=True, exist_ok=True)
    temp_executor_path = _build_temp_executor_dir(session_path)

    attempts = max(int(max_startup_retries or 0), 0) + 1
    last_error: FijiSubprocessError | None = None
    failed_attempt_artifacts: list[Path] = []

    for attempt in range(1, attempts + 1):
        run_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        code_path = temp_executor_path / f"fiji_executor_{run_id}.py"
        result_path = temp_executor_path / f"fiji_executor_{run_id}_result.json"
        log_path = temp_executor_path / f"fiji_executor_{run_id}.log"
        code_path.write_text(code_str, encoding="utf-8")

        command = [
            sys.executable,
            "-u",
            str(ROOT_DIR / "scripts" / "run_generated_imagej_code.py"),
            "--real",
            "--code-file",
            str(code_path),
            "--output-dir",
            str(output_path),
            "--session-dir",
            str(session_path),
            "--workdir",
            str(workdir_path),
            "--result-json",
            str(result_path),
            "--force-exit",
        ]

        started = time.monotonic()
        log_handle = log_path.open("w", encoding="utf-8", errors="replace")
        live_say_messages: list[str] = []
        tail_lines: deque[str] = deque(maxlen=400)
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output_thread = _start_output_forwarder(
            proc=proc,
            log_handle=log_handle,
            say_capture=say_capture,
            live_say_messages=live_say_messages,
            tail_lines=tail_lines,
        )

        try:
            try:
                proc.wait(timeout=max(float(timeout_seconds), 1.0))
                output_thread.join(timeout=5)
            except subprocess.TimeoutExpired as exc:
                _kill_process_tree(proc)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                output_thread.join(timeout=5)
                log_handle.close()
                duration = time.monotonic() - started
                stdout = _tail_from_lines(tail_lines) or _read_text_tail(log_path)
                payload = {
                    "reason": "fiji_executor_timeout",
                    "timeout_seconds": timeout_seconds,
                    "duration_seconds": duration,
                    "pid": proc.pid,
                    "command": command,
                    "code_path": str(code_path),
                    "result_path": str(result_path),
                    "log_path": str(log_path),
                    "stdout_tail": str(stdout or "")[-4000:],
                    "say_messages": list(live_say_messages),
                    "attempt": attempt,
                    "max_attempts": attempts,
                }
                raise FijiSubprocessTimeout(
                    f"Fiji executor timed out after {timeout_seconds:.1f}s and was terminated.",
                    payload=payload,
                ) from exc
            finally:
                if not log_handle.closed:
                    log_handle.close()

            duration = time.monotonic() - started
            child_payload = _read_result_json(result_path)
            metadata = child_payload.get("metadata") or {}
            say_messages = list(child_payload.get("say_messages") or live_say_messages)
            artifacts = list(child_payload.get("artifacts") or [])
            result = FijiSubprocessResult(
                command=command,
                code_path=code_path,
                result_path=result_path,
                log_path=log_path,
                returncode=int(proc.returncode or 0),
                duration_seconds=duration,
                stdout=_read_text_tail(log_path),
                say_messages=say_messages,
                metadata=metadata,
                attempts=attempt,
            )

            if proc.returncode == 0:
                _merge_child_metadata(storage_manager, metadata)
                _replay_say_messages(say_capture, say_messages, skip=len(live_say_messages))
                _replay_artifacts(artifact_emitter, artifacts)
                _cleanup_attempt_artifacts(failed_attempt_artifacts)
                _cleanup_attempt_artifacts([code_path, result_path, log_path])
                return result

            _replay_say_messages(say_capture, say_messages, skip=len(live_say_messages))
            payload = {
                **result.payload(),
                "reason": "fiji_executor_nonzero_exit",
                "child_error": child_payload.get("error") or "",
                "attempt": attempt,
                "max_attempts": attempts,
            }
            error = FijiSubprocessError(
                f"Fiji executor failed with exit code {proc.returncode}.",
                payload=payload,
            )
            last_error = error
            failed_attempt_artifacts.extend([code_path, result_path, log_path])

            if attempt >= attempts:
                raise error

            if not _is_probably_transient_startup_failure(
                returncode=result.returncode,
                duration_seconds=duration,
                stdout_tail=result.stdout,
                say_messages=say_messages,
                child_payload=child_payload,
            ):
                raise error

            time.sleep(max(float(startup_retry_backoff_seconds), 0.0))
        except FijiSubprocessError:
            raise

    if last_error is not None:
        raise last_error
    raise FijiSubprocessError("Fiji executor failed before a result could be produced.")
