"""In-process executor timeout selection for microscope runtime modes.

demo/mock run virtual microscopes where long synthetic acquisitions (large
Z-stacks, mosaics, retry loops) can legitimately exceed a short budget, so
they use a larger virtual budget (default 600s) to stay safe against runaway
generated loops. Real hardware mode is intentionally UNLIMITED by default
because legitimate acquisitions can run for many hours (e.g. 24h time-lapse);
setting ``in_process_executor_timeout_seconds`` to a positive value re-enables
a cap.
"""

from typing import Any, Optional

DEFAULT_IN_PROCESS_EXECUTOR_TIMEOUT_SEC = 0.0  # 0 = no limit (real mode default)
DEFAULT_VIRTUAL_EXECUTOR_TIMEOUT_SEC = 600.0


def resolve_in_process_executor_timeout_seconds(
    *,
    role_mode: str,
    in_process_timeout_seconds: Any,
    virtual_timeout_seconds: Any = DEFAULT_VIRTUAL_EXECUTOR_TIMEOUT_SEC,
) -> Optional[float]:
    """Return the in-process executor timeout for a runtime role mode.

    ``demo`` and ``mock`` role modes get the (larger) virtual budget. Every
    other mode keeps the configured in-process budget; when that value is 0 or
    absent the executor runs without a timeout (None), which is required for
    long real-hardware acquisitions. Invalid/negative values fall back to the
    documented defaults.
    """
    virtual = _coerce_timeout(virtual_timeout_seconds, DEFAULT_VIRTUAL_EXECUTOR_TIMEOUT_SEC)
    real = _coerce_timeout(in_process_timeout_seconds, DEFAULT_IN_PROCESS_EXECUTOR_TIMEOUT_SEC)
    if str(role_mode or "").strip().lower() in ("demo", "mock"):
        return virtual
    return real if real > 0 else None


def _coerce_timeout(value: Any, default: float) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num != num or num < 0:
        return default
    return num
