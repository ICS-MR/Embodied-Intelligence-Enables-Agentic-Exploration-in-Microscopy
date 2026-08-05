import datetime
import uuid
from pathlib import Path
from typing import Tuple


def create_runtime_session_paths(
    history_root: str | Path,
    output_dir_name: str | Path,
) -> Tuple[str, str, str]:
    base_history_root = Path(history_root)
    output_leaf_name = Path(output_dir_name).name or "output"
    session_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_dir = base_history_root / session_id
    session_output_dir = session_dir / output_leaf_name
    session_output_dir.mkdir(parents=True, exist_ok=True)
    return session_id, str(session_dir), str(session_output_dir)
