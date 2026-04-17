from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_USER_TOOL_DOCS_RELATIVE = Path("prompts") / "generated" / "user_tools"
DEFAULT_USER_TOOL_DOCS_DIR = ROOT_DIR / DEFAULT_USER_TOOL_DOCS_RELATIVE
