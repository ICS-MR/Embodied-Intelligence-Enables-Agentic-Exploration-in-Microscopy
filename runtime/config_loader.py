from typing import Any, Dict

from bootstrap.config import load_runtime_settings
from runtime.config import build_runtime_config


def load_runtime_config() -> Dict[str, Any]:
    return build_runtime_config(load_runtime_settings())
