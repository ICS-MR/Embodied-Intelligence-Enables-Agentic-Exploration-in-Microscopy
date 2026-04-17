from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SystemStatus:
    initialized: bool = False
    initializing: bool = False
    error: Optional[str] = None
    message: str = "Waiting for configuration"
    system_phase: str = "unconfigured"
    failure_step: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
