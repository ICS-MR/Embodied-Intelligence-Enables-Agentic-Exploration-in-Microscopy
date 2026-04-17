from dataclasses import dataclass
from typing import Any, Dict

from adapters.tool_registry import ToolRegistry
from agent.experiment_checker import ExperimentCheckAgent
from agent.experiment_planner import ExperimentPlanAgent
from core_tool.tool_utils import SayCapture
from services.task_orchestrator import TaskOrchestrator
from utils.memory_manager import HistoryManager, StorageManager


@dataclass
class RuntimeContext:
    session_id: str
    session_dir: str
    output_dir: str
    runtime: Dict[str, Any]
    llm_client: Any
    vlm_client: Any
    say_capture: SayCapture
    env_olympus: Any
    env_imagej: Any
    env_cellpose: Any
    storage_manager: StorageManager
    history_manager: HistoryManager
    task_manager: ExperimentPlanAgent
    tool_registry: ToolRegistry
    checker: ExperimentCheckAgent
    task_orchestrator: TaskOrchestrator


SystemComponents = RuntimeContext
