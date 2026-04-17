import asyncio
from dataclasses import dataclass, field
from typing import Optional

from api.models import TaskExecutionResponse


@dataclass
class SessionState:
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_asking_user: bool = False
    first_connection_made: bool = False


@dataclass
class TaskState:
    running: bool = False
    current_task_id: Optional[str] = None
    last_result: Optional[TaskExecutionResponse] = None


@dataclass
class AppState:
    session: SessionState = field(default_factory=SessionState)
    task: TaskState = field(default_factory=TaskState)
