import asyncio
from dataclasses import dataclass, field
from typing import Optional

from api.models import TaskExecutionResponse


@dataclass
class SessionState:
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    output_subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)
    input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_asking_user: bool = False
    pending_user_prompt: dict | None = None
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
