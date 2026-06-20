from typing import Any, Dict, Literal

from pydantic import BaseModel, Field


TaskExecutionStatus = Literal["executed", "cancelled", "failed"]
SystemPhase = Literal["unconfigured", "ready_to_start", "initializing", "ready", "init_failed", "resetting"]
PreviewPhase = Literal["idle", "starting", "live", "failed", "stopped"]


class CommandRequest(BaseModel):
    command: str


class RuntimeInitializationResponse(BaseModel):
    initialized: bool
    initializing: bool = False
    message: str
    system_phase: SystemPhase = "unconfigured"
    failure_step: str = ""


class SystemShutdownResponse(BaseModel):
    shutting_down: bool
    message: str


class PreviewStartResponse(BaseModel):
    started: bool
    message: str
    preview_phase: PreviewPhase = "idle"


class SystemStatusResponse(BaseModel):
    configured: bool
    initialized: bool
    initializing: bool = False
    error: bool
    message: str
    system_phase: SystemPhase = "unconfigured"
    preview_phase: PreviewPhase = "idle"
    failure_step: str = ""


class TaskExecutionResponse(BaseModel):
    status: TaskExecutionStatus
    retry_times: int
    summary: str
    task_id: str
    model_name: str


class UserInputResponse(BaseModel):
    status: str
    message: str


class ConfigSaveRequest(BaseModel):
    mm_dir: str = ""
    config_path: str = ""
    fiji_path: str = ""
    camera_device: str = ""
    xy_stage_device: str = ""
    objective_device: str = ""
    transmittedIllumination: str = ""
    focus_drive: str = ""
    Dichroic: str = ""
    openai_api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_model_name: str = ""
    clarify_enabled: bool = False
    checker_enabled: bool = True
    simulation_mode: bool = True
    startup_objective: str = ""
    startup_channel: str = ""
    startup_exposure: float | None = None
    startup_brightness: int | None = None
    startup_start_preview: bool = True


class AgentConfigView(BaseModel):
    Simulation_mode: bool
    clarify_enabled: bool = False
    checker_enabled: bool = True
    openai_api_key: str
    base_url: str
    model_name: str
    vlm_api_key: str
    vlm_base_url: str
    vlm_model_name: str
    masked: Dict[str, str] = Field(default_factory=dict)


class StartupConfigView(BaseModel):
    objective: str = ""
    channel: str = ""
    exposure: float = 0.0
    brightness: int = 0
    z_position: float = 0.0
    x_position: float = 0.0
    y_position: float = 0.0
    start_preview: bool = True


class ConfigStatusResponse(BaseModel):
    configured: bool
    initialized: bool
    initializing: bool = False
    error: bool = False
    status_message: str
    system_phase: SystemPhase = "unconfigured"
    preview_phase: PreviewPhase = "idle"
    failure_step: str = ""
    system: Dict[str, Any]
    agent: AgentConfigView
    startup: StartupConfigView


class ConfigSaveResponse(BaseModel):
    saved: bool
    initialized: bool
    initializing: bool = False
    message: str
    effective_config_path: str = ""
    system_phase: SystemPhase = "unconfigured"
    preview_phase: PreviewPhase = "idle"
    failure_step: str = ""


class ConfigUploadResponse(BaseModel):
    config_path: str
    stored_config_path: str = ""
    original_filename: str = ""
    suggestions: Dict[str, Dict[str, Any]]
    objective_labels: Dict[str, int]
    dichroic_colors: Dict[str, Any]


class PreviewStatusResponse(BaseModel):
    available: bool
    initialized: bool
    stream_state: str
    status_text: str
    detail: str = ""
    healthy: bool = False
    preview_running: bool = False
    acquisition_running: bool = False
    auto_restart_enabled: bool = True
    thread_alive: bool = False
    has_frame: bool = False
    fallback_active: bool = False
    simulation_mode: bool = True
    last_frame_age_sec: float | None = None
    time_since_preview_start_sec: float | None = None
    last_error: str = ""
    preview_phase: PreviewPhase = "idle"

