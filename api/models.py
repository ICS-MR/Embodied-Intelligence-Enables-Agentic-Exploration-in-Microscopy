from typing import Any, Dict, Literal

from pydantic import BaseModel as PydanticBaseModel, Field


TaskExecutionStatus = Literal["executed", "cancelled", "failed"]
SystemPhase = Literal[
    "unconfigured",
    "ready_to_start",
    "initializing",
    "ready",
    "executing",
    "releasing",
    "failed",
]
PreviewPhase = Literal["idle", "starting", "live", "failed", "stopped"]


class BaseModel(PydanticBaseModel):
    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        dump = getattr(super(), "model_dump", None)
        if callable(dump):
            return dump(*args, **kwargs)
        return self.dict(*args, **kwargs)


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
    focus_drive: str = ""
    Dichroic: str = ""
    Max_X_position: float | None = None
    Min_X_position: float | None = None
    Max_Y_position: float | None = None
    Min_Y_position: float | None = None
    Max_Z_position: float | None = None
    Min_Z_position: float | None = None
    Max_brightness: int | None = None
    Min_brightness: int | None = None
    Max_exposure: int | None = None
    Min_exposure: int | None = None
    openai_api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_model_name: str = ""
    clarify_enabled: bool = False
    checker_enabled: bool = True
    microscope_mode: Literal["demo", "real", "mock"] = "demo"
    image_analysis_mode: Literal["real", "mock"] = "mock"
    segmentation_mode: Literal["real", "mock"] = "mock"
    objectives: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    channels: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    transmitted_light: Dict[str, Any] = Field(default_factory=dict)
    demo_environment: Dict[str, Any] = Field(default_factory=dict)
    startup_objective: str = ""
    startup_channel: str = ""
    startup_exposure: float | None = None
    startup_brightness: int | None = None
    startup_z_position: float | None = None
    startup_x_position: float | None = None
    startup_y_position: float | None = None


class LLMConnectionTestRequest(BaseModel):
    openai_api_key: str = ""
    base_url: str = ""
    model_name: str = ""


class LLMConnectionTestResponse(BaseModel):
    ok: bool
    detail: str = ""


class VLMConnectionTestRequest(BaseModel):
    vlm_api_key: str = ""
    vlm_base_url: str = ""
    vlm_model_name: str = ""


class VLMConnectionTestResponse(BaseModel):
    ok: bool
    detail: str = ""


class AgentConfigView(BaseModel):
    microscope_mode: Literal["demo", "real", "mock"] = "demo"
    image_analysis_mode: Literal["real", "mock"] = "mock"
    segmentation_mode: Literal["real", "mock"] = "mock"
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
    real_system_draft: Dict[str, Any] = Field(default_factory=dict)
    real_startup_draft: Dict[str, Any] = Field(default_factory=dict)
    demo_system: Dict[str, Any] = Field(default_factory=dict)
    demo_startup: Dict[str, Any] = Field(default_factory=dict)
    mock_capabilities: Dict[str, Any] = Field(default_factory=dict)
    transmitted_light_runtime: Dict[str, Any] = Field(default_factory=dict)
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


MappingDraftSource = Literal["core", "rule", "runtime", "ai", "current_config", "manual_required"]
MappingDraftConfidence = Literal["high", "medium", "low", "unknown"]


class ConfigMappingDraftField(BaseModel):
    value: str = ""
    candidates: list[str] = Field(default_factory=list)
    source: MappingDraftSource = "manual_required"
    confidence: MappingDraftConfidence = "unknown"
    reason: str = ""
    needs_review: bool = True
    rule_value: str = ""
    ai_value: str = ""
    current_value: str = ""

    class Config:
        # AI model responses occasionally include extra keys (e.g. a nested
        # transmitted_light block). Tolerate them instead of failing validation;
        # the merge logic only reads the known fields.
        extra = "allow"


class ConfigMappingAnalysis(BaseModel):
    ai_status: Literal["completed", "not_configured", "unavailable"] = "not_configured"
    hardware_inspection_status: Literal["completed", "skipped", "unavailable"] = "skipped"
    inspected_device_count: int = 0
    fields: Dict[str, ConfigMappingDraftField] = Field(default_factory=dict)
    objectives: Dict[str, ConfigMappingDraftField] = Field(default_factory=dict)
    channels: Dict[str, ConfigMappingDraftField] = Field(default_factory=dict)
    transmitted_light: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class ConfigUploadResponse(BaseModel):
    config_path: str
    mapping: ConfigMappingAnalysis


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
    microscope_mode: Literal["demo", "real", "mock"] = "demo"
    image_analysis_mode: Literal["real", "mock"] = "mock"
    segmentation_mode: Literal["real", "mock"] = "mock"
    mode_summary: str = ""
    last_frame_age_sec: float | None = None
    time_since_preview_start_sec: float | None = None
    last_error: str = ""
    preview_phase: PreviewPhase = "idle"

