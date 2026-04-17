from bootstrap.config import load_detection_targets, load_system_config


_config = load_system_config()

MM_DIR = _config.MM_DIR
CONFIG_PATH = _config.CONFIG_PATH
MAVEN_BIN = _config.MAVEN_BIN
objective_labels = _config.objective_labels
dichroic_colors = _config.dichroic_colors
camera_device = _config.camera_device
xy_stage_device = _config.xy_stage_device
objective_device = _config.objective_device
transmittedIllumination = _config.transmittedIllumination
focus_drive = _config.focus_drive
Dichroic = _config.Dichroic
Max_X_position = _config.Max_X_position
Min_X_position = _config.Min_X_position
Max_Y_position = _config.Max_Y_position
Min_Y_position = _config.Min_Y_position
Max_Z_position = _config.Max_Z_position
Min_Z_position = _config.Min_Z_position
Max_brightness = _config.Max_brightness
Min_brightness = _config.Min_brightness
Max_exposure = _config.Max_exposure
Min_exposure = _config.Min_exposure
PSF_40X = _config.PSF_40X
PSF_60X = _config.PSF_60X
PSF_100X = _config.PSF_100X
FIJI_PATH = _config.FIJI_PATH
fiji_executor_timeout_seconds = _config.fiji_executor_timeout_seconds


def get_detection_targets() -> dict[str, dict]:
    return load_detection_targets()


def get_detection_target_spec(target_type: str) -> dict:
    targets = load_detection_targets()
    normalized = str(target_type).strip()
    if normalized in targets:
        return dict(targets[normalized])
    lowered = normalized.lower()
    for key, value in targets.items():
        if key.lower() == lowered:
            return dict(value)
    raise KeyError(f"Unknown detection target: {target_type}. Available: {', '.join(targets.keys())}")
