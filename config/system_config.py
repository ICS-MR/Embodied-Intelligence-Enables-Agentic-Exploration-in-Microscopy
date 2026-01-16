
MM_DIR = r""
CONFIG_PATH = r""

objective_labels = {
    '1-UPLFLN4XPH': 4,
    '2-SOB': 10,
    '3-LUCPLFLN20XRC': 20,
    '4-LUCPLFLN40X': 40,
    '5-LUCPLFLN60X': 60,
    '6-UPLSAPO30XS': 30
}

# Channel and color mapping (RGB values)
dichroic_colors = {
    '1-NONE': (128, 128, 128),
    '2-U-FUNA': (0, 0, 255),
    '3-U-FBNA': (0, 255, 0),
    '4-U-FGNA': (255, 0, 0),
}

# Device names (adjust according to the configuration file)
camera_device = 'Camera-1'
xy_stage_device = 'XYStage'
objective_device = 'Objective'
transmittedIllumination = 'TransmittedIllumination 2'
focus_drive = 'FocusDrive'
Dichroic = 'Dichroic 2'

# Axis range parameters (read from hardware during initialization, not hard-coded)
Max_X_position: float = 500000
Min_X_position: float = 0
Max_Y_position: float = 500000
Min_Y_position: float = 0
Max_Z_position: float = 10000
Min_Z_position: float = 0

# Brightness adjustment range
Max_brightness = 250
Min_brightness = 0
Max_exposure = 1000
Min_exposure = 0

TUMOR_MODEL_CONFIG = ""
TUMOR_MODEL_CHECKPOINT = ""

# Lesion-specific model config and checkpoint
LESION_MODEL_CONFIG = ""
LESION_MODEL_CHECKPOINT = ""

# Bacteria-specific model config and checkpoint
BACTERIA_MODEL_CONFIG = ""
BACTERIA_MODEL_CHECKPOINT = ""

# 2D cell-specific model config and checkpoint
CELL_2D_MODEL_CONFIG = ""
CELL_2D_MODEL_CHECKPOINT = ""

# Organoid-specific model config and checkpoint
ORGANOID_MODEL_CONFIG = ""
ORGANOID_MODEL_CHECKPOINT = ""

# ===================== Configuration Constants (Modify according to actual paths) =====================
PSF_40X = r""
PSF_60X = r""
PSF_100X = r""

FIJI_PATH = r''