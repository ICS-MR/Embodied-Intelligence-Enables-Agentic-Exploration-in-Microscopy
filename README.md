# Embodied Intelligence Enables Agentic Exploration in Microscopy

**Embodied Intelligence Microscope System (EIMS)** is an intelligent platform that reconceptualizes the microscope from a passive imaging tool into an autonomous explorer capable of closing the loop between intent, perception, and action. Through natural language interaction, it enables a **fully automated closed-loop workflow**—from understanding experimental instructions and task planning to automatic image acquisition and real-time analysis.

This project is not merely a control script—it is an AI-powered "experimentalist" equipped with **perception, decision-making, execution, and self-correction** capabilities, designed to address key pain points in traditional biological experiments: complex microscope operation, cumbersome workflows, and the lack of fully autonomous closed-loop control.

## ✨ Core Features

- **🗣️ Natural Language Interaction**: Accepts complex experimental commands directly in natural language (English or Chinese).
- **🧠 Autonomous Task Planning**: The built-in `TaskManager` automatically decomposes abstract goals into executable sequences, including stage movement, focusing, channel switching, Z-stack scanning, and more.
- **🔄 Closed-Loop Self-Correction**: Performs real-time validation during execution (e.g., blur detection, object tracking). If failure occurs, it autonomously generates corrective actions and retries—instead of simply throwing an error.
- **🔬 Multi-Modal Imaging Support**: Natively supports brightfield and multi-channel fluorescence imaging (DAPI, FITC, TRITC) with full multidimensional acquisition (XY-Z-T-C).
- **🧩 Integrated Advanced Analysis**: Seamlessly integrates **Fiji (ImageJ)** for image processing and combines **Cellpose / MMDetection** for high-precision cell segmentation and object detection—enabling true “what you see is what you get” intelligent targeting.

## 🛠️ System Architecture

The system adopts a clean three-layer modular design, ensuring clear responsibilities and efficient collaboration:

### 1. Agent Layer (Core Decision-Making)

- **Task Manager**: Parses natural language instructions via LLM and orchestrates task decomposition and step scheduling.
- **Language Model Program (LMP)**: Dynamically generates executable Python code to drive underlying tools.
- **Checker**: Performs real-time visual and logical quality control (e.g., focus validation) and handles exceptions to ensure workflow closure.

### 2. Tool Platform Layer (Functional Implementation)

- **Microscope Platform** (`tool/microscope.py`): Core hardware control module for focusing, exposure adjustment, stage movement, Z-stack scanning, etc.
- **Image Analysis Platform** (`tool/fiji.py`): Fiji/ImageJ wrapper for preprocessing, signal quantification, and other analyses.
- **Cell Segmentation Platform** (`tool/cellpose_tool.py`): Integrates Cellpose for cell segmentation, counting, and phenotypic analysis.

### 3. Hardware Driver Layer (Low-Level Abstraction)

Built on `pymmcore-plus`, providing standardized control over mainstream microscopes (e.g., Olympus), minimizing device-specific integration effort.

## 🚀 Quick Deployment Guide

Follow these steps to initialize the system:

### 1. Prerequisites

- **OS**: Windows 10/11 (recommended for optimal Micro-Manager driver support)
- **Python**: Version 3.10 or higher
- **External Software**:
  - [Micro-Manager 2.0](https://micro-manager.org/) (required for hardware control)
  - [Fiji (ImageJ)](https://imagej.net/software/fiji/) (required for image processing)
- **Hardware**: NVIDIA GPU with CUDA support (recommended for accelerating Cellpose and MMDetection inference)

### 2. Install Dependencies

Use a dedicated Conda environment to avoid conflicts:

```bash
# 1. Create environment
conda create -n miscope python=3.10
conda activate miscope

# 2. Install base dependencies
pip install -r requirements.txt

# 3. Install MMDetection stack (via mim)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### 3. Core Configuration (Critical Step)

Before running, update the following configuration files to match your setup:

#### A. System Paths (`config/system_config.py`)

Edit key paths to reflect your installation:

```python
# 1. Micro-Manager hardware config file (.cfg)
CONFIG_PATH = r"C:\path\to\Your_Microscope_Config.cfg"

# 2. Micro-Manager root directory (contains device adapters)
MM_DIR = r"C:\Program Files\Micro-Manager-2.0"

# 3. Fiji (ImageJ) application directory
FIJI_PATH = r"D:\Software\Fiji.app"
```

#### B. Model Weights (`config/system_config.py`)

Place model weights in the `weights/` folder or update paths accordingly:

```python
TUMOR_MODEL_CONFIG = "configs/tumor_model.py"
TUMOR_MODEL_CHECKPOINT = "weights/tumor_best.pth"
# ... configure organoid, 2Dcell, etc., as needed
```

#### C. LLM API Keys (`config/agent_config.py`)

Configure your LLM service:

```python
openai_api_key = 'sk-...'       # Your LLM API key
base_url = 'https://...'        # API endpoint (e.g., OpenAI or OpenRouter)
model_name = 'gpt-4o'           # Recommended: gpt-4o or claude-3-5-sonnet
```

#### D. Adding New Tools (e.g., FRAP)

The system provides `create_tool.py` to auto-generate LMP prompts for new tools marked with `@tool_func`, enabling seamless integration.

1. Place your tool class (e.g., `Frap.py`) in the `tool/` directory:

```python
# tool/Frap.py
from tool.base import BaseTool, tool_func
class Frap(BaseTool):
    """Example tool for image capture and motion control."""
    
    def __init__(self, device_id: int = 0, save_dir: str = "./output"):
        self.device = cv.VideoCapture(device_id)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @tool_func
    def capture_image(self, filename: str, resize: tuple = (640, 480)) -> bool:
        ret, frame = self.device.read()
        if not ret:
            return False
        frame = cv.resize(frame, resize)
        save_path = os.path.join(self.save_dir, filename)
        return cv.imwrite(save_path, frame)

    @tool_func
    def adjust_exposure(self, exposure_value: int) -> None:
        if 0 <= exposure_value <= 100:
            self.device.set(cv.CAP_PROP_EXPOSURE, exposure_value)
        else:
            raise ValueError("Exposure must be between 0 and 100")

    @tool_func
    def close_device(self) -> None:
        self.device.release()
```

2. Generate tool metadata:

```bash
python create_tool.py
```

3. Register the new tool in `main.py` within `initialize_system`:

```python
def initialize_system(history_dir, filename):
    # ... existing code ...
    env_newtool = NewTool(storageManager, OUTPUT_DIR)  # ← Add instance

    # Map public methods
    newtool_methods = env_newtool.get_public_methods()
    newtool_vars = {k: getattr(env_newtool, k) for k in newtool_methods}

    # Initialize LMP for new tool
    prompt_newtool = LMP(
        'prompt_newtool',
        cfg_tabletop['lmps']['prompt_newtool'],
        cfg_tabletop['lmps']['fgen'],
        fixed_vars,
        newtool_vars,
        llm_client,
        historymanager
    )

    # Update module map
    module_map = {
        'Microscope Control': prompt_olympus,
        'Cell Segmentation': prompt_cellpose,
        'Image Analysis': prompt_imagej,
        'Data Analysis': prompt_newtool  # ← New mapping
    }

    return {
        # ... existing returns ...
        'env_newtool': env_newtool,
    }
```

4. Add LMP config in `config/agent_config.py`:

```python
cfg_tabletop = {
    'lmps': {
        'prompt_newtool': {
            'prompt_path': 'prompts/prompt_newtool.py',
            'temperature': 0.2,
            'max_tokens': 2000
        },
        # ...
    }
}
```

## 📖 Key Specifications & Guidelines

### 1. Tool Class Interface Requirements

All custom tools must adhere to the following:

- **Required**: `__init__` must accept `storage_manager` and `output_dir`.
- **Recommended**: Include `xxx_initialize()` and `xxx_shutdown()` methods.
- **Methods**: Use type hints; return serializable types (dict, bool, str, etc.).
- **File I/O**: Always use `storage_manager`—never direct file system access.

#### Standard Implementation Example

**Step 1: Initialize with StorageManager**

```python
class NewTool(BaseTool):
    def __init__(self, storage_manager: StorageManager, output_dir: str):
        self._storagemanger = storage_manager
        self.output_dir = output_dir
```

**Step 2: Register generated files**

```python
def generate_result_file(self, data: pd.DataFrame, filename: str, desc: str):
    file_path = Path(self.output_dir, filename)
    data.to_csv(file_path, index=False)
    
    self._storagemanger.register_file(
        filename=filename,
        description=desc,
        tool_name='newtool',
        file_type='csv'
    )
    return file_path
```

**Step 3: Use cache for intermediate results**

```python
def run_analysis(self, input_data):
    self._storagemanger.clear_cache()
    # ... process ...
    cached_files = self._storagemanger.read_cache()
    if not cached_files:
        raise RuntimeError("No output generated!")
```

**Step 4: Clean up on shutdown**

```python
def shutdown(self):
    self._storagemanger.clear_cache()
    print("NewTool resources released")
```

## 📋 User Guide

### 1. Launch the System

Run from the project root:

```bash
python main.py
```

After seeing “System initialized successfully,” interact via natural language.

Two operating modes are supported via the `human_mode` flag:

- **Fully Autonomous Mode (`human_mode=False`)**: User provides only a high-level goal. The system designs, validates, and executes the full experiment plan.
- **Interactive Mode (`human_mode=True`)**: User issues step-by-step commands for fine-grained control or debugging.

> **Recommendation**: Start with `human_mode=True` to learn system behavior, then switch to autonomous mode for efficiency.

## 📂 Project Structure

```plaintext
llm_miscope/
├── main.py                         # Entry point
├── requirements.txt                # Dependencies
├── Demo.ipynb                      # Usage examples
├── agent/                          # Core agent logic
│   ├── experiment_designer.py      # Experimental design
│   ├── experiment_planner.py       # Task decomposition
│   ├── experiment_executor.py      # Code generation & execution
│   └── experiment_checker.py       # Validation & feedback
├── core_tool/                      # Built-in tools
│   ├── microscope.py               # MicroscopeController
│   ├── fiji.py                     # ImageJProcessor
│   ├── cellpose_tool.py            # Cellpose2D
│   └── tool_utils.py               # Utilities (e.g., sharpness metric)
├── tool/                           # Custom tools
│   ├── base.py                     # BaseTool class
│   └── ...                         # User-defined tools
├── config/                         # Configuration
│   ├── system_config.py            # Hardware & paths
│   ├── agent_config.py             # LLM settings
│   └── task_config.py              # Task constants
├── prompts/                        # LMP prompts
│   ├── task_manager_full.py
│   └── ...
├── RAG_documents/                  # RAG knowledge base
└── weights/                        # Model checkpoints (user-provided)
```

## ⚠️ Important Notes & Disclaimer

- **Hardware Safety**: Always set physical limits (e.g., `Max_Z_position`) in `system_config.py` to prevent objective crashes.
- **Model Weights**: MMDetection-based features require pre-downloaded weights. Ensure paths are correctly configured.
- **Driver Compatibility**: Verify that your Micro-Manager `.cfg` file works in the official GUI before use.

**Open-Source Licensing**:
- **Fiji** is distributed under the **GNU GPL**, with its ImageJ2 core under the **BSD 2-Clause License**. Plugins may have individual licenses. See [Fiji Licensing](https://imagej.net/licensing).
- **Micro-Manager (μManager)** is a free, open-source project hosted on GitHub under a **BSD-style license**, suitable for both academic and commercial use.

This system only calls public APIs of Fiji and Micro-Manager without modifying their source code, fully complying with their respective licenses.

## 🤝 Contributions

Issues and Pull Requests are welcome!
