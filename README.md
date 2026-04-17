# Embodied Intelligence Enables Agentic Exploration in Microscopy

**Embodied Intelligence Microscope System (EIMS)** is an intelligent platform that reconceptualizes the microscope from a passive imaging tool into an autonomous explorer capable of closing the loop between intent, perception, and action. Through natural language interaction, it enables a **fully automated closed-loop workflow**—from understanding experimental instructions and task planning to automatic image acquisition and real-time analysis.

This project is not merely a control script—it is an AI-powered "experimentalist" equipped with **perception, decision-making, execution, and self-correction** capabilities, designed to address key pain points in traditional biological experiments: complex microscope operation, cumbersome workflows, and the lack of fully autonomous closed-loop control.

## ✨ Core Features

- **🗣️ Natural Language Interaction**: Accepts complex experimental commands directly in natural language (English or Chinese).
- **🧠 Autonomous Task Planning**: The built-in `TaskManager` automatically decomposes abstract goals into executable sequences, including stage movement, focusing, channel switching, Z-stack scanning, and more.
- **📝 Confirm-Before-Execute Flow**: The runtime first generates a plan, rewrites it into a short Scopebot preview, waits for confirmation or revision, and only then executes.
- **🧩 Skill-Guided Planning**: Users can add planning guidance documents under `user_skills/planning/` to influence how the planner decomposes tasks.
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

- **Microscope Platform** (`core_tool/microscope.py` in real mode, `Empty_function.py` in virtual mode): Core hardware control module for focusing, exposure adjustment, stage movement, Z-stack scanning, etc.
- **Image Analysis Platform** (`core_tool/fiji.py` in real mode, `Empty_function.py` in virtual mode): Fiji/ImageJ wrapper for preprocessing, signal quantification, and other analyses.
- **Cell Segmentation Platform** (`core_tool/cellpose_tool.py` in real mode, `Empty_function.py` in virtual mode): Integrates Cellpose for cell segmentation, counting, and phenotypic analysis.

### 3. Hardware Driver Layer (Low-Level Abstraction)

Built on `pymmcore-plus`, providing standardized control over mainstream microscopes (e.g., Olympus), minimizing device-specific integration effort.

## Developer Docs

For contributors working on runtime internals, API contracts, or new tool integrations, see [`docs/developer-guide.md`](docs/developer-guide.md).

The guide covers:

- the current post-refactor module layout
- runtime and API contracts
- extension points for tools, planner skills, and execution safety
- recommended test commands and maintenance rules

## 🚀 Quick Deployment Guide

Follow these steps to initialize the system:

### 1. Prerequisites

- **OS**: Windows 10/11 (recommended for optimal Micro-Manager driver support)
- **Python**: Version 3.10 or higher
- **External Software**:
  - [Micro-Manager 2.0](https://micro-manager.org/) (required only for real hardware control)
  - [Fiji (ImageJ)](https://imagej.net/software/fiji/) (required only for real image-processing runtime)
- **Hardware**: NVIDIA GPU with CUDA support (recommended for accelerating Cellpose and MMDetection inference)

### 2. Install Dependencies

The default `uv sync` flow installs the officially validated GPU environment for this project on Windows. Before syncing, make sure your NVIDIA driver is installed and `nvidia-smi` works from a terminal.

This repository is now organized around `uv`, with Python 3.10 pinned via `.python-version`:

```bash
# 1. Create or reuse the project virtual environment
uv venv --python 3.10

# 2. Sync all pinned dependencies from pyproject.toml
uv sync

# 3. One-click install a compatible Micro-Manager build for the current pymmcore version
uv run python system_config_wizard.py --install-mmcore

# 4. Open the installed Micro-Manager GUI (MMStudio / ImageJ)
uv run python system_config_wizard.py --open-mmstudio
```

Notes:

- `requirements.txt` is kept as a compatibility snapshot, but `pyproject.toml` is the primary dependency source.
- The default `uv sync` path is pinned to `torch==2.1.0`, CUDA 11.8, `mmcv==2.1.0`, `mmengine==0.10.7`, `mmdet==3.3.0`, and `numpy==1.26.4`.
- PyTorch packages are resolved from the official CUDA 11.8 index, and the `mmcv` wheel is pinned to the matching `cu118/torch2.1.0` build.
- If your machine does not have a usable NVIDIA GPU, see `CPU Compatibility Mode` below instead of changing the default `uv sync` path.
- The one-click installer above calls the official `mmcore install` command, installs Micro-Manager under `%LOCALAPPDATA%\\EIMS\\Micro-Manager\\` by default on Windows, and writes the detected install path back to `config/runtime_config.json` as `MM_DIR`.
- If you already have a working Micro-Manager installation, you can skip the one-click step and point `MM_DIR` to your existing install.
- If you want a specific nightly or only the test adapters, use `uv run python system_config_wizard.py --install-mmcore --mmcore-release 20210219` or add `--test-adapters`.
- If `--mmcore-dest` already contains `Micro-Manager*` directories, installation now stops by default to avoid accidental overwrite.
- Use `uv run python system_config_wizard.py --install-mmcore --reuse-existing` to directly reuse the latest existing install.
- Use `uv run python system_config_wizard.py --install-mmcore --clean-dest` to remove existing `Micro-Manager*` directories and reinstall.
- To open a different install explicitly, use `uv run python system_config_wizard.py --open-mmstudio --mm-dir "C:\\Path\\To\\Micro-Manager"`.

### 2.1 GPU Validation

After `uv sync`, validate the official GPU stack with these commands:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python scripts/test_cellpose_api.py --model cpsam --gpu --skip-segment
uv run python -c "import mmcv, mmengine, mmdet; print(mmcv.__version__); print(mmengine.__version__); print(mmdet.__version__)"
```

Expected results:

- PyTorch reports `2.1.0`
- CUDA reports `11.8`
- `torch.cuda.is_available()` is `True`
- Cellpose initialization succeeds and prints `_use_GPU: True`
- `mmcv`, `mmengine`, and `mmdet` import successfully as `2.1.0`, `0.10.7`, and `3.3.0`

### 2.2 CPU Compatibility Mode

CPU-only installation is supported as a fallback for demo, debugging, or environments without a usable NVIDIA GPU, but it is not the recommended runtime. In particular, `cpsam` segmentation is much slower on CPU.

If you need a CPU-only environment, create a separate virtual environment and install the CPU variants manually instead of using the default project `uv sync`. Keep `numpy==1.26.4` unchanged, use the CPU PyTorch index, and swap the `mmcv` wheel to the matching CPU build for `torch==2.1.0`.

### 3. Core Configuration (Critical Step)

Before running, update the runtime configuration to match your setup.

Current recommendation:

- Use the Web UI configuration page when running `uvicorn app:app --reload`
- Or edit `config/runtime_config.json` through the bootstrap config helpers

Static Python config modules in `config/` are still used for prompt/task defaults, but the hardware paths, startup state, and model credentials used by the runtime are primarily driven by the saved runtime configuration.

Configuration priority is:

- built-in defaults
- `config/runtime_config.json`
- `.env`
- real process environment variables

Practical guidance:

- End users should primarily edit `config/runtime_config.json`
- Use `.env` only for secrets or endpoint overrides such as API keys and base URLs
- `model.Simulation_mode` should be configured in `config/runtime_config.json`, not in `.env`
- `model.Simulation_mode=true` keeps the system on the virtual runtime chain powered by `Empty_function.py`
- `model.Simulation_mode=false` switches the runtime to the real hardware chain under `core_tool/`

#### A. System Paths (`config/runtime_config.json`)

Edit key paths to reflect your installation:

```json
{
  "system": {
    "CONFIG_PATH": "C:/path/to/Your_Microscope_Config.cfg",
    "MM_DIR": "C:/Program Files/Micro-Manager-2.0",
    "FIJI_PATH": "D:/Software/Fiji.app"
  }
}
```

#### B. Model Weights (`config/system_config.py`)

Place model weights in the `weights/` folder or update paths accordingly:

```python
TUMOR_MODEL_CONFIG = "configs/tumor_model.py"
TUMOR_MODEL_CHECKPOINT = "weights/tumor_best.pth"
# ... configure organoid, 2Dcell, etc., as needed
```

#### B.1 Minimal Mitosis Reproduction

For a minimal standalone mitosis-model verification workflow, keep the files in this layout:

```plaintext
configs/
  mitosis_rtmdet.py
weights/
  mitosis_best.pth
evaluation/
  mitosis_infer.py
  mitosis_testset/
    images/
      mitosis_test_001.jpg
      ...
    annotations.json
```

Recommended runtime configuration:

```json
{
  "system": {
    "MITOSIS_MODEL_CONFIG": "configs\\mitosis_rtmdet.py",
    "MITOSIS_MODEL_CHECKPOINT": "weights\\mitosis_best.pth"
  }
}
```

The COCO test subset should follow these conventions:

- images are stored under `evaluation/mitosis_testset/images/`
- annotations are stored in `evaluation/mitosis_testset/annotations.json`
- the category name is `mitosis`

Run inference from the project root:

```bash
python evaluation/mitosis_infer.py
```

Common optional arguments:

```bash
python evaluation/mitosis_infer.py --score-thr 0.3
python evaluation/mitosis_infer.py --device cpu
python evaluation/mitosis_infer.py --device cuda:0
```

Expected outputs:

- `evaluation/mitosis_predictions.json`: exported detection results in JSON format
- `evaluation/mitosis_visualizations/`: images with red bounding boxes and confidence labels

#### C. LLM API Keys and Endpoints

You can configure these either in `config/runtime_config.json` or override them from `.env`.
Recommended practice is to keep secrets in `.env`.

Example `.env`:

```dotenv
EIMS_OPENAI_API_KEY=your-openai-compatible-api-key
EIMS_BASE_URL=https://api.openai.com/v1
EIMS_MODEL_NAME=gpt-4.1
EIMS_VLM_API_KEY=your-vlm-api-key
EIMS_VLM_BASE_URL=https://api.openai.com/v1
EIMS_VLM_MODEL_NAME=gpt-4.1
```

Example `config/runtime_config.json`:

```json
{
  "model": {
    "openai_api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4.1",
    "vlm_api_key": "",
    "vlm_base_url": "https://api.openai.com/v1",
    "vlm_model_name": "gpt-4.1",
    "Simulation_mode": true,
    "checker_enabled": true
  }
}
```

#### D. Planner Skills (`user_skills/planning/`)

The planner supports metadata-aware, user-maintained planning skills.

Supported layouts:

```plaintext
user_skills/planning/
  brightfield.md
  fluorescence.json
  fluorescence_first_pass/
    SKILL.md
```

Supported formats:

- Plain `.md`, `.txt`, or `.json` skill files
- Directory-style skill packages with `SKILL.md`

Skill selection now considers more than raw keyword overlap. The planner combines:

- name matches
- trigger phrases
- tags
- example queries
- content overlap
- explicit priority

The most relevant skills are injected into the planning prompt with structured metadata such as description, trigger conditions, example queries, and guidance.

Example package skill:

```md
---
name: Brightfield Focus Workflow
description: Preferred brightfield sequencing
tags: brightfield, focus
triggers: brightfield image, overview scan
examples: capture a brightfield image
priority: 3
---

- Focus before brightfield capture.
- Use low magnification for overview scans.
- Avoid assuming fluorescence channels unless the user explicitly requests them.
```

Optional JSON format:

```json
{
  "name": "Fluorescence First Pass",
  "description": "Conservative fluorescence planning workflow",
  "tags": ["fluorescence", "preview"],
  "triggers": ["fluorescence imaging", "stain preview"],
  "examples": ["capture a fluorescence preview of the stained sample"],
  "priority": 5,
  "content": "Start with a low-exposure preview before high-intensity acquisition."
}
```

See `user_skills/planning/README.md` for the current convention.

#### E. (optional) Adding New Tools (e.g., FRAP)

The runtime now distinguishes between:

- **system tools**: the built-in microscope, image analysis, and segmentation roles defined under `system_tools` in `config/tool_manifest.json`
- **user tools**: extension tools implemented under `tool/` as `BaseTool` subclasses

User tools are implemented under `tool/`, but they only participate in runtime registration and planner injection when they are explicitly listed in `config/tool_manifest.json` with `enabled: true`. The configured `tool_id` is the single shared module name used in planner examples, planner task steps, and runtime executor lookup.

1. Implement a user tool under `tool/`:

```python
# tool/frap.py
from tool.base import BaseTool, tool_func


class Frap(BaseTool):
    """Example extension tool for FRAP point-plan execution."""

    planning_hint = (
        "Use this tool when a task already has FRAP points or needs to save, "
        "validate, map, preview, or execute a point plan."
    )
    execution_hint = (
        "Create or load a FRAP point plan, validate it, build an image-to-screen "
        "mapping, preview the points, then run the click plan in dry-run mode "
        "before real execution."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir

    @tool_func
    def create_empty_plan(self, image_width: int, image_height: int) -> dict:
        """Create an empty FRAP point plan."""
        return {
            "plan_kind": "frap_point_plan",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "targets": [],
        }

    @tool_func
    def validate_point_plan(self, point_plan: dict) -> dict:
        """Validate and normalize a FRAP point plan."""
        return point_plan

    @tool_func
    def build_linear_mapping(
        self,
        image_width: int,
        image_height: int,
        screen_left: int,
        screen_top: int,
        screen_width: int,
        screen_height: int,
    ) -> dict:
        """Define a linear image-to-screen mapping."""
        return {
            "mapping_kind": "frap_linear_mapping",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "screen_left": int(screen_left),
            "screen_top": int(screen_top),
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
        }

    @tool_func
    def execute_click_plan(self, click_plan_path: str, dry_run: bool = True) -> dict:
        """Execute a saved FRAP click plan."""
        return {"status": "ok", "dry_run": bool(dry_run)}
```

2. Register the tool through the CLI-first onboarding flow:

```bash
# Launch the default interactive CLI wizard
python create_tool.py

# Or register explicitly from the command line
python create_tool.py register --class-path tool.frap:Frap --tool-id "FRAP Tool" --dry-run

# Append the entry to config/tool_manifest.json
python create_tool.py register --class-path tool.frap:Frap --tool-id "FRAP Tool"

# Inspect or update existing entries
python create_tool.py list
python create_tool.py enable "FRAP Tool"
```

The CLI validates that the class resolves to a `BaseTool` subclass and that it exposes at least one `@tool_func` method. The configured `tool_id` appears directly in planner `module` fields, so it should be readable as a planner-facing module name. After a successful registration, prompt artifacts are generated automatically under `prompts/generated/user_tools/`.

A minimal `user_tools` entry now looks like this:

```json
{
  "tool_id": "frap",
  "class_path": "tool.frap:Frap",
  "enabled": true,
  "planning_hint": "Use this tool for FRAP point-plan validation, mapping, preview, and execution.",
  "execution_hint": "Validate the FRAP point plan, build a mapping, preview it, and run the click plan in dry-run mode before real execution."
}
```

3. Generate user-tool prompt artifacts manually when needed:

```bash
python create_tool.py generate-docs --output-dir prompts/generated/user_tools
```

This generates per-tool artifacts such as:

- `prompts/generated/user_tools/frap.executor_prompt.txt`
- `prompts/generated/user_tools/frap.planner_summary.txt`

Prompt behavior is now:

- **system tools** continue to use their fixed prompt sources under `prompts/`
- **user tools** prefer generated artifacts from `prompts/generated/user_tools/`; `create_tool.py register` writes them automatically by default
- if an executor artifact is missing, the runtime falls back to `BaseTool.get_execution_prompt_context()`
- planner-side runtime injection prefers generated `planner_summary.txt` sections and otherwise falls back to the stable planning helpers on `BaseTool`
- the planner summary is injected at runtime into `# Submodule Functions` and the example section via fixed placeholders

#### F. Program Execution

To facilitate a quick understanding of the entire program workflow, we provide a runnable example implemented in a Jupyter Notebook. Together with detailed supporting documentation, it clearly describes the step-by-step procedure of program execution, how EIMS autonomously conducts planning, decision-making and microscope control, as well as the experimental results obtained from real‑world deployments.

We recommend that interested researchers prioritize this section for a rapid and comprehensive grasp of the project. For the complete execution of the CLI version (which requires hardware support and adaptation), please refer to the section in the user guide.

```python
Hardware-Free-Demo.ipynb
```

## 📖 Key Specifications & Guidelines

### 1. User Tool Requirements

All user extension tools should follow the current `BaseTool` contract:

- **Required**: inherit from `tool.base.BaseTool`
- **Required**: expose at least one public method with `@tool_func`
- **Constructor**: runtime-supported constructor parameters are `storage_manager` and `output_dir`; any other required constructor argument will prevent runtime registration
- **Recommended**: add type hints and clear docstrings for each `@tool_func` method
- **Optional**: set `planning_hint` and `execution_hint` on the class for better generated prompts
- **Preferred**: return JSON-serializable values such as `dict`, `list`, `bool`, `str`, `int`, or `float`
- **Recommended**: use `storage_manager` when outputs should be tracked by the runtime session

#### Minimal Implementation Example

```python
from tool.base import BaseTool, tool_func


class NewTool(BaseTool):
    planning_hint = "Use this tool for report generation tasks."
    execution_hint = "Call run before export if both are needed."

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir

    @tool_func
    def run(self, text: str) -> str:
        """Process the input text and return a short result string."""
        return f"processed: {text}"
```

#### Working with Runtime Outputs

```python
def save_result(self, filename: str, content: str) -> str:
    file_path = Path(self.output_dir, filename)
    file_path.write_text(content, encoding="utf-8")
    if self.storage_manager is not None:
        self.storage_manager.register_file(
            filename=filename,
            description="Generated by NewTool",
            tool_name="newtool",
            file_type="txt",
        )
    return str(file_path)
```

#### Prompt Generation for User Tools

User-tool prompts now come from two layers:

- **executor prompt**: detailed API-facing prompt for code generation
- **planner summary**: concise planning-facing capability summary

By default both layers are derived from the `BaseTool` method signatures, docstrings, and optional hints. If generated artifacts exist in `prompts/generated/user_tools/`, runtime uses them first.

## 📋 User Guide

### 1. Launch the System

Run from the project root:

```bash
# Web runtime
uvicorn app:app --reload

# CLI runtime
python main.py
```

Web runtime:

- Open `http://127.0.0.1:8000`
- Complete the configuration form in the browser
- Scopebot will stream plan previews, execution updates, checker warnings, and final summaries
- For a lightweight mock demo flow that uses the same `config/runtime_config.json` settings but does not run the full planner/executor/checker stack, `uvicorn app_mock:app --reload` is also available

CLI runtime:

- Start `python main.py`
- Enter a natural-language instruction
- Scopebot will preview a plan, wait for confirmation or revision, then execute step by step

### 2. Runtime Flow

The current runtime flow is:

1. User enters a command.
2. The planner generates a structured plan.
3. Scopebot rewrites the plan into a short user-facing preview.
4. The user can confirm execution, cancel, or append revisions.
5. After confirmation, sub-agents generate code and execute each plan step.
6. After each sub-agent finishes, Scopebot emits a short completion summary.
7. If `model.checker_enabled=true` and new microscope images are produced, the checker validates them and may trigger auto-correction plus retry.
8. After the full task completes, Scopebot emits a short final summary.

### 3. Runtime Session Artifacts

Each system startup creates a new isolated runtime session under `te/`.

Example:

```plaintext
te/
└── run_20260318_170847_cd443080/
    ├── agent_interactions.json
    ├── meta.json
    └── output/
```

Important notes:

- Every startup is treated as a fresh session.
- The runtime does not reuse the previous session's `meta.json` or agent history.
- `agent_interactions.json` records planner / executor / checker interactions for the current run, rather than only storing generated code.

## 📂 Project Structure

```plaintext
llm_miscope/
├── app.py                          # FastAPI entry point for the Web UI
├── main.py                         # CLI entry point
├── front/                          # Web frontend
├── api/                            # FastAPI routes
├── services/                       # Runtime manager and task orchestration
├── user_skills/
│   └── planning/                   # User-provided planner skill documents
├── te/
│   └── run_*/                      # Per-startup runtime session folders
├── agent/                          # Core agent logic
│   ├── experiment_planner.py       # Task decomposition
│   ├── experiment_executor.py      # Code generation & execution
│   └── experiment_checker.py       # Validation & feedback
├── core_tool/                      # Built-in tools
│   ├── microscope.py               # MicroscopeController
│   ├── fiji.py                     # ImageJProcessor
│   ├── cellpose_tool.py            # Cellpose2D
│   └── tool_utils.py               # Utilities (e.g., sharpness metric)
├── tool/                           # Custom BaseTool extensions
│   ├── base.py                     # BaseTool class
│   └── ...                         # User-defined tools
├── prompts/                        # Planner, executor, and generated user-tool prompts
│   ├── generated/user_tools/       # executor_prompt.txt / planner_summary.txt
├── config/                         # Static prompt / task configuration
├── bootstrap/                      # Runtime configuration loading/saving
├── prompts/                        # Planner and executor prompts
│   ├── task_manager_full.py
│   └── ...
├── utils/                          # Runtime setup, logging, storage, sessions
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
