# Embodied Intelligence Enables Agentic Exploration in Microscopy

Embodied Intelligence Microscope System (EIMS) is a research platform for agentic
microscopy. It connects natural-language experimental intent with microscope control,
image analysis, segmentation, validation, and iterative correction.

EIMS is not only a hardware-control script. It is organized as a runtime system with a
Web interface, CLI entry point, simulation mode, real-hardware mode, tool registration,
planner skills, session history, and structured runtime configuration.

## Highlights

- Natural-language task planning in English or Chinese
- Confirm-before-execute workflow before microscope actions run
- Web runtime with configuration, initialization, preview, execution updates, and summaries
- CLI runtime for direct research and debugging workflows
- Hardware-free simulation through `Empty_function.py`
- Real microscope runtime through Micro-Manager and `pymmcore-plus`
- Fiji/ImageJ integration for image processing and analysis
- Cellpose and MMDetection integration for segmentation and target detection
- Session-isolated history under `history/run_*`
- User extension tools through `tool/`, `BaseTool`, and `config/tool_manifest.json`
- Planner guidance through files under `user_skills/planning/`

## How It Works

```text
User intent
  |
  v
Planner
  natural language -> structured task plan
  |
  v
User confirmation
  approve, cancel, or revise the plan
  |
  v
Executors
  task step -> constrained Python code for a specific tool
  |
  v
Tool platforms
  microscope, Fiji/ImageJ, Cellpose, MMDetection, user tools
  |
  v
Checker and history
  validate outputs, record artifacts, support correction
```

The main runtime path is managed by `services/runtime_manager.py` and
`services/task_orchestrator.py`. Tool environments are built in `utils/runtime_factory.py`
from the runtime configuration and `config/tool_manifest.json`.

## Architecture

```text
.
|-- app.py                         # FastAPI Web runtime
|-- app_mock.py                    # lightweight mock Web runtime
|-- main.py                        # CLI runtime
|-- api/                           # API routes and response models
|-- services/                      # runtime manager and task orchestration
|-- agent/                         # planner, executor, checker, clarifier
|-- core_tool/                     # real microscope, Fiji, Cellpose tools
|-- tool/                          # user-defined BaseTool extensions
|-- adapters/                      # tool registry and LLM client adapters
|-- bootstrap/                     # runtime configuration loading and saving
|-- config/                        # runtime example, tool manifest, static task config
|-- prompts/                       # planner and executor prompts
|-- user_skills/                   # planning skills
|-- docs/test_tasks/               # representative task prompts
|-- evaluation/                    # model evaluation helpers
|-- weights/                       # model checkpoints
`-- history/                       # per-run runtime history and outputs
```

## Runtime Modes

EIMS has two runtime modes:

- `Simulation_mode=true`: uses the hardware-free mock tool chain in `Empty_function.py`.
- `Simulation_mode=false`: uses real tools under `core_tool/`.

Use simulation mode when developing prompts, tools, and task plans. Switch to real mode
only after Micro-Manager, hardware limits, Fiji, model paths, and API credentials are
configured.

## Configuration Model

Configuration is intentionally layered:

1. `bootstrap/config.py` defines schema and safe defaults.
2. `config/runtime_config.json` stores local runtime settings written by the UI or helper scripts.
3. `.env` and process environment variables provide runtime-only overrides, especially secrets.

Environment variable overrides are applied when the runtime loads settings, but they are not
written back into `config/runtime_config.json` when settings are saved.

`config/runtime_config.json` is ignored by git. Use
`config/runtime_config.example.json` as the template for a new machine.

Important environment variables:

```dotenv
EIMS_OPENAI_API_KEY=your-api-key
EIMS_BASE_URL=https://api.openai.com/v1
EIMS_MODEL_NAME=gpt-4.1
EIMS_VLM_API_KEY=your-vlm-api-key
EIMS_VLM_BASE_URL=https://api.openai.com/v1
EIMS_VLM_MODEL_NAME=gpt-4.1
EIMS_SIMULATION_MODE=true
EIMS_CHECKER_ENABLED=true
```

Detection target definitions are configured under `detection_targets`. Each target can
define its own model paths and confidence threshold:

```json
{
  "detection_targets": {
    "organoid": {
      "target_class_id": 0,
      "target_class_name": "organoid",
      "score_thr": 0.2,
      "output_filename": "organoid_locations_list.json",
      "model_config": "configs/organoid.py",
      "model_checkpoint": "weights/organoid.pth"
    }
  }
}
```

## Quick Start

### Requirements

- Windows 10/11 is recommended for real Micro-Manager hardware integration.
- Python `>=3.10,<3.11`
- `uv`
- Micro-Manager 2.0 for real hardware mode
- Fiji/ImageJ for real image-analysis mode
- NVIDIA GPU with CUDA is recommended for Cellpose and MMDetection

### Install Dependencies

```bash
uv venv --python 3.10
uv sync
```

Useful validation commands:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
uv run python -c "import mmcv, mmengine, mmdet; print(mmcv.__version__, mmengine.__version__, mmdet.__version__)"
uv run python -c "import cellpose; print(cellpose.__version__)"
```

### Configure Micro-Manager

Install a compatible Micro-Manager build:

```bash
uv run python system_config_wizard.py --install-mmcore
```

If the default install destination already contains `Micro-Manager*` directories,
the installer now cleans them and reinstalls by default. To reuse the latest existing
install instead, run:

```bash
uv run python system_config_wizard.py --install-mmcore --reuse-existing
```

Open the installed Micro-Manager GUI:

```bash
uv run python system_config_wizard.py --open-mmstudio
```

If you already have Micro-Manager installed, set `MM_DIR` in
`config/runtime_config.json` or through the Web configuration page.

### Configure Fiji

Install Fiji manually from <https://imagej.net/software/fiji/>, then let the helper
detect it, update `FIJI_PATH`, and validate the Java/pyimagej stack:

```bash
uv run python system_config_wizard.py --setup-fiji
uv run python system_config_wizard.py --check-java
uv run python system_config_wizard.py --check-fiji
```

To point at a specific Fiji install:

```bash
uv run python system_config_wizard.py --detect-fiji --fiji-dir "C:\Path\To\Fiji.app"
uv run python system_config_wizard.py --open-fiji
```

Fiji-backed processing requires a Java/JDK visible in the same terminal. The helper
does not install Java; it reports whether `java -version`, JPype, Maven/scyjava, and
pyimagej initialization are ready.

### Run the Web Runtime

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

For a lighter mock UI flow:

```bash
uvicorn app_mock:app --reload
```

### Run the CLI Runtime

```bash
python main.py
```

### Hardware-Free Notebook

```text
Hardware-Free-Demo.ipynb
```

## Runtime History

Each startup creates a new session under `history/`.

```text
history/
`-- run_YYYYMMDD_HHMMSS_xxxxxxxx/
    |-- agent_interactions.json
    |-- meta.json
    `-- output/
```

The session records generated plans, generated executor code, execution results, checker
feedback, registered output files, and cache metadata.

## Extending EIMS

### Planner Skills

Planner skills live under `user_skills/planning/`. They guide task decomposition without
changing runtime code.

Supported formats:

- `.md`
- `.txt`
- `.json`
- directories containing `SKILL.md`

Example:

```md
---
name: Brightfield Tracking Workflow
description: Preferred planning pattern for brightfield tracking
tags: brightfield, tracking, autofocus
triggers: brightfield time-lapse, mitosis tracking
priority: 3
---

- Start with a low-exposure brightfield preview.
- Confirm focus before repeated acquisition.
- Reuse detected positions when revisiting targets.
```

### User Tools

User tools inherit from `tool.base.BaseTool` and expose public methods decorated with
`@tool_func`.

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
        """Process text and return a short result."""
        return f"processed: {text}"
```

Register a tool:

```bash
python create_tool.py register --class-path tool.new_tool:NewTool --tool-id "new_tool" --dry-run
python create_tool.py register --class-path tool.new_tool:NewTool --tool-id "new_tool"
python create_tool.py list
```

## Example Task Types

Representative task prompts live in `docs/test_tasks/task.txt`.

- capture multi-channel fluorescence images and merge channels
- scan a brightfield area, detect target regions, and revisit them at higher magnification
- segment cells with Cellpose and export masks or statistics
- perform long-running time-lapse imaging
- acquire Z-stacks and produce projected or deconvolved images
- detect organoids, lesions, bacteria, cells, or mitotic events and record coordinates

## Safety Notes

Real microscope operation can damage samples or hardware if configuration is wrong.
Before running with `Simulation_mode=false`:

- verify the Micro-Manager `.cfg` in the official Micro-Manager GUI
- configure objective, XY, Z, brightness, and exposure limits
- confirm stage coordinate conventions
- test low-risk movement commands first
- keep emergency stop procedures available
- validate Fiji and model checkpoint paths

Generated code execution is constrained, but it should still be treated as experimental
automation. Use simulation mode for new workflows before running on hardware.

## Licensing Notes

This project calls public APIs of external tools such as Fiji/ImageJ and Micro-Manager.

- Fiji is distributed under GPL-related licensing, with ImageJ2 components under BSD-style
  licenses. Individual plugins may have their own licenses.
- Micro-Manager is an open-source project under a BSD-style license.

Check upstream licenses for external distribution or commercial deployment.

## Contributions

Issues, pull requests, tool integrations, planning skills, test tasks, and documentation
improvements are welcome.
