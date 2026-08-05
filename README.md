# Embodied Intelligence Enables Agentic Exploration in Microscopy

Embodied Intelligence Microscope System (EIMS) is a research platform for agentic
microscopy. It connects natural-language experimental intent with microscope control,
image analysis, segmentation, validation, and iterative correction.

EIMS is not only a hardware-control script. It is organized as a runtime system with a
Web interface, CLI entry point, subsystem runtime modes, tool registration,
planner skills, session history, and structured runtime configuration.

## Highlights

- Natural-language task planning in English or Chinese
- Confirm-before-execute workflow before microscope actions run
- Web runtime with configuration, initialization, preview, execution updates, and summaries
- CLI runtime for direct research and debugging workflows
- Micro-Manager demo mode for hardware-free runtime validation on the real MMCore path
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
  image quality checks, plan-trace checks, code repair routing, history
```

The main runtime path is managed by `services/runtime_manager.py` and
`services/task_orchestrator.py`. Tool environments are built in `runtime/tool_factory.py`
from the runtime configuration and `config/tool_manifest.json`.

## Architecture

```text
.
|-- app.py                         # FastAPI Web runtime
|-- app_mock.py                    # internal mock Web runtime helper for development
|-- main.py                        # CLI runtime
|-- api/                           # API routes and response models
|-- services/                      # runtime manager and task orchestration
|-- runtime/                       # runtime config, asset checks, tools, agents, context assembly
|-- agent/                         # planner, executor, image checker, plan checker, code repair, clarifier
|-- core_tool/                     # real microscope, Fiji, Cellpose tools
|-- tool/                          # user-defined BaseTool extensions
|-- tooling/                       # tool manifest and user-tool prompt generation
|-- storage/                       # session history and artifact metadata storage
|-- interfaces/                    # CLI interaction, logging, and local preview glue
|-- skill_runtime/                 # skill parsing and prompt formatting
|-- adapters/                      # tool registry and LLM client adapters
|-- bootstrap/                     # runtime configuration loading and saving
|-- config/                        # runtime example and tool manifest
|-- prompts/                       # planner and executor prompts
|-- user_skills/                   # planning skills
|-- docs/                          # paper materials, model tests, and supplementary scripts
|-- embedding_model/               # local download target for semantic retrieval model assets
|-- detector_models/               # per-detector configs and local checkpoints
`-- history/                       # per-run runtime history and outputs
```

## Runtime Modes

EIMS separates runtime mode by subsystem:

- `microscope_mode=demo|real`
- `image_analysis_mode=mock|real`
- `segmentation_mode=mock|real`

The default local development profile is:

- `microscope_mode=demo`
- `image_analysis_mode=mock`
- `segmentation_mode=mock`

This means the microscope still runs through the real Micro-Manager MMCore chain using the
built-in demo configuration, while Fiji and Cellpose can stay mocked until you need their
real runtimes.

Startup validation is mode-scoped: demo/mock subsystems are checked against their own
required assets and do not require real Fiji, Cellpose, or hardware-only dependencies.
The single validation entry point is `runtime.asset_check`; startup, config save, and
status APIs all derive readiness from `AssetCheckResult.ready`.

Micro-Manager demo mode uses the official `DemoCamera` adapter. Its `DStage`
focus device is limited by the adapter implementation to `-300..300` um, so
demo-mode Z limits and startup Z positions should stay within that range. This
range is not controlled by EIMS and cannot be expanded by editing
`runtime_config.json`; use real hardware mode or a custom Micro-Manager device
adapter if you need a larger virtual Z travel range.

EIMS demo mode relies on the original DemoCamera instance names
`DCam/DXYStage/DStage/DObjective/DStateDevice`. Do not rename those devices to
generic aliases inside the `.cfg`; on the current DemoCamera build, renaming
breaks `Fluorescent Beads` XY image coupling. In demo mode, `XY/Z/brightness`
come from Micro-Manager directly, while `objective/channel` image differences
are applied by EIMS software postprocessing.

Current `.cfg` directory roles:

- `demo_cfg/`: managed built-in demo configuration used by `microscope_mode=demo`
- `uploaded_cfg/`: runtime landing area for user-uploaded Micro-Manager `.cfg` files
- `docs/cfg/`: local reference or experimental `.cfg` files that are not part of the upload flow

## Installation

Run the setup and runtime commands from the repository root directory. If you just
cloned the project, enter the project directory first:

```bash
git clone https://github.com/ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy.git
cd Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy
```

### Requirements

- Windows 10/11 is recommended for real Micro-Manager hardware integration.
- Python `>=3.10,<3.11`
- `uv`
- Micro-Manager 2.0 for microscope `demo` and `real` modes
- Fiji/ImageJ for image-analysis `real` mode
- NVIDIA GPU with CUDA is recommended for Cellpose and MMDetection

### Install Dependencies

```bash
uv venv --python 3.10
powershell -ExecutionPolicy Bypass -File scripts/install_mmcv_with_fallback.ps1
```

The installer tries the official OpenMMLab `mmcv` wheel first and
automatically falls back to the project GitHub Release if needed.

### Download Model Weights

Detector weights are distributed through the `detector-weights` prerelease.

Restore the detector weights locally with:

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_detector_weights.ps1
```

This installs the current checkpoints to:

```text
detector_models/cell2d/weights.pth
detector_models/organoid/weights.pth
detector_models/mitosis/weights.pth
```

Only the detector `config.py` files are intended to live in the repository.
The `weights.pth` files are local runtime assets restored by the download script
and are ignored by git.

### Configure Micro-Manager

Install a compatible Micro-Manager build:

```bash
uv run python system_config_wizard.py --install-mmcore
```

If the default install destination contains `Micro-Manager*` directories, the installer
cleans them and reinstalls by default. To reuse the latest existing install, run:

```bash
uv run python system_config_wizard.py --install-mmcore --reuse-existing
```

Open the installed Micro-Manager GUI:

```bash
uv run python system_config_wizard.py --open-mmstudio
```

If you already have Micro-Manager installed, set `MM_DIR` and `system.CONFIG_PATH`
in `config/runtime_config.json`, or select the same `.cfg` through the Web
configuration page.

#### Configure EIMS To Match Micro-Manager

Do not rename labels in a working Micro-Manager `.cfg` just to match EIMS
defaults. Treat the `.cfg` as the hardware source of truth, then update
`config/runtime_config.json` so EIMS maps stable semantic keys to your existing
Micro-Manager labels.

EIMS uses stable semantic keys such as `"40x"`, `"brightfield"`, and `"fitc"`
in `runtime_config.json`, then injects the mapped real Micro-Manager labels into
the microscope prompt at runtime. The microscope controller executes those
confirmed real labels directly at execution time.

Example:

```json
{
  "system": {
    "objectives": {
      "40x": {
        "label": "Your-40x-Objective-Label",
        "magnification": 40,
        "display_name": "40x objective"
      }
    },
    "channels": {
      "brightfield": {
        "label": "Your-Brightfield-Preset",
        "display_name": "Brightfield",
        "color": [128, 128, 128],
        "illumination": "transmitted"
      },
      "fitc": {
        "label": "Your-FITC-Preset",
        "display_name": "FITC / 488 nm",
        "color": [0, 255, 0],
        "illumination": "fluorescence"
      }
    }
  },
  "startup": {
    "objective": "40x",
    "channel": "brightfield"
  }
}
```

If your transmitted-light illuminator exposes a device-specific intensity
property, configure it explicitly. In Micro-Manager demo mode, EIMS uses
`Camera.BeadBrightness` as a managed brightness surrogate so brightness changes
still affect the generated bead images. In real hardware mode,
`transmitted_light.device` and `transmitted_light.intensity_property` must point
to the real illuminator property; otherwise EIMS falls back to a no-brightness
microscope prompt.

```json
{
  "system": {
    "transmitted_light": {
      "device": "Your-Transmitted-Light-Device",
      "intensity_property": "Your-Intensity-Property",
      "min": 0,
      "max": 250
    }
  }
}
```

Use the Micro-Manager GUI or the Web configuration page to inspect the existing
device names and state labels, then copy those names into the EIMS mapping.

### Configure Fiji

Install or reuse Fiji:

```bash
uv run python system_config_wizard.py --setup-fiji
```

`--setup-fiji` reuses an existing local Fiji installation when possible. If none is
found, it downloads Fiji from the official `stable` channel, updates `FIJI_PATH`, and
validates the Fiji runtime.

To verify the Java and Fiji environment:

```bash
uv run python system_config_wizard.py --check-java
uv run python system_config_wizard.py --check-fiji
```

- `--check-java` verifies that Java/JDK is visible in the current terminal.
- `--check-fiji` initializes Fiji and reports missing optional Fiji capabilities or plugins.

To point at a specific Fiji install:

```bash
uv run python system_config_wizard.py --detect-fiji --fiji-dir "C:\Path\To\Fiji.app"
uv run python system_config_wizard.py --open-fiji
```

If you prefer to manage Fiji manually, you can still download it from:

<https://imagej.net/software/fiji/>

Notes:

- The helper installs or reuses Fiji itself, but it does not silently install third-party Fiji plugins.
- Some EIMS workflows require optional Fiji plugins such as DeconvolutionLab2 for Richardson-Lucy deconvolution.
- On Windows, the default automatic download location is typically:

```text
C:\Users\<YourUserName>\AppData\Local\EIMS\Fiji
```

### Configure Local Models

Download the local semantic similarity model:

```bash
uv run python scripts/setup_models.py
```

By default, EIMS uses:

```text
embedding_model/bge-m3
```

If the download helper dependency is missing, install it first and rerun the setup:

```bash
uv add huggingface_hub
uv run python scripts/setup_models.py
```

The model is downloaded from:

```text
https://huggingface.co/BAAI/bge-m3
```

Notes:

- `embedding_model/` is a local download directory, not a repository asset bundle.
- The semantic similarity model is expected under `embedding_model/`.
- The setup helper downloads only the files required by the current semantic similarity path.
- Optional ONNX/OpenVINO artifacts are skipped to reduce download size and timeout risk.

### Optional: Restore VLA ACT Assets

The `docs/VLA/ACT_for_microscopy/` asset bundle is distributed through the Hugging Face
repository [`404lzh/ACT_for_microscopy`](https://huggingface.co/404lzh/ACT_for_microscopy).
Download or clone that repository separately and place its contents under:

```text
docs/VLA/ACT_for_microscopy
```

### Real Hardware Checklist

Before the first real run on a machine:

- Confirm `config/runtime_config.json` contains valid local paths for this machine.
- Confirm `FIJI_PATH`, `MM_DIR`, model paths, and other local dependency paths exist and are accessible.
- If your workflow calls external APIs, confirm the required keys are configured in `.env`.
- If configuration or hardware state is uncertain, start with Micro-Manager demo mode instead of connecting to real hardware immediately.

Before each real microscope execution:

- Verify in the Micro-Manager GUI that the devices are controllable before starting EIMS automation.
- Confirm stage coordinate conventions, objective selection, illumination source, exposure settings, and Z-direction definitions are correct.
- If motion or acquisition behavior looks wrong, stop and check configuration, travel limits, and coordinate direction before retrying.
- If you suspect a travel-limit or collision risk, stop the automated workflow immediately and inspect the current and target positions manually in the GUI.

## Run EIMS

Before running EIMS for the first time:

1. Copy `config/runtime_config.example.json` to `config/runtime_config.json`
2. Copy `.env.example` to `.env`
3. Fill `base_url`, `model_name`, `vlm_base_url`, and `vlm_model_name` in `config/runtime_config.json`
4. Fill `EIMS_OPENAI_API_KEY` and `EIMS_VLM_API_KEY` in `.env`

Recommended `config/runtime_config.json` model settings:

```json
{
  "model": {
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4.1",
    "vlm_base_url": "https://api.openai.com/v1",
    "vlm_model_name": "gpt-4.1"
  }
}
```

Recommended `.env` settings:

```dotenv
EIMS_OPENAI_API_KEY=your-openai-compatible-api-key
EIMS_VLM_API_KEY=your-vlm-api-key
```

Then start EIMS in either Web mode or CLI mode. You do not need to run both.

### Run the Web Runtime

```bash
uv run uvicorn app:app --reload
```

The first browser open may be slower than usual while the backend finishes startup and
loads configuration. Please wait a moment and avoid repeated refreshes or duplicate clicks.

Then open:

```text
http://127.0.0.1:8000
```

### Run the CLI Runtime

```bash
uv run python main.py
```

## Hardware-Free Notebook

```text
Hardware-Free-Demo.ipynb
```

This notebook is a hardware-free conceptual demo. For current configuration and
execution steps, follow this README, `.env.example`,
`config/runtime_config.example.json`, and `system_config_wizard.py`.

## Configuration Reference

Configuration is intentionally layered:

1. `bootstrap/config.py` defines schema and safe defaults.
2. `config/runtime_config.json` stores local runtime settings written by the UI or helper scripts.
3. `.env` and process environment variables provide secrets and a small number of runtime-only switches.

`config/runtime_config.json` and the configuration page are the only source of truth for
hardware paths, device mappings, and the three subsystem mode fields.

The intended split is:

- `config/runtime_config.json`: hardware paths, device mappings, startup defaults, mode fields, and model endpoint settings
- `.env`: secrets and a small set of runtime-only switches such as `EIMS_OPENAI_API_KEY`, `EIMS_VLM_API_KEY`, optional `EIMS_SKILL_MODE`, and `EIMS_CHECKER_ENABLED`

`config/runtime_config.json` is ignored by git. Use
`config/runtime_config.example.json` as the template for a new machine.

Important environment variables:

```dotenv
EIMS_OPENAI_API_KEY=your-api-key
EIMS_VLM_API_KEY=your-vlm-api-key
EIMS_SKILL_MODE=disabled
EIMS_CHECKER_ENABLED=true
```

Field meanings:

- `base_url`: main LLM API endpoint
- `model_name`: main LLM model name
- `vlm_base_url`: vision-language model API endpoint
- `vlm_model_name`: vision-language model name

Detection target definitions are configured under `detection_targets`. Each target can
define its own model paths and confidence threshold:

```json
{
  "detection_targets": {
    "2Dcell": {
      "target_class_id": 0,
      "target_class_name": "2Dcell",
      "score_thr": 0.2,
      "output_filename": "2Dcell_locations_list.json",
      "model_config": "detector_models/cell2d/config.py",
      "model_checkpoint": "detector_models/cell2d/weights.pth"
    },
    "organoid": {
      "target_class_id": 0,
      "target_class_name": "organoid",
      "score_thr": 0.2,
      "output_filename": "organoid_locations_list.json",
      "model_config": "detector_models/organoid/config.py",
      "model_checkpoint": "detector_models/organoid/weights.pth"
    },
    "mitosis": {
      "target_class_id": 0,
      "target_class_name": "mitosis",
      "score_thr": 0.2,
      "output_filename": "mitosis_locations_list.json",
      "model_config": "detector_models/mitosis/config.py",
      "model_checkpoint": "detector_models/mitosis/weights.pth"
    }
  }
}
```

The `model_checkpoint` path is a local runtime path. Large model weights such as
`detector_models/cell2d/weights.pth` are distributed through GitHub Releases. Download the release asset
and place it at the path referenced by `model_checkpoint`.

## Runtime History

Each startup creates a new session under `history/`.

```text
history/
`-- run_YYYYMMDD_HHMMSS_xxxxxxxx/
    |-- agent_interactions.json
    |-- meta.json
    `-- output/
```

The session records generated plans, generated executor code, execution results, image
checker feedback, plan-trace/code-repair diagnostics, registered output files, and cache
metadata.

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

### Fiji Capability Declarations

Fiji-backed tool methods that depend on optional Fiji plugins should declare those
dependencies next to the method implementation in `core_tool/fiji.py`. The declaration
is used by `--check-fiji` and is also checked again at runtime before the method runs.

```python
@tool_func
@requires_fiji_capability(
    id="plugin_id",
    label="Plugin Display Name",
    required_for="short workflow description",
    command="ImageJ Command Name",
    # or: java_class="plugin.package.ClassName",
    install_hint="Install this plugin in Fiji, then restart EIMS.",
)
def plugin_dependent_method(...):
    ...
```

## Safety Notes

Real microscope operation can damage samples or hardware if configuration is wrong.
Before running in real hardware mode:

- verify the Micro-Manager `.cfg` in the official Micro-Manager GUI
- configure objective, XY, Z, brightness, and exposure limits
- confirm stage coordinate conventions
- test low-risk movement commands first
- keep emergency stop procedures available
- validate Fiji and model checkpoint paths

Generated code execution is constrained, but it should still be treated as experimental
automation. Use `microscope_mode=demo` for new workflows before running on hardware.

## Acknowledgements

EIMS builds on a broad open-source scientific software ecosystem. We gratefully
acknowledge the developers and maintainers of
[Micro-Manager](https://micro-manager.org/),
[Fiji/ImageJ](https://imagej.net/software/fiji/),
[pyimagej](https://github.com/imagej/pyimagej),
[pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus),
[Cellpose](https://www.cellpose.org/),
[OpenMMLab/MMDetection](https://github.com/open-mmlab/mmdetection),
[PyTorch](https://pytorch.org/), and the Python scientific-computing libraries
that support this project.

These tools make reproducible microscope control, image processing, model
inference, and web-based scientific workflows possible. Please refer to the
respective upstream projects for their documentation, licenses, and citation
requirements.

## Licensing Notes

Unless otherwise noted, the original source code developed specifically for EIMS
is made available under the BSD 3-Clause License. See
[LICENSE.BSD-3-Clause](LICENSE.BSD-3-Clause). The broader project-level licensing
context, including combined distributions with external dependencies, is described
in [LICENSE](LICENSE).

Some runtime workflows depend on third-party software, datasets, models, or
plugins that are licensed separately from EIMS. In particular:

- Fiji/ImageJ and its plugins are subject to their own upstream licenses.
- Micro-Manager is subject to its own upstream license.
- Individual model weights, detector checkpoints, and external tools referenced by
  EIMS may have additional license terms or redistribution restrictions.

This repository's license does not replace or override the licenses of those
third-party components. If you download, bundle, redistribute, or deploy EIMS
together with external dependencies, you are responsible for reviewing and
complying with the applicable upstream license terms.

## Contributions

Contributions are welcome. You can help by reporting bugs, fixing issues,
improving documentation, adding test tasks, refining planner skills, or extending
tool integrations.

For changes that affect real hardware control, image-analysis behavior,
generated-code execution, or Fiji plugin-dependent features, please include the
relevant validation steps and dependency declarations with the contribution.
