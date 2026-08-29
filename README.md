# EIMS

**Embodied Intelligence Enables Agentic Exploration in Microscopy**

<p align="left">
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Micro-Manager 2.0" src="https://img.shields.io/badge/Micro--Manager-2.0-2F6F8F">
  <img alt="Combined distribution: GPLv3" src="https://img.shields.io/badge/Combined%20distribution-GPLv3-A42E2B">
  <img alt="Original EIMS code: BSD 3-Clause" src="https://img.shields.io/badge/Original%20EIMS%20code-BSD%203--Clause-4C8C4A">
</p>

EIMS is a research platform that connects natural-language experimental intent with
microscope control, image analysis, segmentation, validation, and iterative correction.
It provides a Web interface and a CLI, supports a simulated microscope, Micro-Manager
demo devices, and real hardware, and keeps every run in an isolated history directory
for inspection.

> EIMS is experimental research software. Validate new workflows in Demo mode before
> using real hardware, and always confirm generated plans before execution.

[Quick Start](#quick-start) · [Capabilities](#what-eims-can-do) · [Runtime Modes](#runtime-modes) ·
[Real Microscope Setup](#real-microscope-setup) · [Configuration](#configuration-reference) ·
[Public Materials](docs_public/README.md) · [Extending EIMS](#extending-eims)

## ✦ At a Glance

| Area | What EIMS provides |
| --- | --- |
| Interaction | Natural-language planning in English or Chinese, with confirmation before execution |
| Microscope | Pure simulation, Micro-Manager DemoCamera validation, and `pymmcore-plus` real-hardware control |
| Analysis | Fiji/ImageJ integration, Cellpose segmentation, and MMDetection target detection |
| Reliability | Image-quality checks, plan-trace checks, code-repair routing, and mode-scoped startup validation |
| Interfaces | Web configuration, initialization, preview, live execution updates, summaries, and a CLI for research and debugging |
| Reproducibility | Session-isolated plans, code, results, artifacts, diagnostics, and cache metadata |
| Extensibility | Planner skills under `user_skills/planning/` and registered `BaseTool` extensions under `tool/` |

<a id="what-eims-can-do"></a>

## What EIMS Can Do

EIMS turns natural-language microscopy requests into confirmed, traceable experimental
workflows. Depending on the configured runtime modes and available hardware, it can:

- Configure objectives, channels, exposure, illumination, autofocus, and XY/Z stage motion.
- Acquire single images, stitched regions, Z-stacks, time series, and plate-style scans.
- Route acquired images to Fiji/ImageJ, Cellpose, or MMDetection for processing,
  segmentation, and target detection.
- Revisit detected targets, run closed-loop quality checks, and trigger replanning or code
  repair after failures.
- Review recorded baseline workflows in `docs_public/` for FRAP or MP-285A comparisons.
- Preserve plans, generated code, artifacts, diagnostics, and metadata for later inspection.

## 🔬 How EIMS Works

```text
Natural-language intent
          |
          v
       Planner              converts intent into a structured task plan
          |
          v
 User confirmation          approve, cancel, inspect, or revise
          |
          v
      Executors             generate constrained code for each selected tool
          |
          v
   Tool platforms           microscope, Fiji, Cellpose, MMDetection, user tools
          |
          v
 Checker and history        validate results, route repairs, and record the run
```

The main runtime path is managed by `services/runtime_manager.py` and
`services/task_orchestrator.py`. Tool environments are assembled by
`runtime/tool_factory.py` from the runtime configuration and
`config/tool_manifest.json`.

## 🧭 Choose a Starting Path

| Goal | Recommended path |
| --- | --- |
| Try the Web application without microscope hardware | Use the default `demo / mock / mock` profile and follow [Quick Start](#quick-start) |
| Use the pure simulated microscope | Select `mock` for Microscope Mode; persisted system and startup values remain active |
| Connect a physical microscope | Complete Quick Start, then follow [Real Microscope Setup](#real-microscope-setup) |
| Work directly from a terminal | Start with `uv run python main.py` after configuration |
| Inspect the conceptual hardware-free example | Open `Hardware-Free-Demo.ipynb` |
| Browse public datasets and qualitative examples | Open [docs_public/README.md](docs_public/README.md) |
| Add planner skills or tool integrations | See [Extending EIMS](#extending-eims) |

<a id="quick-start"></a>

## 🚀 Quick Start

The commands below are intended to be run from the repository root. Windows 10/11 is
recommended for real Micro-Manager integration.

### 1. Clone the Repository

```bash
git clone https://github.com/ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy.git
cd Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy
```

### 2. Check the Requirements

- Python `>=3.10,<3.11`
- [`uv`](https://docs.astral.sh/uv/)
- Micro-Manager 2.0 for microscope `demo` and `real` modes; it is not required for `mock`
- Fiji/ImageJ for image-analysis `real` mode
- An NVIDIA GPU with CUDA is recommended for Cellpose and MMDetection

The default Quick Start profile uses the Micro-Manager DemoCamera with mocked image
analysis and segmentation. It does not require physical microscope hardware,
Fiji/ImageJ, detector checkpoint files, Cellpose, or a CUDA-capable GPU at runtime.
The standard environment setup may still install CUDA/PyTorch and MMDetection
dependencies. Install the other mode-dependent assets only when enabling real
image-analysis, segmentation, or workflows that need them; see [Mode-dependent
Components](#mode-dependent-components).
Demo mode still requires a working local Micro-Manager installation with `MM_DIR`
pointing to it. EIMS uses the managed `demo_cfg/MMConfig_demo.cfg` for `CONFIG_PATH`;
no physical microscope or hardware-specific `.cfg` is required.

### 3. Install Python Dependencies

```bash
uv venv --python 3.10
powershell -ExecutionPolicy Bypass -File scripts/install_mmcv_with_fallback.ps1
```

The installer first tries the official OpenMMLab `mmcv` wheel and automatically falls
back to the project GitHub Release when needed.
`uv venv` creates the project environment under `.venv/` by default, and the installer
runs `uv sync --frozen` from the repository root before installing `mmcv` and `mmdet`
into that environment. If the GitHub Release fallback is needed, the downloaded wheel is
cached under `.runtime/downloads/` before installation.

### 4. Create Local Configuration

```powershell
Copy-Item config/runtime_config.example.json config/runtime_config.json
Copy-Item .env.example .env
```

Set the main LLM and VLM endpoints in `config/runtime_config.json`:

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

Set the corresponding secrets in `.env`:

```dotenv
EIMS_OPENAI_API_KEY=your-openai-compatible-api-key
EIMS_VLM_API_KEY=your-vlm-api-key
```

The configuration page can test the current LLM and VLM values before the system is
started. These tests do not require saving the form first.

### 5. Install Micro-Manager

Install or reuse a compatible Micro-Manager build:

```powershell
uv run python system_config_wizard.py --install-mmcore --reuse-existing
```

This reuses the most recently modified existing local installation when available. If no
existing installation is available, it installs a new compatible build.

For a clean reinstall, use the default command:

```powershell
uv run python system_config_wizard.py --install-mmcore
```

The default command removes existing `Micro-Manager*` directories in the destination
before reinstalling. Use it only when a clean reinstall is intended.

By default, `--install-mmcore` uses `%LOCALAPPDATA%\EIMS\Micro-Manager` as the parent
installation directory and writes the detected Micro-Manager install root back to
`config/runtime_config.json` as `MM_DIR`. Use `--mmcore-dest` to choose a different parent
directory, or `--skip-config-update` to install without updating the runtime config.

For Micro-Manager's own interface, hardware configuration, and acquisition workflow, see
the official [Micro-Manager 2.0 User Guide](https://micro-manager.org/Version_2.0_Users_Guide)
and [documentation overview](https://micro-manager.org/Overview_of_the_documentation).

For a step-by-step walkthrough of Micro-Manager installation and real-hardware
configuration, refer to Supplementary Video 7.

### 6. Start the Web Runtime

```bash
uv run uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>.

The first browser load may be slower while the backend completes startup and reads the
configuration. Wait for it to finish instead of repeatedly refreshing or submitting the
same action.

In the configuration page:

1. Configure and test the Main LLM and Vision Model under **AI Services**.
2. Keep `Microscope Mode` set to **Demo** for the first run.
3. Review **Micro-Manager System** and **Startup Configuration**.
4. Click **Save Configuration**, then click **Start System**.

`Save Configuration` is the only Web UI action that writes `config/runtime_config.json`.

### CLI Alternative

The Web runtime and CLI runtime are alternatives; they do not both need to be started.

```bash
uv run python main.py
```

<a id="runtime-modes"></a>

## ⚙️ Runtime Modes

EIMS selects the runtime mode of each subsystem independently:

| Setting | Values | Controls |
| --- | --- | --- |
| `microscope_mode` | `demo`, `mock`, `real` | Managed DemoCamera devices, the simulated microscope, or the configured real microscope |
| `image_analysis_mode` | `mock`, `real` | Mock processing or the real Fiji/ImageJ runtime |
| `segmentation_mode` | `mock`, `real` | Mock segmentation or real Cellpose/model-backed segmentation |

The default development profile is:

```text
microscope_mode=demo
image_analysis_mode=mock
segmentation_mode=mock
```

The microscope still uses the real Micro-Manager MMCore chain in Demo mode, but with the
built-in DemoCamera configuration. Fiji and Cellpose may remain mocked until their real
runtimes are needed.

Mock microscope mode uses `simulation/microscope.py` with built-in virtual objectives and
channels, so it does not require a Micro-Manager cfg, device names, or physical state
labels. It preserves the operating limits and startup values saved in
`config/runtime_config.json` without applying a managed overlay. External Micro-Manager
cfg import is available only in Real mode.

Startup validation is mode-scoped. Demo and mock subsystems are checked only against the
assets they actually require and do not require real Fiji, Cellpose, or hardware-only
dependencies. `runtime.asset_check` is the single validation entry point; startup,
configuration-save, and status APIs all derive readiness from
`AssetCheckResult.ready`.

### Micro-Manager Demo Mode

Micro-Manager Demo mode uses the official `DemoCamera` adapter. Its `DStage` focus device
is limited by the adapter implementation to `-300..300` µm. Demo-mode Z limits and startup
Z positions must remain inside that range. Editing `runtime_config.json` cannot extend the
adapter's travel; use real hardware mode or a custom Micro-Manager device adapter for a
larger virtual Z range.

EIMS relies on the original DemoCamera instance names:

```text
DCam / DXYStage / DStage / DObjective / DStateDevice
```

The managed cfg also retains the original `DLightPath` and `DShutter` instances. They are
cfg devices, not additional public EIMS role fields that users must configure.

External cfg import is disabled in Demo and Mock modes in both the Web UI and backend.
Switch to Real mode before importing a hardware cfg.

Do not rename those devices to generic aliases in the demo `.cfg`. With the current
DemoCamera build, renaming them breaks the `Fluorescent Beads` XY image coupling. In Demo
mode, XY position, Z position, and brightness come from Micro-Manager directly;
objective- and channel-dependent image differences are applied by EIMS postprocessing.
Demo brightness is implemented through the managed `DCam.BeadBrightness` property.

The cfg directories have distinct roles:

| Directory | Purpose |
| --- | --- |
| `demo_cfg/` | Managed built-in configuration used by `microscope_mode=demo` |
| `uploaded_cfg/` | Runtime landing area for user-uploaded Micro-Manager `.cfg` files |

<a id="real-microscope-setup"></a>

## 🔌 Real Microscope Setup

Real hardware should first work correctly in the official Micro-Manager GUI. EIMS then
maps its stable semantic names onto that known-good configuration.

### 1. Validate Micro-Manager Independently

Open the configured Micro-Manager GUI:

```bash
uv run python system_config_wizard.py --open-mmstudio
```

If Micro-Manager is already installed, set `MM_DIR` and `system.CONFIG_PATH` in
`config/runtime_config.json`, or select the same `.cfg` from the Web configuration page.
Confirm in Micro-Manager that the camera, XY stage, focus drive, objectives, channels,
illumination, and basic acquisition work before involving EIMS.

### 2. Configure EIMS to Match Micro-Manager

Do not rename labels in an existing working Micro-Manager `.cfg` to match EIMS defaults.
Import the validated `.cfg` in Real mode. EIMS parses it and generates an editable
mapping draft; review the mappings and save the confirmed values to
`config/runtime_config.json`.

EIMS uses stable semantic keys such as `"40x"`, `"brightfield"`, and `"fitc"`. At runtime,
it injects the mapped real Micro-Manager labels into the microscope prompt and executes
the user-confirmed plan with those real labels.

```text
Working Micro-Manager cfg
          |
          v
cfg parser and inventory  device names, Core roles, state labels, properties
          |
          v
Optional MMCore inspection runtime property metadata when requested
          |
          v
Rule mapping              fills unambiguous bindings and semantic labels
          |
          v
Optional AI recommendation corrects uncertain cfg-backed values
          |
          v
Editable Web form         user reviews or changes the mapping draft
          |
          v
Save Configuration        writes the confirmed values to runtime_config.json
```

The following is a partial example of objective and channel mappings:

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

### 3. Review the cfg Mapping

The Web configuration page provides an editable cfg import workflow for real microscopes.
AI recommendations are optional; without a configured Main LLM, EIMS can still parse the
cfg and produce a rule-based mapping draft:

1. Switch `Microscope Mode` to **Real**.
2. Configure the Main LLM under **AI Services** if AI recommendations are desired, and use
   **Test LLM** to verify it.
3. Choose a copy of the already validated `.cfg`.
4. Click **Import, Inspect & AI Map**, confirm the hardware connection, and wait for the
   inspection indicator to finish.
5. Review the populated device, objective, channel, and illumination fields.
6. Change any incorrect value directly or select another cfg-backed candidate.
7. Click **Save Configuration** only after the mappings are correct.
8. Start the system separately when ready.

The import has deliberately narrow behavior:

- The current unsaved Main LLM key, endpoint, and model values in the form are used for
  this request only; importing does not persist them. If they are not configured, the
  draft uses deterministic cfg parsing and rule-based mapping.
- EIMS parses the `.cfg` locally, then loads the uploaded copy in an isolated MMCore
  instance to inspect the Device Adapters. This initializes those adapters and may cause
  vendor-defined hardware activity, so the Web UI requires explicit confirmation. An
  active EIMS microscope runtime is safely released before inspection.
- A copy of the uploaded file is retained under `uploaded_cfg/`; the source `.cfg` is
  never modified.
- A Micro-Manager cfg does not necessarily persist every property exposed by a loaded
  Device Adapter. The import inventory contains only properties actually declared in the
  cfg; EIMS does not invent missing adapter properties.
- Only structured device names, Core roles, state labels, Property metadata, current EIMS
  mappings, and rule candidates are sent to the configured Main LLM. Runtime Property
  metadata includes names, types, read-only/PreInit flags, allowed values, and limits;
  current hardware Property values are not read or sent.
- Raw cfg text, local paths, cfg comments, API settings, and unrelated runtime
  configuration are not sent to the model.
- Unambiguous Core bindings and clear semantic labels are filled by deterministic rules.
- For uncertain mappings, the LLM can select any legal value present in the structured
  cfg inventory, including a value outside a weak rule candidate set.
- AI-selected values are marked with confidence and a short reason. Review warnings and
  low-confidence fields before saving.
- If the model returns a device, label, or property outside the parsed inventory, the
  backend rejects that recommendation and keeps the validated parser draft.
- Import and inspection only update the Web form. They do not apply EIMS startup
  positions, acquire images, change modes, modify the original `.cfg`, or write
  `runtime_config.json`. Device Adapter initialization is limited to the explicit
  inspection operation and the temporary MMCore instance is unloaded afterward.
- **Save Configuration** remains the only Web UI action that persists the form to
  `runtime_config.json`.

The import workflow is available only in Real mode. Demo mode uses the managed cfg and
known device mapping, while Mock mode does not initialize external Micro-Manager hardware.
The backend rejects external cfg imports in both Demo and Mock modes.

#### CLI: ai-map

The same cfg parsing and mapping workflow is available from the terminal through the `ai-map`
subcommand of `system_config_wizard.py`:

```powershell
# Preview a mapping draft (nothing is written):
uv run python system_config_wizard.py ai-map --cfg /path/to/your.cfg

# Inspect Device Adapters first and save the draft JSON:
uv run python system_config_wizard.py ai-map --cfg /path/to/your.cfg --inspect --output mapping.json

# Write the confirmed mappings to config/runtime_config.json (asks y/N unless --yes):
uv run python system_config_wizard.py ai-map --cfg /path/to/your.cfg --apply
```

- `--cfg` selects the Micro-Manager `.cfg`; it defaults to `system.CONFIG_PATH` in
  the runtime config.
- `--config` points at an alternative runtime config file to read from and (with
  `--apply`) write to; it defaults to `config/runtime_config.json`.
- `--inspect` loads the cfg in an isolated MMCore to collect real Device Adapter
  property metadata before the AI analysis (initializes adapters, like the Web UI).
- Without `--apply` the command is a dry run. With `--apply` it writes only after a
  y/N confirmation (or `--yes`).
- The CLI asks for explicit consent before sending the structured inventory to the
  configured Main LLM; if no API key/model is configured, it falls back to the
  rule-based draft without calling the LLM.
- AI output is a recommendation. Review `REVIEW`-marked, low-confidence, and
  `manual_required` fields before applying.
- Use `ai-map` for a validated real-hardware cfg; Demo mode uses the managed cfg and does
  not accept external cfg mapping.

The mapping draft is a recommendation, not hardware verification. Test the resulting
mapping with low-risk operations before automated acquisition.

### 4. Configure Transmitted-Light Control When Needed

A transmitted-light device identifies the Micro-Manager device responsible for
illumination. Its intensity property identifies the device-specific property EIMS may set
to control brightness. Both fields are optional in real hardware mode because not every
illuminator exposes a writable intensity property.

If the cfg declares a property such as `Brightness`, `Intensity`, or `Power`, the parser
uses it as an initial candidate. Hardware inspection then calls
`getDevicePropertyNames()` on every loaded Device Adapter and collects type, read-only,
PreInit, allowed-value, and limit metadata. Only writable, runtime-settable numeric
properties can become automatic intensity controls. A unique strong candidate is filled
directly; multiple candidates are ranked by AI and remain editable.

The inspection draft is available before the first save. At startup, EIMS validates the
saved property against the loaded Micro-Manager device adapter. If a configured device
has no `intensity_property`, initialization fails and the mapping must be completed in
the configuration workflow. For real microscopes with transmitted illumination, configure
this mapping before fluorescence workflows; EIMS turns transmitted light off when
switching away from brightfield and will skip that step with a warning when no software
control is available, so the operator should turn off a manual brightfield illuminator
before fluorescence imaging.

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

In Demo mode, EIMS uses `DCam.BeadBrightness` as a managed brightness surrogate so
brightness changes remain visible in generated bead images.

### 5. Complete the Real-Hardware Checklist

Before the first real run on a machine:

- Confirm that `MM_DIR` and `CONFIG_PATH` in `config/runtime_config.json` point to the
  intended Micro-Manager installation and validated real-hardware cfg.
- If `image_analysis_mode=real`, confirm that `FIJI_PATH`, detector configuration files,
  detector checkpoints, and their dependencies exist and are accessible. See
  [Detector Weights](#detector-weights) for the checkpoint download instructions.
- If `segmentation_mode=real`, confirm that the Cellpose runtime and its required
  dependencies are available.
- Confirm required API keys in `.env` when the workflow calls external APIs.
- Use Micro-Manager Demo mode first whenever hardware or configuration state is uncertain.

Before each real microscope execution:

- Verify in the official Micro-Manager GUI that every required device is controllable.
- Confirm stage coordinate conventions, objective selection, illumination source,
  exposure settings, and Z-direction definitions.
- Test low-risk motion and acquisition commands before a full automated workflow.
- Stop and inspect the current and target positions manually if behavior is wrong or a
  travel-limit or collision risk is suspected.
- Keep the laboratory's emergency-stop procedure available.

<a id="mode-dependent-components"></a>

## 🧩 Mode-dependent Components

These components are optional only while their corresponding subsystem is mocked. Fiji
and configured detector model files are required when `image_analysis_mode=real`;
model-backed segmentation assets are required when `segmentation_mode=real`.

### Detector Weights

Detector weights are distributed through the
[`detector-weights` GitHub Release](https://github.com/ICS-MR/Embodied-Intelligence-Enables-Agentic-Exploration-in-Microscopy/releases/tag/detector-weights).
Restore them locally with:

```bash
powershell -ExecutionPolicy Bypass -File scripts/download_detector_weights.ps1
```

The script installs:

```text
detector_models/cell2d/weights.pth
detector_models/organoid/weights.pth
detector_models/mitosis/weights.pth
detector_models/cell2d_brightfield/weights.pth
detector_models/organoid_fluorescence/weights.pth
```

Downloads are staged under `.runtime/downloads/detector-weights/` and then copied into
the final `detector_models/` paths above. Use the script's `-TargetRoot` parameter only if
you intentionally want the final checkpoint files somewhere else.

Only detector `config.py` files are intended to live in the repository. Checkpoint
`weights.pth` files are local runtime assets, restored by the download script and ignored
by git.

### Fiji / ImageJ

Install or reuse Fiji:

```bash
uv run python system_config_wizard.py --setup-fiji
```

`--setup-fiji` reuses an existing installation when possible. Otherwise it downloads
Fiji from the official `stable` channel, updates `FIJI_PATH`, and validates the runtime.

Check Java and Fiji independently:

```bash
uv run python system_config_wizard.py --check-java
uv run python system_config_wizard.py --check-fiji
```

- `--check-java` verifies that Java/JDK is visible in the current terminal.
- `--check-fiji` initializes Fiji and reports missing optional capabilities or plugins.

To point EIMS at a specific Fiji installation or open it:

```bash
uv run python system_config_wizard.py --detect-fiji --fiji-dir "C:\Path\To\Fiji.app"
uv run python system_config_wizard.py --open-fiji
```

Fiji can also be installed manually from <https://imagej.net/software/fiji/>.

The helper installs or reuses Fiji itself, but it does not silently install third-party
plugins. Some EIMS workflows require optional plugins such as DeconvolutionLab2 for
Richardson-Lucy deconvolution. On Windows, the default automatic download location is
typically:

```text
C:\Users\<YourUserName>\AppData\Local\EIMS\Fiji
```

The wizard downloads and extracts Fiji under that parent directory, then records the final
Fiji root in `config/runtime_config.json` as `FIJI_PATH` unless `--skip-config-update` is
used.

### Local Semantic Similarity Model

This model is only needed when Clarifier / C3 semantic-consistency planning is
enabled, for example with `clarify_enabled=true`. It is not required for the
default Quick Start profile.

Download the local semantic similarity model:

```bash
uv run python scripts/setup_models.py
```

By default, EIMS expects `BAAI/bge-m3` under:

```text
embedding_model/bge-m3
```

If the helper dependency is unavailable, restore the locked environment and rerun setup:

```bash
uv sync --frozen
uv run python scripts/setup_models.py
```

The source repository is <https://huggingface.co/BAAI/bge-m3>. `embedding_model/` is a
local download directory, not a repository asset bundle. The helper writes the Hugging
Face snapshot directly under `embedding_model/bge-m3`, downloads only the files used by
the current semantic similarity path, and skips optional ONNX/OpenVINO artifacts to
reduce download size and timeout risk.

## 📓 Hardware-Free Notebook

`Hardware-Free-Demo.ipynb` is a hardware-free conceptual demonstration. For the current
configuration and runtime procedure, use this document together with `.env.example`,
`config/runtime_config.example.json`, and `system_config_wizard.py`.

<a id="configuration-reference"></a>

## ⚙️ Configuration Reference

Configuration is intentionally layered:

1. `bootstrap/config.py` defines the schema and safe defaults.
2. `config/runtime_config.json` stores local runtime settings written by the Web UI or
   helper scripts.
3. `.env` and process environment variables provide secrets and a small set of
   runtime-only switches. Process environment values take precedence over `.env`.

`config/runtime_config.json` and the Web configuration page are the source of truth for
hardware paths, hardware mappings, startup values, model endpoints, and the three
subsystem mode fields. The file is ignored by git; use
`config/runtime_config.example.json` as the template for a new machine.

The intended ownership split is:

| Location | Store here |
| --- | --- |
| `config/runtime_config.json` | Hardware paths, device mappings, startup defaults, mode fields, model endpoints, and detection settings |
| `.env` | Secrets and runtime-only switches such as API keys, `EIMS_SKILL_MODE`, and `EIMS_CHECKER_ENABLED` |

Important environment variables:

```dotenv
EIMS_OPENAI_API_KEY=your-api-key
EIMS_VLM_API_KEY=your-vlm-api-key
EIMS_SKILL_MODE=disabled
EIMS_CHECKER_ENABLED=true
```

Model field meanings:

| Field | Meaning |
| --- | --- |
| `base_url` | Main LLM API endpoint |
| `model_name` | Main LLM model name |
| `vlm_base_url` | Vision-language model API endpoint |
| `vlm_model_name` | Vision-language model name |

Startup objective and channel values use EIMS semantic keys, not hardware labels:

```json
{
  "startup": {
    "objective": "10x",
    "channel": "brightfield"
  }
}
```

Known legacy label values are normalized to their matching semantic keys when loaded.
`system.Min_exposure` and `system.Max_exposure` define the exposure range enforced by the
microscope controller, alongside the existing motion and brightness limits.

EIMS currently registers five detector presets: `2Dcell`, `organoid`, `mitosis`, `2Dcell_brightfield`, and `organoid_fluorescence`.
Their defaults are defined centrally in `bootstrap.config.DEFAULT_DETECTION_TARGETS`.
`config/runtime_config.json` may override the class, confidence threshold, output
filename, and local model paths for each registered target:

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
    },
    "2Dcell_brightfield": {
      "target_class_id": 0,
      "target_class_name": "2D_cell",
      "score_thr": 0.2,
      "output_filename": "2Dcell_brightfield_locations_list.json",
      "model_config": "detector_models/cell2d_brightfield/config.py",
      "model_checkpoint": "detector_models/cell2d_brightfield/weights.pth"
    },
    "organoid_fluorescence": {
      "target_class_id": 0,
      "target_class_name": "Organoids",
      "score_thr": 0.2,
      "output_filename": "organoid_fluorescence_locations_list.json",
      "model_config": "detector_models/organoid_fluorescence/config.py",
      "model_checkpoint": "detector_models/organoid_fluorescence/weights.pth"
    }
  }
}
```

`model_checkpoint` is a local runtime path. Large detector checkpoints are distributed
through GitHub Releases and must be downloaded to the referenced local path.
Reviewer-facing qualitative examples for all five presets are available in
`docs_public/detector_model_examples/`; they do not introduce additional detector weights.

## 🏗️ Architecture

```text
.
|-- app.py                         # FastAPI Web runtime
|-- main.py                        # CLI runtime
|-- api/                           # API routes and response models
|-- front/                         # Web UI static assets
|-- services/                      # runtime manager and task orchestration
|-- runtime/                       # config, asset checks, tools, agents, context assembly
|-- agent/                         # planner, executors, checkers, repair, clarification
|-- core_tool/                     # real microscope, Fiji, and Cellpose tools
|-- simulation/                    # mock microscope and image-analysis backends
|-- tool/                          # user-defined BaseTool extensions
|-- tooling/                       # tool manifest and user-tool prompt generation
|-- scripts/                       # setup, packaging, and utility scripts
|-- storage/                       # session history and artifact metadata
|-- interfaces/                    # CLI interaction, logging, and local preview glue
|-- skill_runtime/                 # skill parsing and prompt formatting
|-- adapters/                      # tool registry and LLM client adapters
|-- bootstrap/                     # runtime configuration loading and saving
|-- config/                        # runtime example and tool manifest
|-- prompts/                       # planner and executor prompts
|-- user_skills/                   # planning skills
|-- docs_public/                   # published datasets, outcomes, and evaluation materials
|-- embedding_model/               # local semantic retrieval model assets
|-- detector_models/               # detector configs and local checkpoints
`-- history/                       # per-run runtime history and outputs
```

## 🗂️ Runtime History

Runtime initialization creates an isolated session when the runtime context is assembled;
starting Uvicorn alone does not create one. Initialized sessions are stored under
`history/`:

```text
history/
`-- run_YYYYMMDD_HHMMSS_xxxxxxxx/
    |-- agent_interactions.json
    |-- meta.json
    `-- output/
```

The session records generated plans, executor code, execution results, image-checker
feedback, plan-trace and code-repair diagnostics, registered output files, and cache
metadata.

<a id="extending-eims"></a>

## 🧰 Extending EIMS

### Planner Skills

Planner skills under `user_skills/planning/` guide task decomposition without changing
runtime code. Supported formats are `.md`, `.txt`, `.json`, and directories containing a
`SKILL.md` file.

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
`@tool_func`:

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

Register and inspect a tool with:

```bash
uv run python create_tool.py register --class-path tool.new_tool:NewTool --tool-id "new_tool" --dry-run
uv run python create_tool.py register --class-path tool.new_tool:NewTool --tool-id "new_tool"
uv run python create_tool.py list
```

### Fiji Capability Declarations

Methods in `core_tool/fiji.py` that require optional Fiji plugins should declare those
dependencies beside the implementation. `--check-fiji` uses the declaration, and the
runtime checks it again before invoking the method.

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

## ⚠️ Safety Notes

Incorrect real-hardware configuration can damage samples or microscope hardware. Before
using real mode:

- Verify the `.cfg` with the official Micro-Manager GUI.
- Configure objective, XY, Z, brightness, and exposure limits.
- Confirm stage coordinate conventions and Z direction.
- Test low-risk movement and acquisition commands first.
- Keep emergency-stop procedures available.
- Validate Fiji and model checkpoint paths used by the workflow.

Generated-code execution is constrained but remains experimental automation. Use
`microscope_mode=demo` when validating a new workflow.

## Acknowledgements

EIMS builds on a broad open-source scientific software ecosystem. We gratefully
acknowledge the developers and maintainers of
[Micro-Manager](https://micro-manager.org/),
[Fiji/ImageJ](https://imagej.net/software/fiji/),
[pyimagej](https://github.com/imagej/pyimagej),
[pymmcore-plus](https://github.com/pymmcore-plus/pymmcore-plus),
[Cellpose](https://www.cellpose.org/),
[OpenMMLab/MMDetection](https://github.com/open-mmlab/mmdetection),
[PyTorch](https://pytorch.org/), and the Python scientific-computing libraries that
support this project.

These tools make reproducible microscope control, image processing, model inference, and
Web-based scientific workflows possible. Refer to their upstream documentation for
usage, license, and citation requirements.

## Licensing Notes

The root [LICENSE](LICENSE) states that the combined EIMS distribution is licensed under
GPLv3 because it incorporates GPLv3 components. Original code specifically developed for
EIMS is additionally available under the BSD 3-Clause License; see
[LICENSE.BSD-3-Clause](LICENSE.BSD-3-Clause). For a combined distribution, the GPLv3
terms described by the root license remain controlling.

Third-party software, datasets, models, and plugins retain their own licenses:

- Fiji/ImageJ and its plugins are subject to their upstream licenses.
- Micro-Manager is subject to its upstream license.
- Model weights, detector checkpoints, and external tools may impose additional terms or
  redistribution restrictions.

The EIMS license does not replace third-party licenses. Anyone downloading, bundling,
redistributing, or deploying EIMS with external dependencies is responsible for reviewing
and complying with the applicable upstream terms.

## Contributions

Contributions are welcome, including bug reports, fixes, documentation improvements, test
tasks, planner skills, and tool integrations.

Changes affecting real hardware control, image-analysis behavior, generated-code
execution, or Fiji plugin-dependent features should include relevant validation steps and
dependency declarations.

## Citation

Embodied Intelligence Enables Agentic Exploration in Microscopy, 09 February 2026, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-8617009/v1]
