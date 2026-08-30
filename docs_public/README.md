# EIMS Public Documents

Public display and release materials for EIMS (Embodied Intelligence Enables Agentic Exploration in Microscopy).
This directory contains only polished/published content.

## Index

| Path | Content |
| --- | --- |
| `c3_calibration/` | Conformal prediction calibration: `calibration_overview.json` (calibration dataset) and `compute_conformal_threshold.py` (non-conformity / conformal threshold calculation). |
| `c3_domain_prior/` | C3 Domain Prior: reviewed exemplar set (`domain_prior_reviewed.json`, 22 cases), used by the Clarifier via local bge-m3 retrieval. |
| `documents/` | Organized dataset and experimental-outcome release: 5 task datasets + 15 experimental outcome collections (detailed below). |
| `Demo_test/` | End-to-end Demo workflow showcases with display Markdown and generated image/detection artifacts. |
| `frap/` | FRAP runtime dependency: UI profile `frap_ui_profile.json` (read by `tool/frap.py`), click-coordinate recorder, startup verification helpers (`capture_frap_startup_reference.py`, `calibrate_frap_startup_reference.py`), verification reference images (`references/`), and `README.md` documenting the startup visual verification and lifecycle state verification workflows. |
| `different_low_level_policy/` | Standalone low-level execution policy baselines: VLM localization/focus/brightness comparisons and ACT/VLA micromanipulation source package. |
| `detector_model_examples/` | Reviewer-facing qualitative examples for currently connected preset detector models: 2Dcell, organoid, mitosis, 2Dcell_brightfield, and organoid_fluorescence. |

## Detector Model Examples

Primary index: [detector_model_examples/README.md](detector_model_examples/README.md).

`detector_model_examples/` is a qualitative visualization aid, not a formal benchmark. The uploadable image subsets are self-contained under `detector_model_examples/testset/<target>/`:

| Target | Public sample data | Model source |
| --- | --- | --- |
| `2Dcell` | 2D fluorescence COCO-format examples | Existing system checkpoint: `detector_models/cell2d/weights.pth` |
| `organoid` | Organoid brightfield COCO-format examples | Existing system checkpoint: `detector_models/organoid/weights.pth` |
| `mitosis` | Original annotated mitosis qualitative subset | Existing system checkpoint: `detector_models/mitosis/weights.pth` |
| `2Dcell_brightfield` | 2D brightfield COCO-format examples | Existing system checkpoint: `detector_models/cell2d_brightfield/weights.pth` |
| `organoid_fluorescence` | Organoid fluorescence COCO-format examples | Existing system checkpoint: `detector_models/organoid_fluorescence/weights.pth` |

No detector weights are copied into `docs_public/`; inference uses the system-registered detector presets.

## Demo Workflow Showcases

Primary index: [Demo_test/README.md](Demo_test/README.md).

`Demo_test/` contains self-contained, reviewer-facing execution showcases for bright-spot detection with 60x follow-up acquisition and multi-channel fluorescence merging. These are illustrative workflow records and generated artifacts, not benchmark datasets.

## FRAP Runtime Dependency

Primary index: [frap/README.md](frap/README.md).

`frap/` contains the runtime configuration that `tool/frap.py` loads to drive the Olympus cellSens FRAP interface through simulated mouse clicks. The UI profile (`frap_ui_profile.json`) and verification reference images (`references/`) are machine-specific captures and must be recalibrated on each workstation using the bundled recorder and reference-capture helpers. Verified lifecycle stages cover startup readiness, laser-on after start, laser-off after stop, and actual process exit on close.

## Low-Level Policy Baselines

Primary index: [different_low_level_policy/README.md](different_low_level_policy/README.md).

`different_low_level_policy/` contains standalone baseline materials for low-level execution-policy comparisons. These baselines are separate from the default EIMS runtime:

| Path | Content / Release status |
| --- | --- |
| `different_low_level_policy/VLM/` | VLM localization, focus, and brightness comparison workflows; focus/brightness includes representative public test images. |
| `different_low_level_policy/ACT_VLA/Micromanipulation_tool/` | Public source package for ACT-style micromanipulation data collection, dataset conversion, training, and inference. |
| `different_low_level_policy/ACT_VLA/ACT_for_microscopy/` | Layout notes for the external ACT/VLA weight bundle. |

Large ACT/VLA datasets, checkpoints, hardware SDKs, and generated outputs are intentionally not bundled in `docs_public/`; the micromanipulation README points to the external Hugging Face resources and required local hardware configuration.

## documents/ (Dataset & Outcomes)

Primary index: [documents/README.md](documents/README.md).

### Task Datasets (`documents/task_datasets/`)

| Directory | Reference | Content / Status |
| --- | --- | --- |
| `different_sample_task_dataset/` | Table 1, S1 | Extracted task set (`task_set.json`) and dataset README. |
| `generalization_dataset/` | Table S2 | Extracted task set (`task_set.json`) and dataset README. |
| `ambiguous_task_dataset/` | Table S3 | Extracted user-input task set (`task_set.json`) and dataset README. |
| `imaging_perturbation_dataset/` | - | Perturbation sample images + state_metadata.json. |
| `teleoperation_dataset/` | - | Download link (external Hugging Face asset, too large to bundle). |

### Experimental Outcomes (`documents/experimental_outcomes/`)

| Directory | Reference | Content / Status |
| --- | --- | --- |
| `sample_type_task_set/` | Table 1 | Model-type task outcomes. |
| `automated_experimental_workflows/` | Fig. 2 | Representative automated experimental workflows and multidimensional imaging records. |
| `long_horizon_task_evaluation/` | Fig. 3a | 12 model-comparison records with per-task `Elapsed time (s)`. |
| `function_call_usage_comparison/` | Fig. 3b | EIMS vs Function-Calling comparison. |
| `planning_module_importance/` | Fig. 3b | High-level planning module vs baseline. |
| `ambiguity_task_comparison/` | Fig. 3d | Ambiguity detection task outcomes. |
| `hierarchy_module_comparison/` | Fig. 4a | Hierarchical architecture ablation. |
| `components_of_MPP_comparison/` | Fig. 4b | Tool-definition paradigm ablation. |
| `prompt_component_ablation/` | Fig. 4b | Prompt component constraint ablation. |
| `model_size_comparison/` | Fig. 4b | Model-size performance comparison. |
| `model_comparison/` | Fig. 4b, 4c | Foundation-model performance comparison. |
| `frap_and_micromanipulation/` | Fig. 5c, 5d, S2 | FRAP test display MDs (`frap/frap_test/`), archived FRAP / MP285 prompt snapshots (`frap/prompts/`, `micromanipulation_system/prompts/`), and micromanipulation-system interaction records (`micromanipulation_system/test/`). VLA/ACT baseline code and assets live under `docs_public/different_low_level_policy/ACT_VLA/`. |
| `different_disturbance_detection/` | Fig. 7c | Environmental/imaging perturbation detection outcomes. |
| `sparse_organoid_scanning/` | Fig. 7f | Organoid collection workflow dialogue (display). |
| `mitotic_cell_collection/` | Fig. 8a | Mitosis skill before/after comparison dialogues (display). |

## Note

`docs_public/frap/` is the runtime dependency location for the FRAP tool; the published `documents/.../frap/` copy intentionally does not duplicate these files.
