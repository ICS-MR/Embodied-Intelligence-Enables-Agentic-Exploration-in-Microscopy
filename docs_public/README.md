# EIMS Public Documents

Public display and release materials for EIMS (Embodied Intelligence Enables Agentic Exploration in Microscopy).
This directory contains only polished/published content.

## Index

| Path | Content |
| --- | --- |
| `c3_calibration/` | Conformal prediction calibration: `calibration_overview.json` (calibration dataset) and `compute_conformal_threshold.py` (non-conformity / conformal threshold calculation). |
| `c3_knowledge_base/` | KnowledgeBase reviewed cases (`knowledge_base_reviewed.json`), used by the Clarifier via local bge-m3 retrieval. |
| `documents/` | Organized dataset and experimental-outcome release: 7 task datasets + 14 experimental outcome collections (detailed below). |
| `frap/` | FRAP runtime dependency: `frap_ui_profile.json` (read by `tool/frap.py`) and `record_frap_click_once.py`. |
| `vlm_location_comparison/` | VLM location comparison evaluation: toolkit, tests, README, requirements. |
| `vlm_focus_and_brightness/` | VLM focus & brightness evaluation scripts (traditional and VLM benchmarks). |
| `mitosis_detector_evaluation/` | Mitosis detection evaluation: `infer.py` + `testset/` (annotations + 10 test images). |
| `VLA/` | VLA micromanipulation research tool code: `Mircomanipulation_tool/` (data recording -> processing -> DETR training -> inference). `ACT_for_microscopy/` links to the external VLA weight bundle. |

## documents/ (Dataset & Outcomes)

Primary index: [documents/README.md](documents/README.md).

### Task Datasets (`documents/task_datasets/`)

| Directory | Reference | Content / Status |
| --- | --- | --- |
| `different_sample_task_dataset/` | Table 1, S1 | Extracted task set (task_set.json/md). |
| `generalization_dataset/` | Table S2 | Extracted task set. |
| `ambiguous_task_dataset/` | Table S3 | Extracted user-input task set. |
| `imaging_perturbation_dataset/` | - | Perturbation sample images + state_metadata.json. |
| `conformal_prediction_dataset/` | - | Calibration dataset (calibration_overview.json). |
| `conformal_prediction/` | - | Non-conformity / conformal threshold calculation script. |
| `teleoperation_dataset/` | - | Download link (external Hugging Face asset, too large to bundle). |

### Experimental Outcomes (`documents/experimental_outcomes/`)

| Directory | Reference | Content / Status |
| --- | --- | --- |
| `sample_type_task_set/` | Table 1 | Model-type task outcomes. |
| `long_horizon_task_evaluation/` | Fig. 3a | 12 model-comparison records with per-task `Elapsed time (s)`. |
| `function_call_usage_comparison/` | Fig. 3b | EIMS vs Function-Calling comparison. |
| `planning_module_importance/` | Fig. 3b | High-level planning module vs baseline. |
| `ambiguity_task_comparison/` | Fig. 3d | Ambiguity detection task outcomes. |
| `hierarchy_module_comparison/` | Fig. 4a | Hierarchical architecture ablation. |
| `components_of_MPP_comparison/` | Fig. 4b | Tool-definition paradigm ablation. |
| `prompt_component_ablation/` | Fig. 4b | Prompt component constraint ablation. |
| `model_size_comparison/` | Fig. 4b | Model-size performance comparison. |
| `model_comparison/` | Fig. 4b, 4c | Foundation-model performance comparison. |
| `frap_and_micromanipulation/` | Fig. 5c, 5d, S2 | FRAP test display MDs (`frap/frap_test/`) and MP-285 interaction records (`mp285/`). VLA code and assets live under `docs_public/VLA/`. |
| `different_disturbance_detection/` | Fig. 7c | Environmental/imaging perturbation detection outcomes. |
| `sparse_organoid_scanning/` | Fig. 7f | Organoid collection workflow dialogue (display). |
| `mitotic_cell_collection/` | Fig. 8a | Mitosis skill before/after comparison dialogues (display). |

## Note

`docs_public/frap/` is the runtime dependency location for the FRAP tool; the published `documents/.../frap/` copy intentionally does not duplicate these files.
