# Organized Dataset Release

This folder contains organized dataset and experimental outcome files.

## Top-Level Structure

1. `task_datasets/`

   Task-content datasets corresponding to manuscript supplementary tables and dataset descriptions.

2. `experimental_outcomes/`

   Experimental records and outcomes corresponding to manuscript figures and evaluations.

## Task Datasets

1. `different_sample_task_dataset/`

   Tables: Table 1, Table S1

   Content: extracted task set only.

2. `generalization_dataset/`

   Tables: Table S2

   Content: extracted task set only.

3. `ambiguous_task_dataset/`

   Tables: Table S3

   Content: extracted user-input task set only.

4. `imaging_perturbation_dataset/`

   Content: perturbation sample images, initial-state images where available, and `state_metadata.json`.

   Note: full detection/evaluation records are excluded here and kept under `experimental_outcomes/different_disturbance_detection/`.

5. `conformal_prediction_dataset/`

   Content: calibration dataset only.

6. `conformal_prediction/`

   Content: scripts used to calculate the non-conformity score and conformal threshold.

7. `teleoperation_dataset/`

   Content: external human teleoperation dataset link and clone instructions.

   Note: `Mircomanipulation_tool` is code, not the dataset itself, so it is not included here. The `ACT_for_microscopy` VLA weight bundle is linked at `docs_public/VLA/ACT_for_microscopy/`.

## Experimental Outcomes

1. `sample_type_task_set/`

   Table 1: model-type task outcomes.

2. `long_horizon_task_evaluation/`

   Fig. 3a evidence: 12 model-comparison experiment records with per-task `Elapsed time (s)`.

3. `function_call_usage_comparison/`

   Fig. 3b: EIMS vs Function-Calling comparison.

4. `planning_module_importance/`

   Fig. 3b: high-level planning module vs baseline.

5. `ambiguity_task_comparison/`

   Fig. 3d: ambiguity detection task outcomes.

6. `hierarchy_module_comparison/`

   Fig. 4a: hierarchical architecture ablation.

7. `components_of_MPP_comparison/`

   Fig. 4b: tool-definition paradigm ablation.

8. `prompt_component_ablation/`

   Fig. 4b: prompt component constraint ablation.

9. `model_size_comparison/`

   Fig. 4b: model-size performance comparison.

10. `model_comparison/`

    Fig. 4b, Fig. 4c: foundation-model performance comparison.

11. `frap_and_micromanipulation/`

    Fig. 5c, Fig. 5d, Fig. S2: FRAP test display MDs (`frap/frap_test/`) and MP-285 interaction records (`mp285/`).

    Note: the FRAP runtime dependency (`frap_ui_profile.json` + `record_frap_click_once.py`) lives at `docs_public/frap/` and is not duplicated here. VLA code and assets live under `docs_public/VLA/`.

12. `different_disturbance_detection/`

    Fig. 7c: environmental/imaging perturbation detection outcomes.

13. `sparse_organoid_scanning/`

    Fig. 7f: organoid collection workflow display dialogue.

14. `mitotic_cell_collection/`

    Fig. 8a: skill before/after comparison dialogues.

