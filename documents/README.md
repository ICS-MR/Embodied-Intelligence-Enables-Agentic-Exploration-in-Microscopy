# Organized Dataset Release

This folder is a copied, non-destructive organization of the dataset and experimental outcome files. The original files under `docs/` are left unchanged.

## Top-Level Structure

1. `task_datasets/`

   Task-content datasets corresponding to manuscript supplementary tables and dataset descriptions.

2. `experimental_outcomes/`

   Experimental records and outcomes corresponding to manuscript figures and evaluations.

## Task Datasets

1. `different_sample_task_dataset/`

   Tables: Table 1, Table S1

   Content: extracted task set only.

   Source: `docs/experiment_records/10_sample_type_task_set/task_set.json`, `docs/experiment_records/10_sample_type_task_set/task_set.md`

2. `generalization_dataset/`

   Tables: Table S2

   Content: extracted task set only.

   Source: `docs/experiment_records/01_model_comparison/claude-full/task_set.json`, `docs/experiment_records/01_model_comparison/claude-full/task_set.md`

3. `ambiguous_task_dataset/`

   Tables: Table S3

   Content: extracted user-input task set only.

   Source: `docs/experiment_records/08_ambiguity_task_comparison/`

4. `imaging_perturbation_dataset/`

   Content: perturbation sample images, initial-state images where available, and `state_metadata.json`.

   Source: `docs/experiment_records/06_different_disturbance_detection/`

   Note: full detection/evaluation records are excluded here and kept under `experimental_outcomes/different_disturbance_detection/`.

5. `conformal_prediction_dataset/`

   Content: calibration dataset only.

   Source: `docs/c3_calibration/calibration_overview.json`

6. `conformal_prediction/`

   Content: scripts used to calculate the non-conformity score and conformal threshold.

   Source: `docs/c3_calibration/compute_conformal_threshold.py`

7. `teleoperation_dataset/`

   Content: human teleoperation demonstration dataset placeholder.

   Source: `docs/VLA/ACT_for_microscopy/`

   Note: `Mircomanipulation_tool` is code, not the dataset itself, so it is not included here.

## Experimental Outcomes

1. `sample_type_task_set/`

   Source: `docs/experiment_records/10_sample_type_task_set/`

2. `long_horizon_task_evaluation/`

   Placeholder. Runtime and success-rate results still need to be added.

3. `function_call_usage_comparison/`

   Source: `docs/experiment_records/07_function_call_usage_comparison/`

4. `planning_module_importance/`

   Source: `docs/experiment_records/09_planning_module_importance/`

5. `ambiguity_task_comparison/`

   Source: `docs/experiment_records/08_ambiguity_task_comparison/`

6. `hierarchy_module_comparison/`

   Source: `docs/experiment_records/05_hierarchy_module_comparison/`

7. `components_of_MPP_comparison/`

   Source: `docs/experiment_records/02_components_of_MPP_comparison/`

8. `prompt_component_ablation/`

   Source: `docs/experiment_records/04_prompt_component_ablation/`

9. `model_size_comparison/`

   Source: `docs/experiment_records/03_model_size_comparison/`

10. `model_comparison/`

    Source: `docs/experiment_records/01_model_comparison/`

11. `frap_and_micromanipulation/`

    Sources: `docs/frap/`, `docs/VLA/Mircomanipulation_tool/`

12. `different_disturbance_detection/`

    Source: `docs/experiment_records/06_different_disturbance_detection/`

13. `sparse_organoid_scanning/`

    Placeholder. Source files still need to be added.

14. `mitotic_cell_collection/`

    Placeholder. Source files still need to be added.

## Placeholder Items

The following directories contain only README placeholders for now:

1. `experimental_outcomes/long_horizon_task_evaluation/`
2. `experimental_outcomes/sparse_organoid_scanning/`
3. `experimental_outcomes/mitotic_cell_collection/`
