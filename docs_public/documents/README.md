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

5. `teleoperation_dataset/`

   Content: external human teleoperation dataset link and clone instructions.

   Note: `Micromanipulation_tool` is the ACT/VLA micromanipulation code package, not the teleoperation dataset itself, so it is not included here. The `ACT_for_microscopy` VLA weight bundle is linked at `docs_public/different_low_level_policy/ACT_VLA/ACT_for_microscopy/`.

## Experimental Outcomes

1. `sample_type_task_set/`

   Table 1: model-type task outcomes.

2. `automated_experimental_workflows/`

   Fig. 2: representative automated experimental workflows and multidimensional imaging records.

3. `long_horizon_task_evaluation/`

   Fig. 3a: 12 model-comparison experiment records with per-task `Elapsed time (s)`.

4. `function_call_usage_comparison/`

   Fig. 3b: EIMS vs Function-Calling comparison.

5. `planning_module_importance/`

   Fig. 3b: high-level planning module vs baseline in Function-Calling comparison.

6. `ambiguity_task_comparison/`

   Fig. 3d: ambiguity detection task outcomes.

7. `hierarchy_module_comparison/`

   Fig. 4a: hierarchical architecture ablation.

8. `components_of_MPP_comparison/`

   Fig. 4b: tool-definition paradigm ablation.

9. `prompt_component_ablation/`

   Fig. 4b: prompt component constraint ablation.

10. `model_size_comparison/`

   Fig. 4b: model-size performance comparison.

11. `model_comparison/`

    Fig. 4b, Fig. 4c: foundation-model performance comparison.

12. `frap_and_micromanipulation/`

    Fig. 5c, Fig. 5d, Fig. S2: FRAP test display MDs (`frap/frap_test/`), archived FRAP / MP285 prompt snapshots, and micromanipulation-system interaction records (`micromanipulation_system/`).

    Note: the FRAP runtime dependency (`frap_ui_profile.json` + `record_frap_click_once.py`) lives at `docs_public/frap/` and is not duplicated here.

13. `different_disturbance_detection/`

    Fig. 7c: environmental/imaging perturbation detection outcomes.

14. `sparse_organoid_scanning/`

    Fig. 7f: organoid collection workflow display dialogue.

15. `mitotic_cell_collection/`

    Fig. 8a: mitotic cell collection tasks dialogues with and without the use of a Skill.

