# C3 conformal-prediction calibration overview

This directory holds the documented calibration view for the C3 (Cross-Sample Consistency Check) self-correction module.

## Files
- calibration_overview.json : the single self-contained view used for display and review.
- README.md : this note.
- compute_conformal_threshold.py : reproduces the threshold from this overview. Default path recomputes every non-conformity score with bge-m3 (matching `agent/clarifier.py::Clarify._compare_commands`); `--no-recompute` reads the stored `nonconformity_score` values directly. Running `uv run python docs_public/c3_calibration/compute_conformal_threshold.py --no-recompute` prints `threshold (round) = 0.029`, the value linked from `bootstrap/config.py` `task_similarity_threshold`.

## calibration_overview.json
A self-contained JSON document with header-level parameters plus a per-command record list.

### Header fields
- alpha : conformal prediction significance level; 0.1 => 90% statistical guarantee.
- n : number of calibration commands; 80.
- score_definition : `1 - min_pairwise_candidate_plan_similarity` (nonconformity score s).
- selected_threshold : alpha-quantile of the calibration s-distribution; 0.029 (raw 0.0285875201225281).
- semantic_model : embedding model used for pairwise plan similarity; model/bge-m3.

### Per-record fields (array `records`, n = 80)
- request_id : stable hash identifier for the calibration command.
- user_request : the natural-language experiment instruction (unambiguous command).
- num_solutions : number of candidate plans generated per command (3).
- candidate_plans : array of length num_solutions; each entry is a list of subtasks for one candidate plan.
  - Each subtask has: subtask_index, module, command.
- pairwise_scores : three pairwise candidate-plan similarities.
- min_similarity : the minimum of pairwise_scores.
- nonconformity_score : 1 - min_similarity; the nonconformity score s used for the threshold.

## Runtime link
bootstrap/config.py : task_similarity_threshold = 0.029 points back here.
The 0.029 threshold is the 73rd smallest of the 80 non-conformity scores (split conformal quantile k = ceil((n + 1) * (1 - alpha)), alpha = 0.1), verified by `compute_conformal_threshold.py --no-recompute`.
