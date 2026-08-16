"""Compute the C3 conformal-prediction calibration threshold.

Reproduces the non-conformity score distribution and the alpha-quantile
threshold used by the C3 (Cross-Sample Consistency Check) self-correction
module, as documented in docs/c3_calibration/README.md.

The score definition and pipeline mirror the live runtime:

  1. Load the calibration overview (80 commands, 3 candidate plans each).
  2. For every pair of candidate plans, compute pairwise similarity using
     the exact routine from agent/clarifier.py Clarify._compare_commands:
       - a candidate plan is a list of subtasks {module, command};
         two plans must have equal subtask counts and identical module
         sequences, otherwise their pairwise similarity is 0.0;
       - per matched subtask, embed ENCODING_PROMPT + command with bge-m3
         (normalize_embeddings=True) and take the cosine similarity;
       - the plan-pair similarity is the minimum across matched subtasks.
  3. min_similarity = minimum of the three pairwise plan similarities;
     non-conformity score s = 1 - min_similarity.
  4. threshold = k-th order statistic of the 80 s-values, where
     k = ceil((n + 1) * (1 - alpha)); this is the split conformal
     prediction (1 - alpha) quantile of the calibration distribution.

Verified constants (docs/c3_calibration/README.md):
    n          = 80
    alpha      = 0.1
    k          = 73
    raw        = 0.0285875201225281
    threshold  = 0.029            # round(raw, 3)

Usage (from repo root):

    uv run python docs/c3_calibration/compute_conformal_threshold.py
    uv run python docs/c3_calibration/compute_conformal_threshold.py --no-recompute
    uv run python docs/c3_calibration/compute_conformal_threshold.py --model path/to/bge-m3
    uv run python docs/c3_calibration/compute_conformal_threshold.py --alpha 0.05

Notes
-----
* --no-recompute skips bge-m3 recomputation and reads the stored
  nonconformity_score values from calibration_overview.json directly. It is
  network-free and runs instantly, ideal for regenerating the threshold when
  only reordering data was changed.
* The default recomputation path requires sentence-transformers and
  scikit-learn (the same libraries used by the runtime Clarifier) plus the
  local bge-m3 weights at embedding_model/bge-m3.
* The runtime link (bootstrap/config.py task_similarity_threshold = 0.029)
  is reported to make the round-trip traceable; this script does not modify
  that constant.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Sequence

# Resolve paths relative to this file so the script works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_CALIBRATION = SCRIPT_DIR / "calibration_overview.json"
DEFAULT_MODEL = REPO_ROOT / "embedding_model" / "bge-m3"
# Matches agent/clarifier.py Clarify.PROMPT exactly.
ENCODING_PROMPT = "Represent this sentence for semantic similarity comparison: "

# README reference values, used for the regression check.
README_ALPHA = 0.1
README_RAW_THRESHOLD = 0.0285875201225281
README_THRESHOLD = 0.029


def load_calibration(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Calibration overview not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "records" not in data:
        raise ValueError(f"{path}: expected an object with a 'records' array")
    return data


def plan_to_tasks(plan: Sequence[dict[str, Any]]) -> List[dict[str, str]]:
    """Normalize a candidate plan into the {module, command} task list
    expected by agent/clarifier.py Clarify._compare_commands."""
    tasks: List[dict[str, str]] = []
    for step in plan:
        module = str(step.get("module", "") or "")
        command = str(step.get("command", "") or "").strip() or "[Empty command]"
        tasks.append({"module": module, "command": command})
    return tasks


def compare_plans(model: Any, plan_a: Sequence[dict[str, Any]],
                  plan_b: Sequence[dict[str, Any]]) -> float:
    """Reproduce agent/clarifier.py Clarify._compare_commands exactly.

    Returns 0.0 when the two plans differ in subtask count or module sequence,
    matching the runtime's strict alignment gate. Otherwise returns the minimum
    cosine similarity of command embeddings across matched subtasks.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    tasks_a = plan_to_tasks(plan_a)
    tasks_b = plan_to_tasks(plan_b)
    if len(tasks_a) != len(tasks_b):
        return 0.0
    if [t["module"] for t in tasks_a] != [t["module"] for t in tasks_b]:
        return 0.0

    scores: List[float] = []
    for ta, tb in zip(tasks_a, tasks_b):
        embeddings = model.encode(
            [ENCODING_PROMPT + ta["command"], ENCODING_PROMPT + tb["command"]],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        raw_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        scores.append(float(raw_sim))
    return min(scores) if scores else 0.0


def min_pairwise_plan_similarity(model: Any,
                                 plans: Sequence[Sequence[dict[str, Any]]]) -> float:
    """Minimum over all candidate-plan pairs; matches
    agent/clarifier.py Clarify._run_semantic_model."""
    scores: List[float] = []
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            scores.append(compare_plans(model, plans[i], plans[j]))
    return min(scores) if scores else 1.0


def conformal_threshold(scores: Sequence[float], alpha: float) -> tuple[float, int]:
    """Split conformal prediction threshold.

    threshold = k-th smallest score, k = ceil((n + 1) * (1 - alpha)),
    clamped to [1, n]; the runtime stores this rounded to 3 decimals.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot compute threshold on zero scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    k = min(max(math.ceil((n + 1) * (1.0 - alpha)), 1), n)
    ordered = sorted(scores)
    return float(ordered[k - 1]), k


def recompute_scores(records: Sequence[dict[str, Any]], model: Any,
                     verify: bool = True) -> tuple[List[float], dict[str, Any]]:
    """Recompute the non-conformity score per record and (optionally)
    compare against the values stored in calibration_overview.json."""
    scores: List[float] = []
    max_gap = 0.0
    mismatches = 0
    for rec in records:
        plans = rec.get("candidate_plans")
        if not plans or len(plans) < 2:
            scores.append(1.0)
            continue
        sim = min_pairwise_plan_similarity(model, plans)
        s = 1.0 - sim
        scores.append(s)
        if verify:
            stored = rec.get("nonconformity_score")
            if isinstance(stored, (int, float)):
                gap = abs(s - float(stored))
                if gap > max_gap:
                    max_gap = gap
                if gap > 1e-6:
                    mismatches += 1
    return scores, {"max_gap": max_gap, "mismatches": mismatches}


def load_model(model_dir: Path) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "sentence-transformers is required to recompute scores. Install "
            "it (e.g. `uv pip install sentence-transformers scikit-learn`) "
            "or pass --no-recompute to use the stored scores. Error: "
            f"{exc}"
        ) from exc
    if not (model_dir.exists() and (model_dir / "config.json").exists()):
        raise SystemExit(
            f"bge-m3 model not found at {model_dir}. Download it from "
            "https://huggingface.co/BAAI/bge-m3 into embedding_model/bge-m3, "
            "or pass --model <path>."
        )
    return SentenceTransformer(str(model_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                   help=f"Path to calibration_overview.json (default: {DEFAULT_CALIBRATION})")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help=f"Path to bge-m3 SentenceTransformer folder (default: {DEFAULT_MODEL})")
    p.add_argument("--alpha", type=float, default=None,
                   help="Override alpha (default: read from calibration file)")
    p.add_argument("--no-recompute", action="store_true",
                   help="Skip bge-m3 recomputation; use stored nonconformity_score values")
    p.add_argument("--no-verify", action="store_true",
                   help="With recomputation, do not compare against stored scores")
    p.add_argument("--round-digits", type=int, default=3,
                   help="Decimal places for the rounded threshold (default: 3)")
    p.add_argument("--quiet", action="store_true",
                   help="Only print the rounded threshold value")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    data = load_calibration(Path(args.calibration))
    records = data.get("records", [])
    n = len(records)
    if n == 0:
        raise SystemExit("No calibration records found")

    alpha = float(data["alpha"]) if args.alpha is None else args.alpha
    if not 0.0 < alpha < 1.0:
        raise SystemExit(f"alpha must be in (0, 1), got {alpha}")

    if args.no_recompute:
        scores = [float(r["nonconformity_score"]) for r in records]
        verify_note = "score source: stored nonconformity_score"
    else:
        model = load_model(Path(args.model))
        scores, verify = recompute_scores(records, model,
                                          verify=not args.no_verify)
        if args.no_verify:
            verify_note = "score source: bge-m3 recomputed (verification disabled)"
        else:
            verify_note = (
                "score source: bge-m3 recomputed "
                f"(max gap vs stored = {verify['max_gap']:.3e}, "
                f"records with gap>1e-6 = {verify['mismatches']})"
            )

    raw_threshold, k = conformal_threshold(scores, alpha)
    threshold = round(raw_threshold, args.round_digits)

    if args.quiet:
        print(threshold)
        return 0

    sep = "=" * 64
    print(sep)
    print("C3 conformal-prediction calibration")
    print(sep)
    print(f"calibration file : {args.calibration}")
    print(f"model            : {args.model if not args.no_recompute else '(not used)'}")
    print(verify_note)
    print(f"alpha            : {alpha}")
    print(f"n (records)      : {n}")
    print(f"k = ceil((n+1)(1-alpha)) = {k}")
    print(f"score range      : [{min(scores):.6g}, {max(scores):.6g}]")
    print("-" * 64)
    print(f"threshold (raw)  : {raw_threshold}")
    print(f"threshold (round): {threshold}")
    print("-" * 64)
    readme_ok = threshold == README_THRESHOLD
    flag = "OK" if readme_ok else "MISMATCH"
    print(f"README reference : raw={README_RAW_THRESHOLD}, round={README_THRESHOLD}  [{flag}]")
    print(f"runtime link     : bootstrap/config.py task_similarity_threshold = {README_THRESHOLD}")
    print(sep)
    return 0 if readme_ok else 1


if __name__ == "__main__":
    sys.exit(main())
