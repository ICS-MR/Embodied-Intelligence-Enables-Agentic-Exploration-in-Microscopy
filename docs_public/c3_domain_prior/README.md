# C3 Domain Prior (reviewed exemplars)

This directory holds the human-reviewed domain prior set used by the C3
(Cross-Sample Consistency Check) self-correction module's Clarifier for
retrieval-augmented ambiguity analysis.

## Files
- `domain_prior_reviewed.json` : the reviewed Domain Prior set (22 cases).
- `README.md` : this note.

## Domain Prior Data (`domain_prior_reviewed.json`)
A JSON array of 22 exemplar cases. Each case records a command, its candidate
plans, and a human consistency verdict, so the Clarifier can retrieve similar
past cases as few-shot references when judging whether the current candidate
plans reveal a genuinely unresolved ambiguity.

### Per-record fields (array, n = 22)
- `command` : the natural-language experiment instruction.
- `candidate_plans` : array of length 3; each entry is a Planner-Generated candidate plan (a list of subtasks, each with `subtask_index`, `module`, `command`).
- `consistency_label` : human-reviewed verdict; `consistent` (12) or `inconsistent` (10).
- `introspective_rationale` : the reasoning behind the label.
- `state` : hardware state at execution time (`objective`, `channel`, `exposure`, `brightness`, ...).

Label distribution: `consistent` = 12, `inconsistent` = 10.

## Runtime link
The Domain Prior is loaded from `bootstrap/config.py` `DEFAULT_DOMAIN_PRIOR_PATH = "docs_public/c3_domain_prior/domain_prior_reviewed.json"`; `agent/clarifier.py` loads it as a `DomainPrior` (`agent/domain_prior.py`) when the Clarify module is enabled (`bootstrap/config.py` `clarify_enabled`). The `DomainPrior` embeds the 22 `command` fields with bge-m3 and retrieves the top-3 most similar cases for the current query during consistency analysis (`Clarify._analyze_semantic_consistency`). It is read-only at runtime; no entries are added automatically.

## When it is used
Only when `clarify_enabled` is true. For each user request the Planner generates candidate plans; if a lightweight semantic-similarity gate does not already confirm consistency, the Clarifier retrieves the most similar reviewed cases from this Domain Prior set and includes them as few-shot context for the LLM-based consistency/ambiguity judgment. This corresponds to the ambiguity detection experiments in `experimental_outcomes/ambiguity_task_comparison/` (Fig. 3d).
