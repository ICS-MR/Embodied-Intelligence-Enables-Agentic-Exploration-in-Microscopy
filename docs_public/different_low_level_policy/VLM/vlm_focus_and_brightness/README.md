# VLM Focus and Brightness Benchmark

This directory contains a pairwise benchmark comparing VLM-based autofocus and
brightness adjustment against traditional image-quality metrics on a real
microscope. Both scripts share the same search strategy and hardware setup so
that the only variable is the scoring method.

## Scripts

| Script | Scoring Method |
| --- | --- |
| `run_traditional_focus_brightness_benchmark.py` | Classical image-quality metrics (Tenengrad, Laplacian variance, adaptive Tenengrad, brightness fitness). |
| `run_vlm_focus_brightness_benchmark.py` | VLM visual judgment via an OpenAI-compatible chat completions API. |

## Search Strategy

Both scripts use the same iterative candidate-search loop:

1. Start from a center value (Z position for focus, brightness level for brightness).
2. Generate candidates at fixed offsets from the center.
3. Capture a microscope image for each candidate.
4. Score each image with the selected method.
5. Select the best candidate, reduce the step size, and repeat (up to 4 iterations).

The traditional script scores images with closed-form metrics:

- Focus: `tenengrad`, `laplacian`, or `adaptive_tenengrad`.
- Brightness: `brightness_fitness` (target gray-ratio 0.5 with a Gaussian tolerance).

The VLM script builds a 3x3 mosaic of candidate images, sends it to the VLM with a
task-specific prompt, and parses the returned filename to identify the selected
candidate. The VLM client and model name are injected from the EIMS runtime context
(`runtime_context.vlm_client`, `settings.model.vlm_model_name`); no API credentials
are hardcoded in these scripts.

## Configuration

Each script has a `RUN_CONFIG` dictionary near the top of the file. Edit it directly
before running. Key fields:

| Field | Description |
| --- | --- |
| `mode` | `"focus"`, `"brightness"`, or `"both"`. |
| `trial_count` | Number of repeated trials per run. |
| `channel` | Microscope channel to use (default `"brightfield"`). |
| `output_dir` | Directory for CSV summaries and preview frames. |
| `focus` / `brightness_search` | Per-task parameters: initial step, min step, max iterations, candidate offsets. |
| `metric` (traditional only) | Focus metric name and brightness fitness parameters. |
| `vlm_temperature` / `vlm_max_tokens` (VLM only) | VLM generation parameters. |

## Running

Both scripts import from the EIMS runtime (`bootstrap.config`, `runtime.factory`,
`runtime.hardware_lifecycle`) and must be run from the repository root so those
imports resolve correctly.

```powershell
.venv\Scripts\python.exe docs_public\different_low_level_policy\VLM\vlm_focus_and_brightness\run_traditional_focus_brightness_benchmark.py
```

```powershell
.venv\Scripts\python.exe docs_public\different_low_level_policy\VLM\vlm_focus_and_brightness\run_vlm_focus_brightness_benchmark.py
```

## Output

Each run creates a timestamped directory under `output_dir` containing:

- `summary.csv`: per-trial results (selected value, score, iteration count).
- Preview frames and candidate mosaics for visual inspection.

## Requirements

Both scripts require a connected microscope initialized through the EIMS runtime
and its dependencies (OpenCV, NumPy, Pillow). The VLM script additionally requires
a configured VLM endpoint accessible through the EIMS runtime settings.
