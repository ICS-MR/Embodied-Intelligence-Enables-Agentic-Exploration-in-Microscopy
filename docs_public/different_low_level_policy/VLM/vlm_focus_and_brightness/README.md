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

## Public Test Dataset

Representative static image sequences are bundled under `test_dataset/` for
public inspection of the two visual-selection tasks:

| Path | Content |
| --- | --- |
| `test_dataset/focus/` | Brightfield Z-stack images for focus selection; filenames encode the Z position. |
| `test_dataset/brightness/` | Brightfield images across brightness settings; filenames encode the brightness value. |

These images are qualitative test examples for reviewing focus and brightness
selection behavior. They are not a complete statistical benchmark, and the online
benchmark scripts below still acquire fresh candidate images from a connected
microscope when executed.

## Search Strategy

Both scripts use the same iterative candidate-search loop:

1. Start from a center value (Z position for focus, brightness level for brightness).
2. Generate candidates at fixed offsets from the center.
3. Capture a microscope image for each candidate (in `source="testset"` mode this is
   simulated by loading the static testset image whose encoded value is closest
   to the candidate).
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
| `output_dir` | Directory for CSV summaries and preview frames (default: `docs_public/different_low_level_policy/VLM/vlm_focus_and_brightness/outputs/<method>`). |
| `focus` / `brightness_search` | Per-task parameters: initial step, min step, max iterations, candidate offsets. |
| `metric` (traditional only) | Focus metric name and brightness fitness parameters. |
| `vlm_temperature` / `vlm_max_tokens` (VLM only) | VLM generation parameters. |
| `source` | `"system"` connects the microscope backend (current online flow) or `"testset"` evaluates the static `test_dataset/` without starting any backend. |
| `testset_dir` | Path to the static test dataset used when `source="testset"`. |
| `max_testset_images` | Cap on the testset images available for nearest-value matching when `source="testset"` (0 = use all). |

## Testset (Offline) Mode

Set `"source": "testset"` in `RUN_CONFIG` to evaluate the bundled static
`test_dataset/` without connecting the microscope backend:

- No microscope, preview, or EIMS runtime hardware is initialized.
- `test_dataset/focus/` and `test_dataset/brightness/` filenames encode the true
  value (`pos` = Z position, `bri` = brightness); each script scores those images
  with its own method and reports the selected value plus per-image results.
- Both scripts mirror the online system search: start from the midpoint of the
  testset value range, generate 9 candidates per iteration (center ± step), and
  for each candidate load the testset image whose encoded value is closest to
  that candidate (simulated capture). Selection, step-halving, and termination
  match the online mode exactly.
- Traditional script scores the loaded image with the configured metric
  (tenengrad / laplacian / adaptive_tenengrad / brightness_fitness).
- VLM script builds the 3x3 mosaic and asks the configured VLM endpoint to pick
  the best image; credentials come from `vlm_api_config.json` (see the
  `vlm_location_comparison` README), so no EIMS runtime is required.
- `testset_initial_z_um` / `testset_initial_brightness` override the starting
  center (default: midpoint of the available testset values).
- Outputs use the same `summary.csv` / `summary.json` / `result.json` layout as
  the online mode.

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

- `source="system"`: requires a connected microscope initialized through the EIMS
  runtime and its dependencies (OpenCV, NumPy, Pillow). The VLM script additionally
  requires a configured VLM endpoint accessible through the EIMS runtime settings.
- `source="testset"`: no microscope or EIMS runtime hardware is needed; both scripts
  only require OpenCV/NumPy/Pillow, and the VLM script additionally reads its VLM
  endpoint credentials from `vlm_api_config.json` (see the `vlm_location_comparison`
  README).
