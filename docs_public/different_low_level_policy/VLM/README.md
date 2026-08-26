# VLM Low-Level Policy Baseline

This directory contains VLM (Vision-Language Model) baseline experiments that
compare VLM-based visual perception against traditional methods on real
microscope hardware. These baselines are **not** part of the EIMS runtime; they
are provided solely for comparative analysis with the EIMS
planner/executor/checker workflow.

## Directory Structure

| Path | Content |
| --- | --- |
| `vlm_location_comparison/` | VLM (Qwen-VL) vs local MMDetection localization comparison with COCO evaluation. |
| `vlm_focus_and_brightness/` | VLM vs traditional image-quality metrics for autofocus and brightness adjustment, with representative public test images. |

## vlm_location_comparison

Compares VLM-based object localization against a tiled local MMDetection model
on the same microscopy image. Three modes are supported:

- `vlm`: Qwen-VL localization with 0-999 normalized bounding boxes.
- `model`: tiled MMDetection inference with global NMS.
- `compare`: COCO-style metric comparison (precision, recall, F1, localization error).

Includes a `localization_toolkit/` package with CLI and Python API, plus optional
implementation self-checks that are not evaluation data. See
`vlm_location_comparison/README.md` for full usage instructions.
VLM endpoint credentials are configured in the gitignored `vlm_api_config.json`
(copy the committed `vlm_api_config.example.json` and fill it in); never commit
real credentials.

## vlm_focus_and_brightness

Pairwise benchmark on a real microscope comparing two scoring methods for
autofocus and brightness adjustment:

- **Traditional**: Tenengrad, Laplacian variance, adaptive Tenengrad, brightness fitness.
- **VLM**: 3x3 mosaic sent to an OpenAI-compatible VLM endpoint for visual judgment.

Both scripts share the same iterative candidate-search strategy so the only
variable is the scoring method. See `vlm_focus_and_brightness/README.md` for
configuration, run instructions, and the bundled `test_dataset/` image examples.

## Relationship to EIMS

Where EIMS decomposes natural-language instructions into tool-mediated plans
executed by a checker-verified executor, the VLM baselines test direct
visual-perception approaches, using VLM either as a detector or as an
image-quality judge, without a planning layer. The comparison between these
paradigms is discussed in the manuscript.

## External Dependencies

Both subdirectories require dependencies beyond the EIMS runtime:

- `vlm_location_comparison`: MMDetection, MMCV (CUDA-compatible build), pycocotools, matplotlib.
- `vlm_focus_and_brightness`: OpenCV and, for online acquisition runs, a connected microscope via the EIMS runtime and a configured VLM endpoint.

Model weights, checkpoints, generated outputs, and large localization assets are
not stored in this repository. The focus/brightness directory includes a compact
public image subset for qualitative inspection.
