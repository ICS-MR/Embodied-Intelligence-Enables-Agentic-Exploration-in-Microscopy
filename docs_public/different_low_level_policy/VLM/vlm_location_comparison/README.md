# Location Comparison

This directory contains the localization comparison workflow for testing VLM-based
localization against a local MMDetection model.

The workflow supports three actions:

- `vlm`: run Qwen-VL localization on an input microscopy image.
- `model`: run tiled local MMDetection localization on the same image.
- `compare`: compare VLM and local model predictions against COCO ground truth.

## Contents

- `localization_toolkit/cli.py`: command-line entry point.
- `localization_toolkit/pipeline.py`: public Python API.
- `localization_toolkit/model_inference.py`: tiled MMDetection inference, global NMS, COCO result export, and visualization.
- `localization_toolkit/vlm_inference.py`: Qwen-VL localization, coordinate conversion, raw JSON export, and visualization.
- `localization_toolkit/evaluation.py`: COCO-style prediction comparison and error plots.
- `requirements.txt`: minimal runtime dependencies outside MMDetection.

## Command Line Usage

Install the dependencies from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe -r docs_public\different_low_level_policy\VLM\vlm_location_comparison\requirements.txt
```

The local-model mode also requires the CUDA/PyTorch-compatible MMCV build documented
in the repository's main README.

Before using `--mode vlm`, replace the configuration placeholders at the top of
`localization_toolkit/vlm_inference.py`:

```python
API_KEY = "<your-vlm-api-key>"
API_URL = "<your-vlm-api-endpoint>"
MODEL_NAME = "<your-vlm-model-name>"
```

```powershell
python -m localization_toolkit.cli `
  --mode vlm `
  --image path/to/image.jpg `
  --output-dir localization_output `
  --category-id 1 `
  --queries cell
```

```powershell
python -m localization_toolkit.cli `
  --mode model `
  --image path/to/image.jpg `
  --output-dir localization_output `
  --category-id 1 `
  --config path/to/config.py `
  --checkpoint path/to/epoch.pth
```

```powershell
python -m localization_toolkit.cli `
  --mode compare `
  --gt path/to/test.json `
  --model-pred localization_output/model_detection_result.json `
  --vlm-pred localization_output/vlm_output_coco.json `
  --output-dir localization_output
```

## Python Usage

```python
from localization_toolkit import (
    LocalizationConfig,
    compare_localizations,
    run_model_localization,
    run_vlm_localization,
)

cfg = LocalizationConfig(
    image_path=r"path/to/image.jpg",
    output_dir="localization_output",
    image_id=1,
    category_id=1,
    config_file=r"path/to/config.py",
    checkpoint_file=r"path/to/epoch.pth",
    gt_annotation_file=r"path/to/test.json",
    query_texts=("cell",),
)

run_vlm_localization(cfg)
run_model_localization(cfg)
compare_localizations(cfg)
```

## Outputs

- `model_detection_result.json`
- `model_result.jpg`
- `vlm_detections.json`
- `vlm_output_coco.json`
- `vlm_result.jpg`
- `error_results.json`
- `error_analysis.png`

Use the same input image and ground-truth annotation when comparing methods so
the reported localization errors are directly comparable.

`error_results.json` includes matched-box localization errors plus `gt_count`,
`prediction_count`, `true_positive`, `false_positive`, `false_negative`,
`precision`, `recall`, and `f1`. Predictions and annotations are matched only
when both their image IDs and category IDs agree. When no boxes are matched,
the localization-error fields are `null` rather than zero.

## Notes

- Run commands from this directory so Python can import `localization_toolkit`.
- MMDetection, model configs, checkpoints, input images, and COCO annotation files
  are external test assets and are not stored here.
- VLM credentials and endpoint details are configured at the top of
  `localization_toolkit/vlm_inference.py`. Replace all three placeholders locally,
  and never commit real credentials to this repository.
- The VLM request asks for bounding boxes normalized to the `0-999` coordinate range.
  The configured endpoint and model must support that response convention.
- For the default single-class workflow, set `--category-id` to the cell category ID
  used by the COCO ground-truth file. Both MMDetection and VLM results use this ID.
