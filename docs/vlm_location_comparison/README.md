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

```bash
python -m localization_toolkit.cli ^
  --mode vlm ^
  --image path/to/image.jpg ^
  --output-dir localization_output ^
  --queries cell
```

```bash
python -m localization_toolkit.cli ^
  --mode model ^
  --image path/to/image.jpg ^
  --output-dir localization_output ^
  --config path/to/config.py ^
  --checkpoint path/to/epoch.pth
```

```bash
python -m localization_toolkit.cli ^
  --mode compare ^
  --gt path/to/test.json ^
  --model-pred localization_output/model_detection_result.json ^
  --vlm-pred localization_output/vlm_output_coco.json ^
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
    category_id=0,
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

## Notes

- Run commands from this directory so Python can import `localization_toolkit`.
- MMDetection, model configs, checkpoints, input images, and COCO annotation files
  are external test assets and are not stored here.
- `localization_toolkit/vlm_inference.py` contains Qwen-VL API configuration. Move
  credentials to an environment variable before sharing or publishing this code.
