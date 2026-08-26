# Location Comparison

This directory contains the localization comparison workflow for testing VLM-based
localization against a local MMDetection model.

The workflow supports three actions:

- `vlm`: run Qwen-VL localization on an input microscopy image.
- `model`: run tiled local MMDetection localization on the same image.
- `compare`: compare VLM and local model predictions against COCO ground truth.

## Contents

- `localization_toolkit/__init__.py`: public API re-exports.
- `localization_toolkit/cli.py`: command-line entry point.
- `localization_toolkit/pipeline.py`: public Python API.
- `localization_toolkit/model_inference.py`: tiled MMDetection inference, global NMS, COCO result export, and visualization.
- `localization_toolkit/vlm_inference.py`: Qwen-VL localization, coordinate conversion, raw JSON export, and visualization.
- `localization_toolkit/evaluation.py`: COCO-style prediction comparison and error plots.
- `self_checks/check_localization_toolkit.py`: implementation self-checks for parsing, COCO conversion, and comparison metrics. This is not an evaluation dataset.
- `requirements.txt`: runtime dependencies, including `mmdet` and `mmengine`. The CUDA/PyTorch-compatible `mmcv` build must be installed separately (see below).

## Command Line Usage

Install the dependencies from the repository root:

```powershell
uv pip install --python .venv\Scripts\python.exe -r docs_public\different_low_level_policy\VLM\vlm_location_comparison\requirements.txt
```

The local-model mode also requires the CUDA/PyTorch-compatible MMCV build documented
in the repository's main README.

Before using `--mode vlm`, create the local API config file from the committed
template and fill in the three fields:

```powershell
Copy-Item docs_public\different_low_level_policy\VLM\vlm_api_config.example.json docs_public\different_low_level_policy\VLM\vlm_api_config.json
```

```json
{
  "api_key": "<your-vlm-api-key>",
  "api_url": "<your-vlm-api-endpoint>",
  "model_name": "<your-vlm-model-name>"
}
```

`vlm_api_config.json` is gitignored; never commit real credentials to this repository.

Recommended preset workflow using `docs_public/detector_model_examples` assets:

```powershell
$env:PYTHONPATH = "docs_public/different_low_level_policy/VLM/vlm_location_comparison"
$out = "localization_output"

python -m localization_toolkit.cli --mode vlm `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out

python -m localization_toolkit.cli --mode model `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out

python -m localization_toolkit.cli --mode compare `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out
```

With `--target`, the CLI resolves the image path, COCO annotation file,
`image_id`, `category_id`, VLM query text, detector config/checkpoint paths, and
registered detector score threshold from `detector_model_examples` and
`bootstrap.config.DEFAULT_DETECTION_TARGETS`. Use explicit flags such as
`--score-thr`, `--queries`, `--image-id`, or `--category-id` only when you need to
override the preset.

If the VLM call fails with a proxy error such as `Cannot connect to proxy`, retry
with environment proxies disabled for that API request:

```powershell
python -m localization_toolkit.cli --mode vlm `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out `
  --no-env-proxy
```

List available preset targets:

```powershell
python -m localization_toolkit.cli --list-targets
```

Manual mode is still available for custom images and annotations:

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

## Using the detector_model_examples testset

The repository ships a self-contained qualitative image set under
`docs_public/detector_model_examples/testset/<target>/`, each with a COCO-format
`annotations.json`. It is the recommended source of test images and ground truth
for this comparison workflow. The toolkit is single-image by design: pick one
image from a target folder and run the three steps against it.

Per-target parameters (image IDs are read from each `annotations.json`):

| target | testset dir | `category_id` | VLM query | detector config | detector checkpoint |
| --- | --- | --- | --- | --- | --- |
| `2Dcell` | `testset/2Dcell` | 1 | `2D_cell` | `detector_models/cell2d/config.py` | `detector_models/cell2d/weights.pth` |
| `mitosis` | `testset/mitosis` | 0 | `mitosis` | `detector_models/mitosis/config.py` | `detector_models/mitosis/weights.pth` |
| `organoid` | `testset/organoid` | 1 | `Organoids` | `detector_models/organoid/config.py` | `detector_models/organoid/weights.pth` |
| `2Dcell_brightfield` | `testset/2Dcell_brightfield` | 1 | `2D_cell` | `detector_models/cell2d_brightfield/config.py` | `detector_models/cell2d_brightfield/weights.pth` |
| `organoid_fluorescence` | `testset/organoid_fluorescence` | 1 | `Organoids` | `detector_models/organoid_fluorescence/config.py` | `detector_models/organoid_fluorescence/weights.pth` |

The detector config and checkpoint paths mirror
`bootstrap.config.DEFAULT_DETECTION_TARGETS`, so the `model` step uses the same
registered weights as the preset detectors.

Example: a single `2Dcell` image (`Image_12106.jpg`, `image_id 1`), run from the
repository root. Set `PYTHONPATH` so `localization_toolkit` imports correctly.
The preset form is preferred because it reads the image/category IDs and detector
assets automatically:

```powershell
$env:PYTHONPATH = "docs_public/different_low_level_policy/VLM/vlm_location_comparison"
$out = "localization_output"

python -m localization_toolkit.cli --mode vlm `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out

python -m localization_toolkit.cli --mode model `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out

python -m localization_toolkit.cli --mode compare `
  --target 2Dcell --image-name Image_12106.jpg --output-dir $out
```

Notes for this testset:

- The `detector_model_examples` set is a qualitative reviewer-facing sample, not a
  held-out benchmark (see its README). Treat the metrics from `compare` as
  illustrative, not as formal mAP/precision/recall reporting.
- `image_id` and `category_id` must match the chosen `annotations.json`; otherwise
  `compare` matches nothing and every box counts as false positive or false negative.
- `2Dcell`, `organoid`, and `organoid_fluorescence` images are 512x512, so the VLM
  path sends them without resizing. `mitosis` images are about 1882x1891 and
  `2Dcell_brightfield` images are 2048x2048; both are downscaled to a 512 edge
  before the VLM call, which shrinks their small targets; keep that scale tradeoff
  in mind when interpreting `mitosis` and `2Dcell_brightfield` results.

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

## Implementation Self-Checks

The `self_checks/` folder contains small implementation checks for parser,
converter, and metric behavior. It is included to guard the workflow code against
accidental breakage and should not be interpreted as a held-out benchmark or
scientific test set.

Run it only when modifying the toolkit code:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s self_checks -p "check_*.py"
```

## Notes

- Run commands from the repository root with `PYTHONPATH` set as shown above, or
  run from this directory directly so Python can import `localization_toolkit`.
- MMDetection, model configs, checkpoints, input images, and COCO annotation files
  are external test assets and are not stored here.
- VLM credentials and endpoint details are read from
  `docs_public/different_low_level_policy/VLM/vlm_api_config.json` (gitignored),
  created by copying `vlm_api_config.example.json` and filling in the fields.
  Never commit real credentials to this repository.
- The VLM request asks for bounding boxes normalized to the `0-999` coordinate range.
  The configured endpoint and model must support that response convention.
- For the default single-class workflow, set `--category-id` to the cell category ID
  used by the COCO ground-truth file. Both MMDetection and VLM results use this ID.
