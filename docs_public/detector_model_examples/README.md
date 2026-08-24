# Preset Detector Qualitative Examples

This folder provides reviewer-facing qualitative examples for the detector presets currently connected to the system: `2Dcell`, `organoid`, and `mitosis`.

These examples are not a held-out benchmark, not a statistically representative test set, and should not be used to report formal detection metrics such as mAP, precision, or recall.

The uploadable qualitative image subsets live directly under `testset/<target>/`. They are intentionally modality-specific rather than equal-sized: `testset/2Dcell/` uses 2D fluorescence data, while `testset/organoid/` uses organoid brightfield data.

## Preset Detectors

The example manifest only stores image/annotation paths. Model config and checkpoint paths are loaded from `bootstrap.config.DEFAULT_DETECTION_TARGETS`, so this folder does not introduce any new detector weights.

| Target | Class | Config | Checkpoint |
| --- | --- | --- | --- |
| `2Dcell` | `2Dcell` | `detector_models/cell2d/config.py` | `detector_models/cell2d/weights.pth` |
| `organoid` | `organoid` | `detector_models/organoid/config.py` | `detector_models/organoid/weights.pth` |
| `mitosis` | `mitosis` | `detector_models/mitosis/config.py` | `detector_models/mitosis/weights.pth` |

## Run Examples

List available qualitative targets:

```bash
python docs_public/detector_model_examples/infer.py --list-targets
```

Validate image paths and registered detector assets without running inference:

```bash
python docs_public/detector_model_examples/infer.py --validate --all
```

Run the retained mitosis example set:

```bash
python docs_public/detector_model_examples/infer.py
```

Run all three currently connected detector presets:

```bash
python docs_public/detector_model_examples/infer.py --all
```

Run one detector target:

```bash
python docs_public/detector_model_examples/infer.py --target 2Dcell
python docs_public/detector_model_examples/infer.py --target organoid
python docs_public/detector_model_examples/infer.py --target mitosis
```

Outputs are written under `docs_public/detector_model_examples/outputs/`:

- `summary.json`: top-level run summary for selected targets.
- `<target>/predictions.json`: per-target detection results.
- `<target>/visualizations/`: input images overlaid with red prediction boxes.
- `<target>/summary.json`: per-target model metadata and output paths.

## Notes

- The qualitative image data is self-contained under `testset/`; it does not depend on internal-only source folders.
- `testset/2Dcell/` uses a small 2D fluorescence COCO-format qualitative subset.
- `testset/organoid/` uses a small organoid brightfield COCO-format qualitative subset.
- `testset/mitosis/` retains the original annotated mitosis qualitative subset.
- Example counts are allowed to differ across targets; the folder is for qualitative inspection, not balanced evaluation.
- Detector weights are not copied into this folder; inference uses the existing system-registered checkpoints.
