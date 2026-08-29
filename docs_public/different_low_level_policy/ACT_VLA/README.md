# ACT / VLA Low-Level Policy Baseline

This directory contains the ACT (Action Chunking with Transformers) baseline used to
compare end-to-end VLA-style low-level execution against the EIMS
planner/executor/checker workflow. The ACT pipeline learns a visuomotor policy from
human demonstrations and runs it in closed loop on either a serial robot arm or an
Olympus microscope, without a natural-language planning layer.

## Directory Structure

| Path | Content |
| --- | --- |
| `Micromanipulation_tool/` | Full ACT pipeline: data collection, dataset conversion, policy training, and real-time inference. |
| `Micromanipulation_tool/configs/` | Tracked example YAML configuration for training. |
| `ACT_for_microscopy/` | Placeholder for the external pretrained-weight bundle hosted on Hugging Face. |

## Micromanipulation_tool

The pipeline consists of four sequential scripts:

| Step | Script | Description |
| --- | --- | --- |
| 1 | `1_recorde.py` | Record keyboard-driven demonstrations from hardware (Daheng camera + robot arm or Olympus microscope). |
| 2 | `2_data_processing.py` | Convert recorded episodes into HDF5 training datasets. |
| 3 | `3_model_train.py` | Train the ACT (DETR/CVAE) policy on the HDF5 dataset. |
| 4 | `4_model_inference.py` | Run the trained policy in closed loop on hardware with video and action logging. |

### Key Subdirectories

| Path | Content |
| --- | --- |
| `model/` | ACT policy implementation (`policy.py`, `constants.py`, `utils.py`) and the DETR/CVAE backbone (`model/detr/`, modified from [facebookresearch/detr](https://github.com/facebookresearch/detr) under Apache 2.0). |
| `utils/` | Hardware interfaces: `camera.py` (Daheng), `robot.py` (serial arm), `olympus.py` (microscope via Micro-Manager), `agent.py` (runtime sync), `task_interfaces.py` (task routing), `image_processing.py`. |
| `scripts/` | Auxiliary batch scripts such as `run_loop_train.sh`. |
| `test/` | Utility scripts for data inspection, HDF5 export, and pkl processing. |

### Task Configuration

Training and inference settings are defined in `model/constants.py` via the
`TASK_CONFIGS` dictionary. Command-line arguments can override individual training
parameters. The `--config` flag in `3_model_train.py` accepts the tracked example YAML
at `Micromanipulation_tool/configs/training_config.example.yaml`, or any compatible
external YAML you provide.

### Environment

Python 3.10 with conda is recommended. See `Micromanipulation_tool/README.md` for full
setup instructions, including hardware-specific dependencies (Daheng SDK,
Micro-Manager).

## ACT_for_microscopy

This subdirectory is a placeholder. Pretrained weights and training configurations are
hosted externally on Hugging Face and should be downloaded separately:

```bash
git clone https://huggingface.co/404lzh/ACT_for_microscopy
```

Place the cloned contents under this directory.

## Relationship to EIMS

The ACT baseline is not part of the EIMS runtime. It is provided solely as a
comparative low-level policy baseline. Where EIMS decomposes a natural-language
instruction into a tool-mediated plan executed by a checker-verified executor, the ACT
pipeline maps images and states directly to actions through a learned policy.

## Data and Output Policy

Large artifacts such as raw recordings, HDF5 datasets, checkpoints, videos, and logs
are excluded from the repository via `.gitignore`. Only source code, configuration, and
documentation are published. Datasets and pretrained weights are distributed through
Hugging Face.
