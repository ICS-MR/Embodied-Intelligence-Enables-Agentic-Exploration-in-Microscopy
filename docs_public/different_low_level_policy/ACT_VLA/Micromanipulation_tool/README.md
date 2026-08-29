# Micromanipulation Tool

This project provides a hardware-in-the-loop micromanipulation workflow for data collection, dataset conversion, ACT-style policy training, and real-time model inference. It supports both robot-arm manipulation tasks and microscope-control tasks through a unified task interface.

## Features

- High-frequency image and state acquisition from a Daheng camera and serial robot arm.
- Unified task adapter for robot-arm and Olympus microscope workflows.
- Keyboard-driven demonstration recording with synchronized image, action, and state data.
- Dataset conversion from recorded episodes to HDF5 training files.
- ACT policy training based on the DETR/CVAE model stack.
- Closed-loop inference with video recording and action logging.

## Pretrained Models and Datasets

The policies used in this project are ACT action-chunking models. The supported tasks include micromanipulator operations, such as push-to-target, and Olympus microscope-control tasks, such as focusing, stage movement, and brightness adjustment. The weights for these tasks were trained with ACT, and the corresponding training parameters are saved with each set of weights.

The pretrained weights and their training configurations are available on Hugging Face:

```bash
git clone https://huggingface.co/404lzh/ACT_for_microscopy
```

No suitable open-source datasets are currently available for these tasks. Therefore, more than 150 demonstration episodes were collected for each task type. The datasets are available on Hugging Face:

```bash
git clone https://huggingface.co/datasets/404lzh/Data_of_Micromanipulation
```

## Project Structure

```text
.
├── 1_recorde.py              # Record demonstrations from hardware
├── 2_data_processing.py      # Convert recorded episodes to HDF5 datasets
├── 3_model_train.py          # Train ACT policy checkpoints
├── 4_model_inference.py      # Run policy inference on hardware
├── model/                    # Policy, dataset utilities, and DETR/CVAE model code
├── utils/                    # Hardware interfaces and task adapters
│   ├── agent.py              # Robot-arm runtime and synchronization logic
│   ├── camera.py             # Daheng camera interface
│   ├── robot.py              # Serial robot-arm interface
│   ├── olympus.py            # Olympus microscope interface
│   ├── task_interfaces.py    # Unified robot/microscope task selection layer
│   └── image_processing.py   # Microscope image-processing helpers
├── configs/                  # Tracked training configuration examples
├── scripts/                  # Legacy batch helper; requires user-provided YAML files
├── test/                     # Utility scripts for data inspection and conversion checks
├── requirements.txt          # Pip dependencies
├── environment.yml           # Conda environment definition
└── .gitignore                # Open-source ignore rules for caches, data, logs, and outputs
```

## Environment Setup

The recommended runtime is Python 3.10 in a conda environment.

```bash
conda env create -f environment.yml
conda activate micromanipulation
```

Or install manually:

```bash
conda create -n micromanipulation python=3.10
conda activate micromanipulation
pip install -r requirements.txt
```

Hardware-specific dependencies may require separate installation:

- Daheng camera SDK: install the vendor SDK and make sure `gxipy` is importable.
- Olympus/Micro-Manager: install Micro-Manager and configure the environment variables before starting Python.
- The microscope/API 75 package set in `requirements.txt` pins `pymmcore==11.14.0.75.0` and `pymmcore-plus==0.16.0`.

Linux/macOS:

```bash
export MICRO_MANAGER_DIR=/path/to/Micro-Manager
export MICRO_MANAGER_CONFIG=/path/to/config.cfg
```

Windows PowerShell:

```powershell
$env:MICRO_MANAGER_DIR = "C:\path\to\Micro-Manager"
$env:MICRO_MANAGER_CONFIG = "C:\path\to\config.cfg"
```

Before running Python, open the same Micro-Manager configuration in the GUI and confirm that the camera and stage initialize on the expected COM ports.

## Basic Workflow

Paths beginning with `/path/to/` in the commands below are placeholders, not directories included in this repository. Replace them with paths on your own system. Windows paths such as `C:\data\micromanipulation` can also be used.

### 1. Record Demonstrations

Use `1_recorde.py` to collect synchronized action, image, and state data.

List the tasks that the recording entry point can resolve without opening hardware:

```bash
python 1_recorde.py --list_tasks
```

Robot-arm task:

```bash
python 1_recorde.py \
  --task_name task_Push_to_target \
  --backend robot \
  --control_mode xy \
  --root_folder /path/to/recordings
```

Microscope task:

```bash
python 1_recorde.py \
  --task_name task_Cell_set_z_none \
  --backend microscope \
  --control_mode z \
  --root_folder /path/to/recordings
```

`--backend auto` can select the interface from the task name. Robot-style tasks use the robot arm and camera. Cell/microscope tasks use the Olympus microscope interface.

Keyboard controls during recording:

- `y`: start recording
- `n`: stop recording
- `delete`: mark current episode for deletion
- `esc` or `q`: exit, depending on the active backend

For microscope `set_z` tasks, stopping recording with `n` returns Z to the task's configured stop position. Other microscope tasks do not reset Z on stop.

Recorded data is saved under:

```text
<root_folder>/<task_name>/epoch_N/
├── Action/
├── Observations/
│   ├── img/
│   └── qpos/
└── ...
```

### 2. Convert Episodes to Dataset

Convert a recorded task folder to ACT HDF5 episodes:

```bash
python 2_data_processing.py \
  --task Push_to_target \
  --root_folder /path/to/recordings_root \
  --dataset_folder /path/to/hdf5_dataset
```

By default, image frames are stored as raw HDF5 arrays. To create ACT/ALOHA-style compressed HDF5 files, store each frame as padded JPEG bytes and write `/compress_len`:

```bash
python 2_data_processing.py \
  --task Cell_set_z_none \
  --root_folder /path/to/recordings_root \
  --dataset_folder /path/to/hdf5_dataset \
  --compress \
  --jpeg_quality 50
```

`model/utils.py` automatically detects `attrs["compress"]` when loading HDF5 episodes, so training can read both raw and compressed datasets.

`--root_folder` must point to the parent directory that contains the recorded task folder. For example, if recordings live in `/run/media/nova26/LuZhihui/task/task_Cell_set_z_none/epoch_0/...`, then set `--root_folder /run/media/nova26/LuZhihui/task` and `--task_name task_Cell_set_z_none`.

The script converts recorded episode folders into HDF5 files with the following structure:

```text
/action
/observations/qpos
/observations/images/top
/compress_len              # compressed datasets only
```

Before running conversion, check:

- `task`
- `root_folder`
- `task_name`
- `dataset_folder`
- `compress`
- `jpeg_quality`
- episode image/action/state lengths

### 3. Train the Policy

Pass the training settings directly as command-line arguments. Always specify `--dataset_dir` and `--ckpt_dir` because the defaults in `3_model_train.py` are local development paths and are not portable.

```bash
python 3_model_train.py \
  --dataset_dir /path/to/hdf5_dataset \
  --ckpt_dir /path/to/output_directory \
  --batch_size 8 \
  --num_epochs 1000 \
  --lr 1e-4 \
  --chunk_size 30
```

Training arguments:

| Argument | Description |
| --- | --- |
| `--dataset_dir` | Directory containing the prepared HDF5 episode files. |
| `--ckpt_dir` | Output directory for checkpoints, dataset statistics, the saved run configuration, and training plots. |
| `--batch_size` | Number of samples in each training and validation batch. |
| `--num_epochs` | Number of training epochs. |
| `--lr` | Policy learning rate. |
| `--chunk_size` | Number of future actions predicted in each ACT action chunk. |
| `--kl_weight` | Weight applied to the CVAE KL-divergence loss. |
| `--hidden_dim` | Transformer hidden dimension. |
| `--dim_feedforward` | Transformer feed-forward dimension. |
| `--seed` | Random seed used for reproducible training. |

Alternatively, copy or edit the tracked configuration example and replace both path values before running it:

```bash
python 3_model_train.py --config configs/training_config.example.yaml
```

The `.example` suffix indicates that the file is a task-neutral template. In particular, `/path/to/hdf5_dataset` and `/path/to/output_directory` are not real directories and must be replaced with paths on your system. The example groups the same command-line settings into path, training, and ACT model sections.

Configuration values are applied after command-line parsing, so a YAML value overrides a command-line argument with the same name. Avoid specifying the same setting in both places. Only arguments declared in `3_model_train.py` can be loaded from this YAML file. The task-level values `episode_len` and `camera_names` remain configured in `model/constants.py`.

Training outputs checkpoints, policy statistics, the resolved run configuration, and training plots into the directory supplied through `--ckpt_dir`.
The saved `config.yaml` in that directory is also used by inference when it is present.

### 4. Run Model Inference

Robot-arm inference:

```bash
python 4_model_inference.py \
  --task_name Push_to_target \
  --backend robot \
  --control_mode xy \
  --ckpt_dir /path/to/checkpoint_directory \
  --run_id 09 \
  --video_filename /path/to/output/robot_inference.mp4
```

Microscope inference:

```bash
python 4_model_inference.py \
  --task_name Cell_set_z_none \
  --backend microscope \
  --control_mode z \
  --ckpt_dir /path/to/checkpoint_directory \
  --run_id 09 \
  --video_filename /path/to/output/microscope_inference.mp4
```

Use `--video_filename` to select the video output path. If it is omitted, the script uses a local development default that may not exist on another system.
`--run_id` is used as an inference run label for default video/log paths. It applies to both robot-arm and microscope inference. `--record_epoch` remains available as a deprecated compatibility alias.
When `--ckpt_dir/config.yaml` exists, `4_model_inference.py` reads the saved training values and uses the checkpoint's `chunk_size`, `kl_weight`, `hidden_dim`, and `dim_feedforward` automatically.
The checkpoint directory must contain at least `policy_best.ckpt` and `dataset_stats.pkl`; `config.yaml` is preferred.

For repeated training runs, `scripts/run_loop_train.sh` loads configuration files matching `configs/Push_to_target*.yaml` and writes logs to `logs/`. The tracked `training_config.example.yaml` is a template and is not matched by this script.

## Task and Interface Selection

The task-routing logic is implemented in `utils/task_interfaces.py`. Task names are resolved through an explicit registry, not by open-ended string guessing. The resolver accepts recording names such as `task_Cell_set_z_none` and dataset artifact names such as `dataset_Cell_set_z_none_compressed.zip`.

Supported collection tasks:

| Task key | Backend | Control mode |
| --- | --- | --- |
| `2d_move_none` | `microscope` | `xy` |
| `2d_set_brightness_none` | `microscope` | `brightness` |
| `2d_set_z_none` | `microscope` | `z` |
| `cell_move_none` | `microscope` | `xy` |
| `cell_set_brightness_none` | `microscope` | `brightness` |
| `cell_set_z_none` | `microscope` | `z` |
| `slice_move_none` | `microscope` | `xy` |
| `slice_set_brightness_none` | `microscope` | `brightness` |
| `slice_set_z_none` | `microscope` | `z` |
| `push_to_target` | `robot` | `xy` |
| `cell_move_funa` | `microscope` | `xy` |

If `--backend` or `--control_mode` conflicts with the registered task profile, the script raises a clear error before opening the hardware.

Common modes:

| Backend | Control mode | Typical task |
| --- | --- | --- |
| `robot` | `xy` | Arm-based push-to-target manipulation |
| `microscope` | `z` | Focus or vertical-stage control |
| `microscope` | `brightness` | Illumination adjustment |
| `microscope` | `exposure` | Camera exposure adjustment |
| `microscope` | `xy` | Microscope stage movement |

Examples:

```bash
python 1_recorde.py --task_name task_Push_to_target --backend auto
python 1_recorde.py --task_name task_Cell_set_brightness_none --backend auto
python 1_recorde.py --task_name task_Cell_move_funa --backend auto
```

To add a new task, update `SUPPORTED_TASK_PROFILES` in `utils/task_interfaces.py`. Keep the task name, backend, control mode, interval, and hardware presets in one profile so recording and inference use the same interface.

## Important Files to Modify

### `utils/task_interfaces.py`

Main entry point for choosing robot or microscope interfaces. Modify this file when adding:

- new task names
- backend-selection rules
- control modes
- microscope presets such as dichroic mirror, brightness, exposure, or initial Z position
- recording interval per task

### `1_recorde.py`

Recording entry point. Modify this file when changing:

- command-line arguments for recording
- episode naming and save paths
- data fields saved per frame
- recording frequency behavior

### `2_data_processing.py`

Dataset conversion entry point. Modify this file when changing:

- source recording path
- output dataset path
- episode length handling
- HDF5 field layout
- camera names or image layout

### `3_model_train.py`

Training entry point. Pass common run settings through command-line arguments. Modify this file only when changing behavior that is not exposed as an argument:

- dataset path
- checkpoint path
- batch size
- learning rate
- chunk size
- number of epochs
- model hyperparameters

### `4_model_inference.py`

Inference entry point. Modify this file when changing:

- checkpoint loading
- policy rollout timing
- action post-processing
- video/log output path
- hardware backend arguments

### `utils/robot.py`

Serial robot-arm interface. Modify this file only when changing:

- serial protocol
- port or baud-rate defaults
- position parsing
- command format
- motion limits or speed

### `utils/camera.py`

Daheng camera interface. Modify this file when changing:

- camera initialization
- frame acquisition logic
- image resolution or color conversion
- OpenCV display behavior

### `utils/olympus.py`

Olympus microscope interface. Modify this file when changing:

- Micro-Manager device names
- stage, focus, brightness, or exposure commands
- microscope-specific keyboard behavior
- safety limits for microscope motion

### `model/constants.py`

Task-level model constants. Update this file when changing:

- `episode_len`
- `camera_names`
- default dataset configuration

## Data and Output Policy

Large generated artifacts should not be committed:

- raw recordings
- HDF5 datasets
- videos
- checkpoints
- logs
- Python caches
- local environment folders

These are covered by `.gitignore`. Keep only source code, configuration files, and documentation in the public repository.

## Notes

- This repository assumes direct access to the robot arm, Daheng camera, and/or Olympus microscope hardware.
- Default paths in scripts are local development paths. Replace every `/path/to/...` placeholder in this README with a path on your system, and pass path arguments explicitly where available.
- No dataset or trained checkpoint is required to read the code, but training and inference require prepared HDF5 datasets and checkpoint directories.
- The current model code keeps function names and behavior stable for compatibility with existing checkpoints and scripts.
