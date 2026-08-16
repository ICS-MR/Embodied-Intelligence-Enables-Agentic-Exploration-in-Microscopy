# Mircomanipulation_tool

本项目提供一套面向硬件闭环微操作任务的完整流程，覆盖数据采集、数据集转换、ACT 风格策略训练和实时模型推理。项目通过统一的任务接口同时支持机械臂操作任务和 Olympus 显微镜控制任务。

## 项目功能

- 从 Daheng 相机和串口机械臂高频读取图像与状态。
- 使用统一任务适配层管理机械臂和 Olympus 显微镜接口。
- 支持键盘控制的数据示教采集，同步保存图像、动作、阶段标签和状态。
- 将采集 episode 转换为 HDF5 训练数据集。
- 基于 DETR/CVAE 模型结构训练 ACT 策略。
- 支持硬件闭环推理，并保存推理视频与运行日志。

## 项目结构

```text
.
├── 1_recorde.py              # 数据采集入口
├── 2_data_processing.py      # episode 数据转 HDF5 数据集
├── 3_model_train.py          # ACT 策略训练入口
├── 4_model_inference.py      # 硬件闭环模型推理入口
├── model/                    # 策略、数据工具和 DETR/CVAE 模型代码
├── utils/                    # 硬件接口与任务适配层
│   ├── agent.py              # 机械臂运行时与同步逻辑
│   ├── camera.py             # Daheng 相机接口
│   ├── robot.py              # 串口机械臂接口
│   ├── olympus.py            # Olympus 显微镜接口
│   ├── task_interfaces.py    # 机械臂/显微镜统一任务选择层
│   └── image_processing.py   # 显微镜图像处理工具
├── configs/                  # 训练配置文件
├── scripts/                  # 辅助批处理脚本
├── test/                     # 数据检查与转换辅助脚本
├── requirements.txt          # pip 依赖
├── environment.yml           # conda 环境配置
└── .gitignore                # 开源仓库忽略规则
```

## 环境配置

推荐使用 Python 3.8 和 conda 环境。

```bash
conda env create -f environment.yml
conda activate micromanipulation
```

也可以手动创建环境：

```bash
conda create -n micromanipulation python=3.8
conda activate micromanipulation
pip install -r requirements.txt
```

硬件相关依赖可能需要单独安装：

- Daheng 相机 SDK：需要安装厂商 SDK，并确保 Python 可以导入 `gxipy`。
- Olympus/Micro-Manager：需要安装 Micro-Manager，并配置环境变量：

```bash
export MICRO_MANAGER_DIR=/path/to/Micro-Manager
export MICRO_MANAGER_CONFIG=/path/to/config.cfg
```

## 使用流程

### 1. 采集示教数据

使用 `1_recorde.py` 采集同步的动作、图像、阶段标签和状态数据。

机械臂任务示例：

```bash
python 1_recorde.py \
  --task_name task_Splicing_3 \
  --backend robot \
  --control_mode xy \
  --root_folder /home/nova/mir
```

显微镜任务示例：

```bash
python 1_recorde.py \
  --task_name task_Cell_set_z_none \
  --backend microscope \
  --control_mode z \
  --root_folder /home/nova/mir
```

`--backend auto` 会根据任务名称自动选择接口。普通微操作或拼接类任务默认使用机械臂与相机；细胞或显微镜控制类任务默认使用 Olympus 显微镜接口。

采集时常用按键：

- `y`：开始记录
- `n`：停止记录
- `space`：切换阶段标签
- `delete`：标记当前 episode 删除
- `esc` 或 `q`：退出，具体取决于当前后端

采集数据保存结构：

```text
<root_folder>/<task_name>/epoch_N/
├── Action/
├── Observations/
│   ├── img/
│   ├── qpos/
│   └── stage/
└── ...
```

### 2. 转换数据集

运行前先修改 `2_data_processing.py` 顶部的任务名和路径变量，然后执行：

```bash
python 2_data_processing.py
```

脚本会将采集的 episode 文件夹转换为 HDF5 数据集，主要字段如下：

```text
/action
/observations/qpos
/observations/stage
/observations/images/top
```

转换前建议检查：

- `task`
- `root_folder`
- `task_name`
- `dataset_folder`
- 每个 episode 的图像、动作和状态长度

### 3. 训练策略模型

训练可以通过命令行参数或 `configs/` 下的 YAML 配置文件控制。

```bash
python 3_model_train.py \
  --dataset_dir /home/nova/mir/dataset/dataset_Splicing_2 \
  --ckpt_dir /home/nova/mir/result/Splicing_2/cs30_1e-04 \
  --batch_size 8 \
  --num_epochs 1000 \
  --lr 1e-4 \
  --chunk_size 30
```

使用配置文件：

```bash
python 3_model_train.py --config configs/Splicing_2_cs30_1e-04.yaml
```

批量训练辅助脚本位于 `scripts/`：

```bash
bash scripts/run_loop_train.sh
```

训练结果会保存到 checkpoint 目录，包括模型权重、日志和数据统计信息。

### 4. 运行模型推理

机械臂推理示例：

```bash
python 4_model_inference.py \
  --task_name Splicing_2 \
  --backend robot \
  --control_mode xy \
  --ckpt_dir /home/nova/mir/result/Splicing_2/cs30_1e-04 \
  --record_epoch 09
```

显微镜推理示例：

```bash
python 4_model_inference.py \
  --task_name Cell_set_z_none \
  --backend microscope \
  --control_mode z \
  --ckpt_dir /home/nova/mir/result/Cell_set_z_none/cs30_1e-04
```

推理默认将视频和日志保存到 `/home/nova/videos/`。如需修改输出位置，可以使用 `--video_filename`。

## 任务与接口选择

任务路由逻辑位于 `utils/task_interfaces.py`。

常见模式：

| 后端 | 控制模式 | 典型任务 |
| --- | --- | --- |
| `robot` | `xy` | 机械臂微操作或拼接任务 |
| `microscope` | `z` | 显微镜聚焦或 Z 轴控制 |
| `microscope` | `brightness` | 亮度控制 |
| `microscope` | `exposure` | 曝光控制 |
| `microscope` | `xy` | 显微镜载物台移动 |

示例：

```bash
python 1_recorde.py --task_name task_Splicing_3 --backend auto
python 1_recorde.py --task_name task_Cell_set_brightness_none --backend auto
python 1_recorde.py --task_name task_Cell_move_funa --backend auto
```

新增任务时，优先修改 `utils/task_interfaces.py` 中的 `resolve_task_profile()`。建议将任务名、后端、控制模式、采样间隔和硬件预设集中写在同一个 profile 中，保证采集和推理使用一致的接口。

## 重要文件与修改位置

### `utils/task_interfaces.py`

统一选择机械臂或显微镜接口的核心入口。新增以下内容时优先修改该文件：

- 新任务名称
- 后端选择规则
- 控制模式
- 显微镜预设，例如 dichroic、亮度、曝光、初始 Z 位置
- 不同任务的采样间隔

### `1_recorde.py`

数据采集入口。修改以下内容时使用该文件：

- 采集相关命令行参数
- episode 命名与保存路径
- 每一帧保存的数据字段
- 采集频率行为

### `2_data_processing.py`

数据集转换入口。修改以下内容时使用该文件：

- 原始采集数据路径
- 输出数据集路径
- episode 长度处理方式
- HDF5 字段布局
- 相机名称或图像布局

### `3_model_train.py`

模型训练入口。修改该文件或 `configs/` 下的 YAML 文件可以调整：

- 数据集路径
- checkpoint 保存路径
- batch size
- 学习率
- chunk size
- 训练轮数
- 模型超参数

### `4_model_inference.py`

模型推理入口。修改以下内容时使用该文件：

- checkpoint 加载方式
- 策略 rollout 时间逻辑
- 动作后处理
- 视频和日志输出路径
- 硬件后端参数

### `utils/robot.py`

串口机械臂接口。只有在修改以下内容时建议调整该文件：

- 串口协议
- 默认端口或波特率
- 位置解析方式
- 控制命令格式
- 运动限制或速度

### `utils/camera.py`

Daheng 相机接口。修改以下内容时使用该文件：

- 相机初始化
- 帧读取逻辑
- 图像分辨率或颜色转换
- OpenCV 显示行为

### `utils/olympus.py`

Olympus 显微镜接口。修改以下内容时使用该文件：

- Micro-Manager 设备名称
- 载物台、焦距、亮度或曝光控制命令
- 显微镜专用键盘行为
- 显微镜运动安全限制

### `model/constants.py`

任务级模型配置。修改以下内容时使用该文件：

- `episode_len`
- `num_episodes`
- `camera_names`
- 默认数据集配置

## 数据与输出文件管理

以下大文件或生成文件不建议提交到开源仓库：

- 原始采集数据
- HDF5 数据集
- 推理视频
- checkpoint
- 日志
- Python 缓存
- 本地环境目录

这些内容已经由 `.gitignore` 覆盖。公开仓库中建议只保留源码、配置文件和文档。

## 注意事项

- 本项目默认运行环境可以访问机械臂、Daheng 相机和/或 Olympus 显微镜硬件。
- 脚本中的默认路径包含本地开发路径。为了增强可迁移性，优先使用命令行参数；没有参数的位置需要在运行前手动修改路径变量。
- 阅读代码不需要数据集或训练权重，但训练和推理需要准备好的 HDF5 数据集与 checkpoint 目录。
- 当前模型代码保持函数名称和主要逻辑稳定，以兼容已有 checkpoint 和运行脚本。
