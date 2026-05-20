# open-audio-opd 中文说明

工业级 ASR 在线策略蒸馏训练代码。

## 项目概览

`open-audio-opd` 是用于音频 ASR 场景的在线策略蒸馏训练仓库。核心训练入口是：

```bash
scripts/train/train_ark_asr_opd_fsdp2_resume.py
```

本项目基于 [THUNLP/OPD](https://github.com/thunlp/OPD/) 以及
[verl](https://github.com/volcengine/verl)。仓库内包含精简后的 vendored
`verl/`，用于 FSDP2 封装、梯度裁剪和 checkpoint 管理，因此不依赖另一份本机
`verl` checkout。

仓库不包含模型权重、音频文件、JSONL 数据集或私有机器路径。所有模型、数据和输出目录都需要显式传入。

## 训练在做什么

ASR OPD 的训练流程是：

```text
audio batch
  -> student 无梯度生成转写 token
  -> teacher 在同一音频和 student 转写条件下打分
  -> student 对自己的转写重新打分并保留梯度
  -> 构造 teacher/student union top-k support
  -> 在对齐后的转写位置优化 KL(teacher || student)
  -> 保存可恢复的 FSDP2 checkpoint
```

关键点是 teacher 不是提供一份静态标签，而是对 student 当前在线生成的结果打分。
因此 student 学到的是自己当前策略下的 teacher 偏好分布。

## Student 和 Teacher 是什么

`--student_model` 是要训练的音频 ASR 学生模型。它需要能通过
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载，并且
processor/tokenizer 支持脚本里的音频 prompt 格式。student 会被 FSDP2 包裹并参与反向传播。

`--teacher_model` 是更强的 ASR 教师模型，只用于评分，不参与训练。支持的 teacher backend：

- `qwen3_asr_teacher_forcing`: 默认生产路径，适合 Qwen3-ASR 风格 teacher。
- `qwen3_asr_transformers`: Qwen3-ASR Transformers backend。
- `qwen3_asr_vllm`: 需要额外安装匹配 vLLM 栈。
- `hf_causal_lm`: 通用 Hugging Face causal LM teacher 路径。

使用 `qwen3_asr_*` backend 时，需要通过 `--qwen3_asr_code_path` 传入本地
Qwen3-ASR Transformers backend 代码路径。该 backend 代码没有 vendored 到本仓库。

## 仓库结构

```text
scripts/train/train_ark_asr_opd_fsdp2_resume.py      # 主训练脚本
scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh # 多机启动脚本
configs/hostfile.example                             # hostfile 示例
verl/                                                # vendored verl 运行代码
README.md / README_zh.md                             # 使用文档
```

## 安装

先准备匹配集群的 CUDA/PyTorch 环境，然后安装本仓库依赖：

```bash
pip install -e .
```

如果你的环境需要单独安装 vendored `verl`：

```bash
pip install -e ./verl
```

如果使用 `qwen3_asr_vllm`：

```bash
pip install -e ".[vllm]"
```

## 数据格式

训练数据为 JSONL，每行一个 ASR 样本：

```json
{"audio":"/path/to/audio.wav","text":"reference transcript","task":"asr","begin_time":-1,"end_time":-1}
```

字段说明：

- `audio`: 必需，音频路径。
- `text`: 必需，参考转写文本，用于 ASR 监督和元信息。
- `task`: 可选；如果存在，必须为 `asr`。
- `begin_time`: 可选，音频切片起点，单位秒。`-1` 表示整段音频。
- `end_time`: 可选，音频切片终点，单位秒。`-1` 表示整段音频。

脚本遇到缺失音频会直接报错，不会用 fallback 音频静默替换坏样本。

## 单机训练

```bash
torchrun --nproc_per_node 8 scripts/train/train_ark_asr_opd_fsdp2_resume.py \
  --student_model /path/to/student_model \
  --teacher_model /path/to/qwen3_asr_model \
  --qwen3_asr_code_path /path/to/qwen3-asr/backend \
  --train_data /path/to/train.jsonl \
  --output_dir runs/ark_asr_opd_fsdp2 \
  --teacher_backend qwen3_asr_teacher_forcing \
  --calibrate_only False \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-6 \
  --opd_top_k 32 \
  --asr_opd_max_new_tokens 256 \
  --save_freq 1000
```

建议先用小 batch 和较短 `--asr_opd_max_new_tokens` 起步，确认生成长度、非空生成比例、
teacher 对齐情况和 `opd_valid_topk_mean` 正常后再扩大规模。

## 多机 hostfile 启动

hostfile 示例：

```text
node0 slots=8
node1 slots=8
node2 slots=8
```

启动命令：

```bash
HOSTFILE=/path/to/hostfile \
STUDENT_MODEL=/path/to/student_model \
TEACHER_MODEL=/path/to/qwen3_asr_model \
QWEN3_ASR_CODE_PATH=/path/to/qwen3-asr/backend \
TRAIN_DATA=/path/to/train.jsonl \
OUTPUT_DIR=runs/ark_asr_opd_fsdp2 \
NCCL_SOCKET_IFNAME=hpn0 \
GLOO_SOCKET_IFNAME=hpn0 \
scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
```

启动脚本要求显式提供 `HOSTFILE`、`STUDENT_MODEL`、`TEACHER_MODEL` 和 `TRAIN_DATA`。
当 `TEACHER_BACKEND` 以 `qwen3_asr_` 开头时，还必须提供 `QWEN3_ASR_CODE_PATH`。

## 恢复训练

从指定 checkpoint 恢复：

```bash
torchrun --nproc_per_node 8 scripts/train/train_ark_asr_opd_fsdp2_resume.py \
  --student_model /path/to/student_model \
  --teacher_model /path/to/qwen3_asr_model \
  --qwen3_asr_code_path /path/to/qwen3-asr/backend \
  --train_data /path/to/train.jsonl \
  --output_dir runs/ark_asr_opd_fsdp2 \
  --resume_from_checkpoint runs/ark_asr_opd_fsdp2/checkpoints/global_step_1000 \
  --calibrate_only False
```

从 `output_dir/checkpoints` 下最新 checkpoint 恢复：

```bash
--resume_from_checkpoint latest
```

`auto` 和 `latest` 等价。

## Calibration 模式

默认 `--calibrate_only True`。该模式只做前向和指标打印，不执行 optimizer step。
建议在正式训练前先跑 calibration，确认模型加载、数据加载、student rollout、teacher scoring
和 OPD 对齐都正常。

正式训练时传入：

```bash
--calibrate_only False
```

## 关键参数

- `--student_model`: 需要训练的 student ASR 模型路径或 HF repo id。
- `--teacher_model`: teacher ASR 模型路径或 HF repo id。
- `--teacher_backend`: teacher scoring 实现。
- `--qwen3_asr_code_path`: Qwen3-ASR teacher backend 所需代码路径。
- `--train_data`: JSONL 训练数据。
- `--output_dir`: 日志和 FSDP2 checkpoint 输出目录。
- `--hf_cache_dir`: Hugging Face datasets cache 目录。
- `--opd_top_k`: teacher/student top-k support 大小。
- `--opd_temperature`: OPD 分布温度。
- `--asr_block_token_id_from`: 生成时屏蔽非 ASR token id 的起点。
- `--asr_opd_max_new_tokens`: student rollout 最大长度。
- `--save_freq`: checkpoint 保存间隔，`-1` 表示不保存。
- `--resume_from_checkpoint`: checkpoint 目录、`latest` 或 `auto`。

## Smoke 检查

不需要模型权重的检查：

```bash
python3 -m py_compile scripts/train/train_ark_asr_opd_fsdp2_resume.py
bash -n scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
python scripts/train/train_ark_asr_opd_fsdp2_resume.py --help
```

其中 `--help` 需要在已安装训练依赖的环境中运行，包括 `numpy`、`torch`、
`datasets`、`transformers`、`omegaconf` 和 `pyproject.toml` 中列出的 `verl` 依赖。
