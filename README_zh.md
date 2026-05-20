# open-audio-opd 中文说明

工业级音频在线策略蒸馏训练代码。

本项目基于 [THUNLP/OPD](https://github.com/thunlp/OPD/) 以及 [verl](https://github.com/volcengine/verl)。仓库内已 vendored `verl/`，用于真实 ASR OPD FSDP2 训练脚本的 FSDP2 封装、梯度裁剪和 checkpoint 管理。

## 核心训练脚本

真实训练入口是：

```bash
scripts/train/train_ark_asr_opd_fsdp2_resume.py
```

这个脚本来自实际 ASR OPD 训练流程，支持：

- Qwen3-ASR teacher forcing / Transformers / vLLM teacher backend；
- 学生模型 FSDP2 训练；
- sparse union-support KL；
- checkpoint 保存与 `latest`/`auto` 恢复；
- 多机 hostfile 启动脚本。

模型路径、训练数据、输出目录和 Qwen3-ASR backend 代码路径都必须显式传入；仓库不内置私有数据、模型权重或本机路径。

## 单机启动示例

```bash
torchrun --nproc_per_node 8 scripts/train/train_ark_asr_opd_fsdp2_resume.py \
  --student_model /path/to/student_model \
  --teacher_model /path/to/qwen3_asr_model \
  --qwen3_asr_code_path /path/to/qwen3-asr/backend \
  --train_data /path/to/train.jsonl \
  --output_dir runs/ark_asr_opd_fsdp2 \
  --teacher_backend qwen3_asr_teacher_forcing \
  --calibrate_only False \
  --save_freq 1000
```

## 多机 hostfile 启动

```bash
HOSTFILE=/path/to/hostfile \
STUDENT_MODEL=/path/to/student_model \
TEACHER_MODEL=/path/to/qwen3_asr_model \
QWEN3_ASR_CODE_PATH=/path/to/qwen3-asr/backend \
TRAIN_DATA=/path/to/train.jsonl \
OUTPUT_DIR=runs/ark_asr_opd_fsdp2 \
scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
```

hostfile 示例：

```text
node0 slots=8
node1 slots=8
node2 slots=8
```

## 恢复训练

```bash
torchrun --nproc_per_node 8 scripts/train/train_ark_asr_opd_fsdp2_resume.py \
  --student_model /path/to/student_model \
  --teacher_model /path/to/qwen3_asr_model \
  --qwen3_asr_code_path /path/to/qwen3-asr/backend \
  --train_data /path/to/train.jsonl \
  --output_dir runs/ark_asr_opd_fsdp2 \
  --resume_from_checkpoint latest \
  --calibrate_only False
```

## 数据格式

训练 JSONL 至少需要：

- `audio`: 音频文件路径；
- `text`: 参考转写文本；
- `task`: 可选，存在时必须为 `asr`；
- `begin_time` / `end_time`: 可选，音频切片起止时间，单位秒。

## 安装

建议在 CUDA/PyTorch 环境中安装：

```bash
pip install -e .
```

如果需要独立安装 vendored verl：

```bash
pip install -e ./verl
```

Qwen3-ASR teacher backend 依赖外部 Qwen3-ASR Transformers backend 代码，请通过 `--qwen3_asr_code_path` 显式传入。
