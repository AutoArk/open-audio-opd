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

## 实验结果

Ark-ASR 是 0.6B 参数规模的 ASR student 模型。本组 OPD 实验只使用了 10 万小时
ASR 音频数据。公开的 Qwen3-ASR technical report 没有披露实际 wall-clock 训练耗时，
但披露了 Qwen3-ASR 采用多阶段训练流程，其中仅 AuT encoder 预训练阶段就使用约
4000 万小时伪标注 ASR 音频数据，之后还包括 Omni training、ASR SFT 和 ASR RL。
在这个对比下，Ark-ASR 使用的数据规模约为 Qwen3-ASR 公开披露 ASR 预训练音频规模的
1/400，却已经达到与 Qwen3-ASR-0.6B baseline 可比甚至更优的水平。

`Ark-Base` 表示在 10 万小时 ASR 数据上 SFT 得到的 0.6B checkpoint。`TD`
表示 teacher-data adaptation，即在 Ark-Base 上用 2000 小时 teacher-generated
ASR 数据做适配。`OPD` 表示使用 Qwen-ASR teacher 的 on-policy distillation。

| 模型 | aishell-1 (CER) | Wenet-meeting (CER) | Wenet-net (CER) | Libri-clean (WER) | Libri-other (WER) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ark-Base (0.6B) | 3.48% | 10.22% | 7.74% | 3.75% | 7.17% |
| Ark-Base+OPD (0.6B) | 3.00% | 7.18% | 6.13% | 2.88% | 5.50% |
| Ark-Base+TD+OPD (0.6B) | 1.94% | 6.11% | 5.41% | 2.77% | 4.88% |
| Qwen3-ASR-1.7B | 1.50% | 4.69% | 4.55% | 2.20% | 4.05% |
| Qwen3-ASR-0.6B | 2.07% | 5.57% | 5.45% | 2.81% | 5.05% |

CER/WER 越低越好。

主要结论：

- Ark-ASR 只使用 10 万小时音频，就已经达到与 Qwen3-ASR 大规模训练体系可对比的水平，
  体现了 OPD 路线在数据效率上的优势。
- Ark-Base 是直接的 10 万小时监督微调基线。在同一个 0.6B student 上继续引入 OPD 后，
  所有评测集都明显优于 Ark-Base，说明 OPD 能在标准 SFT 之外进一步迁移 teacher 的 ASR 能力。
- Ark-Base+OPD 从 Ark-Base 出发，再用 Qwen-ASR teacher 在同一数据集上进行 OPD。
  这个版本在 LibriSpeech clean/other 上已经接近 Qwen3-ASR-0.6B，说明在远小于 4000
  万小时公开预训练规模的数据条件下，OPD 仍然能有效迁移 ASR 能力。
- Ark-Base+TD+OPD 是更优路线：aishell-1 从 3.00% CER 提升到 1.94%，Wenet-meeting
  从 7.18% CER 提升到 6.11%，Wenet-net 从 6.13% CER 提升到 5.41%，Libri-clean
  从 2.88% WER 提升到 2.77%，Libri-other 从 5.50% WER 提升到 4.88%。
- 在同为 0.6B 参数规模的对比下，Ark-Base+TD+OPD 整体已经超过 Qwen3-ASR-0.6B，
  在 aishell-1、Wenet-net、Libri-clean、Libri-other 上均取得更好结果。
- Qwen3-ASR-1.7B 仍是表中最强模型，但它参数规模更大，且背后是更大规模的公开训练流程。
  当前结果说明，TD + OPD 能用 10 万小时数据把 0.6B ASR 模型推到接近
  大模型基线的水平，是一条高数据效率的训练路径。

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

## 推理

只做 ASR 推理、不计算指标：

```bash
python scripts/infer/ark_asr_transformers.py \
  --input /path/to/input.jsonl \
  --output runs/infer/predictions.jsonl \
  --model_path /path/to/student_or_exported_model \
  --processor_path /path/to/student_or_exported_model \
  --batch_size 40 \
  --dtype float16 \
  --attn_impl sdpa
```

输出 JSONL 会保留输入 metadata，并新增：

- `pred_text`: 清洗后的预测文本；
- `pred_text_raw`: 清洗前的原始 decode 文本。

## 评测

对单个 JSONL 运行 J/WER 评测：

```bash
python scripts/eval/eval_jwer_ark_asr_transformers.py \
  --input /path/to/test_aishell.jsonl \
  --output runs/eval/test_aishell_result.jsonl \
  --model_path /path/to/student_or_exported_model \
  --processor_path /path/to/student_or_exported_model \
  --batch_size 40 \
  --dtype float16 \
  --attn_impl sdpa
```

评测输出会按 `cer_errors` 从高到低排序，方便优先查看 bad case。每行包含
`ref_text`、`pred_text`、清洗后的文本、`wer_errors`、`cer_errors`、`ref_words`
和 `ref_chars`。

如果 `text_process` 文本归一化在单独环境中，传入：

```bash
--text_normalize_python /path/to/wetext/bin/python
```

多 GPU 五套集评测可以使用开源化后的 launcher，不包含任何硬编码数据路径：

```bash
MODEL_PATH=/path/to/exported_or_checkpoint_model \
EVAL_DATA_DIR=/path/to/eval_jsonl_dir \
OUTPUT_DIR=runs/eval/arkasr_step30000 \
SUFFIX=step30000 \
GPUS="0 1 2 3 4" \
PRESETS="aishell clean meeting net other" \
scripts/eval/run_arkasr_eval.sh
```

launcher 会读取 `EVAL_DATA_DIR` 下的 `test_${preset}.jsonl`，并把日志、pid
和结果 JSONL 写到 `OUTPUT_DIR`。仓库不包含任何评测数据文件。

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
python3 -m py_compile scripts/infer/ark_asr_transformers.py
python3 -m py_compile scripts/eval/eval_jwer_ark_asr_transformers.py
bash -n scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
bash -n scripts/eval/run_arkasr_eval.sh
python scripts/train/train_ark_asr_opd_fsdp2_resume.py --help
```

其中 `--help` 需要在已安装训练依赖的环境中运行，包括 `numpy`、`torch`、
`datasets`、`transformers`、`omegaconf` 和 `pyproject.toml` 中列出的 `verl` 依赖。
