# open-audio-opd

Industrial ASR online policy distillation training code.

中文文档: [README_zh.md](README_zh.md)

## Overview

`open-audio-opd` contains the production ASR OPD training stack used to
distill an audio ASR student model from a stronger ASR teacher model. The core
training script is:

```bash
scripts/train/train_ark_asr_opd_fsdp2_resume.py
```

The repository is based on [THUNLP/OPD](https://github.com/thunlp/OPD/) and
[verl](https://github.com/volcengine/verl). A trimmed vendored copy of `verl/`
is included so the training script can use FSDP2 wrapping, gradient clipping,
and checkpoint management without depending on another local checkout.

No model weights, audio files, JSONL datasets, or private machine paths are
included. All model/data/output paths are explicit command-line arguments.

## What The Training Does

ASR OPD trains a student ASR model using online rollouts and teacher scores:

```text
audio batch
  -> student generates transcript tokens with no grad
  -> teacher scores the same audio plus the student transcript
  -> student scores its own transcript with gradients
  -> build teacher/student union top-k support
  -> optimize KL(teacher || student) on aligned transcript positions
  -> save FSDP2 checkpoints that can be resumed
```

The key point is that the teacher is not used to provide a static transcript
label. It scores what the student actually generated online, so the student is
trained on its own current behavior.

## Student And Teacher

`--student_model` is the trainable audio-capable ASR model. It must be loadable
with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and its
processor/tokenizer must support the audio prompt format used by the script.
The student is wrapped with FSDP2 and receives gradients.

`--teacher_model` is the stronger ASR model used for scoring. It is loaded in
eval mode and does not receive gradients. Supported teacher backends are:

- `qwen3_asr_teacher_forcing`: default production path for Qwen3-ASR-style teachers.
- `qwen3_asr_transformers`: Transformers backend for Qwen3-ASR.
- `qwen3_asr_vllm`: vLLM backend when the matching vLLM stack is installed.
- `hf_causal_lm`: generic Hugging Face causal LM teacher path.

For `qwen3_asr_*` backends, pass `--qwen3_asr_code_path` to the local Qwen3-ASR
Transformers backend code. That backend code is not vendored here.

## Repository Layout

```text
scripts/train/train_ark_asr_opd_fsdp2_resume.py   # main FSDP2 ASR OPD trainer
scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh  # multi-node launcher
configs/hostfile.example                          # hostfile format example
verl/                                             # vendored verl runtime code
README.md / README_zh.md                          # usage docs
```

## Install

Use a CUDA/PyTorch environment that matches your cluster. Then install this
repository and its Python dependencies:

```bash
pip install -e .
```

If your workflow expects `verl` to be installed as its own editable package:

```bash
pip install -e ./verl
```

For `qwen3_asr_vllm`, install a compatible vLLM stack separately:

```bash
pip install -e ".[vllm]"
```

## Data Format

Training data is JSONL. Each line is one ASR sample:

```json
{"audio":"/path/to/audio.wav","text":"reference transcript","task":"asr","begin_time":-1,"end_time":-1}
```

Fields:

- `audio`: required audio path.
- `text`: required reference transcript used for ASR supervision and metadata.
- `task`: optional; if present, it must be `asr`.
- `begin_time`: optional segment start in seconds. Use `-1` for full audio.
- `end_time`: optional segment end in seconds. Use `-1` for full audio.

The script fails on missing audio paths. It does not silently replace bad
samples with fallback audio.

## Single-Node Training

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

Start with a small batch and small `--asr_opd_max_new_tokens`, then scale after
checking generation length, non-empty generation ratio, teacher alignment, and
`opd_valid_topk_mean`.

## Multi-Node Hostfile Launch

Create a hostfile:

```text
node0 slots=8
node1 slots=8
node2 slots=8
```

Launch:

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

The launcher requires `HOSTFILE`, `STUDENT_MODEL`, `TEACHER_MODEL`, and
`TRAIN_DATA`. It also requires `QWEN3_ASR_CODE_PATH` when `TEACHER_BACKEND`
starts with `qwen3_asr_`.

## Resume Training

Resume a specific checkpoint:

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

Resume the latest checkpoint under `output_dir/checkpoints`:

```bash
--resume_from_checkpoint latest
```

`auto` is accepted as an alias for `latest`.

## Calibration Mode

By default, `--calibrate_only True`. Calibration runs forward passes and prints
initial loss/generation metrics without optimizer steps. Use it before a real
run to verify model loading, data loading, rollout, teacher scoring, and OPD
alignment.

For actual training, pass:

```bash
--calibrate_only False
```

## Important Arguments

- `--student_model`: trainable student ASR model path or HF repo id.
- `--teacher_model`: teacher ASR model path or HF repo id.
- `--teacher_backend`: teacher scoring implementation.
- `--qwen3_asr_code_path`: required for Qwen3-ASR teacher backends.
- `--train_data`: JSONL training data.
- `--output_dir`: logs and FSDP2 checkpoints.
- `--hf_cache_dir`: Hugging Face datasets cache directory.
- `--opd_top_k`: teacher/student top-k support size.
- `--opd_temperature`: temperature for OPD distribution.
- `--asr_block_token_id_from`: masks non-ASR token ids during student generation.
- `--asr_opd_max_new_tokens`: rollout length cap.
- `--save_freq`: checkpoint save interval. Set `-1` to disable saving.
- `--resume_from_checkpoint`: checkpoint dir, `latest`, or `auto`.

## Smoke Checks

These checks do not require model weights:

```bash
python3 -m py_compile scripts/train/train_ark_asr_opd_fsdp2_resume.py
bash -n scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
python scripts/train/train_ark_asr_opd_fsdp2_resume.py --help
```

The final `--help` command must be run in an environment with the training
dependencies installed, including `numpy`, `torch`, `datasets`,
`transformers`, `omegaconf`, and the `verl` dependencies listed in
`pyproject.toml`.
