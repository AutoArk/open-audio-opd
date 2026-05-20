# open-audio-opd

Industrial audio online policy distillation training code.

中文文档: [README_zh.md](README_zh.md)

`open-audio-opd` is an adapter-first project for online policy distillation on
audio tasks. The first supported workflow is ASR OPD: a student ASR model
generates a transcript from audio, a stronger teacher scores that transcript
under the same audio condition, and the student is trained with sparse
union-support KL. This repository also vendors the `verl` utilities needed by
the production FSDP2 resume training script.

This project is based on [THUNLP/OPD](https://github.com/thunlp/OPD/) and
[verl](https://github.com/volcengine/verl).

## Production ASR OPD Script

The industrial training entrypoint is:

```bash
scripts/train/train_ark_asr_opd_fsdp2_resume.py
```

It uses the vendored `verl/` package for FSDP2 wrapping and checkpoint
management. Model paths, data paths, output paths, and Qwen3-ASR backend code
paths are explicit CLI arguments; private local defaults are intentionally not
embedded.

Minimal launch shape:

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

Multi-node hostfile launch:

```bash
HOSTFILE=/path/to/hostfile \
STUDENT_MODEL=/path/to/student_model \
TEACHER_MODEL=/path/to/qwen3_asr_model \
QWEN3_ASR_CODE_PATH=/path/to/qwen3-asr/backend \
TRAIN_DATA=/path/to/train.jsonl \
OUTPUT_DIR=runs/ark_asr_opd_fsdp2 \
scripts/run/run_ark_asr_opd_fsdp2_resume_hostfile.sh
```

Resume latest checkpoint:

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

Expected JSONL fields:

- `audio`: audio file path.
- `text`: reference transcript.
- `task`: optional, must be `asr` when present.
- `begin_time` and `end_time`: optional segment boundaries in seconds.

This repository is intentionally clean-room. It does not copy private training
code or hard-code private model paths. Real models are connected through adapters.

## Current Model Setup

The built-in model path is a toy smoke test only:

- Student: `toy.student`, a tiny trainable PyTorch module.
- Teacher: `toy.teacher`, a deterministic sparse scorer.
- Purpose: verifies config loading, rollout, scoring, OPD loss, backward, and CLI.

The intended real ASR setup is:

- Student: your audio-capable ASR CausalLM or seq2seq policy, exposed through
  `StudentPolicy`.
- Teacher: a stronger audio ASR teacher, for example Qwen3-ASR, exposed through
  `TeacherScorer`.
- Data: JSONL records containing audio paths and optional language/reference text.

For Qwen3-ASR-style teachers, the teacher adapter should build a teacher-only
context like:

```text
audio + language <lang><asr_text> + student_rollout_text
```

The student should not be trained to emit `language <lang><asr_text>`. OPD should
cover only the rollout transcript positions after the teacher text marker.

## Install

Recommended:

```bash
uv venv
uv pip install -e ".[dev]"
```

Plain pip:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

On a training machine, install PyTorch/CUDA and model-specific dependencies
before installing real adapters.

## Quick Start: Toy Smoke

Run this first on any machine. It does not download models or read audio files.

```bash
open-audio-opd validate-config --config configs/toy_smoke.yaml
open-audio-opd validate-data --data configs/toy_train.jsonl
open-audio-opd smoke --config configs/toy_smoke.yaml
```

Expected output includes:

```text
"ok": true
"steps": 1
"loss": <finite number>
"opd_valid_topk_mean": > opd_top_k
```

`opd_valid_topk_mean > opd_top_k` means union support is active: the loss uses
teacher top-k plus student top-k, not teacher-only top-k.

## Data Format

The v1 built-in schema is ASR-oriented JSONL:

```json
{"audio_path":"/data/audio/0001.wav","text":"reference transcript","language":"English","duration":3.2}
```

Required:

- `audio_path`: absolute path or path relative to the JSONL file.

Optional:

- `text`: reference transcript. Online OPD treats it as metadata unless your
  adapter explicitly uses it.
- `language`: language hint for teacher prompts.
- `duration`: seconds, used for filtering.
- extra fields: preserved in `metadata`.

Validate data:

```bash
open-audio-opd validate-data --data /path/to/train.jsonl --require-audio-exists
```

## Real ASR OPD Workflow

1. Prepare a JSONL ASR dataset.
2. Implement a student adapter that loads your ASR student model.
3. Implement a teacher adapter that loads your ASR teacher model.
4. Create a YAML config pointing to adapter factories with `module:attr`.
5. Run a one-step smoke with a tiny batch.
6. Increase batch size and steps only after monitoring signals are healthy.
7. Export a complete checkpoint into an inference-compatible model directory.

The online OPD step is:

```text
audio batch
  -> student no-grad rollout
  -> teacher scores same audio + rollout
  -> student teacher-forced forward on its rollout
  -> build teacher/student union support
  -> KL(teacher || student) on aligned rollout positions
  -> backward into student
```

## Adapter Contract

Student adapters implement:

```python
class MyStudent(torch.nn.Module):
    vocab_size: int

    def rollout(self, samples, max_new_tokens):
        ...

    def score_rollouts(self, samples, rollouts):
        # return logits shaped [batch, time, vocab]
        ...
```

Teacher adapters implement:

```python
class MyTeacher:
    def score(self, samples, rollouts, student_logits, top_k):
        # return TeacherScores
        ...
```

`TeacherScores` must include:

- teacher top-k token ids and logprobs;
- student logprobs on teacher top-k ids;
- student top-k token ids and logprobs;
- teacher logprobs on student top-k ids;
- a mask for valid rollout positions.

Do not assume teacher and student token ids match globally. Map only comparable
text tokens for ASR. Student-only audio/TTS codec tokens should be excluded from
ASR OPD unless the teacher has matching semantics.

See [docs/adapters.md](docs/adapters.md).

## Example Real Config

`configs/qwen3_asr_teacher.example.yaml` shows the intended shape:

```yaml
data:
  train_data: /path/to/asr_train.jsonl
  max_audio_seconds: 30
  train_max_samples: -1
  shuffle: true

adapters:
  student: your_private_adapters.ark_audio:build_student
  teacher: your_private_adapters.qwen3_asr:build_teacher
  options:
    student:
      model_name_or_path: /path/to/student
      block_token_id_from: 151670
    teacher:
      model_name_or_path: /path/to/qwen3-asr
      forced_prefix_template: "language {language}<asr_text>"
      top_k: 32

training:
  output_dir: runs/qwen3_asr_opd
  max_steps: 1000
  per_device_train_batch_size: 1
  learning_rate: 0.00001
  opd_top_k: 32
  max_new_tokens: 64
  device: cuda
```

Run:

```bash
open-audio-opd validate-config --config configs/qwen3_asr_teacher.example.yaml
open-audio-opd train --config configs/qwen3_asr_teacher.example.yaml
```

## Monitoring

Do not judge OPD training only by loss. Track:

- `loss` and `opd_loss`.
- `opd_valid_topk_mean`: should be greater than `opd_top_k` when union support
  is active.
- generated token length: sudden collapse to very short outputs is a failure.
- non-empty rollout ratio.
- teacher alignment mismatch rate.
- teacher language distribution and fallback rate.
- examples from exported checkpoints, not only training curves.

Bad signs:

- loss decreases while generated text becomes very short;
- `opd_valid_topk_mean` stays exactly equal to `opd_top_k`;
- teacher/student token mapping silently drops most positions;
- ASR output becomes only an end token or repetitive fragments.

## Checkpoint Export

FSDP checkpoints are usually not directly loadable as inference model folders.
The expected export flow is:

1. Choose a complete `global_step_*` checkpoint.
2. Copy the original student model directory as an inference template.
3. Merge all FSDP shards, for example `model_world_size_*_rank_*.pt`.
4. Save into a new target directory with the step number.
5. Load the exported model and run a short ASR test.

The current CLI documents the contract:

```bash
open-audio-opd export-fsdp \
  --checkpoint-dir runs/my_run/checkpoints/global_step_1000 \
  --template-model-dir /path/to/base_student \
  --target-dir /path/to/exported_student_step1000
```

Model-stack-specific merging should live in adapter or deployment code.

## TTS Support

TTS support is coming soon. The planned shape is:

- text or prompt input replaces ASR audio-only prompting;
- student rollout emits acoustic/code/token sequences;
- teacher scores comparable acoustic/code/token positions;
- adapters map teacher/student token spaces before creating `TeacherScores`;
- the core union-support OPD loss remains unchanged.

## Project Status

This is v1 scaffolding plus a working toy smoke path. ASR is the first documented
task. Real single-node or multi-node training requires user-provided adapters and
model-specific dependencies. Multi-node FSDP is documented as an integration
pattern, not hard-coded to a private cluster.
