#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${PY:-python3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT/scripts/eval/eval_jwer_ark_asr_transformers.py}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the ASR model checkpoint or HF repo id}"
PROCESSOR_PATH="${PROCESSOR_PATH:-$MODEL_PATH}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:?Set EVAL_DATA_DIR to the directory containing eval JSONL files}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for eval logs and result JSONL files}"
SUFFIX="${SUFFIX:-arkasr}"
PRESETS="${PRESETS:-aishell clean meeting net other}"
GPUS="${GPUS:-0 1 2 3 4}"
BATCH_SIZE="${BATCH_SIZE:-40}"
DTYPE="${DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
TEXT_NORMALIZE_PYTHON="${TEXT_NORMALIZE_PYTHON:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-40}"

mkdir -p "$OUTPUT_DIR"

read -r -a preset_array <<< "$PRESETS"
read -r -a gpu_array <<< "$GPUS"
if [ "${#gpu_array[@]}" -lt "${#preset_array[@]}" ]; then
  echo "GPUS must provide at least one GPU id per preset" >&2
  exit 1
fi

for index in "${!preset_array[@]}"; do
  preset="${preset_array[$index]}"
  gpu="${gpu_array[$index]}"
  input="$EVAL_DATA_DIR/test_${preset}.jsonl"
  output="$OUTPUT_DIR/test_${preset}_${SUFFIX}_result.jsonl"
  log="$OUTPUT_DIR/${preset}.log"
  if [ ! -f "$input" ]; then
    echo "Missing input JSONL: $input" >&2
    exit 1
  fi
  echo "[LAUNCH] gpu=${gpu} preset=${preset} input=${input} output=${output} log=${log}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    cmd=(
      "$PY" "$EVAL_SCRIPT"
      --input "$input"
      --output "$output"
      --model_path "$MODEL_PATH"
      --processor_path "$PROCESSOR_PATH"
      --batch_size "$BATCH_SIZE"
      --dtype "$DTYPE"
      --attn_impl "$ATTN_IMPL"
      --max_new_tokens "$MAX_NEW_TOKENS"
      --max_audio_seconds "$MAX_AUDIO_SECONDS"
    )
    if [ -n "$TEXT_NORMALIZE_PYTHON" ]; then
      cmd+=(--text_normalize_python "$TEXT_NORMALIZE_PYTHON")
    fi
    "${cmd[@]}"
  ) >"$log" 2>&1 &
  echo $! > "$OUTPUT_DIR/${preset}.pid"
done

echo "Launched ${#preset_array[@]} Ark-ASR eval jobs. Logs: $OUTPUT_DIR"
