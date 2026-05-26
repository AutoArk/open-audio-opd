#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-/data/yumu/model/trained_model/ark_asr_td_opd}"
CONDA_BIN="${CONDA_BIN:-/root/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-asr_vlm}"

cd "${REPO_ROOT}"
exec "${CONDA_BIN}" run -n "${CONDA_ENV}" python -m scripts.vllm.ark_asr_vllm.service \
  --model "${MODEL}" \
  "$@"
