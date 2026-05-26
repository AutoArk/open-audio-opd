#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ACTION="${1:-start}"
MODEL="${MODEL:-/data/yumu/model/trained_model/ark_asr_td_opd}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8025}"
GPU="${GPU:-1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniforge3/envs/asr_vlm/bin/python}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/vllm}"
PID_FILE="${PID_FILE:-${RUN_DIR}/ark_asr_vllm_${PORT}.pid}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/ark_asr_vllm_${PORT}.log}"

is_running() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

start_service() {
  mkdir -p "${RUN_DIR}"
  cd "${REPO_ROOT}"
  if is_running; then
    echo "Ark-ASR vLLM service is already running: pid=$(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" nohup setsid "${PYTHON_BIN}" -m scripts.vllm.ark_asr_vllm.service \
    --model "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    > "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Started Ark-ASR vLLM service: pid=$(cat "${PID_FILE}") port=${PORT} gpu=${GPU}"
  echo "Log: ${LOG_FILE}"
}

stop_service() {
  if ! [[ -s "${PID_FILE}" ]]; then
    echo "No PID file: ${PID_FILE}"
    return 0
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}"
  echo "Stopped Ark-ASR vLLM service for port=${PORT}"
}

status_service() {
  if is_running; then
    echo "running pid=$(cat "${PID_FILE}") port=${PORT}"
  else
    echo "not running port=${PORT}"
  fi
  echo "PID: ${PID_FILE}"
  echo "Log: ${LOG_FILE}"
}

case "${ACTION}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  status)
    status_service
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 2
    ;;
esac
