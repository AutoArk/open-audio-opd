#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HOSTFILE="${HOSTFILE:?Set HOSTFILE to a torchrun hostfile with one host per line and optional slots=N}"
PY="${PY:-python3}"
RUN_ID="${RUN_ID:-ark_asr_opd_fsdp2_resume_hostfile_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/$RUN_ID}"
LOG="$RUN_DIR/launch.log"

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE")
if [ "${#HOSTS[@]}" -lt 1 ]; then
  echo "No hosts found in $HOSTFILE" >&2
  exit 1
fi

NNODES="${NNODES:-${#HOSTS[@]}}"
if [ "$NNODES" -ne "${#HOSTS[@]}" ]; then
  echo "NNODES=$NNODES does not match host count ${#HOSTS[@]} from $HOSTFILE" >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"
MASTER_PORT="${MASTER_PORT:-29503}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk 'NF && $1 !~ /^#/ {for (i=1; i<=NF; i++) if ($i ~ /^slots=/) {split($i,a,"="); print a[2]; exit}}' "$HOSTFILE")}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

STUDENT_MODEL="${STUDENT_MODEL:?Set STUDENT_MODEL to the audio ASR student model path or HF repo id}"
TEACHER_MODEL="${TEACHER_MODEL:?Set TEACHER_MODEL to the teacher ASR model path or HF repo id}"
TEACHER_BACKEND="${TEACHER_BACKEND:-qwen3_asr_teacher_forcing}"
QWEN3_ASR_CODE_PATH="${QWEN3_ASR_CODE_PATH:-}"
TEACHER_VLLM_GPU_MEMORY_UTILIZATION="${TEACHER_VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
TRAIN_DATA="${TRAIN_DATA:?Set TRAIN_DATA to the JSONL ASR training data path}"
EVAL_DATA="${EVAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-40}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-False}"
DATALOADER_MULTIPROCESSING_CONTEXT="${DATALOADER_MULTIPROCESSING_CONTEXT:-}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
SHUFFLE_TRAIN="${SHUFFLE_TRAIN:-True}"
MAX_STEPS="${MAX_STEPS:-0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.005}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
CALIBRATE_ONLY="${CALIBRATE_ONLY:-False}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-10}"
OPD_LOSS_WEIGHT="${OPD_LOSS_WEIGHT:-1.0}"
OPD_EOS_LOSS_WEIGHT="${OPD_EOS_LOSS_WEIGHT:-0.0}"
OPD_APPEND_TEACHER_EOS_FOR_STOPPED_ROLLOUTS="${OPD_APPEND_TEACHER_EOS_FOR_STOPPED_ROLLOUTS:-False}"
ASR_TERMINAL_LOSS_WEIGHT="${ASR_TERMINAL_LOSS_WEIGHT:-0.0}"
OPD_TOP_K="${OPD_TOP_K:-32}"
OPD_TEMPERATURE="${OPD_TEMPERATURE:-1.0}"
OPD_STUDENT_SCORE_MODE="${OPD_STUDENT_SCORE_MODE:-teacher_forcing}"
ASR_OPD_MAX_NEW_TOKENS="${ASR_OPD_MAX_NEW_TOKENS:-64}"
DEBUG_GENERATION_STEPS="${DEBUG_GENERATION_STEPS:-0}"
DEBUG_OPD_STEPS="${DEBUG_OPD_STEPS:-0}"
ASR_BLOCK_TOKEN_ID_FROM="${ASR_BLOCK_TOKEN_ID_FROM:-151670}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-30}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
STUDENT_ATTN_IMPLEMENTATION="${STUDENT_ATTN_IMPLEMENTATION:-sdpa}"
TEACHER_ATTN_IMPLEMENTATION="${TEACHER_ATTN_IMPLEMENTATION:-sdpa}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hpn0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-hpn0}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$RUN_DIR/hf_datasets_cache}"
HF_HOME="${HF_HOME:-$RUN_DIR/hf_home}"

if [[ "$TEACHER_BACKEND" == qwen3_asr_* && -z "$QWEN3_ASR_CODE_PATH" ]]; then
  echo "QWEN3_ASR_CODE_PATH is required when TEACHER_BACKEND starts with qwen3_asr_" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "[$(date)] RUN_ID=$RUN_ID"
  echo "[$(date)] RUN_DIR=$RUN_DIR"
  echo "[$(date)] HOSTFILE=$HOSTFILE"
  echo "[$(date)] HOSTS=${HOSTS[*]}"
  echo "[$(date)] NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE WORLD_SIZE=$((NNODES * NPROC_PER_NODE))"
  echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
  echo "[$(date)] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
  echo "[$(date)] STUDENT_MODEL=$STUDENT_MODEL"
  echo "[$(date)] TEACHER_MODEL=$TEACHER_MODEL TEACHER_BACKEND=$TEACHER_BACKEND"
  echo "[$(date)] TRAIN_DATA=$TRAIN_DATA"
  echo "[$(date)] TRAIN_MAX_SAMPLES=$TRAIN_MAX_SAMPLES SHUFFLE_TRAIN=$SHUFFLE_TRAIN"
  echo "[$(date)] CALIBRATE_ONLY=$CALIBRATE_ONLY SAVE_FREQ=$SAVE_FREQ RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
  echo "[$(date)] PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE ASR_OPD_MAX_NEW_TOKENS=$ASR_OPD_MAX_NEW_TOKENS"
  echo "[$(date)] LEARNING_RATE=$LEARNING_RATE OPD_LOSS_WEIGHT=$OPD_LOSS_WEIGHT OPD_EOS_LOSS_WEIGHT=$OPD_EOS_LOSS_WEIGHT OPD_APPEND_TEACHER_EOS_FOR_STOPPED_ROLLOUTS=$OPD_APPEND_TEACHER_EOS_FOR_STOPPED_ROLLOUTS ASR_TERMINAL_LOSS_WEIGHT=$ASR_TERMINAL_LOSS_WEIGHT OPD_TOP_K=$OPD_TOP_K"
  echo "[$(date)] LR_SCHEDULER_TYPE=$LR_SCHEDULER_TYPE WARMUP_STEPS=$WARMUP_STEPS MIN_LR_RATIO=$MIN_LR_RATIO"
} | tee -a "$LOG"

pids=()
for node_rank in "${!HOSTS[@]}"; do
  host="${HOSTS[$node_rank]}"
  safe_host="${host//./_}"
  node_log="$RUN_DIR/node_${node_rank}_${safe_host}.out"
  echo "[$(date)] launch node_rank=$node_rank host=$host log=$node_log" | tee -a "$LOG"
  ssh -o BatchMode=yes "$host" "
    set -euo pipefail
    cd '$ROOT'
    mkdir -p '$RUN_DIR' '$HF_DATASETS_CACHE' '$HF_HOME'
    env \
      CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' \
      TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM' \
      PYTHONUNBUFFERED=1 \
      HYDRA_FULL_ERROR=1 \
      NCCL_DEBUG='$NCCL_DEBUG' \
      NCCL_SOCKET_IFNAME='$NCCL_SOCKET_IFNAME' \
      GLOO_SOCKET_IFNAME='$GLOO_SOCKET_IFNAME' \
      NCCL_SHM_DISABLE='$NCCL_SHM_DISABLE' \
      TORCH_NCCL_BLOCKING_WAIT='$TORCH_NCCL_BLOCKING_WAIT' \
      NCCL_TIMEOUT='$NCCL_TIMEOUT' \
      PYTORCH_CUDA_ALLOC_CONF='$PYTORCH_CUDA_ALLOC_CONF' \
      HF_DATASETS_CACHE='$HF_DATASETS_CACHE' \
      HF_HOME='$HF_HOME' \
      '$PY' -m torch.distributed.run \
        --nnodes='$NNODES' \
        --node_rank='$node_rank' \
        --nproc_per_node='$NPROC_PER_NODE' \
        --master_addr='$MASTER_ADDR' \
        --master_port='$MASTER_PORT' \
        scripts/train/train_ark_asr_opd_fsdp2_resume.py \
        --student_model '$STUDENT_MODEL' \
        --teacher_model '$TEACHER_MODEL' \
        --teacher_backend '$TEACHER_BACKEND' \
        --qwen3_asr_code_path '$QWEN3_ASR_CODE_PATH' \
        --teacher_vllm_gpu_memory_utilization '$TEACHER_VLLM_GPU_MEMORY_UTILIZATION' \
        --train_data '$TRAIN_DATA' \
        --eval_data '$EVAL_DATA' \
        --output_dir '$OUTPUT_DIR' \
        --hf_cache_dir '$HF_DATASETS_CACHE' \
        --train_max_samples '$TRAIN_MAX_SAMPLES' \
        --shuffle_train '$SHUFFLE_TRAIN' \
        --per_device_train_batch_size '$PER_DEVICE_TRAIN_BATCH_SIZE' \
        --dataloader_num_workers '$DATALOADER_NUM_WORKERS' \
        --dataloader_prefetch_factor '$DATALOADER_PREFETCH_FACTOR' \
        --dataloader_persistent_workers '$DATALOADER_PERSISTENT_WORKERS' \
        --dataloader_multiprocessing_context '$DATALOADER_MULTIPROCESSING_CONTEXT' \
        --max_steps '$MAX_STEPS' \
        --total_epochs '$TOTAL_EPOCHS' \
        --learning_rate '$LEARNING_RATE' \
        --lr_scheduler_type '$LR_SCHEDULER_TYPE' \
        --warmup_steps '$WARMUP_STEPS' \
        --min_lr_ratio '$MIN_LR_RATIO' \
        --weight_decay '$WEIGHT_DECAY' \
        --grad_clip '$GRAD_CLIP' \
        --calibrate_only '$CALIBRATE_ONLY' \
        --calibration_batches '$CALIBRATION_BATCHES' \
        --opd_loss_weight '$OPD_LOSS_WEIGHT' \
        --opd_eos_loss_weight '$OPD_EOS_LOSS_WEIGHT' \
        --opd_append_teacher_eos_for_stopped_rollouts '$OPD_APPEND_TEACHER_EOS_FOR_STOPPED_ROLLOUTS' \
        --asr_terminal_loss_weight '$ASR_TERMINAL_LOSS_WEIGHT' \
        --opd_top_k '$OPD_TOP_K' \
        --opd_temperature '$OPD_TEMPERATURE' \
        --opd_student_score_mode '$OPD_STUDENT_SCORE_MODE' \
        --asr_opd_max_new_tokens '$ASR_OPD_MAX_NEW_TOKENS' \
        --debug_generation_steps '$DEBUG_GENERATION_STEPS' \
        --debug_opd_steps '$DEBUG_OPD_STEPS' \
        --asr_block_token_id_from '$ASR_BLOCK_TOKEN_ID_FROM' \
        --max_audio_seconds '$MAX_AUDIO_SECONDS' \
        --save_freq '$SAVE_FREQ' \
        --resume_from_checkpoint '$RESUME_FROM_CHECKPOINT' \
        --model_dtype '$MODEL_DTYPE' \
        --student_attn_implementation '$STUDENT_ATTN_IMPLEMENTATION' \
        --teacher_attn_implementation '$TEACHER_ATTN_IMPLEMENTATION'
  " > "$node_log" 2>&1 &
  pids+=("$!")
  echo "$!" > "$RUN_DIR/ssh_node_${node_rank}.pid"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "[$(date)] hostfile launch finished status=$status" | tee -a "$LOG"
exit "$status"
