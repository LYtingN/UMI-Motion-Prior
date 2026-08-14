#!/usr/bin/env bash
# Poll GPUs until N cards are free, then auto-launch DDP training and exit.
#
# "free" = utilization <= MAX_UTIL%  AND  free memory >= MIN_FREE_MB.
# On success it starts torchrun on the N freest qualifying GPUs using the full
# config and the train_feat dataset, then exits 0. Training keeps running.
set -euo pipefail

NEED_GPUS="${NEED_GPUS:-3}"
MAX_UTIL="${MAX_UTIL:-40}"
MIN_FREE_MB="${MIN_FREE_MB:-40000}"
POLL_SEC="${POLL_SEC:-60}"
MEM_FRACTION="${MEM_FRACTION:-0.30}"   # 3 procs share; keep modest to be a good neighbour
AMP="${AMP:-bf16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
TRAIN_PY="$SCRIPT_DIR/train_delta_flow.py"
CONFIG="$SCRIPT_DIR/../configs/config_achieve/delta69_full.yaml"
FEAT_ROOT="${FEAT_ROOT:-/data/nas_ray/home/eason.nai/train_feat}"
TRAIN_LOG="${TRAIN_LOG:-$REPO_ROOT/train_ddp_3gpu.log}"

echo "[watch] need=$NEED_GPUS free-GPUs (util<=${MAX_UTIL}%, free>=${MIN_FREE_MB}MB), poll every ${POLL_SEC}s"
echo "[watch] will launch: config=$CONFIG feat_root=$FEAT_ROOT amp=$AMP mem_frac=$MEM_FRACTION"

while true; do
  mapfile -t ROWS < <(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.free \
    --format=csv,noheader,nounits)
  CANDIDATES=()
  for row in "${ROWS[@]}"; do
    idx=$(awk -F',' '{gsub(/ /,"",$1); print $1}' <<< "$row")
    util=$(awk -F',' '{gsub(/ /,"",$2); print $2}' <<< "$row")
    free=$(awk -F',' '{gsub(/ /,"",$3); print $3}' <<< "$row")
    if (( util <= MAX_UTIL )) && (( free >= MIN_FREE_MB )); then
      CANDIDATES+=("$free $idx")
    fi
  done

  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  n=${#CANDIDATES[@]}
  if (( n >= NEED_GPUS )); then
    SELECTED=$(printf '%s\n' "${CANDIDATES[@]}" | sort -rn | head -n "$NEED_GPUS" \
               | awk '{print $2}' | paste -sd, -)
    echo "[watch] $ts  $n free GPUs -> launching on: $SELECTED"
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="$SELECTED" nohup torchrun \
      --standalone \
      --nproc_per_node="$NEED_GPUS" \
      "$TRAIN_PY" \
      --config "$CONFIG" \
      --feat-root "$FEAT_ROOT" \
      --gpu-memory-fraction "$MEM_FRACTION" \
      --num-workers 0 \
      --amp "$AMP" \
      > "$TRAIN_LOG" 2>&1 &
    echo "[watch] launched torchrun (pid $!), training log: $TRAIN_LOG"
    echo "$SELECTED" > "$REPO_ROOT/.last_train_gpus"
    exit 0
  else
    echo "[watch] $ts  only $n/$NEED_GPUS free; waiting..."
  fi
  sleep "$POLL_SEC"
done
