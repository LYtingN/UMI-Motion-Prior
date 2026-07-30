#!/usr/bin/env bash
# Launch multi-GPU (DDP) training on the currently-free GPUs.
#
# It probes nvidia-smi, picks GPUs whose utilisation and free memory clear the
# thresholds below, then starts one torchrun process per selected GPU. Because
# GPU occupancy on a shared box changes over time, selection happens ONCE at
# launch. To pick up newly-freed cards later: stop, then re-launch (optionally
# --resume <ckpt>) and it re-probes.
#
# Usage:
#   bash scripts/launch_ddp.sh [--gpus "4,7"] [-- <extra train_delta_flow.py args>]
#
# Examples:
#   # auto-pick free GPUs, full config
#   bash Prior_Recon/Masked_Flow/scripts/launch_ddp.sh
#   # force GPUs 4 and 7
#   bash Prior_Recon/Masked_Flow/scripts/launch_ddp.sh --gpus "4,7"
#   # resume
#   bash .../launch_ddp.sh --gpus "4,7" -- --resume checkpoints/.../emft_best.pt
set -euo pipefail

# ── Tunables ──────────────────────────────────────────────────────────────────
MAX_UTIL="${MAX_UTIL:-40}"          # only use GPUs with util% <= this
MIN_FREE_MB="${MIN_FREE_MB:-40000}" # only use GPUs with free mem(MB) >= this
MAX_GPUS="${MAX_GPUS:-2}"           # cap number of GPUs to use
MEM_FRACTION="${MEM_FRACTION:-0.45}" # per-process memory cap (avoid OOM'ing neighbours)

# ── Repo layout ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # .../Motion_Prior_Manipualtion
TRAIN_PY="$SCRIPT_DIR/train_delta_flow.py"
CONFIG="$SCRIPT_DIR/../configs/delta69_full.yaml"
FEAT_ROOT="${FEAT_ROOT:-/data/nas_ray/home/eason.nai/train_feat}"

# ── Parse args ──────────────────────────────────────────────────────────────
FORCE_GPUS=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) FORCE_GPUS="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ── Select GPUs ───────────────────────────────────────────────────────────────
if [[ -n "$FORCE_GPUS" ]]; then
  SELECTED="$FORCE_GPUS"
  echo "[launch] using forced GPUs: $SELECTED"
else
  echo "[launch] probing GPUs (util<=${MAX_UTIL}%, free>=${MIN_FREE_MB}MB, max ${MAX_GPUS})"
  mapfile -t ROWS < <(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.free \
    --format=csv,noheader,nounits)
  CANDIDATES=()
  for row in "${ROWS[@]}"; do
    idx=$(echo "$row"  | awk -F',' '{gsub(/ /,"",$1); print $1}')
    util=$(echo "$row" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    free=$(echo "$row" | awk -F',' '{gsub(/ /,"",$3); print $3}')
    if (( util <= MAX_UTIL )) && (( free >= MIN_FREE_MB )); then
      # sort key: prefer most-free memory (printed first)
      CANDIDATES+=("$free $idx")
      echo "  GPU $idx: util=${util}%  free=${free}MB  -> eligible"
    else
      echo "  GPU $idx: util=${util}%  free=${free}MB  -> skip"
    fi
  done
  if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
    echo "[launch] ERROR: no GPU meets the thresholds. Relax MAX_UTIL / MIN_FREE_MB." >&2
    exit 1
  fi
  # sort by free mem desc, take top MAX_GPUS indices
  SELECTED=$(printf '%s\n' "${CANDIDATES[@]}" | sort -rn | head -n "$MAX_GPUS" \
             | awk '{print $2}' | paste -sd, -)
  echo "[launch] selected GPUs: $SELECTED"
fi

NPROC=$(awk -F',' '{print NF}' <<< "$SELECTED")

# ── Launch ────────────────────────────────────────────────────────────────────
echo "[launch] nproc_per_node=$NPROC  mem_fraction=$MEM_FRACTION"
echo "[launch] config=$CONFIG"
echo "[launch] feat_root=$FEAT_ROOT"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$SELECTED" torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  "$TRAIN_PY" \
  --config "$CONFIG" \
  --feat-root "$FEAT_ROOT" \
  --gpu-memory-fraction "$MEM_FRACTION" \
  "${EXTRA_ARGS[@]}"
