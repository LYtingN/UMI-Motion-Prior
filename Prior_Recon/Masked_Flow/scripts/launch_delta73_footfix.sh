#!/usr/bin/env bash
# Launch 8-GPU (DDP) training for delta73_footfix.
#
# launch_ddp.sh hard-codes delta69_full and ignores --config, so delta73 runs
# go through this dedicated script that torchruns directly.
#
# KEY: FEAT_CACHE_MAX_FILES caps the PER-WORKER in-RAM source-file cache. The
# code default (100000) makes every DataLoader worker resident-cache the whole
# 23962-file dataset (~9GB decompressed each). With 8 gpus x num_workers that
# multiplies into hundreds of GB and OOM-kills the container (256GiB cap, takes
# sshd down with it). Capping at 1000 files (~0.37GB/worker) keeps it bounded;
# the 6.2GB dataset stays warm in OS page cache / the 160G /dev/shm anyway.
#
# Usage:
#   bash scripts/launch_delta73_footfix.sh --gpus "0,1,2,3,4,5,6,7"
#   bash scripts/launch_delta73_footfix.sh --gpus "0,1,2,3,4,5,6,7" -- --resume checkpoints/.../emft_best.pt
set -euo pipefail

# ── Tunables ──────────────────────────────────────────────────────────────────
# FULL per-worker cache (code default 100000 >> 23962 files -> whole dataset resident,
# ~8GB/worker, epoch 2+ is zero-decompress). This is the speed knob: npz is DEFLATE-
# compressed so a cache miss re-decompresses (~2.3ms), and a small cap thrashes every
# epoch. RAM is bounded via num_workers instead: 8 gpus x 2 workers x 8GB ~= 128GB < 256GiB.
# Leave unset to use the code default; override to a small number only on a RAM-tight run.
export FEAT_CACHE_MAX_FILES="${FEAT_CACHE_MAX_FILES:-100000}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"                          # per-process GPU memory cap

# ── Repo layout ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # .../Motion_Prior_Manipualtion
TRAIN_PY="$SCRIPT_DIR/train_delta_flow.py"
CONFIG="$SCRIPT_DIR/../configs/delta73_footfix.yaml"
FEAT_ROOT="${FEAT_ROOT:-/data/nas_ray/home/eason.er/train_delta_feat_v3}"

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

if [[ -z "$FORCE_GPUS" ]]; then
  echo "[launch] ERROR: pass --gpus \"0,1,2,3,4,5,6,7\"" >&2
  exit 1
fi
SELECTED="$FORCE_GPUS"
NPROC=$(awk -F',' '{print NF}' <<< "$SELECTED")

echo "[launch] gpus=$SELECTED  nproc_per_node=$NPROC  mem_fraction=$MEM_FRACTION"
echo "[launch] config=$CONFIG"
echo "[launch] feat_root=$FEAT_ROOT"
echo "[launch] FEAT_CACHE_MAX_FILES=$FEAT_CACHE_MAX_FILES"

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$SELECTED" FEAT_CACHE_MAX_FILES="$FEAT_CACHE_MAX_FILES" torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  "$TRAIN_PY" \
  --config "$CONFIG" \
  --feat-root "$FEAT_ROOT" \
  --gpu-memory-fraction "$MEM_FRACTION" \
  "${EXTRA_ARGS[@]}"
