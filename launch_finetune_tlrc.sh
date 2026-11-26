#!/usr/bin/env bash
set -ex

NNODES=${NNODES:-1}
NPROC=${NPROC:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-12341}
NODE_RANK=${NODE_RANK:-0}
if [[ -z "${WANDB_MODE:-}" ]]; then
  if [[ -n "${WANDB_API_KEY:-}" ]] || { [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; }; then
    WANDB_MODE="online"
  else
    WANDB_MODE="offline"
  fi
fi

export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_DIST_INIT_BARRIER=1
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${SCRIPT_DIR}/outputs}"

source "${SCRIPT_DIR}/.venv/bin/activate"

CONFIG_NAME="${1:-dreamdojo_2b_480_640_tlrc_yam}"
shift || true

torchrun --nnodes=$NNODES --nproc_per_node=$NPROC \
  --master_port=$MASTER_PORT --master_addr $MASTER_ADDR \
  --node_rank=$NODE_RANK -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=$CONFIG_NAME \
  job.wandb_mode=$WANDB_MODE \
  ~dataloader_train.dataloaders \
  "$@"
