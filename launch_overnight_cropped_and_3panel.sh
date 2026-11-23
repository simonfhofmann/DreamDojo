#!/usr/bin/env bash
set -euo pipefail

# Zero-duplicate single-dataset converter + overnight trainer for:
#   Single Dataset: datasets/tlrc_bimanual_yam_multicam (all 3 rectified cameras saved once, incremental!)
#   Exp 1: camera_layout=head_crop   (Column B tabletop crop on-the-fly: x[0.25:0.75], y[0.32:0.92] -> 640x480)
#   Exp 2: camera_layout=three_panel (on-the-fly 16px-aligned 640x480: top 288x640 + bottom 192x320 wrists)
#
# Usage:
#   ./launch_overnight_cropped_and_3panel.sh                  # Incrementally update multicam dataset + train Exp 1 + Exp 2
#   ./launch_overnight_cropped_and_3panel.sh --convert-only   # Only convert/update single multicam dataset (CPU-only)
#   ./launch_overnight_cropped_and_3panel.sh --skip-convert   # Skip conversion, run Exp 1 then Exp 2 training
#   ./launch_overnight_cropped_and_3panel.sh --exp1-only      # Run Exp 1 only (plus incremental convert unless --skip-convert)
#   ./launch_overnight_cropped_and_3panel.sh --exp2-only      # Run Exp 2 only (plus incremental convert unless --skip-convert)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MCAP_DIR="${MCAP_DIR:-raw_mcap}"
DS_MULTICAM="${DS_MULTICAM:-datasets/tlrc_bimanual_yam_multicam}"

CONVERT_ONLY=0
SKIP_CONVERT=0
RUN_EXP1=1
RUN_EXP2=1

for arg in "$@"; do
  case "$arg" in
    --convert-only) CONVERT_ONLY=1 ;;
    --skip-convert) SKIP_CONVERT=1 ;;
    --exp1-only)    RUN_EXP2=0 ;;
    --exp2-only)    RUN_EXP1=0 ;;
    -h|--help)
      sed -n '3,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/overnight_${TS}"
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"

source "${SCRIPT_DIR}/.venv/bin/activate"

if [[ "$SKIP_CONVERT" -eq 0 ]]; then
  echo "============================================================"
  echo "[Convert] Incrementally syncing $MCAP_DIR -> $DS_MULTICAM (camera-mode=multicam)"
  echo "============================================================"
  python tools/convert_mcap_to_lerobot.py \
    --input "$MCAP_DIR" \
    --output-dir "$DS_MULTICAM" \
    --camera-mode multicam \
    --incremental \
    --workers "${CONVERT_WORKERS:-6}" \
    2>&1 | tee "$LOG_DIR/convert_multicam.log"
fi

if [[ "$CONVERT_ONLY" -eq 1 ]]; then
  echo "Dataset conversion complete (--convert-only specified). Not starting training."
  exit 0
fi

if [[ "$RUN_EXP1" -eq 1 ]]; then
  echo "============================================================"
  echo "[Train Exp 1] dreamdojo_2b_480_640_tlrc_yam_cropped (camera_layout=head_crop)"
  echo "============================================================"
  MASTER_PORT="${MASTER_PORT_EXP1:-12341}" \
    ./launch_finetune_tlrc.sh dreamdojo_2b_480_640_tlrc_yam_cropped \
    2>&1 | tee "$LOG_DIR/train_exp1_cropped.log"
fi

if [[ "$RUN_EXP2" -eq 1 ]]; then
  echo "============================================================"
  echo "[Train Exp 2] dreamdojo_2b_480_640_tlrc_yam_3panel (camera_layout=three_panel)"
  echo "============================================================"
  MASTER_PORT="${MASTER_PORT_EXP2:-12342}" \
    ./launch_finetune_tlrc.sh dreamdojo_2b_480_640_tlrc_yam_3panel \
    2>&1 | tee "$LOG_DIR/train_exp2_3panel.log"
fi

echo "============================================================"
echo "All scheduled stages finished! Logs: $LOG_DIR"
echo "============================================================"
