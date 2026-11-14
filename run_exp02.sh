#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
DATA_ROOT="${MEDPACK_DATA_ROOT:-$ROOT/data/MedPackID}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/exp02_fastvit_384}"
BASELINE_EVAL="${BASELINE_EVAL:-$ROOT/runs/eval_valid_best}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-fastvit_medpack}"
[[ -d "$DATA_ROOT" ]] || { echo "Extracted dataset missing: $DATA_ROOT" >&2; exit 1; }
[[ -f "$BASELINE_EVAL/metrics.json" && -f "$BASELINE_EVAL/ground_truth.json" ]] || {
  echo "Experiment 01 validation results missing: $BASELINE_EVAL" >&2; exit 1;
}
for filename in best.pth last.pth history.json; do
  [[ ! -e "$RUN_DIR/$filename" ]] || { echo "Run already has checkpoints: $RUN_DIR" >&2; exit 1; }
done
python - <<'PY'
import torch, timm
from pycocotools.cocoeval import COCOeval
if not torch.cuda.is_available(): raise SystemExit('CUDA GPU unavailable')
print('Experiment 02 | GPU:',torch.cuda.get_device_name(0),'| torch:',torch.__version__,'| timm:',timm.__version__)
PY
mkdir -p "$RUN_DIR"
cp "$BASELINE_EVAL/metrics.json" "$RUN_DIR/exp01_validation_reference.json"
python -u src/train_fastvit_medpack.py \
  --config configs/exp02_fastvit_384.json --data "$DATA_ROOT" --out "$RUN_DIR"

# Compare the loss-selected checkpoints first, then the extra mAP-selected candidate.
python -u src/evaluate_fastvit_medpack.py --data "$DATA_ROOT" --checkpoint "$RUN_DIR/best.pth" \
  --split valid --out "$RUN_DIR/eval_valid_loss"
python src/compare_experiments.py --baseline "$BASELINE_EVAL" --candidate "$RUN_DIR/eval_valid_loss" \
  --selection minimum_loss
python -u src/evaluate_fastvit_medpack.py --data "$DATA_ROOT" --checkpoint "$RUN_DIR/best_map.pth" \
  --split valid --out "$RUN_DIR/eval_valid_map"
python src/compare_experiments.py --baseline "$BASELINE_EVAL" --candidate "$RUN_DIR/eval_valid_map" \
  --selection maximum_sampled_map
printf '\nExperiment 02 complete. Validation reports and comparisons: %s\n' "$RUN_DIR"
