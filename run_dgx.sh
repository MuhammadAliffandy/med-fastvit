#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ENV_NAME="${ENV_NAME:-fastvit_medpack}"
DATA_ROOT="${MEDPACK_DATA_ROOT:-$ROOT/data/MedPackID}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/fastvit_medpack_100ep}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
WORKERS="${WORKERS:-12}"
IMAGE_SIZE="${IMAGE_SIZE:-320}"
EPOCHS="${EPOCHS:-100}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python - <<'PY'
import torch, torchvision, timm
print(f"torch={torch.__version__} torchvision={torchvision.__version__} timm={timm.__version__}")
print(f"cuda_available={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
if not torch.cuda.is_available(): raise SystemExit("CUDA is unavailable in this environment")
PY
export MEDPACK_DATA_ROOT="$DATA_ROOT"
export RUN_DIR
python scripts/download_medpackid.py
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
python src/train_fastvit_medpack.py \
  --data "$DATA_ROOT" --out "$RUN_DIR" --epochs "$EPOCHS" \
  --image-size "$IMAGE_SIZE" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
