#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-fastvit_medpack}"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. Load your DGX Conda module or initialize Conda first." >&2
  exit 1
fi
conda env create -n "$ENV_NAME" -f "$ROOT/environment.yml" || conda env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
printf '\nEnvironment ready. Activate with: conda activate %s\n' "$ENV_NAME"
