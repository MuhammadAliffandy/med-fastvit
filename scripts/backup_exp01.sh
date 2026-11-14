#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEST="${1:-$HOME/xxdrive/MedPackID/experiments/exp01_fastvit_320}"
ARCHIVE="$ROOT/runs/archives/exp01_fastvit_320"
command -v rclone >/dev/null || { echo 'rclone is not on PATH' >&2; exit 1; }
python scripts/archive_experiment.py --out "$ARCHIVE"
rclone copy "$ARCHIVE" "$DEST" --immutable --checksum --progress --stats 5s --transfers 2
rclone check "$ARCHIVE" "$DEST" --download --one-way
printf '\nExperiment 01 copy verified at: %s\n' "$DEST"
