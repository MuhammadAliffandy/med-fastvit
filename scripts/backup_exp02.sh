#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEST="${1:-$HOME/xxdrive/MedPackID/experiments/exp02_fastvit_384}"
ARCHIVE="$ROOT/runs/archives/exp02_fastvit_384"
if command -v rclone >/dev/null 2>&1; then
  TRANSPORT=rclone
elif [[ "$DEST" != *:* ]] && command -v rsync >/dev/null 2>&1; then
  TRANSPORT=rsync
else
  echo 'Install rclone or make rsync available for copying to a mounted local path.' >&2
  exit 1
fi
python scripts/archive_exp02.py --out "$ARCHIVE"
if [[ "$TRANSPORT" == rclone ]]; then
  rclone copy "$ARCHIVE" "$DEST" --immutable --checksum --progress --stats 5s --transfers 2
  rclone check "$ARCHIVE" "$DEST" --download --one-way
else
  echo 'Using rsync to copy to the mounted destination with progress.'
  mkdir -p "$DEST"
  rsync -r --progress --ignore-existing "$ARCHIVE/" "$DEST/"
  python scripts/archive_experiment.py --verify-dir "$DEST" --checksums "$ARCHIVE/SHA256SUMS"
fi
printf '\nExperiment 02 copy verified at: %s\n' "$DEST"
printf 'Download for conversion: %s/download/exp02_fastvit_384_coreml_source.zip\n' "$DEST"
