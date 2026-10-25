#!/usr/bin/env bash
# Download the three released classifiers into submission/assets/ and verify their checksums.
# Usage: bash scripts/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# file in the release            -> name expected by submission/run.py
declare -A TARGET=(
  [m1_eva02l_clip.pt]=weights.pt
  [m2_eva02l_clip_region.pt]=weights2.pt
  [m3_eva02b_mim.pt]=weights3.pt
)
declare -A URL=(
  [m1_eva02l_clip.pt]="${M1_URL:-TODO_M1_URL}"
  [m2_eva02l_clip_region.pt]="${M2_URL:-TODO_M2_URL}"
  [m3_eva02b_mim.pt]="${M3_URL:-TODO_M3_URL}"
)

for f in "${!TARGET[@]}"; do
  out="submission/assets/${TARGET[$f]}"
  if [ ! -s "$out" ]; then
    echo "downloading $f"
    case "${URL[$f]}" in
      *drive.google.com*) python -m gdown --fuzzy "${URL[$f]}" -O "$out" ;;
      *) curl -L --fail -o "$out" "${URL[$f]}" ;;
    esac
  fi
  echo "$(grep " $f\$" scripts/SHA256SUMS | cut -d' ' -f1)  $out" | sha256sum -c -
done
