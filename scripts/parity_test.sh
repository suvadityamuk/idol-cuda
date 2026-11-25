#!/usr/bin/env bash
# Parity regression test: runs the native parity build in a fresh process per configuration and byte-compares every
# checkpoint against the stage-1 reference dumps (official pipeline, ATTN_BACKEND=sdpa, --no_optimize), plus the
# output files (.ply and _camera.json byte for byte, _input.png pixel for pixel via final.preprocessed_image).
# Exits non-zero on the first mismatch.
#
#   scripts/parity_test.sh REF_DIR OUT_DIR [BIN]
set -euo pipefail
REF=${1:?reference dump dir}
OUT=${2:?output dir}
BIN=${3:-$(dirname "$0")/../native/build/idol_native}
SDK=${IDOL_SDK:-/work/idol-sdk}
TOOLS=$(cd "$(dirname "$0")/../tools" && pwd)
mkdir -p "$OUT"
fail=0
for cfg in "robot synthetic fig-robot-cute 262144 262k" "robot synthetic fig-robot-cute 1048576 1m" \
           "avocado real food-avocado 262144 262k" "avocado real food-avocado 1048576 1m"; do
  set -- $cfg
  tag=$1_$5
  ref=$REF/${tag}_eager_sdpa_a
  [ -f "$ref.safetensors" ] || { echo "SKIP $tag (no reference dump)"; continue; }
  "$BIN" --build parity --image "$SDK/demo/examples/$3.webp" --mode $2 --num_gaussians $4 --out "$OUT" \
         --name "$tag" --dump "$OUT/$tag.safetensors" > "$OUT/$tag.log" 2>&1
  if python "$TOOLS/compare.py" "$ref.safetensors" "$OUT/$tag.safetensors" --quiet > "$OUT/$tag.compare.txt"; then
    c="checkpoints identical ($(tail -1 "$OUT/$tag.compare.txt" | cut -d, -f1 | tr -d '#'))"
  else
    c="CHECKPOINTS DIFFER"; fail=1
  fi
  if cmp -s "$OUT/$tag.ply" "$ref.ply"; then p="ply identical"; else p="PLY DIFFERS"; fail=1; fi
  if cmp -s "$OUT/${tag}_camera.json" "${ref}_camera.json"; then j="camera.json identical"; else j="CAMERA DIFFERS"; fail=1; fi
  echo "$tag: $c; $p; $j"
done
[ $fail -eq 0 ] && echo "PARITY PASS" || { echo "PARITY FAIL"; exit 1; }
