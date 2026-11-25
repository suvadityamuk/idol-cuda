#!/usr/bin/env bash
# Assemble the Space repository in OUT from this source tree:
#   space/assemble.sh OUT [PRECOMPUTED_DIR]
# The kernels are not bundled: the app loads them from the Hub (suvadityamuk/idol-cuda, version 1).
# PRECOMPUTED_DIR: optional robot_fast.ply / robot_official_eager.ply shown on
# page load.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; PRE=${2:-}
SDK_EXAMPLES=${SDK_EXAMPLES:-assets/examples}
rm -rf "$OUT" && mkdir -p "$OUT/fast" "$OUT/examples" "$OUT/kernel_tests"
cp space/app.py space/README.md space/requirements.txt space/test_app.py "$OUT/"  # test_app.py: not needed on the Space
cp fast/*.py "$OUT/fast/"
cp kernel/tests/*.py "$OUT/kernel_tests/"
for f in fig-robot-cute food-avocado obj-backpack toy-train food-donut obj-power-drill fig-fox food-mushroom; do
  cp "$SDK_EXAMPLES/$f.webp" "$OUT/examples/"
done
cp assets/examples/README.md "$OUT/examples/"
if [ -n "$PRE" ]; then mkdir -p "$OUT/precomputed" && cp "$PRE"/robot_fast.ply "$PRE"/robot_official_eager.ply "$OUT/precomputed/"; fi
printf '*.ply filter=lfs diff=lfs merge=lfs -text\n*.so filter=lfs diff=lfs merge=lfs -text\n*.webp filter=lfs diff=lfs merge=lfs -text\n' > "$OUT/.gitattributes"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
du -sh "$OUT"
