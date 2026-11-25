#!/usr/bin/env bash
# Idol image -> 3D Gaussians with the native libtorch C++ binary (no Python in the inference process).
#
#   scripts/run_native.sh [--parity] [idol_native args...]     e.g. --image photo.webp --mode real --out outputs
#
# What is native: the whole request (WebP decode, Pillow-exact resize/crop/composite, BiRefNet, DINOv3 and FLUX.2 VAE
# encoder as TorchScript graphs run by libtorch's interpreter, the denoiser, octree and Gaussian decoders, PLY and
# camera JSON writers) runs in one C++ process linked against libtorch + the idol_cuda kernels.
# What is not: building the binary (CMake finds the libtorch of the installed torch wheel through Python) and the
# one-off export of the three image networks to TorchScript (tools/export_encoders.py, Python, run once). The
# binary uses the torch wheel's shared libraries (libtorch, cuBLAS, cuDNN) at run time.
#
# Env: IDOL_WEIGHTS (default /w/idol), IDOL_TS (TorchScript dir, default /scratch/ts), IDOL_BUILD_DIR, DINOV3, FLUX2.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${IDOL_BUILD_DIR:-$ROOT/native/build}
WEIGHTS=${IDOL_WEIGHTS:-/w/idol}
TS=${IDOL_TS:-/scratch/ts}
MODE=fast
ARGS=()
for x in "$@"; do
  if [ "$x" = "--parity" ]; then MODE=parity; else ARGS+=("$x"); fi
done
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || { echo "error: no NVIDIA GPU / driver found" >&2; exit 2; }
if [ ! -x "$BUILD/idol_native" ]; then
  command -v nvcc >/dev/null || { echo "error: nvcc (CUDA toolkit) not found; needed to build idol_native" >&2; exit 2; }
  command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 2; }
  echo "building idol_native in $BUILD (one-off)"
  TORCH_SITE=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
  PILLOW_LIBS=$(python -c 'import PIL,os;print(os.path.realpath(os.path.join(os.path.dirname(PIL.__file__),"..","pillow.libs")))')
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0a} cmake -S "$ROOT/native" -B "$BUILD" -DTORCH_SITE="$TORCH_SITE" \
     -DPILLOW_LIBS="$PILLOW_LIBS" >/dev/null
  cmake --build "$BUILD" -j"$(nproc)"
fi
if [ ! -f "$TS/birefnet.pt" ]; then
  echo "exporting the image networks to TorchScript in $TS (one-off, Python)"
  python "$ROOT/tools/export_encoders.py" --out "$TS" --weights "$WEIGHTS" ${DINOV3:+--dinov3 $DINOV3} ${FLUX2:+--flux2 $FLUX2}
fi
FA3_ARGS=()
# The fast build uses cuDNN attention. `--fa3 <lib.so>` (FlashAttention-3 from the Hub kernel) is accepted but that
# library does not load without libpython (it binds PyModule_Create2), so the binary then falls back to cuDNN.
[ -n "${IDOL_FA3:-}" ] && [ "$MODE" = fast ] && FA3_ARGS=(--fa3 "$IDOL_FA3")
if [ ${#ARGS[@]} -eq 0 ] || ! printf '%s\n' "${ARGS[@]}" | grep -q -- '--image'; then
  SDK=${IDOL_SDK:-$(python -c 'import idol,os;print(os.path.dirname(os.path.dirname(idol.__file__)))' 2>/dev/null || true)}
  ARGS+=(--image "$SDK/demo/examples/fig-robot-cute.webp")
fi
exec "$BUILD/idol_native" --build "$MODE" --weights "$WEIGHTS" --ts "$TS" "${FA3_ARGS[@]}" "${ARGS[@]}"
