# CPU job: compile native/ against the pinned torch wheel; binary goes to /scratch/bin.
source /work/idol-cuda/jobs/env.sh
(apt-get update -qq && apt-get install -y -qq zlib1g-dev >/dev/null) || true
pip install -q cmake ninja
export TORCH_CUDA_ARCH_LIST="9.0a"
cmake -S native -B /work/build -G Ninja -DTORCH_SITE="$TORCH_SITE" -DPILLOW_LIBS="$PILLOW_LIBS" 2>&1 | tail -20
cmake --build /work/build -j"$(nproc)" 2>&1 | tail -150
mkdir -p /scratch/bin && cp /work/build/idol_native /scratch/bin/idol_native
sha256sum /scratch/bin/idol_native | tee "$RUN/idol_native.sha256"
ldd /work/build/idol_native | grep -E "cudnn|cublas|torch|webp" | tee "$RUN/ldd.txt"
echo BUILD_OK
