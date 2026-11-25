# H200 job: the interleaved same-session benchmark of every build, then rendered-view PSNR/SSIM (gsplat) if it installs.
source /work/idol-cuda/jobs/env.sh
set +e
cd /work/idol-cuda
pip install -q pytest
# kernel tests against the kernel-builder output (torch210-cu130 variant is the one this torch loads)
rm -rf /work/kb && cp -r /scratch/kbuild /work/kb
(cd kernel && IDOL_KERNEL_LOCAL_PATH=/work/kb python -m pytest -v tests -rf 2>&1 | grep -v Warning | tail -50 > $RUN/pytest_kbuild.txt)
tail -3 $RUN/pytest_kbuild.txt
(apt-get update -qq && apt-get install -y -qq zlib1g-dev >/dev/null) || true
pip install -q ninja cmake
export TORCH_CUDA_ARCH_LIST="9.0a"
cmake -S native -B /work/build -G Ninja -DTORCH_SITE="$TORCH_SITE" -DPILLOW_LIBS="$PILLOW_LIBS" > $RUN/cmake.txt 2>&1
cmake --build /work/build -j20 2>&1 | grep -E "error|FAILED" | head -20
cp /work/build/idol_native /work/idol_native && cp /work/build/idol_native /scratch/bin/idol_native
export FA3_SO=$(python -c "
import glob, os
from kernels import get_kernel
k = get_kernel('kernels-community/flash-attn3', version=1)
print(sorted(glob.glob(os.path.join(os.path.dirname(k.__file__), '*.so')))[0])" 2>/dev/null | tail -1)
echo "FA3_SO=$FA3_SO"
# smoke test of the native fast build with FlashAttention-3; on failure the native row falls back to cuDNN
if ! /work/idol_native --build fast --fa3 "$FA3_SO" --image $SDK/demo/examples/fig-robot-cute.webp --out /work/smoke \
     --name smoke --dump /work/smoke/smoke.safetensors > $RUN/native_fa3_smoke.txt 2>&1; then
  echo "native FA3 smoke test FAILED: $(tail -2 $RUN/native_fa3_smoke.txt)"; export FA3_SO=""
else
  grep -E "attention|request" $RUN/native_fa3_smoke.txt
  python tools/closeness.py /scratch/ref/robot_262k_eager_sdpa_a.safetensors /work/smoke/smoke.safetensors \
     --cand_camera /work/smoke/smoke_camera.json | tee $RUN/native_fa3_smoke_close.json
fi
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv | tee $RUN/gpu.txt
python tools/bench.py --native /work/idol_native --sdk $SDK --reps ${REPS:-5} --outdir /work/bench_out \
   --out $RUN/bench.jsonl 2>&1 | grep -v Warning | grep -E '"kind"|rror|Traceback' | cut -c1-260
mkdir -p /scratch/bench_out && cp -r /work/bench_out/* /scratch/bench_out/ 2>/dev/null
# rendered views (gsplat, JIT-compiled CUDA): best effort, bounded
( timeout 900 pip install -q gsplat ninja && echo GSPLAT_INSTALLED ) 2>&1 | tail -3
for cfg in robot_262k avocado_262k robot_1m avocado_1m; do
  B=/work/bench_out
  timeout 900 python tools/render_psnr.py $B/ref_eager/${cfg}_ref_eager_r0.ply $B/ref_eager/${cfg}_ref_eager_r1.ply \
     $B/ref_compiled/${cfg}_ref_compiled_r0.ply $B/ref_compiled/${cfg}_ref_compiled_r1.ply \
     $B/py_fast/${cfg}_py_fast_r0.ply $B/native_fast/${cfg}_native_fast_r0.ply $B/native_parity/${cfg}_native_parity_r0.ply \
     --json $RUN/render_${cfg}.json 2>&1 | grep -E '^\{|rror' | cut -c1-300
done
# the reference's own spread: eager flash-attn vs eager sdpa, compiled run B for the 1M configs, then closeness
EX=$SDK/demo/examples
for c in "robot synthetic fig-robot-cute 262144 262k" "avocado real food-avocado 262144 262k" \
         "robot synthetic fig-robot-cute 1048576 1m" "avocado real food-avocado 1048576 1m"; do
  set -- $c; tag=$1_$5
  python tools/ref_run.py --image $EX/$3.webp --mode $2 --num_gaussians $4 --attn flash_attn --no_optimize \
     --dump /scratch/ref/${tag}_eager_flash_a.safetensors 2>&1 | grep -E "saved|rror" | tail -1
  [ "$5" = 1m ] && python tools/ref_run.py --image $EX/$3.webp --mode $2 --num_gaussians $4 --attn sdpa \
     --dump /scratch/ref/${tag}_compiled_b.safetensors 2>&1 | grep -E "saved|rror" | tail -1
  for v in compiled_a compiled_b eager_flash_a; do
    python tools/closeness.py /scratch/ref/${tag}_eager_sdpa_a.safetensors /scratch/ref/${tag}_${v}.safetensors \
      --json $RUN/spread_${tag}_${v}.json
  done
  python tools/closeness.py /scratch/ref/${tag}_compiled_a.safetensors /scratch/ref/${tag}_compiled_b.safetensors \
      --json $RUN/spread_${tag}_compiled_a_vs_b.json
done
echo BENCH_DONE
