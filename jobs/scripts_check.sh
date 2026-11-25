# H200 job (stage 6): each inference script once in a fresh shell on robot 262k, outputs compared with the reference.
source /work/idol-cuda/jobs/env.sh
set +e
cd /work/idol-cuda
# prerequisites for building the native binary (README): cmake, ninja (optional), zlib headers, CUDA toolkit (image)
(apt-get update -qq && apt-get install -y -qq zlib1g-dev >/dev/null) || true
pip install -q cmake
REF=/scratch/ref/robot_262k_eager_sdpa_a
rm -rf /work/kb && cp -r /scratch/kbuild /work/kb
rm -rf native/build /work/s6 /work/ts_fresh
echo "=== run_native.sh (fast), fresh shell, builds the binary and exports TorchScript from scratch"
env -i PATH="$PATH" HOME="$HOME" IDOL_TS=/work/ts_fresh HF_HUB_OFFLINE=1 bash -c \
  "time bash scripts/run_native.sh --out /work/s6 --name native_fast --dump /work/s6/native_fast.safetensors" 2>&1 | grep -v "^\[" | tail -12
python tools/closeness.py $REF.safetensors /work/s6/native_fast.safetensors --cand_camera /work/s6/native_fast_camera.json
echo "=== run_native.sh --parity, fresh shell"
env -i PATH="$PATH" HOME="$HOME" IDOL_TS=/work/ts_fresh HF_HUB_OFFLINE=1 bash -c \
  "time bash scripts/run_native.sh --parity --out /work/s6 --name native_parity" 2>&1 | tail -6
cmp /work/s6/native_parity.ply $REF.ply && echo "native parity: PLY BYTE-IDENTICAL" || echo "native parity: PLY DIFFERS"
cmp /work/s6/native_parity_camera.json ${REF}_camera.json && echo "native parity: camera JSON BYTE-IDENTICAL"
echo "=== run_kernels.py --local, fresh shell"
env -i PATH="$PATH" HOME="$HOME" HF_HUB_OFFLINE=1 bash -c \
  "time python scripts/run_kernels.py --local --kernel_dir /work/kb --out /work/s6 --name kernels_fast" 2>&1 | grep -v Warning | tail -6
python tools/closeness.py $REF.safetensors /work/s6/kernels_fast.ply
echo "=== run_kernels.py --local --parity, fresh shell"
env -i PATH="$PATH" HOME="$HOME" HF_HUB_OFFLINE=1 bash -c \
  "time python scripts/run_kernels.py --local --kernel_dir /work/kb --parity --out /work/s6 --name kernels_parity --check $REF.ply" 2>&1 | grep -v Warning | tail -6
echo "exit code of the parity check: ${PIPESTATUS[0]}"
cmp /work/s6/kernels_parity_camera.json ${REF}_camera.json && echo "kernels parity: camera JSON BYTE-IDENTICAL"
mkdir -p /scratch/s6 && cp /work/s6/*.json /work/s6/*.png /scratch/s6/ 2>/dev/null
echo SCRIPTS_DONE
# closeness of the benchmark session's own outputs (first timed rep) vs the eager reference, and rendered views
B=/scratch/bench_out
for cfg in robot_262k avocado_262k robot_1m avocado_1m; do
  for row in py_fast native_fast ref_compiled; do
    python tools/closeness.py /scratch/ref/${cfg}_eager_sdpa_a.safetensors $B/$row/${cfg}_${row}_r0.ply --json /work/close_bench_${cfg}_${row}.json | sed "s/^/$cfg $row /"
  done
done
mkdir -p $RUN/close && cp /work/close_bench_*.json $RUN/close/
pip install -q gsplat 2>&1 | tail -1
for cfg in robot_262k avocado_262k robot_1m avocado_1m; do
  timeout 900 python tools/render_psnr.py $B/ref_eager/${cfg}_ref_eager_r0.ply $B/ref_eager/${cfg}_ref_eager_r1.ply \
     $B/ref_compiled/${cfg}_ref_compiled_r0.ply $B/py_fast/${cfg}_py_fast_r0.ply $B/native_fast/${cfg}_native_fast_r0.ply \
     /scratch/ref/${cfg}_eager_flash_a.ply --json $RUN/render_${cfg}.json 2>&1 | grep -E '^\{|rror' | cut -c1-300
done
echo ALL_DONE
