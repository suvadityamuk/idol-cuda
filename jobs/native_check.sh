# H200 job: run_native.sh (fast) once in a fresh shell, from a clean build dir, vs the reference; FA3 fallback check.
source /work/idol-cuda/jobs/env.sh
set +e
cd /work/idol-cuda
(apt-get update -qq && apt-get install -y -qq zlib1g-dev >/dev/null) || true
pip install -q cmake
REF=/scratch/ref/robot_262k_eager_sdpa_a
rm -rf native/build /work/s6
echo "=== run_native.sh (fast), fresh shell, clean build, TorchScript from /scratch/ts"
env -i PATH="$PATH" HOME="$HOME" HF_HUB_OFFLINE=1 bash -c \
  "time bash scripts/run_native.sh --out /work/s6 --name native_fast --dump /work/s6/native_fast.safetensors" 2>&1 | grep -v "^\[" | tail -8
python tools/closeness.py $REF.safetensors /work/s6/native_fast.safetensors --cand_camera /work/s6/native_fast_camera.json
FA3_SO=$(python -c "
import glob, os
from kernels import get_kernel
k = get_kernel('kernels-community/flash-attn3', version=1)
print(sorted(glob.glob(os.path.join(os.path.dirname(k.__file__), '*.so')))[0])" 2>/dev/null | tail -1)
echo "=== --fa3 fallback"
IDOL_FA3=$FA3_SO bash scripts/run_native.sh --out /work/s6 --name native_fast_fa3try 2>&1 | grep -E "warning|attention|request|ply"
cmp /work/s6/native_fast.ply /work/s6/native_fast_fa3try.ply && echo "fallback output identical to the cuDNN run"
echo NATIVE_CHECK_DONE
