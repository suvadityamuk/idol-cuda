# H200 job: release checks run from a fresh export of the release tree (uploaded as code/release.tgz), not from the
# development checkout. Mounts as in launch.sh; the kernel-builder output is expected in the bucket's kbuild/.
#   1. pip install -r requirements.txt of the export; error paths (no CUDA, no token, Hub kernel repo missing)
#   2. scripts/infer.py: fast (robot synthetic, avocado real) with closeness to the eager reference dumps, parity with
#      the byte check, reference eager (byte check) and compiled; --turntable
#   3. the kernel package's tests on the new build
#   4. the Space app logic (space/test_app.py)
#   5. example PLYs (fast + official eager) and turntable frames for the media
set -o pipefail
set +e
export PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN=/scratch/runs/$(date -u +%Y%m%d-%H%M%S)-release_check; mkdir -p $RUN
exec > >(tee -a "$RUN/job.log") 2>&1
echo "== $(date -u) $(hostname) run $RUN"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
mkdir -p /work/release && tar xzf /scratch/code/release.tgz -C /work/release && cd /work/release
REF=/scratch/ref; EX=assets/examples; O=/work/o; mkdir -p $O /scratch/release
rm -rf /work/kb && cp -r /scratch/kbuild /work/kb
grep -h '"archs"' -A8 /work/kb/build/torch210-cu130-x86_64-linux/metadata.json | tr -d ' \n'; echo

echo "== 1. fresh install from requirements.txt"
time pip install -q -r requirements.txt 2>&1 | grep -v "^ *$" | tail -3
pip install -q "gradio==6.32.0" pytest 2>&1 | tail -2
pip freeze > $RUN/pip-freeze.txt
python -c "import torch, kernels, transformers, diffusers, PIL; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), 'kernels', kernels.__version__, 'transformers', transformers.__version__, 'diffusers', diffusers.__version__, 'pillow', PIL.__version__)"
export TORCH_CUDA_ARCH_LIST=9.0
pip install -q gsplat 2>&1 | tail -1
python -c "import gsplat; print('gsplat', gsplat.__version__)"

echo "== 1b. error paths"
CUDA_VISIBLE_DEVICES= python scripts/infer.py $EX/fig-robot-cute.webp -o $O/x.ply; echo "exit $?"
env -u HF_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python -c "from fast import runner; runner.check_gated_access()" 2>&1 | tail -8; echo "exit $?"
python -c "from fast import runner; runner.check_gated_access(); print('gated access OK with the job token')"
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/hub.ply 2>&1 | tail -3; echo "exit $? (Hub kernel repo not published yet)"

echo "== 2. infer.py"
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_fast.ply --mode synthetic --local-kernels /work/kb --warmup --turntable
echo "exit $?"
python tools/closeness.py $REF/robot_262k_eager_sdpa_a.safetensors $O/robot_fast.ply --cand_camera $O/robot_fast_camera.json
python scripts/infer.py $EX/food-avocado.webp -o $O/avocado_fast.ply --mode real --local-kernels /work/kb
echo "exit $?"
python tools/closeness.py $REF/avocado_262k_eager_sdpa_a.safetensors $O/avocado_fast.ply --cand_camera $O/avocado_fast_camera.json
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_parity.ply --mode synthetic --build parity --check $REF/robot_262k_eager_sdpa_a.ply
echo "parity exit $?"
cmp $O/robot_parity_camera.json $REF/robot_262k_eager_sdpa_a_camera.json && echo "parity: camera JSON BYTE-IDENTICAL"
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_ref_eager.ply --mode synthetic --build reference --eager --check $REF/robot_262k_eager_sdpa_a.ply
echo "reference eager exit $?"
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_ref_compiled.ply --mode synthetic --build reference --warmup
echo "reference compiled exit $?"
cp $O/*.gif $O/*_camera.json $RUN/ 2>/dev/null

echo "== 3. kernel tests on the new build (torch210-cu130 variant)"
(cd kernel && IDOL_KERNEL_LOCAL_PATH=/work/kb python -m pytest -q tests -rf -p no:cacheprovider 2>&1 | tail -4)

echo "== 4. Space app logic"
KERNEL_BUILD=/work/kb/build bash space/assemble.sh /work/space
(cd /work/space && python test_app.py --ref $REF/robot_262k_eager_sdpa_a.ply --out /scratch/release/precomputed)
echo "app test exit $?"

echo "== 5. media: example PLYs and turntable frames"
python tools/gen_examples.py --local-kernels /work/kb --out /work/ex --official
for f in /work/ex/*_fast.ply; do
  n=$(basename $f _fast.ply)
  python tools/turntable.py /work/ex/${n}_fast.ply /work/ex/${n}_official.ply --npz /scratch/release/frames_${n}.npz --size 384 --frames 72
done
python tools/render_psnr.py /work/ex/fig-robot-cute_official.ply /work/ex/fig-robot-cute_fast.ply 2>&1 | grep -E '^\{|rror' | cut -c1-300
ls -la /scratch/release /scratch/release/precomputed
echo RELEASE_CHECK_DONE
