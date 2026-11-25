# H200 job: the published path a user takes. Clone the public GitHub repository, install requirements.txt, run
# scripts/infer.py with its default kernel source (the Hub repo suvadityamuk/idol-cuda, version 1), and compare with
# the stored eager reference.
set -o pipefail
set +e
export PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN=/scratch/runs/$(date -u +%Y%m%d-%H%M%S)-hub_check; mkdir -p $RUN
exec > >(tee -a "$RUN/job.log") 2>&1
nvidia-smi --query-gpu=name --format=csv,noheader
(command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1))
git clone -q https://github.com/suvadityamuk/idol-cuda /work/gh && cd /work/gh && git log --oneline | head -1
pip install -q -r requirements.txt 2>&1 | tail -2
REF=/scratch/ref; EX=assets/examples; O=/work/o; mkdir -p $O
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_fast.ply --mode synthetic --warmup
echo "fast (Hub kernels) exit $?"
python tools/closeness.py $REF/robot_262k_eager_sdpa_a.safetensors $O/robot_fast.ply --cand_camera $O/robot_fast_camera.json
python scripts/infer.py $EX/fig-robot-cute.webp -o $O/robot_parity.ply --mode synthetic --build parity --check $REF/robot_262k_eager_sdpa_a.ply
echo "parity exit $?"
(cd kernel && python -m pytest -q tests -rf -p no:cacheprovider 2>&1 | tail -3)
echo "kernel tests (Hub) exit $?"
