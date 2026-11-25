# H200 job: the Space app logic only (space/test_app.py), from the release export; writes the robot PLYs the Space
# shows on page load to the bucket's release/precomputed/.
set -o pipefail
set +e
export PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
RUN=/scratch/runs/$(date -u +%Y%m%d-%H%M%S)-app_check; mkdir -p $RUN
exec > >(tee -a "$RUN/job.log") 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
mkdir -p /work/release && tar xzf /scratch/code/release.tgz -C /work/release && cd /work/release
rm -rf /work/kb && cp -r /scratch/kbuild /work/kb
pip install -q -r requirements.txt "gradio==6.32.0" pytest 2>&1 | grep -v "^ERROR: pip\|spin" | tail -2
python -c "import gradio; print('gradio', gradio.__version__)"
KERNEL_BUILD=/work/kb/build bash space/assemble.sh /work/space
mkdir -p /scratch/release/precomputed
(cd /work/space && python test_app.py --ref /scratch/ref/robot_262k_eager_sdpa_a.ply --out /scratch/release/precomputed)
echo "app test exit $?"
ls -la /scratch/release/precomputed
echo APP_CHECK_DONE
