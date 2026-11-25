# Sourced at the start of every HF Job: clones the reference at the pinned commit, installs the pinned environment.
# Mounts expected: /scratch (the bucket named in launch.sh), /w/idol, /w/dinov3, /w/flux2 (model repos).
set -eo pipefail
export SDK=/work/idol-sdk REPO=/work/idol-cuda HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
export RUN=${RUN:-/scratch/runs/$(date -u +%Y%m%d-%H%M%S)-${JOB_NAME:-job}}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/job.log") 2>&1
echo "== $(date -u) host $(hostname) nproc $(nproc) run $RUN"
SDK_SHA=8db700349a5570aac390e4780fa32e3eed6b587a
if [ ! -d "$SDK" ]; then  # the image has no git: fetch the pinned commit's tarball
  python -c "import urllib.request as u; u.urlretrieve('https://github.com/speridlabs/idol-sdk/archive/$SDK_SHA.tar.gz', '/work/sdk.tgz')"
  mkdir -p "$SDK" && tar xzf /work/sdk.tgz -C "$SDK" --strip-components=1
fi
echo "idol-sdk $SDK_SHA"
bash "$REPO/tools/setup_env.sh" "$SDK"
cp /work/pip-freeze.txt "$RUN/" || true
export TORCH_SITE=$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')
export PILLOW_LIBS=$(python -c 'import PIL,os;print(os.path.realpath(os.path.join(os.path.dirname(PIL.__file__),"..","pillow.libs")))')
# FlashAttention-3 (Hopper) Hub kernel used by the fast build: fetched once per job into the local HF cache (the bucket
# mount cannot hold the cache's symlinks), then loaded offline
HF_HUB_OFFLINE=0 python -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3', version=1)" \
  >/dev/null 2>&1 && echo "flash-attn3 kernel cached" || echo "WARNING: flash-attn3 kernel not fetched"
cd "$REPO"
