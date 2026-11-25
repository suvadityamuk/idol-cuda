#!/usr/bin/env bash
# Pinned Python environment for the reference on HF Jobs.
# Image: pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel (Python 3.12.3, torch 2.10.0+cu130, cuDNN 9.15.1, nvcc 13.0.48)
# Usage (inside the job): bash tools/setup_env.sh /work/idol-sdk
set -eo pipefail
SDK=${1:-/work/idol-sdk}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
echo "torch==2.10.0+cu130" > /tmp/constraints.txt
echo "torchvision==0.25.0+cu130" >> /tmp/constraints.txt
export PIP_CONSTRAINT=/tmp/constraints.txt
IDX="--index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple"
pip install -q --no-cache-dir $IDX "torch==2.10.0+cu130" "torchvision==0.25.0+cu130"
# tensordict / kernels: newest releases compatible with torch 2.10 (the resolver picks them; see pip-freeze.txt)
pip install -q --no-cache-dir $IDX \
  "transformers==5.24.1" "diffusers==0.43.0" "accelerate==1.16.0" "huggingface_hub==2.4.0" \
  "safetensors==0.8.0" "pillow==13.0.0" "numpy==2.5.5" "plyfile==1.1.5" "tqdm==4.70.1" "tensordict==0.14.3" "einops==0.8.1" \
  "kernels==0.19.0" scipy
# flash-attn prebuilt for torch 2.10 / CUDA 13 / cp312 (only needed for the flash_attn reference configuration)
pip install -q --no-cache-dir --no-deps \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  || echo "WARNING: flash-attn wheel not installed (only the optional flash_attn rows need it)"
pip install -q --no-cache-dir --no-deps -e "$SDK"
python - <<'EOF'
import torch, torchvision, transformers, diffusers, PIL, numpy, tensordict
try:
    import flash_attn
except Exception as e:
    flash_attn = type('x', (), {'__version__': f'absent ({e.__class__.__name__})'})
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("torchvision", torchvision.__version__, "transformers", transformers.__version__, "diffusers", diffusers.__version__)
print("pillow", PIL.__version__, "numpy", numpy.__version__, "tensordict", tensordict.__version__, "flash_attn", flash_attn.__version__)
EOF
pip freeze > /work/pip-freeze.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
