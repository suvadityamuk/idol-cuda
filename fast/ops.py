"""Loads the idol_cuda kernels.

IDOL_KERNEL_LOCAL_PATH=<kernel project dir with build/>: the kernel-builder build through kernels.get_local_kernel
(what scripts/run_kernels.py --local uses). IDOL_KERNEL_HUB=1: kernels.get_kernel('suvadityamuk/idol-cuda').
Otherwise: a development build of the same sources with torch.utils.cpp_extension (never published)."""
import importlib.util
import os
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
_K = None


def _dev_build():
    from torch.utils.cpp_extension import load
    src = ROOT / 'kernel'
    name = 'idol_cuda_dev'
    load(name=name, sources=[str(src / 'torch-ext/torch_binding.cpp'), str(src / 'idol_cuda/idol_kernels.cu')],
         extra_include_paths=[str(src / 'dev'), str(src / 'idol_cuda')],
         extra_cflags=['-O2', '-DCUDA_KERNEL', f'-DTORCH_EXTENSION_NAME={name}'],
         extra_cuda_cflags=['-O3', '-DCUDA_KERNEL', f'-DTORCH_EXTENSION_NAME={name}', '-gencode=arch=compute_90a,code=sm_90a'],
         is_python_module=False, verbose=False)
    pkg = types.ModuleType('idol_cuda_dev_pkg')
    pkg.__path__ = []
    opsmod = types.ModuleType('idol_cuda_dev_pkg._ops')
    opsmod.ops = getattr(torch.ops, name)
    sys.modules['idol_cuda_dev_pkg'] = pkg
    sys.modules['idol_cuda_dev_pkg._ops'] = opsmod
    spec = importlib.util.spec_from_file_location('idol_cuda_dev_pkg.api', src / 'torch-ext/idol_cuda/__init__.py')
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'idol_cuda_dev_pkg'
    spec.loader.exec_module(mod)
    mod.SOURCE = 'dev build (torch.utils.cpp_extension)'
    return mod


def kernel():
    global _K
    if _K is None:
        local = os.environ.get('IDOL_KERNEL_LOCAL_PATH')
        if local:
            import kernels
            _K = kernels.get_local_kernel(Path(local), trust_remote_code=True)
            _K.SOURCE = f'kernel-builder build at {local}'
        elif os.environ.get('IDOL_KERNEL_HUB'):
            import kernels
            _K = kernels.get_kernel('suvadityamuk/idol-cuda', version=1, trust_remote_code=True)
            _K.SOURCE = 'Hub suvadityamuk/idol-cuda v1'
        else:
            _K = _dev_build()
    return _K
