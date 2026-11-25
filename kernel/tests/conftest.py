import os
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def kernel():
    """IDOL_KERNEL_LOCAL_PATH: a kernel-builder project directory holding build/<variant>/ (after
    `kernel-builder build-and-copy`); otherwise the Hub repo."""
    if os.environ.get("IDOL_KERNEL_DEV"):  # development build of the same sources (not the builder's output)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fast.ops import kernel as dev_kernel
        return dev_kernel()
    import kernels

    local = os.environ.get("IDOL_KERNEL_LOCAL_PATH")
    if local:
        return kernels.get_local_kernel(Path(local), trust_remote_code=True)
    return kernels.get_kernel("suvadityamuk/idol-cuda", version=1, trust_remote_code=True)
