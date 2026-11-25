"""Shared helpers for scripts/infer.py and the Space demo: environment checks, weight resolution, kernel source,
attention choice and per-stage timing of one request.

Nothing here changes the arithmetic of the builds in fast/pipeline.py; it only decides what to load and measures it.
"""
import os
import time

GATED = ['facebook/dinov3-vith16plus-pretrain-lvd1689m', 'black-forest-labs/FLUX.2-dev']
IDOL_REPO = 'speridlabs/idol'
KERNEL_REPO = 'suvadityamuk/idol-cuda'
KERNEL_VERSION = 1
STAGES = ['preprocess', 'encoders', 'sampling', 'octree', 'gs_decoder', 'to_gaussians']


class SetupError(RuntimeError):
    """A problem the user can fix (no GPU, unsupported GPU, missing gated access, missing kernels)."""


def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise SetupError('No CUDA GPU is visible to PyTorch. idol-cuda needs an NVIDIA GPU (compute capability 8.0 '
                         'or newer; tested on H200, sm_90) and a CUDA build of torch.')
    cap = torch.cuda.get_device_capability(0)
    if cap < (8, 0):
        raise SetupError(f'{torch.cuda.get_device_name(0)} has compute capability {cap[0]}.{cap[1]}; the idol-cuda '
                         'kernels are built for 8.0, 8.6, 8.9, 9.0, 10.0 and 12.0 only. Use --build parity or '
                         '--build reference, which need no custom kernels.')
    return torch.cuda.get_device_name(0), cap


def check_gated_access(dinov3=None, flux2=None):
    """Raises SetupError with the SDK's own explanation when the gated encoders cannot be downloaded (encoders given
    as local folders are not checked)."""
    from idol.conditioning import EncoderAccessError, require_encoder_access
    repos = [r for r, local in zip(GATED, (dinov3, flux2)) if not local]
    try:
        require_encoder_access(repos)
    except EncoderAccessError as e:
        raise SetupError(str(e)) from None


def idol_weights(path=None):
    """Idol weights directory: the given path, /w/idol (HF Jobs mount) or a Hub download (about 3 GB, once)."""
    if path:
        return path
    if os.path.isdir('/w/idol'):
        return '/w/idol'
    from huggingface_hub import snapshot_download
    from idol.pipeline import WEIGHT_FILES
    return snapshot_download(IDOL_REPO, allow_patterns=WEIGHT_FILES)


def encoder_paths(dinov3=None, flux2=None):
    """Local encoder folders if given or mounted (HF Jobs), else None = the Hub ids in Idol's config.json."""
    d = dinov3 or ('/w/dinov3' if os.path.isdir('/w/dinov3') else None)
    f = flux2 or ('/w/flux2' if os.path.isdir('/w/flux2') else None)
    return d, f


def use_kernels(local_path=None):
    """Select where fast/ops.py loads the idol_cuda kernels from: a local kernel-builder project directory (one
    holding build/<variant>/) or the Hub repo suvadityamuk/idol-cuda, version 1."""
    os.environ.pop('IDOL_KERNEL_LOCAL_PATH', None)
    os.environ.pop('IDOL_KERNEL_HUB', None)
    if local_path:
        if not os.path.isdir(os.path.join(local_path, 'build')):
            raise SetupError(f'{local_path}/build not found: --local-kernels takes the kernel project directory '
                             '(the one that contains build/<variant>/), e.g. kernel/')
        os.environ['IDOL_KERNEL_LOCAL_PATH'] = os.path.abspath(local_path)
    else:
        os.environ['IDOL_KERNEL_HUB'] = '1'


def load_kernels():
    """Load the kernels now so that failures are reported clearly (before the weights are loaded)."""
    from .ops import kernel
    try:
        return kernel()
    except Exception as e:
        msg = str(e)
        if 'does not support the current device' in msg or 'CUDA capability' in msg:
            raise SetupError(f'The idol-cuda kernel build has no variant for this GPU: {msg}') from None
        hub = bool(os.environ.get('IDOL_KERNEL_HUB'))
        hint = (f' Could not load {KERNEL_REPO} (version {KERNEL_VERSION}) from the Hub. If the kernel repo is not '
                'published yet, pass --local-kernels kernel/ (a kernel-builder build) instead.') if hub else ''
        raise SetupError(f'Loading the idol-cuda kernels failed ({type(e).__name__}: {msg}).{hint}') from None


def pick_attention(requested='auto'):
    """FlashAttention-3 (kernels-community/flash-attn3) is built for Hopper (sm_90) only; elsewhere cuDNN attention.
    Returns (backend, notice or None)."""
    import torch
    major = torch.cuda.get_device_capability(0)[0]
    if requested == 'cudnn':
        return 'cudnn', None
    if major != 9:
        note = ('FlashAttention-3 is Hopper-only (sm_90); this GPU is sm_%d%d, so the fast build uses cuDNN attention.'
                % torch.cuda.get_device_capability(0))
        if requested == 'fa3':
            note += ' (--attn fa3 ignored)'
        return 'cudnn', note
    try:
        from .model import _fa3
        _fa3()
        return 'fa3', None
    except Exception as e:
        return 'cudnn', (f'FlashAttention-3 could not be loaded from kernels-community/flash-attn3 '
                         f'({type(e).__name__}: {str(e)[:200]}); falling back to cuDNN attention (slower).')


def stage_durations(marks):
    """[(name, perf_counter)] from FastIdolPipeline(timings=...) -> [(stage, seconds)]."""
    return [(marks[i][0], marks[i][1] - marks[i - 1][1]) for i in range(1, len(marks))]


def timed_request(fn, *args, **kwargs):
    """Run one request with CUDA syncs around it; returns (result, seconds)."""
    import torch
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t
