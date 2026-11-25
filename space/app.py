"""idol-cuda demo: the same Idol request through the official pipeline and through the fast build (idol-cuda kernels),
timed on the Space's GPU, both results in a 3D splat viewer.

Unofficial port. Idol, idol-sdk and the official demo (whose viewer fetch patch is reused below) are by Speri Labs,
Apache-2.0: https://github.com/speridlabs/idol-sdk
"""
import os
import random
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import gradio as gr

try:
    import spaces  # ZeroGPU: imported before torch / the model so CUDA placement is deferred
    GPU = spaces.GPU
except ImportError:  # local / HF Jobs: a no-op decorator
    def GPU(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)

import torch  # noqa: E402

from fast import runner  # noqa: E402

ROOT = Path(__file__).parent
os.environ.setdefault('ATTN_BACKEND', 'sdpa')

# Where the idol-cuda kernels come from. None: fast/ops.py loads kernels.get_kernel("suvadityamuk/idol-cuda",
# version=1) from the Hub. A kernel-builder project directory (holding build/<variant>/) uses that build instead.
LOCAL_KERNELS = None

GITHUB = 'https://github.com/suvadityamuk/idol-cuda'
KERNEL_HUB = 'https://huggingface.co/kernels/suvadityamuk/idol-cuda'
MODES = {'Real photo': 'real', 'Synthetic (render / generated)': 'synthetic'}
GAUSSIANS = {'131,072': 131072, '262,144': 262144, '524,288': 524288, '1,048,576': 1048576}
OUT = Path(tempfile.mkdtemp(prefix='idol-cuda-'))
# measured offline: docs/benchmark.md, H200, torch 2.10.0+cu130, CUDA 13.0, robot, 50 steps, 262,144 Gaussians,
# median of 5 interleaved requests, excluding load and the first request
H200 = {'compiled': 6.470, 'eager': 9.599, 'fast': 4.221}

# ------------------------------------------------------------------------------------------------ start-up
STATE = {'error': None, 'notice': None, 'kernels': None, 'load_s': None, 'attn': 'cudnn'}
REF = None
try:
    runner.use_kernels(LOCAL_KERNELS)
    K = runner.load_kernels()
    STATE['kernels'] = K.SOURCE.replace(str(ROOT) + '/', '')
    STATE['kernel_variant'] = Path(K.__file__).parent.name
    print(f'idol-cuda kernels loaded: {STATE["kernels"]} ({STATE["kernel_variant"]})', flush=True)
except Exception as e:  # noqa: BLE001
    STATE['error'] = f'The idol-cuda kernels could not be loaded: {e}'
    print(STATE['error'], flush=True)

if STATE['error'] is None:
    if not os.environ.get('HF_TOKEN'):
        STATE['error'] = (
            'This Space has no HF_TOKEN secret yet. Idol needs two gated image encoders, '
            '<a href="https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m">DINOv3</a> and the '
            '<a href="https://huggingface.co/black-forest-labs/FLUX.2-dev">FLUX.2-dev</a> VAE. The Space owner '
            'needs to add an <code>HF_TOKEN</code> secret (Settings → Variables and secrets) from an account that '
            'has accepted both licences, then restart the Space. Until then nothing can be generated; the kernels '
            'and the GPU self-check below still work.')
    else:
        try:
            t0 = time.perf_counter()
            runner.check_gated_access()
            from idol import IdolPipeline
            REF = IdolPipeline(runner.idol_weights(), device='cuda', optimize=False)
            STATE['load_s'] = time.perf_counter() - t0
            print(f'Idol loaded in {STATE["load_s"]:.1f} s', flush=True)
        except runner.SetupError as e:
            STATE['error'] = 'The HF_TOKEN secret cannot download the gated encoders:<pre>' + str(e) + '</pre>'
        except Exception as e:  # noqa: BLE001
            STATE['error'] = f'Loading Idol failed: {type(e).__name__}: {e}'
            traceback.print_exc()

# ------------------------------------------------------------------------------------------------ GPU side
WORKER = {'fast': None, 'warm': set()}  # lives as long as the ZeroGPU worker process


def _fast_pipeline():
    if WORKER['fast'] is None:
        from fast.pipeline import FastIdolPipeline
        attn, note = runner.pick_attention('auto')
        STATE['attn'], STATE['notice'] = attn, note
        WORKER['fast'] = FastIdolPipeline(ref=REF, attn=attn)
    return WORKER['fast']


def _pool():
    """One long-lived GPU thread per worker process: PyTorch keeps cuBLAS / cuDNN handles per thread, so a fresh
    thread per request would pay their set-up inside the timed request."""
    if WORKER.get('pool') is None:
        from concurrent.futures import ThreadPoolExecutor
        WORKER['pool'] = ThreadPoolExecutor(max_workers=1, thread_name_prefix='idol-gpu')
    return WORKER['pool']


def _run_threaded(fn, progress_fn):
    """Run fn() on the GPU thread; yield ('tick', elapsed, progress) every 0.25 s, then ('done', result, seconds)."""
    from concurrent.futures import TimeoutError as FutureTimeout

    def work():
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        return out, time.perf_counter() - t
    fut = _pool().submit(work)
    t0 = time.perf_counter()
    while True:
        try:
            out, s = fut.result(timeout=0.25)
            break
        except FutureTimeout:
            yield ('tick', time.perf_counter() - t0, progress_fn())
    yield ('done', out, s)


def _request(build, image, kw):
    """Generator for one request of one build on the GPU worker: warm-up (first time in this worker, 2 steps, not
    counted), then the timed request. Yields ('warmup', s) / ('tick', elapsed, progress) / ('done', path, s, info)."""
    if build == 'fast':
        pipe = _fast_pipeline()
        calls = {'n': 0}
        orig = pipe.sampler.velocity

        def counting(*a, **k):
            calls['n'] += 1
            return orig(*a, **k)
        pipe.sampler.velocity = counting
        marks = []
        call = lambda kw2: pipe(image, timings=marks, **kw2)  # noqa: E731
        per_step = 1
    else:
        pipe = REF
        calls = {'n': 0}
        hook = REF.denoiser.register_forward_pre_hook(lambda *_: calls.__setitem__('n', calls['n'] + 1))
        marks = None
        call = lambda kw2: pipe(image, **kw2)  # noqa: E731
        per_step = 2 if kw['cfg_strength'] > 0 else 1  # eager: separate cond / uncond passes
    try:
        if build == 'fast':
            yield ('attn', pipe.sampler.attn)
        if build not in WORKER['warm']:
            _, s = _pool().submit(runner.timed_request, call, dict(kw, steps=2)).result()
            WORKER['warm'].add(build)
            yield ('warmup', s)
            calls['n'] = 0
            if marks is not None:
                marks.clear()
        steps = kw['steps']
        for ev in _run_threaded(lambda: call(kw), lambda: min(steps, calls['n'] // per_step)):
            if ev[0] == 'tick':
                yield ev
            else:
                out, s = ev[1], ev[2]
    finally:
        if build == 'fast':
            pipe.sampler.velocity = orig
        else:
            hook.remove()
    path = OUT / f'{build}-{uuid.uuid4().hex[:8]}.ply'
    (out.save_ply if build == 'fast' else out.gaussian.save_ply)(str(path))
    stages = runner.stage_durations(marks) if marks else []
    yield ('done', str(path), s, {'stages': stages, 'camera': out.camera})


def _duration(build):
    def f(image, mode, seed, steps, gaussians):
        per_step = {'fast': 0.9, 'official': 1.8}[build]  # generous guess for half an RTX PRO 6000 (not measured)
        return int(min(120, 25 + per_step * int(steps)))
    return f


def _gpu(build):
    @GPU(duration=_duration(build))
    def run(image, mode, seed, steps, gaussians):
        kw = dict(mode=MODES[mode], seed=int(seed), steps=int(steps), cfg_strength=3.0, rescale_t=3.0,
                  num_gaussians=GAUSSIANS[gaussians])
        yield ('gpu', torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0)))
        yield from _request(build, image, kw)
    run.__name__ = f'run_{build}'
    return run


RUN = {'fast': _gpu('fast'), 'official': _gpu('official')}


@GPU(duration=90)
def selfcheck():
    """The kernel package's own 44 tests on this GPU (the loaded build), plus device facts."""
    import io
    import contextlib
    import pytest
    os.environ['IDOL_KERNEL_LOCAL_PATH'] = LOCAL_KERNELS or ''
    buf = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        code = pytest.main(['-q', '-p', 'no:cacheprovider', str(ROOT / 'kernel_tests')])
    tail = '\n'.join(buf.getvalue().strip().splitlines()[-6:])
    return (f'GPU: {torch.cuda.get_device_name(0)} (sm_{"".join(map(str, torch.cuda.get_device_capability(0)))}), '
            f'torch {torch.__version__}, CUDA {torch.version.cuda}\nkernels: {STATE["kernels"]} '
            f'[{STATE.get("kernel_variant")}]\npytest exit code {int(code)} in {time.perf_counter() - t:.1f} s\n{tail}')




# ------------------------------------------------------------------------------------------------ UI side
# One idea per screen: the same image through both builds, two big numbers, the speed-up. Everything else (settings,
# explanation, self-check) sits behind two quiet disclosures. No number is shown until it is measured, except the
# labelled offline H200 line.
NAME = {'fast': ('idol-cuda', 'fast build'), 'official': ('Official pipeline', 'eager')}


def _esc(s):
    import html
    return html.escape(str(s))


def card(build, state):
    st = state.get(build) or {}
    name, tag = NAME[build]
    status = st.get('status')
    big, note, cls = '', '', ''
    if status == 'done':
        big = f'{st["s"]:.1f}<small> s</small>'
        note = 'per request'
    elif status == 'running':
        big, cls = f'{st.get("elapsed", 0):.1f}<small> s</small>', ' is-live'
        note = f'running · step {st.get("progress", 0)} of {st.get("steps")}'
    elif status == 'warmup':
        big, cls = 'Warming up', ' is-word'
        note = 'once per GPU worker, not counted'
    elif status == 'waiting':
        big, cls = 'Next', ' is-word is-dim'
        note = 'starts when idol-cuda finishes'
    elif status == 'skipped':
        big, cls = 'Not run', ' is-word is-dim'
        note = 'Compare runs both builds'
    else:
        cls = ' is-empty'
        note = 'example output · not timed yet'
    if st.get('warmup') is not None and status in ('running', 'done'):
        note += f' · {st["warmup"]:.1f} s warm-up not counted'
    return (f'<div class="ic-card ic-{build}"><div class="ic-name">{name} <span>{tag}</span></div>'
            f'<div class="ic-num{cls}">{big or "&nbsp;"}</div><div class="ic-note">{note}</div></div>')


def hero(state):
    f, o = state.get('fast') or {}, state.get('official') or {}
    if f.get('status') == 'done' and o.get('status') == 'done':
        gpu = _esc((state.get('gpu_name') or 'this GPU').replace(' Server Edition', ''))
        return (f'<div class="ic-hero is-result"><h1><span class="ic-x">{o["s"] / f["s"]:.2f}×</span> faster</h1>'
                f'<p>idol-cuda vs the official Idol pipeline (eager), same image and seed, '
                f'measured just now on {gpu}.</p></div>')
    return ('<div class="ic-hero"><h1>Same image. Same model.<br><span class="ic-x">About half the time.</span></h1>'
            '<p>Idol turns one image into a 3D Gaussian splat. Compare runs the official pipeline (eager) and '
            'idol-cuda on this GPU with the same seed, and times both.</p></div>')


def details(state):
    rows = []
    if state.get('gpu'):
        rows.append(f'GPU: {_esc(state["gpu"])}; attention in idol-cuda: '
                    f'{"FlashAttention-3" if state.get("attn") == "fa3" else "cuDNN"}; seed {state.get("seed")}.')
    for build in ('fast', 'official'):
        st = state.get(build) or {}
        if st.get('status') == 'done':
            line = f'{NAME[build][0]}: {st["s"]:.2f} s'
            if st.get('warmup') is not None:
                line += f' (one-off warm-up on this worker, 2 steps, not counted: {st["warmup"]:.1f} s)'
            if st.get('stages'):
                line += '; ' + ', '.join(f'{_esc(n)} {s:.2f} s' for n, s in st['stages'])
            rows.append(line + '.')
    return '<div class="ic-run">' + ''.join(f'<p>{r}</p>' for r in rows) + '</div>' if rows else ''


def generate(image, mode, seed, steps, gaussians, randomize, compare):
    if image is None:
        raise gr.Error('Upload an image or pick an example first.')
    if STATE['error'] or REF is None:
        import re
        raise gr.Error('Generation is unavailable. ' + re.sub(r'<[^>]+>', '', STATE['error'] or 'Model not loaded.'), 15)
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    state = {'fast': None, 'official': {'status': 'waiting' if compare else 'skipped'}, 'seed': int(seed)}
    files = {'fast': None, 'official': None}
    sent = {}

    def emit():
        # Send a .ply only when it changes: re-sending the finished fast splat with every progress tick of the
        # official run made each tick cost seconds (the file is re-served and re-loaded by the viewer every time).
        def once(slot, build):
            if slot in sent and sent[slot] == files[build]:
                return gr.skip()
            sent[slot] = files[build]
            return files[build]
        return (once('vf', 'fast'), once('vo', 'official'), card('fast', state), card('official', state),
                hero(state), details(state), int(seed))

    for build in (['fast', 'official'] if compare else ['fast']):
        state[build] = {'status': 'warmup', 'steps': int(steps)}
        yield emit()
        for ev in RUN[build](image, mode, seed, steps, gaussians):
            if ev[0] == 'gpu':
                state['gpu'] = f'{ev[1]} (sm_{ev[2][0]}{ev[2][1]})'
                state['gpu_name'] = ev[1]
            elif ev[0] == 'attn':
                state['attn'] = ev[1]
            elif ev[0] == 'warmup':
                state[build]['warmup'] = ev[1]
            elif ev[0] == 'tick':
                state[build].update(status='running', elapsed=ev[1], progress=ev[2])
            elif ev[0] == 'done':
                state[build].update(status='done', s=ev[2], stages=ev[3]['stages'])
                files[build] = ev[1]
            yield emit()


def run_diag():
    if STATE.get('kernels') is None:
        return STATE['error'] or 'kernels not loaded'
    return selfcheck()


EXAMPLES = [  # all from idol-sdk demo/examples (see examples/README.md)
    ('fig-robot-cute.webp', 'Synthetic (render / generated)'), ('food-avocado.webp', 'Real photo'),
    ('obj-backpack.webp', 'Synthetic (render / generated)'), ('toy-train.webp', 'Synthetic (render / generated)'),
    ('food-donut.webp', 'Synthetic (render / generated)'), ('obj-power-drill.webp', 'Synthetic (render / generated)'),
    ('fig-fox.webp', 'Synthetic (render / generated)'), ('food-mushroom.webp', 'Synthetic (render / generated)'),
]
PRE = ROOT / 'precomputed'
CLEAR = (0.0, 0.0, 0.0, 1.0)  # = --ic-stage below

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&'
         'family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">')

CSS = """
:root, .light { --ic-bg: #fbfbfd; --ic-fg: #111114; --ic-mute: #6e6e76; --ic-line: #e4e4ea; --ic-accent: #0a66e4;
  --ic-panel: #f0f0f3; --ic-stage: #000; }
.dark { --ic-bg: #121214; --ic-fg: #f5f5f7; --ic-mute: #94949c; --ic-line: #2a2a2f; --ic-accent: #3d9bff;
  --ic-panel: #1c1c1f; --ic-stage: #000; }
body, gradio-app, .gradio-container, .main, .wrap.app { background: var(--ic-bg) !important; }
.gradio-container { font-family: 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
  color: var(--ic-fg) !important; max-width: 1480px !important; margin: 0 auto !important; padding: 0 clamp(16px, 3vw, 40px) !important; }
.gradio-container * { font-family: inherit; }
footer { display: none !important; }
.gradio-container, .gradio-container .main, .gradio-container .wrap, .gradio-container main.contain { min-width: 0 !important; }
.gradio-container .main { padding: 12px 0 0 !important; }
.html-container { padding: 0 !important; }

/* top bar */
.ic-bar { display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: clamp(8px, 2vh, 18px) 0 clamp(4px, 1vh, 8px); }
.ic-mark { font-weight: 700; font-size: 17px; letter-spacing: -0.02em; color: var(--ic-fg) !important; }
.ic-mark b { color: var(--ic-accent) !important; font-weight: 700; }
.ic-links { display: flex; gap: 22px; font-size: 14px; }
.ic-links a { color: var(--ic-mute) !important; text-decoration: none !important; }
.ic-links a:hover { color: var(--ic-fg) !important; }

/* hero */
.ic-hero { padding: clamp(4px, min(1.6vw, 2.4vh), 26px) 0 clamp(14px, min(2vw, 3.4vh), 30px); }
.ic-hero h1 { font-size: clamp(38px, min(5.6vw, 9.5vh), 84px) !important; line-height: 1.0 !important; letter-spacing: -0.045em;
  font-weight: 700 !important; margin: 0 0 14px !important; color: var(--ic-fg) !important; text-wrap: balance; }
.ic-hero.is-result h1 { font-size: clamp(52px, min(8vw, 13vh), 124px) !important; letter-spacing: -0.05em; }
.ic-x { color: var(--ic-accent) !important; font-variant-numeric: tabular-nums; }
.ic-hero p { font-size: clamp(15px, 1.35vw, 20px) !important; line-height: 1.4 !important; color: var(--ic-mute) !important;
  margin: 0 !important; max-width: 72ch; text-wrap: pretty; }

/* stage: input + two dark panes */
#ic-stage { gap: clamp(12px, 1.4vw, 20px) !important; align-items: stretch; flex-wrap: nowrap; }
#ic-stage > .column { gap: 0 !important; }
.ic-pane { background: var(--ic-stage) !important; border-radius: 22px !important; overflow: hidden; }
.ic-pane .block, .ic-pane .html-container, .ic-pane .prose { background: transparent !important; border: 0 !important;
  box-shadow: none !important; padding: 0 !important; }
.ic-pane .model3D { height: clamp(190px, calc(100vh - 500px), 480px) !important; }
.ic-pane > .block, .ic-input > .block { height: auto !important; }
.ic-pane canvas { display: block; width: 100% !important; height: 100% !important; }
.ic-pane .icon-button-wrapper { background: transparent !important; border: 0 !important; box-shadow: none !important;
  top: 10px !important; right: 12px !important; }
.ic-pane .icon-button { color: #8f8f98 !important; background: transparent !important; border: 0 !important; }
.ic-card { padding: clamp(14px, 1.6vw, 24px) clamp(16px, 1.8vw, 28px) 0; color: #f5f5f7 !important; }
.ic-name { font-size: clamp(15px, 1.35vw, 22px); font-weight: 600; letter-spacing: -0.01em; color: #f5f5f7 !important; }
.ic-name span { color: #8f8f98 !important; font-weight: 500; }
.ic-fast .ic-name { color: #5aa9ff !important; }
.ic-num { font-size: clamp(48px, min(5.8vw, 8.6vh), 96px); font-weight: 700; letter-spacing: -0.045em; line-height: 1.02;
  font-variant-numeric: tabular-nums; color: #fff !important; margin-top: 2px; white-space: nowrap; }
.ic-num small { font-size: 0.42em; letter-spacing: -0.01em; font-weight: 600; color: #a1a1aa !important; }
.ic-fast .ic-num { color: #4ea3ff !important; }
.ic-num.is-live, .ic-fast .ic-num.is-live { color: #d4d4d8 !important; }
.ic-num.is-word { font-size: clamp(30px, 3vw, 48px); letter-spacing: -0.035em; padding: 0.18em 0 0.12em; }
.ic-num.is-dim, .ic-fast .ic-num.is-dim { color: #6e6e76 !important; }
.ic-fast .ic-num.is-word { color: #f5f5f7 !important; }
.ic-num.is-empty { display: none; }
.ic-note { font-size: clamp(12px, 1vw, 15px); color: #8f8f98 !important; margin-top: 4px; min-height: 1.4em; }
.ic-input { background: var(--ic-panel) !important; border-radius: 22px !important; padding: 12px !important;
  justify-content: space-between; gap: 12px !important; }
.ic-input > .block { flex: 1 1 auto; display: flex; align-items: flex-start; justify-content: center; }
.ic-input .image-container { height: auto !important; }
.ic-input img { border-radius: 14px; max-height: clamp(150px, calc(100vh - 460px), 420px); }
.ic-input .image-container { position: relative; }
.ic-input .icon-button-wrapper { position: absolute !important; top: 6px !important; right: 6px !important; z-index: 5;
  background: rgba(0,0,0,.45) !important; border: 0 !important; border-radius: 999px !important; }
.ic-input .icon-button { color: #fff !important; }
.ic-input .block { background: transparent !important; border: 0 !important; box-shadow: none !important; }
.ic-input .image-container, .ic-input .upload-container { border-radius: 14px; overflow: hidden; }
#ic-go { flex: 0 0 auto !important; border-radius: 999px !important; font-size: 17px !important; font-weight: 600 !important;
  padding: 14px 18px !important; background: var(--ic-accent) !important; color: #fff !important; border: 0 !important;
  box-shadow: none !important; }
#ic-go:hover { filter: brightness(1.08); }
.ic-fine { font-size: 13px; color: var(--ic-mute) !important; margin: 14px 2px 0; line-height: 1.45; }

/* examples strip */
.ic-strip-label { font-size: 14px; color: var(--ic-mute) !important; margin: 30px 2px 8px; }
#ic-examples { background: transparent !important; border: 0 !important; padding: 0 !important; }
#ic-examples .grid-wrap { padding: 0 !important; overflow: visible !important; max-height: none !important;
  height: auto !important; min-height: 0 !important; }
#ic-examples, #ic-examples .gallery-container { height: auto !important; min-height: 0 !important; }
#ic-examples .grid-container { gap: 10px !important; grid-template-columns: repeat(8, minmax(0, 1fr)) !important;
  min-width: 0 !important; }
#ic-examples .gallery-item, #ic-examples .thumbnail-item { min-width: 0 !important; width: 100% !important;
  height: auto !important; aspect-ratio: 1; }
#ic-examples .thumbnail-item { border-radius: 14px !important; border: 0 !important; box-shadow: none !important;
  background: var(--ic-panel) !important; }
#ic-examples .thumbnail-item:hover { transform: translateY(-2px); }
#ic-examples .icon-button-wrapper, #ic-examples .block-label { display: none !important; }

/* quiet disclosures */
.ic-more { margin-top: 28px !important; gap: 10px !important; }
.ic-more > .block, .ic-acc { background: transparent !important; border: 0 !important; border-top: 1px solid var(--ic-line) !important;
  border-radius: 0 !important; box-shadow: none !important; padding: 6px 0 !important; }
.ic-acc .label-wrap { font-size: 15px !important; color: var(--ic-mute) !important; }
.ic-acc .label-wrap span { font-weight: 500 !important; }
.ic-doc { font-size: 14px; line-height: 1.55; color: var(--ic-mute) !important; max-width: 75ch; }
.ic-doc p { margin: 0 0 10px; }
.ic-doc a, .ic-run a, .ic-banner a { color: var(--ic-accent) !important; padding: 0 !important; margin: 0 !important; }
.ic-acc .form { max-width: 720px; }
.ic-run { font-size: 13px; color: var(--ic-mute) !important; font-family: 'Geist Mono', ui-monospace, monospace !important;
  overflow-wrap: anywhere; }
.ic-run p { margin: 0 0 4px; }
.ic-banner { border-radius: 18px; padding: 16px 20px; background: #fff1ec; color: #7a2410 !important; font-size: 15px;
  line-height: 1.5; overflow-wrap: anywhere; margin-bottom: 18px; }
.dark .ic-banner { background: #2a130c; color: #ffb59e !important; }
.ic-banner pre { white-space: pre-wrap; font-size: 12px; }
.ic-end { height: 40px; }

/* narrow screens: stack, input first, panes full width */
@media (max-width: 900px) {
  #ic-stage { flex-wrap: wrap !important; }
  #ic-stage > .ic-input { flex: 1 1 100% !important; min-width: 100% !important; flex-direction: row !important;
    gap: 14px !important; align-items: center !important; flex-wrap: nowrap !important; }
  #ic-stage > .ic-input > .block { flex: 0 0 120px !important; max-width: 120px; justify-content: flex-start !important; }
  .ic-input img { max-height: 120px; }
  #ic-stage > .ic-input #ic-go { flex: 1 1 0 !important; min-width: 0 !important; align-self: center; }
  .ic-pane .model3D { height: 300px !important; }
  #ic-stage > .ic-pane { flex: 1 1 calc(50% - 10px) !important; min-width: 280px !important; }
  .ic-links { gap: 14px; }
}
@media (max-width: 620px) {
  #ic-stage > .ic-pane { flex: 1 1 100% !important; min-width: 100% !important; }
  #ic-examples .grid-container { grid-template-columns: repeat(4, minmax(0, 1fr)) !important; }
}
"""
# Same fix as the official Idol demo (Apache-2.0): the splat viewer fetches the .ply with credentials omitted, which a
# private Space answers with 404; send the session cookies with file requests.
JS = """() => {
  const f = window.fetch.bind(window);
  window.fetch = (r, o) => { const u = typeof r === 'string' ? r : (r && r.url) || '';
    if (u.includes('/gradio_api/file=')) o = Object.assign({}, o, { credentials: 'same-origin' });
    return f(r, o); };
}"""

BAR = (f'<div class="ic-bar"><div class="ic-mark">idol<b>-</b>cuda</div><div class="ic-links">'
       f'<a href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>'
       f'<a href="{KERNEL_HUB}" target="_blank" rel="noopener">Kernels</a></div></div>')
FINE = (f'Baseline: the official Idol pipeline in eager mode, on this GPU. Measured offline on an H200: official '
        f'compiled (default) {H200["compiled"]:.2f} s, idol-cuda {H200["fast"]:.2f} s, '
        f'{H200["compiled"] / H200["fast"]:.2f}× faster.')
DOC = f"""<div class="ic-doc">
<p>Idol turns one image into a 3D Gaussian splat. idol-cuda keeps its encoders and decoders and
replaces the sampler: context computed once per request, a cheaper unconditional pass, and fused CUDA kernels from the
<a href="{KERNEL_HUB}">Kernel Hub</a>. Both builds get the same image and seed; the splats are close, not identical,
because idol-cuda draws its octree samples differently.</p>
<p>The live baseline is the official pipeline in eager mode (<code>--no_optimize</code>), as the official Idol Space
runs it: its default path uses torch.compile, which ZeroGPU does not support. Measured offline on an H200 (50 steps,
262,144 Gaussians, median of 5): official compiled {H200["compiled"]:.2f} s, official eager {H200["eager"]:.2f} s,
idol-cuda {H200["fast"]:.2f} s.</p>
<p>Times are per request, GPU-synchronised, and exclude model loading and a one-off warm-up per GPU worker. The
viewers open on the robot example, generated beforehand. Source, method and benchmarks:
<a href="{GITHUB}">GitHub</a>.</p></div>"""

theme = gr.themes.Base(
    font=[gr.themes.Font(f) for f in ('Geist', 'ui-sans-serif', 'system-ui', 'sans-serif')],
    font_mono=[gr.themes.Font(f) for f in ('Geist Mono', 'ui-monospace', 'monospace')],
    radius_size=gr.themes.sizes.radius_lg,
).set(body_background_fill='#fbfbfd', body_background_fill_dark='#000000',
      block_background_fill='transparent', block_background_fill_dark='transparent',
      block_border_width='0px', block_shadow='none', block_label_border_width='0px',
      button_primary_background_fill='#0a66e4', button_primary_background_fill_dark='#3d9bff',
      button_primary_text_color='#ffffff', button_primary_text_color_dark='#ffffff',
      color_accent='#0a66e4', color_accent_soft='#e8f0fd', color_accent_soft_dark='#0c2342',
      checkbox_background_color_selected='#0a66e4', slider_color='#0a66e4', slider_color_dark='#3d9bff')

with gr.Blocks(title='idol-cuda: Idol in about half the time') as demo:
    gr.HTML(BAR, padding=False)
    hero_h = gr.HTML(hero({}), padding=False)
    if STATE['error']:
        gr.HTML(f'<div class="ic-banner">{STATE["error"]}</div>', padding=False)
    with gr.Row(elem_id='ic-stage', equal_height=True):
        with gr.Column(scale=3, min_width=200, elem_classes='ic-input'):
            image = gr.Image(value=str(ROOT / 'examples' / 'fig-robot-cute.webp'), type='pil', image_mode='RGBA',
                             show_label=False, sources=['upload'], height=300, container=False, buttons=[],
                             placeholder='Drop an image of one object')
            btn_cmp = gr.Button('Compare', variant='primary', elem_id='ic-go')
        with gr.Column(scale=5, min_width=280, elem_classes='ic-pane'):
            card_f = gr.HTML(card('fast', {}), padding=False)
            view_f = gr.Model3D(show_label=False, height=360, clear_color=CLEAR, container=False,
                                value=str(PRE / 'robot_fast.ply') if (PRE / 'robot_fast.ply').exists() else None)
        with gr.Column(scale=5, min_width=280, elem_classes='ic-pane'):
            card_o = gr.HTML(card('official', {}), padding=False)
            view_o = gr.Model3D(show_label=False, height=360, clear_color=CLEAR, container=False,
                                value=str(PRE / 'robot_official_eager.ply')
                                if (PRE / 'robot_official_eager.ply').exists() else None)
    gr.HTML(f'<p class="ic-fine">{FINE}</p>', padding=False)
    gr.HTML('<div class="ic-strip-label">Or try one of these</div>', padding=False)
    gallery = gr.Gallery([str(ROOT / 'examples' / f) for f, _ in EXAMPLES], columns=8, rows=1, height='auto',
                         allow_preview=False, show_label=False, object_fit='cover', elem_id='ic-examples',
                         interactive=False, buttons=[], container=False)
    with gr.Column(elem_classes='ic-more'):
        with gr.Accordion('Options', open=False, elem_classes='ic-acc'):
            mode = gr.Radio(list(MODES), value='Synthetic (render / generated)', label='Input type')
            gaussians = gr.Dropdown(list(GAUSSIANS), value='262,144', label='Gaussians')
            steps = gr.Slider(10, 50, value=50, step=1, label='Steps')
            seed = gr.Number(value=42, precision=0, label='Seed (both builds use it)')
            randomize = gr.Checkbox(value=False, label='Random seed')
            btn_fast = gr.Button('Run idol-cuda only', variant='secondary', size='sm')
        with gr.Accordion('Details', open=False, elem_classes='ic-acc'):
            gr.HTML(DOC, padding=False)
            info = gr.HTML('', padding=False)
            gr.HTML(f'<div class="ic-doc"><p>Kernel self-check: runs the kernel package\'s 44 tests on this GPU '
                    f'(<code>{_esc(STATE.get("kernels"))}</code>, {_esc(STATE.get("kernel_variant"))}).</p></div>',
                    padding=False)
            diag_btn = gr.Button('Run self-check', variant='secondary', size='sm')
            diag = gr.Textbox(show_label=False, lines=7, max_lines=12, placeholder='Result appears here')
    gr.HTML('<div class="ic-end"></div>', padding=False)

    def pick(evt: gr.SelectData):
        f, m = EXAMPLES[evt.index]
        return str(ROOT / 'examples' / f), m
    gallery.select(pick, None, [image, mode], show_progress='hidden', api_name=False)

    outs = [view_f, view_o, card_f, card_o, hero_h, info, seed]
    ins = [image, mode, seed, steps, gaussians, randomize]
    btn_cmp.click(lambda *a: (yield from generate(*a, compare=True)), ins, outs, concurrency_limit=1,
                  concurrency_id='gpu', api_name='compare', show_progress='hidden')
    btn_fast.click(lambda *a: (yield from generate(*a, compare=False)), ins, outs, concurrency_limit=1,
                   concurrency_id='gpu', api_name='generate_fast', show_progress='hidden')
    diag_btn.click(run_diag, None, diag, concurrency_limit=1, concurrency_id='gpu', api_name='selfcheck')

if __name__ == '__main__':
    demo.queue(max_size=20, default_concurrency_limit=1).launch(css=CSS, js=JS, head=FONTS, theme=theme,
                                                               footer_links=[])
