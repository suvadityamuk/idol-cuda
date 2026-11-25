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
LABEL = {'fast': 'Fast build · idol-cuda kernels', 'official': 'Official pipeline, eager (--no_optimize)'}


def card(build, state, gpu=None):
    st = state.get(build) or {}
    rows = []
    if st.get('status') == 'running':
        prog = st.get('progress', 0)
        rows.append(f'<div class="ic-big">{st.get("elapsed", 0):.1f} s</div>'
                    f'<div class="ic-sub">running · sampling step {prog}/{st.get("steps")}</div>')
    elif st.get('status') == 'warmup':
        rows.append('<div class="ic-big">…</div><div class="ic-sub">one-off warm-up on this GPU worker</div>')
    elif st.get('status') == 'done':
        rows.append(f'<div class="ic-big">{st["s"]:.2f} s</div><div class="ic-sub">per request (image in → '
                    'Gaussians out, GPU-synced; excludes loading and warm-up)</div>')
        if st.get('stages'):
            rows.append('<table class="ic-stages">' + ''.join(
                f'<tr><td>{n}</td><td>{s:.2f} s</td></tr>' for n, s in st['stages']) + '</table>')
    elif st.get('status') == 'skipped':
        rows.append('<div class="ic-sub">Not run. Press “Compare with official” to run the official pipeline '
                    'on the same image and seed.</div>')
    else:
        rows.append('<div class="ic-sub">–</div>')
    if st.get('warmup') is not None:
        rows.append(f'<div class="ic-note">one-off warm-up on this GPU worker (2 steps, not counted): '
                    f'{st["warmup"]:.1f} s</div>')
    return f'<div class="ic-card"><div class="ic-label">{LABEL[build]}</div>{"".join(rows)}</div>'


def summary(state):
    f, o = state.get('fast') or {}, state.get('official') or {}
    gpu = state.get('gpu')
    lines = []
    if f.get('status') == 'done' and o.get('status') == 'done':
        lines.append(f'<div class="ic-speed">{o["s"] / f["s"]:.2f}× faster than the official pipeline (eager), '
                     f'measured now on this Space</div>')
    if gpu:
        lines.append(f'<div class="ic-sub">GPU: {gpu}. Attention in the fast build: '
                     f'{"FlashAttention-3" if state.get("attn") == "fa3" else "cuDNN"}.</div>')
    lines.append(
        '<div class="ic-note">The official pipeline\'s default uses torch.compile. ZeroGPU\'s documentation lists '
        'torch.compile as unsupported, and its first request took about 21 s of compilation on an H200, which would be '
        'paid again by every new GPU worker on visitors\' quota. So the live baseline is the official pipeline in eager '
        'mode (<code>--no_optimize</code>), which is also what the official Idol Space runs. <b>Measured offline on H200</b> (torch 2.10 + CUDA 13.0, 50 steps, 262,144 Gaussians, '
        f'median of 5): official compiled {H200["compiled"]:.2f} s, official eager {H200["eager"]:.2f} s, fast build '
        f'{H200["fast"]:.2f} s, i.e. {H200["compiled"] / H200["fast"]:.2f}× vs compiled and '
        f'{H200["eager"] / H200["fast"]:.2f}× vs eager. On the H200 the fast build also uses FlashAttention-3, '
        'which is Hopper-only.</div>')
    return ''.join(lines)


def _views(state):
    return card('fast', state), card('official', state), summary(state)


def generate(image, mode, seed, steps, gaussians, randomize, compare):
    if image is None:
        raise gr.Error('Upload an image or pick an example first.')
    if STATE['error'] or REF is None:
        import re
        raise gr.Error('Generation is unavailable. ' + re.sub(r'<[^>]+>', '', STATE['error'] or 'Model not loaded.'), 15)
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    state = {'fast': None, 'official': {'status': 'skipped'} if not compare else None}
    files = {'fast': None, 'official': None}

    sent = {}

    def emit():
        # Send a .ply only when it changes: re-sending the finished fast splat with every progress tick of the
        # official run made each tick cost seconds (the file is re-served and re-loaded by the viewer every time).
        f, o, s = _views(state)

        def once(slot, build):
            if slot in sent and sent[slot] == files[build]:
                return gr.skip()
            sent[slot] = files[build]
            return files[build]
        return (once('vf', 'fast'), once('vo', 'official'), f, o, s, once('ff', 'fast'), once('fo', 'official'),
                int(seed))

    for build in (['fast', 'official'] if compare else ['fast']):
        state[build] = {'status': 'warmup', 'steps': int(steps)}
        yield emit()
        for ev in RUN[build](image, mode, seed, steps, gaussians):
            if ev[0] == 'gpu':
                state['gpu'] = f'{ev[1]} (sm_{ev[2][0]}{ev[2][1]})'
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


EXAMPLES = [  # all from idol-sdk demo/examples (Apache-2.0, https://github.com/speridlabs/idol-sdk)
    ('fig-robot-cute.webp', 'Synthetic (render / generated)'), ('food-avocado.webp', 'Real photo'),
    ('obj-backpack.webp', 'Synthetic (render / generated)'), ('toy-train.webp', 'Synthetic (render / generated)'),
    ('food-donut.webp', 'Synthetic (render / generated)'), ('obj-power-drill.webp', 'Synthetic (render / generated)'),
    ('fig-fox.webp', 'Synthetic (render / generated)'), ('food-mushroom.webp', 'Synthetic (render / generated)'),
]
PRE = ROOT / 'precomputed'

CSS = """
.ic-card { border: 1px solid #8884; padding: 14px 16px; min-height: 120px; }
.ic-label { font-family: ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; font-size: 12px;
  opacity: .7; margin-bottom: 6px; }
.ic-big { font-size: 34px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; }
.ic-sub { font-size: 14px; opacity: .8; }
.ic-note { font-size: 13px; opacity: .7; margin-top: 8px; }
.ic-speed { font-size: 22px; font-weight: 700; margin: 4px 0 6px; }
.ic-stages { font-size: 13px; margin-top: 6px; font-variant-numeric: tabular-nums; }
.ic-stages td { padding: 1px 12px 1px 0; border: 0; }
.ic-banner { border: 1px solid #d94a1f; padding: 10px 14px; margin: 6px 0; }
footer { display: none !important; }
"""
# Same fix as the official Idol demo (Apache-2.0): the splat viewer fetches the .ply with credentials omitted, which a
# private Space answers with 404; send the session cookies with file requests.
JS = """() => {
  const f = window.fetch.bind(window);
  window.fetch = (r, o) => { const u = typeof r === 'string' ? r : (r && r.url) || '';
    if (u.includes('/gradio_api/file=')) o = Object.assign({}, o, { credentials: 'same-origin' });
    return f(r, o); };
}"""

HEAD = f"""
<h1 style="margin-bottom:4px">Idol on CUDA: official vs fast kernels</h1>
<p style="margin-top:0">One image in, a 3D Gaussian splat out. The same request runs through the official
<a href="https://github.com/speridlabs/idol-sdk">idol-sdk</a> pipeline and through
the fast build of <a href="{GITHUB}">idol-cuda</a> (restructured sampler + fused CUDA kernels), each timed on this GPU.
Unofficial port of <a href="https://huggingface.co/speridlabs/idol">Speri Labs' Idol</a>. ·
<a href="{GITHUB}">GitHub</a> · <a href="{KERNEL_HUB}">Kernels</a></p>
"""

with gr.Blocks(title='idol-cuda demo') as demo:
    gr.HTML(HEAD)
    if STATE['error']:
        gr.HTML(f'<div class="ic-banner">{STATE["error"]}</div>')
    with gr.Row():
        with gr.Column(scale=4):
            image = gr.Image(value=str(ROOT / 'examples' / 'fig-robot-cute.webp'), type='pil', image_mode='RGBA',
                             label='Input image', sources=['upload', 'clipboard'], height=320)
            mode = gr.Radio(list(MODES), value='Synthetic (render / generated)', label='Input type')
            with gr.Row():
                btn_fast = gr.Button('Generate (fast build)', variant='primary')
                btn_cmp = gr.Button('Compare with official', variant='secondary')
            with gr.Accordion('Settings', open=False):
                gaussians = gr.Dropdown(list(GAUSSIANS), value='262,144', label='Gaussians')
                steps = gr.Slider(10, 50, value=50, step=1, label='Steps')
                seed = gr.Number(value=42, precision=0, label='Seed (both builds use the same seed)')
                randomize = gr.Checkbox(value=False, label='Random seed')
            gr.Examples([[str(ROOT / 'examples' / f), m] for f, m in EXAMPLES], [image, mode],
                        label='Examples (from idol-sdk, Apache-2.0)', examples_per_page=8)
        with gr.Column(scale=8):
            summ = gr.HTML(summary({}))
            with gr.Row():
                with gr.Column():
                    card_f = gr.HTML(card('fast', {}))
                    view_f = gr.Model3D(label='Fast build', height=420, clear_color=(0.05, 0.05, 0.05, 1.0),
                                        value=str(PRE / 'robot_fast.ply') if (PRE / 'robot_fast.ply').exists() else None)
                    file_f = gr.File(label='Download .ply (fast)')
                with gr.Column():
                    card_o = gr.HTML(card('official', {}))
                    view_o = gr.Model3D(label='Official pipeline', height=420, clear_color=(0.05, 0.05, 0.05, 1.0),
                                        value=str(PRE / 'robot_official_eager.ply')
                                        if (PRE / 'robot_official_eager.ply').exists() else None)
                    file_o = gr.File(label='Download .ply (official)')
            gr.Markdown('The viewers start with the robot example generated beforehand on an H200 (seed 42, 262,144 '
                        'Gaussians). Same seed means the same starting noise; the fast build draws its octree '
                        'samples differently, so the two splats are close but not identical.')
    with gr.Accordion('Diagnostics: kernel self-check on this GPU', open=False):
        gr.Markdown(f'Runs the kernel package\'s 44 tests on the loaded build. Kernels: '
                    f'`{STATE.get("kernels")}` `{STATE.get("kernel_variant")}`.')
        diag_btn = gr.Button('Run self-check')
        diag = gr.Textbox(label='Result', lines=8)
    outs = [view_f, view_o, card_f, card_o, summ, file_f, file_o, seed]
    ins = [image, mode, seed, steps, gaussians, randomize]
    btn_fast.click(lambda *a: (yield from generate(*a, compare=False)), ins, outs, concurrency_limit=1,
                   concurrency_id='gpu', api_name='generate_fast', show_progress='minimal')
    btn_cmp.click(lambda *a: (yield from generate(*a, compare=True)), ins, outs, concurrency_limit=1,
                  concurrency_id='gpu', api_name='compare', show_progress='minimal')
    diag_btn.click(run_diag, None, diag, concurrency_limit=1, concurrency_id='gpu', api_name='selfcheck')

if __name__ == '__main__':
    demo.queue(max_size=20, default_concurrency_limit=1).launch(css=CSS, js=JS)
