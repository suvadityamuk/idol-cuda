# HF Jobs log (running tally)

Prices from `hf jobs hardware` (2026-11-24): cpu-basic $0.01/h, cpu-upgrade $0.03/h, cpu-xl $1.00/h,
cpu-performance $1.90/h, l4x1 $0.80/h, a10g-small $1.00/h, l40sx1 $1.80/h, h200 $5.00/h.
Minutes are wall time from `hf jobs inspect` (rounded up); cost = minutes x price.

| # | job id | flavor | purpose | minutes | cost ($) | cumulative ($) |
|---|---|---|---|---|---|---|
| 1 | 6b05f4bf1460ffe1d5e90b54 | cpu-basic | probe: image, mounts, ssh, bucket | 5 | 0.00 | 0.00 |
| 2 | 6b05f5b01460ffe1d5e90d29 | h200 | previous agent: interactive `sleep infinity` session (stage 1/2 work); nothing it produced was persisted (bucket empty, no logs) | ~40 (estimate) | ~3.33 (estimate) | 3.33 |
| 3 | 6b0656451460ffe1d5e9ab91 | cpu-basic | ssh probe | 2 | 0.00 | 3.33 |
| 4 | 6b065746d0582cfbbefd8d03 | cpu-upgrade | build_native: failed (no git in image) | 1 | 0.00 | 3.33 |
| 5 | 6b0657bf1460ffe1d5e9ad29 | cpu-upgrade | build_native: failed (pip PEP 668) | 4 | 0.00 | 3.33 |
| 6 | 6b06587cd0582cfbbefd8e1d | cpu-upgrade | build_native: failed (TORCH_CUDA_ARCH_LIST default) | 5 | 0.00 | 3.33 |
| 7 | 6b0659161460ffe1d5e9ae65 | cpu-upgrade | build_native: script bug (grep exit status) | 3 | 0.00 | 3.33 |
| 8 | 6b065a3ad0582cfbbefd9014 | cpu-upgrade | build_native: OK (binary in bucket bin/) | 9 | 0.01 | 3.34 |
| 9 | 6b0658841460ffe1d5e9adc1 | h200 | stage1: eager/compiled dumps x2, det pin, profiles (parity check skipped: bucket mount drops the exec bit) | 23 | 1.92 | 5.26 |
| 10 | 6b065e281460ffe1d5e9b5d5 | h200 | dev1: native PARITY PASS (4 configs); kernel dev build failed to compile (A/B not run) | 8 | 0.67 | 5.93 |
| 11 | 6b066e4ed0582cfbbefdaae7 | cpu-upgrade | compile_check: kernel dev build compiles | 7 | 0.00 | 5.93 |
| 12 | 6b066ebfd0582cfbbefdab88 | h200 | dev2: native rebuild (reuse), PARITY PASS x2; kernel tests (13 fails: test bugs); fast A/B | 12 | 1.00 | 6.93 |
| 13 | 6b0674611460ffe1d5e9df26 | h200 | dev3: kernel tests (5 fails), fast profile, attention bench, fast pipeline e2e + closeness (4 configs), Python parity PLY identical | 9 | 0.75 | 7.68 |
| 14 | 6b0676fad0582cfbbefdbcf6 | cpu-performance | kbuild: kernel-builder build-and-copy (17 variants), check-builds OK; check-config needs the pinned CLI (image has 0.18) | 7 | 0.22 | 7.90 |
| 15 | 6b0677101460ffe1d5e9e2da | h200 | dev4: kernel tests (4 fails), attention bench, native fast e2e (4.74 s), native parity steady 9.24 s | 10 | 0.83 | 8.73 |
| 16 | 6b06791f1460ffe1d5e9e693 | h200 | dev5: test diagnostics, first test of kernel-builder output (40/44, tolerance fails), FA3 offline-blocked | 5 | 0.42 | 9.15 |
| 17 | 6b067a69d0582cfbbefdc240 | h200 | dev6: 44/44 tests on kernel-builder output (torch210-cu130) and dev build; FA3 / FA3-fp8 A/B | 6 | 0.50 | 9.65 |
| 18 | 6b067be2d0582cfbbefdc558 | h200 | fa3probe: cache kernels-community/flash-attn3 v1 in the bucket, op schema | 2 | 0.17 | 9.82 |
| 19 | 6b067c961460ffe1d5e9ed5f | h200 | dev7: FA3 failed to load from the bucket cache (all fast rows ran cuDNN); Python + native fast e2e closeness, 4 configs | 15 | 1.25 | 11.07 |
| 20 | 6b06820cd0582cfbbefdd312 | cpu-performance | kbuild rerun: cancelled (pinned-rev lookup bug) | 4 | 0.13 | 11.20 |
| 21 | 6b06823c1460ffe1d5e9faac | cpu-performance | kbuild: 17 variants; check-config + check-builds with the pinned kernel-builder 0.19.0-dev1: exit 0 (image CLI 0.18 rejects edition 6) | 7 | 0.22 | 11.42 |
| 22 | 6b06824ed0582cfbbefdd36d | h200 | dev8: FA3 A/B (Python 4.25 s/request); native FA3 dlopen failed (libpython symbols) | 10 | 0.83 | 12.25 |
| 23 | 6b0687b0d0582cfbbefddd09 | h200 | bench: 44/44 tests on builder output; interleaved benchmark (6 builds x 4 configs x 5 rounds); reference spread runs (eager flash, compiled B 1M) | 33 | 2.75 | 15.00 |
| 24 | 6b068facd0582cfbbefdec4c | h200 | scripts_check: stage-6 fresh-shell runs (native fast failed on FA3 load; others OK), bench-output closeness, gsplat renders | 10 | 0.83 | 15.83 |
| 25 | 6b0692481460ffe1d5ea1d8f | h200 | native_check: run_native.sh (fast) in a fresh shell from a clean build: OK; --fa3 falls back to cuDNN | 3 | 0.25 | 16.08 |
| 26 | 6b0693dc1460ffe1d5ea2062 | h200 | final: PARITY PASS on HEAD bd5c29c (4 configs); dev-build vs kernel-builder kernels: PLY byte-identical, same speed (40 min in SCHEDULING, not billed as runtime) | 10 | 0.83 | 16.91 |
| 27 | 6b071208d91f621b93087ae1 | cpu-performance | release kbuild (version 1, archs 8.0-12.0): failed, transient cachix download errors | 15 | 0.48 | 17.39 |
| 28 | 6b0715b7d91f621b93088737 | cpu-performance | release kbuild with retries: attempt 1 failed, attempt 2 OK; 17 variants, check-builds (image and pinned CLI) exit 0 | 17 | 0.54 | 17.93 |
| 29 | 6b0719f2d91f621b930895e9 | h200 | release_check from the release export: pip install -r requirements.txt, infer.py error paths, fast robot/avocado, parity PLY + camera byte-identical, reference eager byte-identical, reference compiled; 44/44 tests on the new torch210-cu130 build; example PLYs + turntable frames (app test step failed: assemble path) | 11 | 0.92 | 18.85 |
| 30 | 6b071c9ab6a90bbe997e23f2 | h200 | app_check: Space app logic, APP TEST PASSED (official run byte-identical to eager reference); timings showed per-thread handle set-up | 5 | 0.42 | 19.27 |
| 31 | 6b071de3d91f621b93089e1a | h200 | app_check rerun with a persistent GPU thread: APP TEST PASSED; fast 4.18 s, official eager 9.74 s inside the app (H200), official byte-identical to eager reference | 4 | 0.33 | 19.60 |
| 32 | 6b072afad91f621b9308b27b | h200 | hub_check from the public GitHub clone with the published Hub kernels: fast robot 4.15 s with FlashAttention-3, Chamfer 1.69e-3 vs eager; parity PLY byte-identical; kernel tests not run (pytest missing in the image) | ~6 | ~0.50 | ~20.1 |
