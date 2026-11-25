# H200 job: parity regression on HEAD's native binary; short interleaved bench: shipped default vs Python fast with
# the dev build vs Python fast with the kernel-builder output; byte-compare their PLYs.
source /work/idol-cuda/jobs/env.sh
set +e
cd /work/idol-cuda
(apt-get update -qq && apt-get install -y -qq zlib1g-dev >/dev/null) || true
pip install -q ninja cmake
export TORCH_CUDA_ARCH_LIST="9.0a"
cmake -S native -B /work/build -G Ninja -DTORCH_SITE="$TORCH_SITE" -DPILLOW_LIBS="$PILLOW_LIBS" > $RUN/cmake.txt 2>&1
cmake --build /work/build -j20 2>&1 | grep -E "error|FAILED" | head
echo "source commit $(cat .source_commit)"
IDOL_SDK=$SDK bash scripts/parity_test.sh /scratch/ref /work/parity_out /work/build/idol_native 2>&1 | tee $RUN/parity_test.txt
echo "parity_test exit ${PIPESTATUS[0]}"
rm -rf /work/kb && cp -r /scratch/kbuild /work/kb
export KB_DIR=/work/kb
python tools/bench.py --native /work/build/idol_native --sdk $SDK --reps 3 --rows ref_compiled,py_fast,py_fast_kb \
   --configs robot_262k --outdir /work/fb --out $RUN/bench_kb.jsonl 2>&1 | grep -E '"kind"|kernels:|rror' | cut -c1-200
for r in r0 r1 r2; do
  cmp /work/fb/py_fast/robot_262k_py_fast_$r.ply /work/fb/py_fast_kb/robot_262k_py_fast_kb_$r.ply && echo "$r: dev-build and kernel-builder PLY byte-identical"
done
echo FINAL_DONE
