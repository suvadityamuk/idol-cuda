# H200 job: stage 1 (reference dumps, run-to-run noise, compiled-vs-eager spread), TorchScript export, native parity
# check against the fresh dumps, profile of the shipped pipeline. Everything is written under $RUN and /scratch.
source /work/idol-cuda/jobs/env.sh
set +e
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$RUN/gpu.txt"
REF=/scratch/ref
mkdir -p $REF /scratch/ts
R="python tools/ref_run.py"
EX=$SDK/demo/examples
cd $SDK   # the reference resolves nothing relative, but keep demo paths short
T=/work/idol-cuda/tools

# --- TorchScript traces for the native binary (one-off preparation)
python $T/export_encoders.py --out /scratch/ts 2>&1 | tail -5

cfgs=("robot synthetic fig-robot-cute 262144 262k" "avocado real food-avocado 262144 262k"
      "robot synthetic fig-robot-cute 1048576 1m" "avocado real food-avocado 1048576 1m")

# --- eager reference (parity target): two fresh processes per config; A also records every denoiser call
for c in "${cfgs[@]}"; do
  set -- $c; tag=$1_$5
  python $T/ref_run.py --image $EX/$3.webp --mode $2 --num_gaussians $4 --attn sdpa --no_optimize --dump_steps \
     --dump $REF/${tag}_eager_sdpa_a.safetensors 2>&1 | grep -v Warning | tail -3
  python $T/ref_run.py --image $EX/$3.webp --mode $2 --num_gaussians $4 --attn sdpa --no_optimize \
     --dump $REF/${tag}_eager_sdpa_b.safetensors 2>&1 | tail -2
  python $T/compare.py $REF/${tag}_eager_sdpa_a.safetensors $REF/${tag}_eager_sdpa_b.safetensors --quiet \
     > $RUN/noise_eager_${tag}.txt || true
  echo "noise eager $tag: $(tail -1 $RUN/noise_eager_${tag}.txt)"; grep -m5 DIFF $RUN/noise_eager_${tag}.txt | cut -c1-300 || true
  cmp $REF/${tag}_eager_sdpa_a.ply $REF/${tag}_eager_sdpa_b.ply && echo "ply A==B $tag" || echo "ply A!=B $tag"
done

# --- deterministic cuDNN pin (robot 262k, twice)
for x in a b; do
  python $T/ref_run.py --image $EX/fig-robot-cute.webp --mode synthetic --num_gaussians 262144 --attn sdpa \
    --no_optimize --deterministic --dump $REF/robot_262k_eager_sdpa_det_$x.safetensors 2>&1 | tail -1
done
python $T/compare.py $REF/robot_262k_eager_sdpa_det_a.safetensors $REF/robot_262k_eager_sdpa_det_b.safetensors \
  --quiet > $RUN/noise_det.txt || true; echo "det a vs b: $(tail -1 $RUN/noise_det.txt)"
python $T/compare.py $REF/robot_262k_eager_sdpa_a.safetensors $REF/robot_262k_eager_sdpa_det_a.safetensors \
  --quiet > $RUN/det_vs_default.txt || true; echo "det vs default: $(tail -1 $RUN/det_vs_default.txt)"
grep -m5 DIFF $RUN/det_vs_default.txt | cut -c1-300 || true

# --- compiled reference as shipped (optimize=True), fresh processes; inductor cache kept for analysis
export TORCHINDUCTOR_CACHE_DIR=/work/inductor_cache
for c in "${cfgs[@]}"; do
  set -- $c; tag=$1_$5
  for x in a b; do
    [ "$5" = 1m ] && [ $x = b ] && continue
    TORCH_LOGS=graph_breaks,recompiles python $T/ref_run.py --image $EX/$3.webp --mode $2 --num_gaussians $4 \
      --attn sdpa --dump $REF/${tag}_compiled_$x.safetensors > $RUN/compiled_${tag}_$x.log 2>&1
    tail -2 $RUN/compiled_${tag}_$x.log
  done
  python $T/compare.py $REF/${tag}_eager_sdpa_a.safetensors $REF/${tag}_compiled_a.safetensors --quiet --allow-diff \
    > $RUN/spread_compiled_${tag}.txt; echo "compiled vs eager $tag: $(tail -1 $RUN/spread_compiled_${tag}.txt)"
done
for tag in robot_262k avocado_262k; do
  python $T/compare.py $REF/${tag}_compiled_a.safetensors $REF/${tag}_compiled_b.safetensors --quiet --allow-diff \
    > $RUN/noise_compiled_${tag}.txt; echo "compiled a vs b $tag: $(tail -1 $RUN/noise_compiled_${tag}.txt)"
done
grep -h -i -E "graph break|recompil" $RUN/compiled_*.log | sort | uniq -c | head -20 || true
tar czf $RUN/inductor_cache.tgz -C /work inductor_cache || true

# --- native parity check against the fresh eager dumps
if [ -x /scratch/bin/idol_native ]; then
  cd /work/idol-cuda
  IDOL_SDK=$SDK bash scripts/parity_test.sh $REF /work/parity_out /scratch/bin/idol_native 2>&1 | tee $RUN/parity_test.txt || true
  cp /work/parity_out/*.log /work/parity_out/*.compare.txt $RUN/ 2>/dev/null || true
  cd $SDK
fi

# --- profiles (steady state, after one warm-up request), compiled and eager
python $T/profile_ref.py --out $RUN/trace_compiled.json.gz > $RUN/profile_compiled.txt 2>&1 || true
head -50 $RUN/profile_compiled.txt
python $T/profile_ref.py --no_optimize --out $RUN/trace_eager.json.gz > $RUN/profile_eager.txt 2>&1 || true
head -30 $RUN/profile_eager.txt
echo STAGE1_DONE
