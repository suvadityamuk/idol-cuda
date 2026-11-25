# CPU job (image ghcr.io/huggingface/kernel-builder-job:latest): real kernel-builder build of kernel/.
set +x
set -o pipefail
RUN=/scratch/runs/$(date -u +%Y%m%d-%H%M%S)-kbuild; mkdir -p $RUN
exec > >(tee -a "$RUN/job.log") 2>&1
export GIT_LFS_SKIP_SMUDGE=1
export NIX_CONFIG="download-attempts = 10"  # cachix downloads failed transiently once
rm -rf /src && mkdir -p /src && cp -r /work/idol-cuda/kernel /src/kernel && cd /src/kernel
rm -rf build dev
git init -q && git add -A && git -c user.name=idol -c user.email=idol@localhost commit -qm "idol-cuda kernel $(cat /work/idol-cuda/.source_commit 2>/dev/null)"
git rev-parse HEAD | tee $RUN/kernel-tree-commit.txt
kernel-builder --version
kernel-builder check-config . 2>&1 | tee $RUN/check-config.log; echo "check-config (image CLI) exit ${PIPESTATUS[0]}"
# the image's CLI may predate edition 6: also run the kernel-builder pinned by flake.lock
KB_REV=034904a6dad37a16d3c973d906adfa7760d6a6db  # nodes.kernel-builder.locked.rev in flake.lock
echo "pinned kernel-builder rev $KB_REV"
KB="nix run github:huggingface/kernels/$KB_REV#kernel-builder --"
$KB --version
$KB check-config . 2>&1 | tee $RUN/check-config-pinned.log; echo "check-config (pinned) exit ${PIPESTATUS[0]}"
kernel-builder list-variants . 2>&1 | tee $RUN/variants.txt
for attempt in 1 2 3; do  # retried: a transient substituter failure aborts the whole build
  kernel-builder build-and-copy --max-jobs 4 --cores 8 -L . > $RUN/build.log 2>&1; rc=$?; echo "build-and-copy attempt $attempt exit $rc"
  [ $rc -eq 0 ] && break
done
tail -30 $RUN/build.log
kernel-builder check-builds . 2>&1 | tee $RUN/check-builds.log; echo "check-builds (image CLI) exit ${PIPESTATUS[0]}"
$KB check-builds . 2>&1 | tee $RUN/check-builds-pinned.log; echo "check-builds (pinned) exit ${PIPESTATUS[0]}"
ls build; du -sh build/* | tee $RUN/sizes.txt
if [ -d build ]; then
  rm -rf /scratch/kbuild && mkdir -p /scratch/kbuild && cp -rL build /scratch/kbuild/build && cp build.toml /scratch/kbuild/
  (cd build && for v in torch*; do python3 -c "import json,sys;m=json.load(open('$v/metadata.json'));print('$v', m['version'], m['backend'].get('archs'))" 2>/dev/null || grep -o '"archs"[^]]*]' $v/metadata.json | tr -d '\n' ; echo; done) | tee $RUN/variant-archs.txt
fi
echo KBUILD_DONE
