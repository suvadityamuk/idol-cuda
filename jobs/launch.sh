#!/usr/bin/env bash
# Mac side: upload the committed tree to the bucket and start a job that runs jobs/<script>.sh.
#   jobs/launch.sh <script> <flavor> <timeout> [extra hf jobs run args...]
# The HF token is passed only as a secret (from the environment), never on the command line.
set -euo pipefail
cd "$(dirname "$0")/.."
S=$1; FLAVOR=$2; TO=$3; shift 3
B=${IDOL_BUCKET:?set IDOL_BUCKET=<user>/<bucket>}
TMP=$(mktemp -d)
git archive --format=tar.gz -o "$TMP/idol-cuda.tgz" HEAD
hf buckets cp "$TMP/idol-cuda.tgz" "hf://buckets/$B/code/idol-cuda.tgz" >/dev/null
rm -rf "$TMP"
hf jobs run -d --ssh --flavor "$FLAVOR" --timeout "$TO" --secrets HF_TOKEN --label owner=idol-cuda --label "purpose=$S" \
  -e JOB_NAME="$S" \
  -v "hf://buckets/$B:/scratch" -v hf://speridlabs/idol:/w/idol \
  -v hf://facebook/dinov3-vith16plus-pretrain-lvd1689m:/w/dinov3 -v hf://black-forest-labs/FLUX.2-dev:/w/flux2 \
  "$@" ${IMAGE:-pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel} \
  bash -c "set +x; mkdir -p /work/idol-cuda && tar xzf /scratch/code/idol-cuda.tgz -C /work/idol-cuda && bash /work/idol-cuda/jobs/$S.sh"
