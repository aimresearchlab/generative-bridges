#!/bin/bash
# CIFAR-10 at n = 50,000, the submitted protocol's sample count.
#
# ROW ORDER IS DELIBERATE, so that stopping early still leaves a defensible
# table rather than a one-sided one:
#
#   1. Improved denoising, ANCESTRAL. The submitted manuscript's main baseline,
#      run with its stronger sampler. On CIFAR at n=2000 ancestral beats DDIM at
#      both step counts measured (18.97 against 25.69 at 50, 38.26 against 47.06
#      at 10) -- the reverse of MNIST, where DDIM won and an earlier version of
#      this script was ordered accordingly. Running DDIM first would mean that an
#      early stop represented the baseline by its weaker sampler, so the ordering
#      is dataset specific.
#   2. Flow matching at 1000 steps, the entry that decides whether the small
#      terminal rise in the bridge row (8.88 at 200 to 9.23 at 1000) is shared
#      with the modern baseline or specific to us.
#   3. Improved denoising + DDIM, and simple denoising -- completing the row set
#      of the submitted table. Neither carries a claim.
#
# The bridge row is already complete and is not re-run; bridges.evaluate resumes
# from whatever is in the output JSON, so nothing finished is repeated.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another queue holds the lock"; exit 1; fi
# Anchor the pattern to the start of the command line. Unanchored, it matches
# any shell whose own arguments quote it -- including the one that wrote this
# file -- and the queue refuses to start against a phantom worker.
if pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; then
  echo "a worker is still alive; refusing to start"; exit 1; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
N=50000
BRIDGE=cifar10_drift_g0.3162277660168379
COOL_MIN=${COOL_MIN:-8}; COOL_ABOVE=${COOL_ABOVE:-70}; COOL_EVERY_H=${COOL_EVERY_H:-3}
LAST_COOL=$(date +%s)
cool () {
  local t0 now h; now=$(date +%s); h=$(( (now-LAST_COOL)/3600 ))
  t0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  if [ "$h" -lt "$COOL_EVERY_H" ] || [ "$t0" -lt "$COOL_ABOVE" ]; then
    echo "[cooldown] GPU ${t0}C, ${h}h -> running on"; return 0; fi
  echo "[cooldown] GPU ${t0}C after ${h}h -> pausing ${COOL_MIN} min"
  sleep $((COOL_MIN*60)); LAST_COOL=$(date +%s)
}

table () {
  $PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples $N \
      --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" --bridge_dir $BRIDGE \
      --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
      > runs/tables_cifar10.log 2>&1
}

sweep () {  # <model dir> <samplers...>
  local M=$1; shift
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N --batch_size 500 --compile --save_grid \
      --out runs/$M/fid_n50k.json --steps $GRID --samplers "$@" \
      >> runs/eval_cifar_n50k.log 2>&1
  table
  cool
}

sweep cifar10_improved improved_ancestral
echo "=== STOP POINT A: table complete but for flow at 1000 ==="

sweep cifar10_flow flow_euler
echo "=== STOP POINT B: headline three complete, degradation question settled ==="

sweep cifar10_improved    improved_ddim
sweep cifar10_ddpm_linear ddpm_ancestral
echo "=== CIFAR SWEEPS COMPLETE ==="
