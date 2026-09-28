#!/bin/bash
# CIFAR-10 sweeps at n = 50,000, the published protocol's sample count.
#
# Row set, trimmed from ten to six. One bridge sampler (Euler-Maruyama, the best
# of the three at every MNIST step count measured), flow matching, and ancestral
# sampling for the diffusion arms. DDIM is dropped for the two plain DDPM arms:
# its collapse past 100 steps is documented on MNIST and repeating it here adds
# nothing. It is KEPT for improved DDPM, because that is the baseline the
# comparison rests on and its best MNIST configuration was DDIM at 100 steps
# (1.24) rather than ancestral (2.02) -- reporting only the weaker sampler would
# understate the one row that matters.
#
# Cost at the estimated CIFAR inference rate: about 8 h per sampler for the full
# grid, so roughly 48 h for six rows. Ordered so the three rows carrying the
# headline comparison finish first.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another queue holds the lock"; exit 1; fi
if pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; then
  echo "a worker is still alive; refusing to start"; exit 1; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
N=50000
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

sweep () {  # <model dir> <samplers...>
  local M=$1; shift
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N --batch_size 500 --compile --save_grid \
      --out runs/$M/fid_n50k.json --steps $GRID --samplers "$@" \
      >> runs/eval_cifar_n50k.log 2>&1
  cool
}

# headline three first
sweep cifar10_g0316_t1000 euler
sweep cifar10_flow        flow_euler
sweep cifar10_improved    improved_ddim
$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples $N \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1
echo "=== CIFAR HEADLINE THREE COMPLETE ==="

sweep cifar10_improved    improved_ancestral
sweep cifar10_ddpm_cosine ddpm_ancestral
sweep cifar10_ddpm_linear ddpm_ancestral

$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples $N \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1
echo "=== CIFAR SWEEPS COMPLETE ==="
