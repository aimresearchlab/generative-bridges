#!/bin/bash
# Finish CIFAR training, then a sanity check. Does NOT launch the full sweeps.
#
# The full grid at n=50,000 is estimated at 81 h across ten sampler rows. That
# estimate comes from scaling the measured MNIST inference rate by the ratio of
# training rates, not from measuring CIFAR inference, and a three-day run should
# not start on an extrapolation. It also should not start before the models are
# known to have trained sensibly: a broken checkpoint would cost the same three
# days and produce nothing.
#
# This script therefore stops after the check and prints what it found.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
while pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; do sleep 60; done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM
PY=.venv/bin/python

# finish the last model if the interrupted queue did not reach it
if [ ! -f runs/cifar10_improved/ckpt_040000.pt ]; then
  $PY -m bridges.improved_ddpm --dataset cifar10 --schedule cosine --steps 40000 \
      --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_improved \
      > runs/cifar10_improved.log 2>&1
fi

echo "=== TRAINING COMPLETE, RUNNING SANITY CHECK ==="

# 1. cheap sweep on every model: is FID plausible, and what is the real rate?
for M in cifar10_g0316_t1000 cifar10_flow cifar10_ddpm_cosine \
         cifar10_ddpm_linear cifar10_improved; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples 2000 --batch_size 500 --compile --save_grid \
      --out runs/$M/fid_sanity.json --steps 10 50 \
      >> runs/cifar_sanity.log 2>&1
done

# 2. the achievable floor at both candidate sample counts
$PY -m bridges.calibrate_fid --dataset cifar10 --counts 2000 10000 25000 \
    >> runs/cifar_sanity.log 2>&1

echo "=== SANITY CHECK COMPLETE ==="
