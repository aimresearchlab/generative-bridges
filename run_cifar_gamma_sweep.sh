#!/bin/bash
# CIFAR gamma sweep, as a fairness requirement rather than an extra.
#
# Every baseline runs published, author-tuned settings: Ho et al.'s linear
# schedule, Nichol & Dhariwal's cosine schedule with a learned variance, flow
# matching's linear path. The bridge was running gamma = 0.3162, carried from
# the submitted manuscript where it was tuned on MNIST -- and the MNIST sweep
# then showed it was not optimal even there, smaller values being better at
# every step count measured, monotonically, with no turnover down to the
# smallest tested. Comparing a method at an inherited hyperparameter against
# baselines at their tuned ones is not a matched comparison.
#
# gamma = 0.1 is already training under the probe. This adds 0.05 and 0.2, so
# the grid brackets the MNIST optimum from below and can find a turnover if one
# exists. gamma = 0.3162 is already trained.
set -x
cd "$(dirname "$0")"
while pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; do sleep 60; done
LOCK=runs/.gpu.lock
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT INT TERM
PY=.venv/bin/python

for G in 0.05 0.2; do
  if [ ! -f runs/cifar10_g$G/ckpt_040000.pt ]; then
    $PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 \
        --lr 2e-4 --gamma $G --compile --out runs/cifar10_g$G \
        > runs/cifar10_g$G.log 2>&1
  fi
  $PY -m bridges.evaluate --ckpt runs/cifar10_g$G/ckpt_040000.pt \
      --n_samples 2000 --batch_size 500 --compile \
      --out runs/cifar10_g$G/fid_sanity.json --steps 10 50 \
      --samplers bridge euler >> runs/cifar_sanity.log 2>&1
done
echo "=== CIFAR GAMMA SWEEP COMPLETE ==="
