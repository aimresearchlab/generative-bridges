#!/bin/bash
# Sequential GPU queue, reordered after the MNIST result.
#
# The monotonicity question is settled through 200 NFE, and both correct
# samplers are within ~1.5x of the n=5000 measurement floor by 50 NFE, so the
# 500-NFE tail buys nothing. Flow matching (the modern matched control) and
# error bars therefore jump ahead of it. Sweeps resume from their JSON, so any
# skipped point backfills later without recomputation.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
N=5000

# 1. Finish the MNIST grid at 1000 NFE (500 skipped; 1000 is what the paper
#    reports, so it is the point the rebuttal needs).
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
    --n_samples $N --batch_size 1000 --compile --save_grid \
    --steps $GRID --samplers bridge euler euler_det >> runs/eval_mnist_t1000.log 2>&1

# 2. Flow matching at matched budget: same UNet, optimiser, EMA, batch size and
#    40k steps, measured through the identical FID path.
$PY -m bridges.flowmatch --dataset mnist --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/mnist_flow > runs/mnist_flow.log 2>&1
$PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
    --n_samples $N --batch_size 1000 --compile --save_grid \
    --steps $GRID > runs/eval_mnist_flow.log 2>&1

# 3. Seed variance on the headline configs, for mean +/- sd (reviewer point 5).
for S in 1 2; do
  $PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
      --n_samples $N --batch_size 1000 --compile --seed $S \
      --out runs/mnist_g0316_t1000/fid_seed$S.json \
      --steps 10 50 200 --samplers bridge euler euler_det >> runs/eval_seeds.log 2>&1
done

# 4. Degenerate time conditioning, t in [0, 0.1) as the original fed it.
#    Prediction: this curve turns over and DEGRADES, where the fixed one only
#    flattens. That is what the published non-monotonicity requires.
$PY -m bridges.train --dataset mnist --steps 40000 --batch_size 128 --lr 2e-4 \
    --t_scale 0.1 --compile --out runs/mnist_g0316_t01 > runs/mnist_g0316_t01.log 2>&1
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t01/ckpt_040000.pt \
    --n_samples $N --batch_size 1000 --compile --save_grid \
    --steps $GRID --samplers bridge euler euler_det > runs/eval_mnist_t01.log 2>&1

# 5. CIFAR-10.
$PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_g0316_t1000 > runs/cifar10_g0316_t1000.log 2>&1
$PY -m bridges.evaluate --ckpt runs/cifar10_g0316_t1000/ckpt_040000.pt \
    --n_samples $N --batch_size 500 --compile --save_grid \
    --steps 1 2 5 10 20 50 100 200 --samplers bridge euler euler_det \
    > runs/eval_cifar10_t1000.log 2>&1

echo "QUEUE COMPLETE"
