#!/bin/bash
# Sequential GPU queue. Ordered so the decisive MNIST scaling curve lands first.
# Sweeps use n=5000: FID is biased at that n, but the bias is constant across
# conditions at fixed n, which is all the monotonicity question needs.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 500 1000"
N=5000

while [ ! -f runs/mnist_g0316_t1000/ckpt_040000.pt ]; do sleep 30; done
sleep 10

# 1. THE question: does FID fall monotonically in sampling steps once the
#    sampler is correct? Step-major, so the shape is readable early.
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
    --n_samples $N --batch_size 1000 --compile --save_grid \
    --steps $GRID --samplers bridge euler euler_det > runs/eval_mnist_t1000.log 2>&1

# 2. Ablation: the original's degenerate time conditioning, t in [0, 0.1).
$PY -m bridges.train --dataset mnist --steps 40000 --batch_size 128 --lr 2e-4 \
    --t_scale 0.1 --compile --out runs/mnist_g0316_t01 > runs/mnist_g0316_t01.log 2>&1
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t01/ckpt_040000.pt \
    --n_samples $N --batch_size 1000 --compile --save_grid \
    --steps $GRID --samplers bridge euler euler_det > runs/eval_mnist_t01.log 2>&1

# 3. CIFAR-10 bridge.
$PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_g0316_t1000 > runs/cifar10_g0316_t1000.log 2>&1
$PY -m bridges.evaluate --ckpt runs/cifar10_g0316_t1000/ckpt_040000.pt \
    --n_samples $N --batch_size 500 --compile --save_grid \
    --steps 1 2 5 10 20 50 100 200 1000 --samplers bridge euler euler_det \
    > runs/eval_cifar10_t1000.log 2>&1

echo "QUEUE COMPLETE"
