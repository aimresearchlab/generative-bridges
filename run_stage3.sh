#!/bin/bash
# Stage 3: the improved-DDPM baseline that was missing, then CIFAR-10.
#
# ddpm_baseline.py gives DDPM with a cosine schedule but a FIXED reverse
# variance. Improved DDPM's defining feature is a LEARNED reverse variance
# trained with a hybrid objective, and that is what buys quality at few steps --
# the paper's headline claim. Comparing against the fixed-variance version would
# repeat the weak-baseline objection we make of the original manuscript, whose
# own README runs improved-diffusion with --learn_sigma True.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do sleep 60; done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"

$PY -m bridges.improved_ddpm --dataset mnist --schedule cosine --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/mnist_improved \
    > runs/mnist_improved.log 2>&1

$PY -m bridges.evaluate --ckpt runs/mnist_improved/ckpt_040000.pt \
    --n_samples 50000 --batch_size 1000 --compile \
    --out runs/mnist_improved/fid_n50k.json --steps $GRID \
    >> runs/eval_n50k.log 2>&1

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md \
    > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
echo "=== IMPROVED DDPM COMPLETE ==="

$PY -m bridges.improved_ddpm --dataset cifar10 --schedule cosine --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_improved \
    > runs/cifar10_improved.log 2>&1
$PY -m bridges.evaluate --ckpt runs/cifar10_improved/ckpt_040000.pt \
    --n_samples 10000 --batch_size 500 --compile \
    --out runs/cifar10_improved/fid_n10k.json --steps $GRID \
    >> runs/eval_cifar_n10k.log 2>&1
echo "STAGE 3 COMPLETE"
