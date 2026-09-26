#!/bin/bash
# Stage 2: close the 1000-NFE gap, then CIFAR-10.
#
# The published grid runs to 1000 sampling steps. Stopping the n=50,000 sweep at
# 200 would leave a referee free to assume the step-count pathology is still
# waiting past the edge of the table, so the 1000 column is filled in first.
#
# A lockfile guards the whole run. The previous crash was caused by a diagnostic
# launched alongside the queue: this box has unified memory, so two concurrent
# jobs each holding an Inception model (which resizes batches to 299x299) plus a
# UNet exhausted all 119 GB and took the machine down rather than raising a CUDA
# OOM. Nothing may run beside the queue.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another GPU job is live (pid $(cat "$LOCK")); refusing to start"
  exit 1
fi
mkdir -p runs && echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
N50=50000

# ---- 1000-NFE column at n=50,000 -------------------------------------------
# ~5.1 h per sampler at ~2700 image-evaluations/s.
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
    --n_samples $N50 --batch_size 1000 --compile \
    --out runs/mnist_g0316_t1000/fid_n50k.json --steps 1000 \
    --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1

$PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
    --n_samples $N50 --batch_size 1000 --compile \
    --out runs/mnist_flow/fid_n50k.json --steps 1000 >> runs/eval_n50k.log 2>&1

for M in mnist_ddpm_cosine mnist_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile \
      --out runs/$M/fid_n50k.json --steps 1000 \
      --samplers ddim ddpm_ancestral >> runs/eval_n50k.log 2>&1
done

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples $N50 \
    --floor 0.15 --steps 1 2 5 10 20 50 100 200 1000 --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md \
    > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png \
    >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact \
    --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
echo "=== MNIST TABLE COMPLETE THROUGH 1000 NFE ==="

# ---- remaining seeds at n=50,000 -------------------------------------------
for S in 1 2; do
  $PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_g0316_t1000/fid_n50k_seed$S.json --steps 10 50 200 \
      --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1
  $PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_flow/fid_n50k_seed$S.json --steps 10 50 200 \
      >> runs/eval_n50k.log 2>&1
done

# ---- CIFAR-10 ---------------------------------------------------------------
$PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_g0316_t1000 > runs/cifar10_g0316_t1000.log 2>&1
$PY -m bridges.flowmatch --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_flow > runs/cifar10_flow.log 2>&1
$PY -m bridges.ddpm_baseline --dataset cifar10 --schedule cosine --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_ddpm_cosine \
    > runs/cifar10_ddpm_cosine.log 2>&1
$PY -m bridges.ddpm_baseline --dataset cifar10 --schedule linear --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_ddpm_linear \
    > runs/cifar10_ddpm_linear.log 2>&1

for M in cifar10_g0316_t1000 cifar10_flow cifar10_ddpm_cosine cifar10_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples 10000 --batch_size 500 --compile \
      --out runs/$M/fid_n10k.json --steps 1 2 5 10 20 50 100 200 1000 \
      >> runs/eval_cifar_n10k.log 2>&1
done

$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples 10000 \
    --floor 0.70 --steps 1 2 5 10 20 50 100 200 1000 --pattern "fid_n10k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1

./make_figures.sh >> runs/figures.log 2>&1
echo "STAGE 2 COMPLETE"
