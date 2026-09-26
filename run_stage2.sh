#!/bin/bash
# Stage 2: the 1000-NFE column, trimmed, then CIFAR-10.
#
# The published grid runs to 1000 sampling steps, so leaving the n=50,000 table
# to stop at 200 would let a referee assume the step-count pathology is waiting
# just past its edge. The column is therefore filled first -- but only for the
# rows that belong in the headline comparison:
#
#   Bridge (Euler-Maruyama)  the best bridge curve: equal or better than the
#                            exact-posterior sampler at every step count measured
#   Flow matching            the matched modern control
#   DDPM cosine + DDIM       the stronger diffusion baseline
#   DDPM linear + DDIM       the baseline the original paper used
#
# Dropped from this column, at ~5.1 h per sampler saved: the exact-posterior and
# deterministic-Euler bridge samplers, and both ancestral DDPM samplers. The
# three-way sampler comparison that needs deterministic Euler already has a
# complete 1 -> 1000 sweep at n=5,000, which is the right place for it: it is a
# diagnostic about samplers, not a claim about methods.
#
# A lockfile guards the run. The earlier crash came from a diagnostic launched
# beside the queue: on unified memory two concurrent jobs each holding an
# Inception model (which resizes batches to 299x299) plus a UNet exhaust all
# 119 GB and hang the machine rather than raising a CUDA OOM.
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
GRID="1 2 5 10 20 50 100 200 1000"

# ---- 1000-NFE column at n=50,000, headline rows only ------------------------
$PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
    --n_samples $N50 --batch_size 1000 --compile \
    --out runs/mnist_g0316_t1000/fid_n50k.json --steps 1000 \
    --samplers euler >> runs/eval_n50k.log 2>&1

$PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
    --n_samples $N50 --batch_size 1000 --compile \
    --out runs/mnist_flow/fid_n50k.json --steps 1000 \
    --samplers flow_euler >> runs/eval_n50k.log 2>&1

for M in mnist_ddpm_cosine mnist_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile \
      --out runs/$M/fid_n50k.json --steps 1000 \
      --samplers ddim >> runs/eval_n50k.log 2>&1
done

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples $N50 \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md \
    > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png \
    >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact \
    --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
echo "=== MNIST HEADLINE COMPLETE THROUGH 1000 NFE ==="

# ---- remaining seeds at n=50,000 -------------------------------------------
for S in 1 2; do
  $PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_g0316_t1000/fid_n50k_seed$S.json --steps 10 50 200 \
      --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1
  $PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_flow/fid_n50k_seed$S.json --steps 10 50 200 \
      --samplers flow_euler >> runs/eval_n50k.log 2>&1
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
      --out runs/$M/fid_n10k.json --steps $GRID >> runs/eval_cifar_n10k.log 2>&1
done

$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples 10000 \
    --floor 0.70 --steps $GRID --pattern "fid_n10k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1

./make_figures.sh >> runs/figures.log 2>&1
echo "STAGE 2 COMPLETE"
