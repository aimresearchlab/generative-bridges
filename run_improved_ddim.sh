#!/bin/bash
# Add the improved-DDPM + DDIM row, once the current queue is done with the GPU.
# Improved DDPM's learned variance is what makes ANCESTRAL sampling work at few
# steps, but DDIM is usually stronger in the very-low-NFE regime. Reporting only
# the ancestral sampler shows the baseline below its best -- at 10 NFE it scored
# 10.99 against plain DDPM+DDIM's 6.96, which is not a fair picture of it.
set -x
cd "$(dirname "$0")"
LOCK=runs/.gpu.lock
while { [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; } \
   || pgrep -f "[.]venv/bin/python -m bridges\\.(evaluate|train|flowmatch|ddpm_baseline|improved_ddpm)" > /dev/null; do
  sleep 120
done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
$PY -m bridges.evaluate --ckpt runs/mnist_improved/ckpt_040000.pt \
    --n_samples 50000 --batch_size 1000 --compile \
    --out runs/mnist_improved/fid_n50k.json --steps 1 2 5 10 20 50 100 200 1000 \
    --samplers improved_ddim >> runs/eval_n50k.log 2>&1

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps 1 2 5 10 20 50 100 200 1000 --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --all_rows --out runs/figures/mnist_allsamplers.png >> runs/figures.log 2>&1
echo "=== IMPROVED DDIM ROW COMPLETE ==="
