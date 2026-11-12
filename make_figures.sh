#!/bin/bash
# Regenerate every figure from whatever sweep JSONs currently exist.
# Safe to run at any point; missing inputs are skipped.
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p runs/figures
FLOOR=1.50   # FID(5000 real held-out, 55000 real) -- matches the model measurement

B=runs/mnist_g0316_t1000/fid_sweep_step40000.json
F=runs/mnist_flow/fid_sweep_step40000.json
A=runs/mnist_g0316_t01/fid_sweep_step40000.json
D1=runs/mnist_ddpm_cosine/fid_sweep_step40000.json
D2=runs/mnist_ddpm_linear/fid_sweep_step40000.json
C=runs/cifar10_g0316_t1000/fid_sweep_step40000.json

[ -f "$B" ] && $PY -m bridges.plot_scaling "$B" --floor $FLOOR \
  --out runs/figures/01_mnist_bridge_samplers.png \
  --title "MNIST bridge, 40k steps - sampler comparison at matched NFE"

[ -f "$B" ] && [ -f "$A" ] && $PY -m bridges.plot_scaling "$B" "$A" \
  --labels "fixed t-scale" "t in [0,0.1) as published" --floor $FLOOR \
  --out runs/figures/02_mnist_time_conditioning.png \
  --title "MNIST - effect of the degenerate time conditioning"

ARGS=""; LBL=""
for f in "$B" "$F" "$D1" "$D2"; do [ -f "$f" ] && ARGS="$ARGS $f"; done
[ -n "$ARGS" ] && $PY -m bridges.plot_scaling $ARGS --floor $FLOOR \
  --out runs/figures/03_mnist_method_comparison.png \
  --title "MNIST - matched budget: same UNet, optimiser, EMA, 40k steps, identical FID path"

[ -f "$C" ] && $PY -m bridges.plot_scaling "$C" --floor 0.70 \
  --out runs/figures/04_cifar10_bridge_samplers.png \
  --title "CIFAR-10 bridge, 40k steps - sampler comparison at matched NFE"

ls -la runs/figures/ 2>/dev/null
