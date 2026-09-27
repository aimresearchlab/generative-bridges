#!/bin/bash
# Final queue: finish MNIST, then the gamma sweep, then CIFAR-10.
#
# The gamma sweep is inserted BEFORE CIFAR because it is far cheaper (~7 h against
# ~26 h) and answers a specific review point. gamma^2 = sigma^2 T is the method's
# only free parameter, and the manuscript never varies it: Section 3 reports
# "T = 0.1 after preliminary parameter tuning" while Section 2.4 defers sigma to
# future work, but those are the same knob.
#
# The 1-D study motivates what to expect. With the conditional expectation known
# in closed form, the deterministic sampler's residual 1-Wasserstein error at
# 2000 steps grew monotonically with gamma -- 1.2x the Monte-Carlo floor at
# gamma = 0.032, 6.9x at 0.316, 41.9x at 1.0 and 114x at 2.0, where the terminal
# standard deviation collapsed from 1.758 to 0.775 -- while the correct samplers
# stayed at the floor throughout. So gamma should trade off: larger values give
# the stochastic samplers more room and punish the deterministic one, smaller
# values converge on flow matching. This sweep tests that on images.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
while pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; do sleep 60; done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
GAMMA_GRID="2 5 10 20 50 100 200"
COOL_MIN=${COOL_MIN:-8}; COOL_ABOVE=${COOL_ABOVE:-70}; COOL_EVERY_H=${COOL_EVERY_H:-3}
LAST_COOL=$(date +%s)
cool () {
  local t0 now since_h; now=$(date +%s); since_h=$(( (now-LAST_COOL)/3600 ))
  t0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  if [ "$since_h" -lt "$COOL_EVERY_H" ] || [ "$t0" -lt "$COOL_ABOVE" ]; then
    echo "[cooldown] $(date +%H:%M:%S) GPU ${t0}C, ${since_h}h -> running on"; return 0; fi
  echo "[cooldown] $(date +%H:%M:%S) GPU ${t0}C after ${since_h}h -> pausing ${COOL_MIN} min"
  sleep $((COOL_MIN*60)); LAST_COOL=$(date +%s)
}

# ---- finish MNIST: improved + DDIM ------------------------------------------
$PY -m bridges.evaluate --ckpt runs/mnist_improved/ckpt_040000.pt \
    --n_samples 50000 --batch_size 1000 --compile \
    --out runs/mnist_improved/fid_n50k.json --steps $GRID \
    --samplers improved_ddim >> runs/eval_n50k.log 2>&1
cool

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
$PY -m bridges.all_curves_figure >> runs/figures.log 2>&1
echo "=== MNIST COMPLETE ==="

# ---- gamma sweep -------------------------------------------------------------
# gamma = 0.3162 is the published setting and is already trained.
for G in 0.1 0.2 0.6 1.0; do
  OUT="mnist_g${G}"
  if [ ! -f "runs/$OUT/ckpt_040000.pt" ]; then
    $PY -m bridges.train --dataset mnist --steps 40000 --batch_size 128 --lr 2e-4 \
        --gamma "$G" --compile --out "runs/$OUT" > "runs/$OUT.log" 2>&1
    cool
  fi
done
# evaluate every gamma at a common, cheaper sample count so the column is
# internally consistent; the published gamma is re-run at that n for comparability.
for OUT in mnist_g0.1 mnist_g0.2 mnist_g0316_t1000 mnist_g0.6 mnist_g1.0; do
  $PY -m bridges.evaluate --ckpt runs/$OUT/ckpt_040000.pt \
      --n_samples 10000 --batch_size 1000 --compile \
      --out runs/$OUT/fid_gamma_n10k.json --steps $GAMMA_GRID \
      --samplers euler euler_det >> runs/eval_gamma.log 2>&1
  cool
done
$PY -m bridges.gamma_figure >> runs/figures.log 2>&1
echo "=== GAMMA SWEEP COMPLETE ==="

# ---- CIFAR-10 ----------------------------------------------------------------
for SPEC in "cifar10_g0316_t1000 train --dataset cifar10" \
            "cifar10_flow flowmatch --dataset cifar10" \
            "cifar10_ddpm_cosine ddpm_baseline --dataset cifar10 --schedule cosine" \
            "cifar10_ddpm_linear ddpm_baseline --dataset cifar10 --schedule linear" \
            "cifar10_improved improved_ddpm --dataset cifar10 --schedule cosine"; do
  set -- $SPEC; OUT=$1; MOD=$2; shift 2
  if [ ! -f "runs/$OUT/ckpt_040000.pt" ]; then
    $PY -m "bridges.$MOD" --steps 40000 --batch_size 128 --lr 2e-4 --compile \
        --out "runs/$OUT" "$@" > "runs/$OUT.log" 2>&1
    cool
  fi
done
for M in cifar10_g0316_t1000 cifar10_flow cifar10_ddpm_cosine cifar10_ddpm_linear cifar10_improved; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt --n_samples 10000 \
      --batch_size 500 --compile --out runs/$M/fid_n10k.json --steps $GRID \
      >> runs/eval_cifar_n10k.log 2>&1
  cool
done
$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples 10000 \
    --floor 0.70 --steps $GRID --pattern "fid_n10k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md > runs/tables_cifar10.log 2>&1
./make_figures.sh >> runs/figures.log 2>&1
echo "ALL WORK COMPLETE"
