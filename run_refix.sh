#!/bin/bash
# Re-run every ancestral row with the respacing fix, then continue to CIFAR.
# DDIM rows are unaffected: they already referenced alpha-bar at the next grid
# point, which is correct respacing. Only the ancestral path was wrong.
set -x
cd "$(dirname "$0")"
LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then exit 1; fi
if pgrep -f "bridges\.(evaluate|train|flowmatch|ddpm_baseline|improved_ddpm)" > /dev/null; then
  echo "a bridges worker is still alive; refusing to start"; exit 1; fi
mkdir -p runs && echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
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

$PY -m bridges.evaluate --ckpt runs/mnist_improved/ckpt_040000.pt \
    --n_samples 50000 --batch_size 1000 --compile \
    --out runs/mnist_improved/fid_n50k.json --steps $GRID >> runs/eval_n50k.log 2>&1
cool
for M in mnist_ddpm_cosine mnist_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples 50000 --batch_size 1000 --compile \
      --out runs/$M/fid_n50k.json --steps $GRID \
      --samplers ddpm_ancestral >> runs/eval_n50k.log 2>&1
  cool
done

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --all_rows --out runs/figures/mnist_allsamplers.png >> runs/figures.log 2>&1
echo "=== MNIST COMPLETE ==="

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
echo "ALL REMAINING WORK COMPLETE"
