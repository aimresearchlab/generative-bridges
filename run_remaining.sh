#!/bin/bash
# Everything still outstanding, with a thermal cooldown between blocks.
#
# The box idles around 70 C and sat at 79 C under sustained load with SM clocks
# at 2457 of 3003 MHz. No throttle flag was set, but this chassis already went
# down once, so each heavy block is followed by a pause and the temperature is
# logged either side of it. Cost is roughly 5 h across the run; cheap against
# ~30 h of compute and a machine that has to stay up unattended.
#
# Resumable: every sweep skips (sampler, n_steps) pairs already present in its
# JSON, and training blocks are skipped outright if their checkpoint exists.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another GPU job is live (pid $(cat "$LOCK")); refusing to start"
  exit 1
fi
# Belt and braces: the lockfile only tracks the queue script, but an orphaned
# worker from a half-killed previous run is just as dangerous. This box has
# unified memory, so two concurrent jobs each holding an Inception model plus a
# UNet exhaust all 119 GB and hang the machine rather than raising a CUDA OOM.
if pgrep -f "bridges\.(evaluate|train|flowmatch|ddpm_baseline|improved_ddpm)" > /dev/null; then
  echo "a bridges worker is still alive; refusing to start"
  pgrep -af "bridges\." | head
  exit 1
fi
mkdir -p runs && echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM

PY=.venv/bin/python
GRID="1 2 5 10 20 50 100 200 1000"
COOL_MIN=${COOL_MIN:-20}

# The risk being managed is a ~30 h unbroken run on a desk rig, not short hot
# spells. So a pause is gated on BOTH how long the card has been working without
# relief and how warm it currently is: long stretches get periodic breaks, short
# blocks get none. Pausing after every block regardless of its length was the
# wrong shape -- it fired ~16 times, including after sweeps lasting minutes.
COOL_ABOVE=${COOL_ABOVE:-70}        # only pause if at least this warm
COOL_EVERY_H=${COOL_EVERY_H:-3}     # and only if this many hours since the last
LAST_COOL=$(date +%s)

cool () {
  local t0 now since_h
  now=$(date +%s)
  since_h=$(( (now - LAST_COOL) / 3600 ))
  t0=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
  if [ "$since_h" -lt "$COOL_EVERY_H" ]; then
    echo "[cooldown] $(date +%H:%M:%S) GPU ${t0}C, ${since_h}h since last break -> running on"
    return 0
  fi
  if [ "$t0" -lt "$COOL_ABOVE" ]; then
    echo "[cooldown] $(date +%H:%M:%S) GPU ${t0}C below ${COOL_ABOVE}C -> running on"
    LAST_COOL=$now
    return 0
  fi
  echo "[cooldown] $(date +%H:%M:%S) GPU ${t0}C after ${since_h}h -> pausing ${COOL_MIN} min"
  sleep $((COOL_MIN * 60))
  LAST_COOL=$(date +%s)
  echo "[cooldown] $(date +%H:%M:%S) GPU $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)C -> resuming"
}

# train only if the checkpoint is not already there
train_if_needed () {  # <outdir> <module> <extra args...>
  local out=$1; shift
  local mod=$1; shift
  if [ -f "runs/$out/ckpt_040000.pt" ]; then
    echo "[skip] runs/$out already trained"; return 0
  fi
  $PY -m "bridges.$mod" --steps 40000 --batch_size 128 --lr 2e-4 --compile \
      --out "runs/$out" "$@" > "runs/$out.log" 2>&1
  cool
}

# ---------------------------------------------------------------- MNIST seeds
for S in 1 2; do
  $PY -m bridges.evaluate --ckpt runs/mnist_g0316_t1000/ckpt_040000.pt \
      --n_samples 50000 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_g0316_t1000/fid_n50k_seed$S.json --steps 10 50 200 \
      --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1
  cool
  $PY -m bridges.evaluate --ckpt runs/mnist_flow/ckpt_040000.pt \
      --n_samples 50000 --batch_size 1000 --compile --seed $S \
      --out runs/mnist_flow/fid_n50k_seed$S.json --steps 10 50 200 \
      --samplers flow_euler >> runs/eval_n50k.log 2>&1
  cool
done

# ------------------------------------------------- improved DDPM, MNIST
train_if_needed mnist_improved improved_ddpm --dataset mnist --schedule cosine
$PY -m bridges.evaluate --ckpt runs/mnist_improved/ckpt_040000.pt \
    --n_samples 50000 --batch_size 1000 --compile \
    --out runs/mnist_improved/fid_n50k.json --steps $GRID >> runs/eval_n50k.log 2>&1
cool

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps $GRID --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md > runs/tables_mnist.log 2>&1
$PY -m bridges.headline_figure --out runs/figures/mnist_headline.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --compact --out runs/figures/mnist_headline_compact.png >> runs/figures.log 2>&1
$PY -m bridges.headline_figure --all_rows --out runs/figures/mnist_allsamplers.png >> runs/figures.log 2>&1
echo "=== MNIST COMPLETE (5 methods, 1 -> 1000 NFE, n=50k) ==="

# ------------------------------------------------------------ CIFAR-10 models
train_if_needed cifar10_g0316_t1000 train        --dataset cifar10
train_if_needed cifar10_flow        flowmatch    --dataset cifar10
train_if_needed cifar10_ddpm_cosine ddpm_baseline --dataset cifar10 --schedule cosine
train_if_needed cifar10_ddpm_linear ddpm_baseline --dataset cifar10 --schedule linear
train_if_needed cifar10_improved    improved_ddpm --dataset cifar10 --schedule cosine

# ------------------------------------------------- CIFAR-10 sweeps, n = 10,000
for M in cifar10_g0316_t1000 cifar10_flow cifar10_ddpm_cosine cifar10_ddpm_linear cifar10_improved; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples 10000 --batch_size 500 --compile \
      --out runs/$M/fid_n10k.json --steps $GRID >> runs/eval_cifar_n10k.log 2>&1
  cool
done

$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples 10000 \
    --floor 0.70 --steps $GRID --pattern "fid_n10k*.json" \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1
./make_figures.sh >> runs/figures.log 2>&1
echo "ALL REMAINING WORK COMPLETE"
