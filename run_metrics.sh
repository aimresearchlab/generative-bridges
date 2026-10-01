#!/bin/bash
# Measures the reviewers ask for that FID cannot supply, across every row of the
# comparison rather than the bridge alone.
#
#   1. Memory (associate editor 4). Peak allocated during one training step and
#      one sampling batch, per method.
#
#   2. Fidelity and coverage separated (Reviewer 1.7-1.8). FID collapses the two
#      into one scalar, so clean-but-partial and complete-but-soft score alike.
#      Precision/recall (Kynkaanniemi et al. 2019), density/coverage (Naeem et
#      al. 2020) and KID (Binkowski et al. 2018, unbiased and so usable at small
#      n) come from the same Inception activations the FID path extracts. The
#      k-NN manifolds are O(n^2), so n = 10,000 is the practical ceiling and the
#      conventional reporting point. Three step counts span the regime where the
#      bridge leads, the crossover, and the regime where it trails.
#
#      Every sampler row of the CIFAR-10 table is covered, plus the bridge under
#      the endpoint objective, which asks whether the objective correction buys
#      fidelity, coverage, or both.
#
#   3. An MNIST-appropriate feature space (associate editor 6). Inception-v3 was
#      fitted to natural colour photographs. A small convolutional classifier
#      trained on MNIST supplies the feature space, and its predicted-label
#      histogram gives a direct mode count.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
LOG=runs/metrics.log

while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 60; done
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do sleep 60; done
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; exit' EXIT INT TERM

echo "### 1. memory" >> $LOG
$PY -m bridges.measure_memory --dataset cifar10 --out runs/memory.json >> $LOG 2>&1

echo "### 2. fidelity/coverage, CIFAR-10, n=10000, every row" >> $LOG
for M in cifar10_drift_g0.3162277660168379:euler \
         cifar10_flow:flow_euler \
         cifar10_improved:improved_ancestral \
         cifar10_improved:improved_ddim \
         cifar10_ddpm_linear:ddpm_ancestral \
         cifar10_g0316_t1000:bridge; do
  D=${M%%:*}; S=${M##*:}
  $PY -m bridges.evaluate --ckpt runs/$D/ckpt_040000.pt --n_samples 10000 \
      --batch_size 500 --compile --diversity 10000 --samplers $S \
      --steps 10 50 200 --out runs/$D/diversity_${S}_n10k.json >> $LOG 2>&1
  echo "--- done $D $S" >> $LOG
done

echo "### 3. MNIST classifier feature space" >> $LOG
$PY -m bridges.mnist_metric --train_only >> $LOG 2>&1

echo "=== METRICS COMPLETE ===" >> $LOG
