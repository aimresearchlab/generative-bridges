#!/bin/bash
# A: re-measure the MNIST Euler rows after the evaluation-count correction.
# B: the digit-domain metric applied to generated samples (associate editor 6).
# C: a sampling-seed standard deviation for CIFAR-10, measured at n = 10,000.
#
# A. The Euler and deterministic Euler samplers read the endpoint with a network
#    evaluation taken after the integration loop while the record counted
#    n_steps. They now reuse the estimate computed at the last grid point, which
#    is how the exact-posterior sampler was always counted. The stale entries are
#    stripped from the JSON (backups at *.pre_nfe_fix) so the re-run writes into
#    the same files and the table shape is unchanged. About 4.5 h, of which the
#    single 1000-evaluation cell is 2.5 h.
#
# C. Full replicates at n = 50,000 cost 8 h each. The seed variability is instead
#    measured at n = 10,000 across three seeds. FID's sampling variance falls with
#    n, so a standard deviation measured at 10,000 is an upper bound on the one at
#    50,000: using it to decide which differences in the n = 50,000 table are
#    resolved is conservative in the right direction.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
LOG=runs/A_B_C.log
GRID_E="1 2 5 10 20 50 100 200 1000"
GRID_D="1 2 5 10 20 50 100 200"

while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 60; done
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another queue holds the lock"; exit 1; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; exit' EXIT INT TERM

echo "### A. MNIST Euler rows, re-measured" >> $LOG
M=runs/mnist_g0316_t1000/ckpt_040000.pt
$PY -m bridges.evaluate --ckpt $M --n_samples 50000 --batch_size 500 --compile \
    --save_grid --samplers euler --steps $GRID_E \
    --out runs/mnist_g0316_t1000/fid_n50k.json >> $LOG 2>&1
$PY -m bridges.evaluate --ckpt $M --n_samples 50000 --batch_size 500 --compile \
    --save_grid --samplers euler_det --steps $GRID_D \
    --out runs/mnist_g0316_t1000/fid_n50k.json >> $LOG 2>&1
for S in 1 2; do
  $PY -m bridges.evaluate --ckpt $M --n_samples 50000 --batch_size 500 --compile \
      --samplers euler euler_det --steps 10 --seed $S \
      --out runs/mnist_g0316_t1000/fid_n50k_seed$S.json >> $LOG 2>&1
done
$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples 50000 \
    --floor 0.15 --steps $GRID_E --pattern "fid_n50k*.json" \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md >> $LOG 2>&1
echo "=== A COMPLETE ===" >> $LOG

echo "### B. digit-domain metric on generated MNIST" >> $LOG
for MS in mnist_g0316_t1000:euler mnist_flow:flow_euler mnist_improved:improved_ancestral; do
  D=${MS%%:*}; S=${MS##*:}
  $PY -m bridges.mnist_metric --ckpt runs/$D/ckpt_040000.pt --sampler $S \
      --steps 10 100 --n_samples 50000 --compile \
      --out runs/$D/mnist_modes_$S.json >> $LOG 2>&1
done
echo "=== B COMPLETE ===" >> $LOG

echo "### C. sampling-seed spread, CIFAR-10, n=10000, three seeds" >> $LOG
for MS in cifar10_drift_g0.3162277660168379:euler cifar10_flow:flow_euler \
          cifar10_improved:improved_ancestral; do
  D=${MS%%:*}; S=${MS##*:}
  for SEED in 0 1 2; do
    $PY -m bridges.evaluate --ckpt runs/$D/ckpt_040000.pt --n_samples 10000 \
        --batch_size 500 --compile --samplers $S --steps 10 50 --seed $SEED \
        --out runs/$D/seedspread_${S}_s$SEED.json >> $LOG 2>&1
  done
done
echo "=== ALL COMPLETE ===" >> $LOG
