#!/bin/bash
# Correctness controls, deferred until after the 50k sweep. See
# run_reviewer_response.sh for why these run last rather than first.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
LOG=runs/reviewer_response.log
while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 120; done
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do sleep 120; done
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"; exit' EXIT INT TERM
echo "### gamma=0 control and checkpoint sweep" >> $LOG
T=runs/cifar10_drift_g0.0
if [ ! -f $T/ckpt_040000.pt ]; then
  $PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 \
      --lr 2e-4 --gamma 0.0 --param drift --compile --out $T > $T.log 2>&1
fi
$PY -m bridges.evaluate --ckpt $T/ckpt_040000.pt --n_samples 2000 \
    --batch_size 500 --compile --out $T/fid_sanity.json \
    --steps 10 50 --samplers euler >> $LOG 2>&1
for M in cifar10_flow cifar10_drift_g0.3162277660168379; do
  for C in 010000 020000 030000 040000; do
    [ -f runs/$M/ckpt_$C.pt ] && $PY -m bridges.evaluate --ckpt runs/$M/ckpt_$C.pt \
        --n_samples 2000 --batch_size 500 --compile \
        --out runs/$M/fid_ckpt_$C.json --steps 50 >> $LOG 2>&1
  done
done
echo "=== ALL COMPLETE ===" >> $LOG
