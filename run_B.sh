#!/bin/bash
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 30; done
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"; exit' EXIT INT TERM
for MS in mnist_g0316_t1000:euler mnist_flow:flow_euler mnist_improved:improved_ancestral; do
  D=${MS%%:*}; S=${MS##*:}
  $PY -m bridges.mnist_metric --ckpt runs/$D/ckpt_040000.pt --sampler $S \
      --steps 10 100 --n_samples 50000 --compile \
      --out runs/$D/mnist_modes_$S.json >> runs/B.log 2>&1
done
echo "=== B COMPLETE ===" >> runs/B.log
