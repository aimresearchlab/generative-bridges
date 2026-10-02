#!/bin/bash
# objective x sampler, with the time conditioning held at t_scale = 1000 in both
# arms so the comparison is not confounded by it.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
LOCK=runs/.gpu.lock
while pgrep -f "^[.]venv/bin/python -m bridges" > /dev/null; do sleep 30; done
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"; exit' EXIT INT TERM
for D in cifar10_g0316_t1000 cifar10_drift_g0.3162277660168379; do
  $PY -m bridges.evaluate --ckpt runs/$D/ckpt_040000.pt --n_samples 10000 \
      --batch_size 500 --compile --samplers euler euler_det --steps 10 50 \
      --out runs/$D/objsampler_n10k.json >> runs/2x2.log 2>&1
done
echo "=== 2x2 COMPLETE ===" >> runs/2x2.log
