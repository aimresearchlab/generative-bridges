#!/bin/bash
# Retrain the CIFAR bridge against the drift, the quantity the sampler integrates.
#
# Diagnosis (runs/mse_by_s_cifar.json). Evaluated as endpoint estimators on a
# common path, the endpoint-trained bridge net and the velocity-trained flow net
# agree to three digits for s <= 0.8 and diverge only near the data end: at
# s = 0.995 the bridge's endpoint MSE is 10x the flow net's, and in drift terms,
# which is what the sampler consumes, 7.17 against 0.70. The cause is the loss,
# not the family: MSE on yhat with s ~ U[0,1) is the drift loss with the terminal
# region discounted by 1/(1-s)^2, up to four orders of magnitude. Flow matching
# carries that weight for free because its target is written in velocity form.
#
# Two gammas. At gamma = 0 the drift objective is conditional flow matching
# exactly -- verified, objective and sampler agree to 7e-7 -- so gamma = 0.05
# tests whether the gap closes, and gamma = 0.3162, the submitted manuscript's
# value, tests what the noise itself costs once the objective is matched. The
# drift target is (y-x) - gamma sqrt(s/(1-s)) eps, so larger gamma buys its
# stochasticity with gradient variance that grows without bound as s -> 1.
set -x
cd "$(dirname "$0")"

LOCK=runs/.gpu.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "another queue holds the lock"; exit 1; fi
if pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; then
  echo "a worker is still alive; refusing to start"; exit 1; fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT INT TERM
PY=.venv/bin/python

for G in 0.05 0.3162277660168379; do
  T=runs/cifar10_drift_g$G
  if [ ! -f $T/ckpt_040000.pt ]; then
    $PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 \
        --lr 2e-4 --gamma $G --param drift --compile --out $T > $T.log 2>&1
  fi
  $PY -m bridges.evaluate --ckpt $T/ckpt_040000.pt --n_samples 2000 \
      --batch_size 500 --compile --save_grid --out $T/fid_sanity.json \
      --steps 10 50 --samplers bridge euler >> runs/cifar_drift.log 2>&1
  echo "=== $T DONE ==="; tail -5 runs/cifar_drift.log
done
echo "=== DRIFT RUNS COMPLETE ==="
