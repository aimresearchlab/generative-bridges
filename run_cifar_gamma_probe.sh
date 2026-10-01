#!/bin/bash
# Is the bridge's poor CIFAR showing a mistuned gamma, or a real limit?
#
# gamma = 0.3162 was carried over from the submitted manuscript, where it was
# tuned on MNIST. The MNIST sweep then found it was not optimal even there:
# smaller values were better at every step count, monotonically, down to the
# smallest tested. CIFAR has different statistics and there is no reason the
# same value should transfer.
#
# One additional model at gamma = 0.1 is enough to tell the two apart. If it
# closes most of the gap to flow matching, the CIFAR result is a tuning artefact
# and the full sweep should not be run at 0.3162. If it does not, the gap is
# real and worth reporting as such. About 4 h against the 52 h the full sweep
# would cost.
set -x
cd "$(dirname "$0")"
LOCK=runs/.gpu.lock
if pgrep -f "[.]venv/bin/python -m bridges" > /dev/null; then echo busy; exit 1; fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT INT TERM
PY=.venv/bin/python

if [ ! -f runs/cifar10_g0.1/ckpt_040000.pt ]; then
  $PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
      --gamma 0.1 --compile --out runs/cifar10_g0.1 > runs/cifar10_g0.1.log 2>&1
fi

# same cheap protocol as the sanity check, so the two are directly comparable
$PY -m bridges.evaluate --ckpt runs/cifar10_g0.1/ckpt_040000.pt \
    --n_samples 2000 --batch_size 500 --compile --save_grid \
    --out runs/cifar10_g0.1/fid_sanity.json --steps 10 50 \
    --samplers bridge euler >> runs/cifar_sanity.log 2>&1
echo "=== GAMMA PROBE COMPLETE ==="
