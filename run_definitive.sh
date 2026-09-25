#!/bin/bash
# Definitive run: the numbers that go in the paper.
#
# Everything below shares one UNet per dataset (verified identical in parameter
# count to the originally vendored net), one training loop, one optimiser, EMA
# decay, batch size and 40k-step budget, and one FID implementation. Only the
# generative process and its sampler differ.
#
# MNIST is evaluated at n=50,000 to match the published protocol's sample count.
# CIFAR-10 is evaluated at n=10,000: the floor there is still well below the
# effects of interest and it saves ~20 h of GPU on a single GB10.
#
# Resumable throughout -- every sweep skips (sampler, n_steps) pairs already in
# its JSON, so this can be killed and restarted without losing work.
set -x
cd "$(dirname "$0")"
PY=.venv/bin/python
GRID50K="1 2 5 10 20 50 100 200"     # 1000 NFE at n=50k costs ~5 h per sampler
N50=50000
N10=10000

# wait for the in-flight time-conditioning sweep to finish (27 = 9 steps x 3 samplers)
while [ "$(grep -c '\"sampler\"' runs/eval_mnist_t01.log 2>/dev/null || echo 0)" -lt 27 ]; do
  sleep 60
done
# retire the previous queue before it starts CIFAR with the old settings
pkill -f 'run_queue\.sh' || true
sleep 5
pkill -f 'bridges\.train --dataset cifar10' || true
sleep 5

# ---------------------------------------------------------------- MNIST models
$PY -m bridges.ddpm_baseline --dataset mnist --schedule cosine --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/mnist_ddpm_cosine \
    > runs/mnist_ddpm_cosine.log 2>&1
$PY -m bridges.ddpm_baseline --dataset mnist --schedule linear --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/mnist_ddpm_linear \
    > runs/mnist_ddpm_linear.log 2>&1

# ------------------------------------------------- MNIST definitive, n = 50,000
for M in mnist_g0316_t1000 mnist_flow mnist_ddpm_cosine mnist_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N50 --batch_size 1000 --compile \
      --out runs/$M/fid_n50k.json --steps $GRID50K \
      --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1
done

# error bars at n = 50,000 on the headline step counts
for S in 1 2; do
  for M in mnist_g0316_t1000 mnist_flow; do
    $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
        --n_samples $N50 --batch_size 1000 --compile --seed $S \
        --out runs/$M/fid_n50k_seed$S.json --steps 10 50 200 \
        --samplers bridge euler euler_det >> runs/eval_n50k.log 2>&1
  done
done

$PY -m bridges.make_tables --dataset MNIST --prefix mnist --n_samples $N50 \
    --floor 0.15 --steps $GRID50K \
    --out_tex runs/tables_mnist.tex --out_md runs/tables_mnist.md \
    > runs/tables_mnist.log 2>&1

# ------------------------------------------------------------ CIFAR-10 models
$PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_g0316_t1000 > runs/cifar10_g0316_t1000.log 2>&1
$PY -m bridges.flowmatch --dataset cifar10 --steps 40000 --batch_size 128 --lr 2e-4 \
    --compile --out runs/cifar10_flow > runs/cifar10_flow.log 2>&1
$PY -m bridges.ddpm_baseline --dataset cifar10 --schedule cosine --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_ddpm_cosine \
    > runs/cifar10_ddpm_cosine.log 2>&1
$PY -m bridges.ddpm_baseline --dataset cifar10 --schedule linear --steps 40000 \
    --batch_size 128 --lr 2e-4 --compile --out runs/cifar10_ddpm_linear \
    > runs/cifar10_ddpm_linear.log 2>&1

# ------------------------------------------------ CIFAR-10 definitive, n=10,000
for M in cifar10_g0316_t1000 cifar10_flow cifar10_ddpm_cosine cifar10_ddpm_linear; do
  $PY -m bridges.evaluate --ckpt runs/$M/ckpt_040000.pt \
      --n_samples $N10 --batch_size 500 --compile \
      --out runs/$M/fid_n10k.json --steps $GRID50K \
      --samplers bridge euler euler_det >> runs/eval_cifar_n10k.log 2>&1
done

$PY -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 --n_samples $N10 \
    --floor 0.70 --steps $GRID50K \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md \
    > runs/tables_cifar10.log 2>&1

./make_figures.sh > runs/figures.log 2>&1
echo "DEFINITIVE RUN COMPLETE"
