# Reproducing the rebuilt evaluation

Every model shares one architecture, optimiser, EMA decay, batch size and
training budget, and is scored through one FID implementation. Only the
generative process and its sampler differ between rows, so no result is
attributable to architecture or training setup.

## Environment

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Measured on one NVIDIA GB10 (DGX Spark, 119 GB unified memory, sm_121).
Memory is unified, so a second GPU job competes with the first for the same
pool: every script here takes `runs/.gpu.lock` and refuses to start while
another holds it. Run one at a time.

Datasets download to `./data` on first use. `runs/` and `fid_cache/` are
generated and are not tracked.

## Training

CIFAR-10, 40,000 steps at batch 128, about 3.9 h per model:

```bash
PY=.venv/bin/python
G=0.3162277660168379

# the bridge: regress the drift, which is what the sampler integrates
$PY -m bridges.train --dataset cifar10 --steps 40000 --batch_size 128 \
    --lr 2e-4 --gamma $G --param drift --compile \
    --out runs/cifar10_drift_g$G

# conditional flow matching at matched budget
$PY -m bridges.flowmatch --dataset cifar10 --steps 40000 --batch_size 128 \
    --lr 2e-4 --compile --out runs/cifar10_flow

# improved denoising diffusion, cosine schedule, learned reverse variance
$PY -m bridges.improved_ddpm --dataset cifar10 --schedule cosine \
    --steps 40000 --batch_size 128 --lr 2e-4 --compile \
    --out runs/cifar10_improved

# simple denoising diffusion, linear schedule
$PY -m bridges.ddpm_baseline --dataset cifar10 --schedule linear \
    --steps 40000 --batch_size 128 --lr 2e-4 --compile \
    --out runs/cifar10_ddpm_linear
```

`--param endpoint` trains against the endpoint instead of the drift. The two
objectives differ by a weight of `(1-s)^-2`; at `--gamma 0` with `--param drift`
the objective is conditional flow matching exactly.

MNIST: substitute `--dataset mnist`.

## Evaluation

```bash
.venv/bin/python -m bridges.evaluate \
    --ckpt runs/<model>/ckpt_040000.pt \
    --n_samples 50000 --batch_size 500 --compile --save_grid \
    --steps 1 2 5 10 20 50 100 200 1000 --samplers euler \
    --out runs/<model>/fid_n50k.json
```

Results are appended to the output JSON after every `(sampler, n_steps)` entry
and re-running skips entries already present, so a sweep can be interrupted at
any point and resumed by re-issuing the same command: it continues at the first
unmeasured entry and repeats nothing.

The Euler and deterministic-Euler samplers formerly took one network evaluation
after the integration loop to read the endpoint while the record wrote
`nfe = n_steps`. They now reuse the estimate computed at the last grid point,
matching how the exact-posterior sampler was always counted. Measurements of
those two samplers made before this change are charged one evaluation fewer than
they spent and should be re-run.

Samplers: `bridge` (exact Brownian-bridge posterior), `euler` (Euler-Maruyama),
`euler_det` (the deterministic map) for bridge checkpoints; `flow_euler`,
`improved_ancestral`, `improved_ddim`, `ddpm_ancestral`, `ddim` for the others.

`--diversity N` additionally reports precision/recall, density/coverage and KID
from the same activations. The k-NN manifolds are O(N^2); N = 10,000 is the
practical ceiling.

## Supporting measurements

```bash
.venv/bin/python -m bridges.calibrate_fid --dataset cifar10 --mode both \
    --counts 2000 5000 10000          # FID floors, both comparators
.venv/bin/python -m bridges.measure_memory --dataset cifar10
.venv/bin/python -m bridges.mnist_metric         # classifier-feature metric
.venv/bin/python -m bridges.sanity_1d            # samplers, closed-form target
```

## Orchestration

| script | what it runs |
| --- | --- |
| `run_cifar_sweeps.sh` | the CIFAR-10 table at n = 50,000, row by row, writing the table after each |
| `run_metrics.sh` | memory, precision/recall and KID across every row, MNIST classifier features |
| `run_controls.sh` | the gamma = 0 control and FID against training step |

Each holds `runs/.gpu.lock`, pauses to cool when the GPU has been above 70 C for
three hours, and can be interrupted with

```bash
kill $(cat runs/.gpu.lock); pkill -f '^\.venv/bin/python -m bridges'
```

## Figures and tables

```bash
.venv/bin/python -m bridges.make_tables --dataset CIFAR-10 --prefix cifar10 \
    --n_samples 50000 --floor 0.15 --pattern "fid_n50k*.json" \
    --bridge_dir cifar10_drift_g0.3162277660168379 \
    --out_tex runs/tables_cifar10.tex --out_md runs/tables_cifar10.md
.venv/bin/python -m bridges.samples_figure --bare
.venv/bin/python -m bridges.drift_figure --bare
```

Tables in `notes/experiments.tex` are `\input` from the generated files rather
than transcribed.
