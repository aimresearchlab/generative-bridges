# Reproducing the rebuilt evaluation

Every model shares one architecture, optimiser, EMA decay, batch size and
training budget, and is scored through one FID implementation. Only the
generative process and its sampler differ between rows, so no result is
attributable to architecture or training setup.

## Environment

The code runs on NVIDIA GPUs under CUDA and on AMD GPUs under ROCm. PyTorch
drives both through the same `torch.cuda` API, so nothing in `bridges/` is
vendor-specific and only the PyTorch install differs.

NVIDIA:

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

AMD: the `torch` on PyPI is the CUDA build, so PyTorch comes from AMD's index,
built for the GPU's architecture, and everything else from
`requirements-common.txt`, which names no PyTorch package:

```bash
python -m venv .venv
AMD=https://stable.repo.amd.com/rocm/whl-next/
.venv/bin/pip install --index-url $AMD rocm-bootstrap
GFX=$(.venv/bin/rocm-bootstrap-detect --unique | head -n 1)     # e.g. gfx1151
.venv/bin/pip install --index-url $AMD "torch[device-$GFX]==2.13.0+rocm10.1.0" \
    "torchvision[device-$GFX]==0.28.0+rocm10.1.0" torchaudio
.venv/bin/pip install -r requirements-common.txt
```

Do not install `requirements.txt` into this environment, as it would replace
the ROCm build. Pin the full local version (`+rocm10.1.0`): the index also
carries ROCm 10.0 builds, and the resolver can pick one of those to satisfy a
looser pin. On gfx1151 the ROCm 10.0 build of torch 2.14 measured about half
the matmul throughput of the 10.1 build. When moving to newer versions, keep
torch and torchvision a matched pair: AMD's torchvision does not pin its torch,
and a mismatched pair installs cleanly but loses torchvision's compiled
operators. torch 2.14.0 with torchvision 0.29.0a0, both `+rocm10.1.0`, is also
a working pair, but on gfx1151 it trained and sampled no faster than the pair
above.

Either way, check the result with

```bash
.venv/bin/python -m bridges.device --compile
```

which names the GPU and the stack, then runs a bf16 training step and a short
sampling run. Every script uses the first GPU; `BRIDGES_DEVICE` selects another
(`cuda:1`), or the CPU (`cpu`) for a check on a machine without one.

The reported results were measured on one NVIDIA GB10 (DGX Spark, 119 GB
unified memory, sm_121). The pipeline also runs unchanged on an AMD Ryzen AI
Max+ 395 (Radeon 8060S, gfx1151, 128 GB unified memory) under ROCm 10.1, where
CIFAR-10 training settles at 1.2-1.3 steps/s against the GB10's 2.85, about 9 h
per model, and sampling runs at about 0.6 of the GB10's rate (MNIST, 3,270
against 5,530 network evaluations per second at batch 1000). Keep `--compile`
there: it nearly doubles sampling throughput on that GPU. The throughput
columns (`imgs_per_s`) are properties of the machine, so do not mix records
from different machines in one table. The MNIST classifier behind
`bridges.mnist_metric` is trained on first use and cached as
`runs/mnist_classifier.pt`; copy that file across rather than retraining it if
its numbers are to be compared with ones measured elsewhere.

Memory is unified, so a second GPU job competes with the first for the same
pool: every script here takes `runs/.gpu.lock` and refuses to start while
another holds it. Run one at a time.

Datasets download to `./data` on first use. Checkpoints, sample grids, logs
and result JSON under `runs/` are generated and are not tracked.

The one exception is `runs/<model>/config.json`, which *is* tracked. Each is
written by the trainer at launch and records every hyperparameter the run
actually used, including the ones left at their defaults. The command lines
below reproduce those configs as the code stands today; the config files are
the durable record, so check a rebuilt run against the committed one rather
than trusting that a default has not moved:

```bash
diff <(jq -S 'del(.out)' runs/<model>/config.json) \
     <(jq -S 'del(.out)' runs/<model>_rebuild/config.json)
```

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
three hours, and can be interrupted with the command below. The temperature
comes from `nvidia-smi` on NVIDIA and from the amdgpu driver's sensor on AMD
(`gpu_temp.sh`); the two are not the same measurement, so set `COOL_ABOVE` for
the machine.

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

The paper's tables are `\input` from these generated files rather than
transcribed.
