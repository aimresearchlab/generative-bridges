# `bridges/` — clean re-implementation

A from-scratch implementation of the Gaussian random-bridge generative scheme,
written to settle whether the sampling-step plateau reported in the paper is a
property of the method or an artefact of the original implementation.

Nothing here is inherited from the original scripts; every component was
re-derived or re-written so that it can be audited independently.

## Layout

| file | contents |
|---|---|
| `bridge.py` | the scheme itself: marginals, training objective, three samplers. Full derivation in the module docstring. |
| `unet.py`   | ADM-style UNet, written out in full (not vendored). |
| `data.py`   | dataset loading under one pixel convention: float `[-1,1]`, `32x32`, both datasets. |
| `ema.py`    | EMA of weights (the original kept none). |
| `fid.py`    | FID with cached real statistics and no PNG round-trip. |
| `train.py`  | training loop. |
| `evaluate.py` | the FID-vs-steps sweep, per sampler. |

## Parameterisation

The bridge marginal, conditional on endpoints `(x, y)` and written in normalised
time `s = t/T`:

    xi_s = (1-s) x + s y + gamma sqrt(s(1-s)) eps,   eps ~ N(0, I)

with `gamma^2 = sigma^2 T`. **`sigma` and `T` are not separately identifiable** —
they enter training and sampling only through `gamma`. The published setting
`sigma = 1, T = 0.1` is `gamma = 0.3162`. The code therefore exposes `gamma`
and nothing else, which also means the "tune `T`" of Section 3 and the "tuning
`sigma` is left to future work" of Section 2.4 are one sensitivity analysis,
not two.

## Samplers

All three cost exactly `n_steps` network evaluations, so they are comparable at
equal cost.

- **`euler`** — Euler–Maruyama on `d xi = (E[Y|xi]-xi)/(1-s) ds + gamma dB`.
  This is Algorithm 2 of the paper.
- **`euler_det`** — the same with the diffusion term deleted. This is what the
  original repository actually evaluated (`sample_bridges.py` passes
  `deterministic=True`). It is **biased**: deleting `gamma dB` without adding
  the score correction `-(gamma^2/2) grad log p_s` does not produce the
  probability-flow ODE, so refining the grid converges to something that is not
  the target distribution.
- **`bridge`** — exact Brownian-bridge posterior step. Given the state `xi_s`
  and predicted endpoint `yhat`,

      xi_u ~ N( xi_s + (u-s)/(1-s) (yhat - xi_s),  gamma^2 (u-s)(1-u)/(1-s) )

  exactly, for any `s < u <= 1`. The variance vanishes at `u = 1`, so the
  terminal point is reached exactly: no `1/(T-t)` singularity and no epsilon
  offset. To first order in `(u-s)` it agrees with `euler`.

## Reproducing the original behaviour

`--t_scale 0.1` feeds the network raw `t in [0, 0.1)` as the original did, which
makes the sinusoidal time embedding degenerate (every channel becomes the same
near-linear function of `t`; the whole code spans 0.0998 against 2.0 for a
DDPM-scale input). The default `--t_scale 1000` restores the dynamic range the
embedding was designed for. Combined with `--samplers euler_det`, this
reproduces the original configuration inside the clean codebase.

## Usage

    python -m bridges.train    --dataset mnist --steps 40000 --compile --out runs/mnist
    python -m bridges.evaluate --ckpt runs/mnist/ckpt_040000.pt --n_samples 50000
