"""Sampler correctness on a 1-D problem with an exact E[Y | xi_s].

Motivation
----------
On image data the terminal distribution depends on both the sampler and the
quality of the learned E[Y | xi_s].  To attribute the step-count behaviour we
need a case where the conditional expectation is exact, so that any residual
error is the sampler's alone.

Setup
-----
Phi = N(0, 1),  Psi = sum_k w_k N(m_k, tau^2),  x independent of y.

    xi_s = (1-s) x + s y + gamma sqrt(s(1-s)) eps

Conditional on Y = y this is N(s y, v_s) with v_s = (1-s)^2 + gamma^2 s(1-s),
so with a Gaussian-mixture target every quantity below is closed form:

    pi_k(xi_s)     ∝ w_k N(xi_s; s m_k, v_s + s^2 tau^2)
    E[Y | xi_s, k] = m_k + s tau^2 / (v_s + s^2 tau^2) * (xi_s - s m_k)
    E[Y | xi_s]    = sum_k pi_k E[Y | xi_s, k]

We then run each sampler with this exact function and compare the terminal
samples to Psi by 1-Wasserstein distance (exact in 1-D by sorting).

A correct sampler must drive W1 -> 0 as n_steps -> infinity.
"""

from __future__ import annotations

import argparse, json, math

import torch


class GaussianMixture1D:
    def __init__(self, means, weights, tau=0.15):
        self.m = torch.tensor(means, dtype=torch.float64)
        self.w = torch.tensor(weights, dtype=torch.float64)
        self.w /= self.w.sum()
        self.tau = float(tau)

    def sample(self, n, generator=None):
        k = torch.multinomial(self.w, n, replacement=True, generator=generator)
        return self.m[k] + self.tau * torch.randn(n, dtype=torch.float64, generator=generator)

    def cond_mean(self, xi, s, gamma):
        """Exact E[Y | xi_s = xi]."""
        v = (1 - s) ** 2 + gamma**2 * s * (1 - s)
        denom = v + s**2 * self.tau**2
        # log N(xi; s m_k, denom) up to a constant common to all k
        logp = -0.5 * (xi[:, None] - s * self.m[None, :]) ** 2 / denom + torch.log(self.w)[None, :]
        pi = torch.softmax(logp, dim=1)
        cond_k = self.m[None, :] + (s * self.tau**2 / denom) * (xi[:, None] - s * self.m[None, :])
        return (pi * cond_k).sum(1)


def w1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Exact 1-Wasserstein between two equal-size empirical measures."""
    return float((a.sort().values - b.sort().values).abs().mean())


def run(psi, gamma, n_steps, sampler, n=200_000, seed=0, eta=1.0):
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(n, dtype=torch.float64, generator=g)
    eps = 1e-7 if sampler == "euler_det" else 0.0
    grid = torch.linspace(0.0, 1.0 - eps, n_steps + 1, dtype=torch.float64)

    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i + 1])
        d, oms = u - s, 1.0 - s
        yhat = psi.cond_mean(xi, s, gamma)
        if sampler in ("euler", "euler_det"):
            xi = xi + (yhat - xi) * (d / oms)
            if sampler == "euler":
                xi = xi + eta * gamma * math.sqrt(d) * torch.randn(n, dtype=torch.float64, generator=g)
        elif sampler == "bridge":
            mean = xi + (yhat - xi) * (d / oms)
            var = gamma**2 * d * (1.0 - u) / oms
            last = i == n_steps - 1
            xi = mean if (last or var <= 0) else mean + eta * math.sqrt(var) * torch.randn(
                n, dtype=torch.float64, generator=g)
        else:
            raise ValueError(sampler)

    if sampler in ("euler", "euler_det"):
        xi = psi.cond_mean(xi, float(grid[-1]), gamma)  # report the endpoint estimate
    return xi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=0.3162277660168379)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--steps", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    p.add_argument("--out", default="runs/sanity_1d.json")
    args = p.parse_args()

    psi = GaussianMixture1D(means=[-2.0, 0.5, 3.0], weights=[0.3, 0.5, 0.2], tau=0.15)
    g = torch.Generator().manual_seed(123)
    ref = psi.sample(args.n, generator=g)
    print(f"target Psi: mean {ref.mean():.4f}  std {ref.std():.4f}   gamma={args.gamma:.4f}")
    # Monte-Carlo floor: W1 between two independent draws of the same size
    ref2 = psi.sample(args.n, generator=torch.Generator().manual_seed(999))
    floor = w1(ref, ref2)
    print(f"sampling-noise floor on W1 (n={args.n}): {floor:.5f}\n")

    rows = []
    print(f"{'sampler':<11}{'steps':>7}{'W1':>12}{'std':>10}{'mean':>10}")
    print("-" * 50)
    for sampler in ("bridge", "euler", "euler_det"):
        for n_steps in args.steps:
            x = run(psi, args.gamma, n_steps, sampler, n=args.n)
            rec = {"sampler": sampler, "n_steps": n_steps, "w1": w1(x, ref),
                   "std": float(x.std()), "mean": float(x.mean()), "floor": floor}
            rows.append(rec)
            print(f"{sampler:<11}{n_steps:>7}{rec['w1']:>12.5f}{rec['std']:>10.4f}{rec['mean']:>10.4f}")
        print()
    json.dump(rows, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
