"""When does clipping a Gaussian bridge substitute for a support-matched driver?

The mask experiment found that a Gaussian bridge which overshoots and is clipped
once at the end beat both support-respecting families. The proposed explanation
was that the target carried most of its mass AT the boundary, so clipping is a
serviceable model of that atom.

That explanation makes a prediction: the advantage should disappear, and
reverse, for targets whose density VANISHES at the boundary. There, clipping
manufactures an atom the target does not have.

Gamma targets on (0, inf) vary exactly this, through the shape k alone:

    k < 1   density diverges at 0   mass piles at the boundary
    k = 1   exponential, density finite and non-zero at 0
    k > 1   density vanishes at 0   no boundary mass

Everything else is held fixed: same mean, same source (a Dirac at 0), same
sampler grid, same number of draws. Two transports are compared,

    Gaussian   xi_s = s y + gamma sqrt(s(1-s)) eps      then clipped at 0
    Gamma      xi_s = y * B,  B ~ Beta(kappa s, kappa(1-s))   support (0, y]

with E[Y | xi_s] exact in both cases from the target on a grid, so no network is
involved. We report 1-Wasserstein distance and, separately, the spurious atom
each method places at the boundary against the target's own.
"""

from __future__ import annotations

import argparse, json, math, os

import torch


def w1(a, b):
    n = min(len(a), len(b))
    return float((a.sort().values[:n] - b.sort().values[:n]).abs().mean())


class GammaTarget:
    """Gamma(k, scale) discretised on a grid, with exact conditional means."""

    def __init__(self, k, mean=1.0, n_grid=400, kappa=8.0, gamma=0.3162277660168379):
        self.k = k
        self.scale = mean / k
        # upper grid edge from a large empirical sample; Gamma has no icdf in torch
        hi = float(torch.distributions.Gamma(
            torch.tensor(k, dtype=torch.float64),
            torch.tensor(1 / self.scale, dtype=torch.float64)
        ).sample((200_000,)).quantile(0.9995))
        edges = torch.linspace(0, hi, n_grid + 1, dtype=torch.float64)
        self.y = (edges[:-1] + edges[1:]) / 2
        d = torch.distributions.Gamma(torch.tensor(k, dtype=torch.float64),
                                      torch.tensor(1 / self.scale, dtype=torch.float64))
        w = d.log_prob(self.y.clamp(min=1e-12)).exp()
        self.w = w / w.sum()
        self.kappa, self.gamma = kappa, gamma
        self.mean = float((self.w * self.y).sum())
        self.sd = float(((self.w * self.y ** 2).sum() - self.mean ** 2).sqrt())

    def sample(self, n, generator=None):
        i = torch.multinomial(self.w, n, replacement=True, generator=generator)
        return self.y[i]

    def cond_mean_gauss(self, xi, s):
        if s <= 0:
            return torch.full_like(xi, self.mean)
        v = (self.gamma * self.sd) ** 2 * s * (1 - s) + 1e-12
        mu = s * self.y                                  # source is a Dirac at 0
        logp = -0.5 * (xi[:, None] - mu[None, :]) ** 2 / v + torch.log(self.w + 1e-300)
        return (torch.softmax(logp, 1) * self.y[None, :]).sum(1)

    def cond_mean_gamma(self, xi, s):
        """xi_s = y B with B ~ Beta(kappa s, kappa(1-s)); the gamma-bridge transition."""
        if s <= 0:
            return torch.full_like(xi, self.mean)
        a, b = self.kappa * s, self.kappa * (1 - s)
        yy = self.y.clamp(min=1e-9)
        B = (xi[:, None] / yy[None, :]).clamp(1e-9, 1 - 1e-9)
        lg = torch.lgamma
        logpdf = ((a - 1) * torch.log(B) + (b - 1) * torch.log1p(-B)
                  + lg(torch.tensor(a + b)) - lg(torch.tensor(a)) - lg(torch.tensor(b))
                  - torch.log(yy)[None, :])
        logpdf = torch.where(xi[:, None] <= yy[None, :], logpdf,
                             torch.full_like(logpdf, -1e30))   # xi <= y, monotone
        logp = logpdf + torch.log(self.w + 1e-300)[None, :]
        return (torch.softmax(logp, 1) * self.y[None, :]).sum(1)


def run(t, n_steps, mode, n=200_000, seed=0, pred_noise=0.0):
    """pred_noise: relative error injected into yhat, standing in for network error.

    With an exact conditional mean the terminal state equals yhat and is always
    in range, so clipping never fires and the experiment cannot see it. A real
    sampler has a fallible yhat; this reintroduces that.
    """
    g = torch.Generator().manual_seed(seed)
    xi = torch.zeros(n, dtype=torch.float64)
    grid = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64)
    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i + 1])
        d, oms = u - s, 1 - s
        if mode.startswith("gaussian"):
            yhat = t.cond_mean_gauss(xi, s)
            if pred_noise:
                yhat = yhat + pred_noise * t.sd * torch.randn(
                    n, dtype=torch.float64, generator=g)
            mean = xi + (yhat - xi) * (d / oms)
            var = (t.gamma * t.sd) ** 2 * d * (1 - u) / oms
            prev = xi
            xi = mean if (i == n_steps - 1 or var <= 0) else \
                mean + math.sqrt(var) * torch.randn(n, dtype=torch.float64, generator=g)
            if mode == "gaussian-mono":
                # same additive noise and same support handling as the free Gaussian;
                # the ONLY change is that steps may not decrease, which is the
                # structural constraint the gamma family carries.
                xi = torch.maximum(xi, prev)
        else:
            yhat = t.cond_mean_gamma(xi, s)
            if pred_noise:
                yhat = (yhat + pred_noise * t.sd * torch.randn(
                    n, dtype=torch.float64, generator=g)).clamp(min=1e-9)
            a, b = t.kappa * d, t.kappa * (1 - u)
            if i == n_steps - 1 or b <= 0:
                xi = yhat
            else:
                A = torch._standard_gamma(torch.full((n,), a, dtype=torch.float64))
                C = torch._standard_gamma(torch.full((n,), b, dtype=torch.float64))
                xi = xi + (yhat - xi) * (A / (A + C).clamp(min=1e-30))
    return xi.clamp(min=0) if mode.startswith("gaussian") else xi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--pred_noise", type=float, default=0.15)
    p.add_argument("--kappas", type=float, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--out", default="runs/boundary_mass_1d.json")
    a = p.parse_args()

    print("Gamma targets, mean fixed at 1; only the boundary behaviour changes.")
    print(f"{a.steps} sampling steps, n={a.n}.\n")
    print(f"{'shape k':>8}{'density at 0':>15}{'W1 gauss+clip':>16}{'W1 gamma':>11}"
          f"{'gauss MONOTONE':>16}{'gamma':>11}{'mono/free':>10}{'gamma/mono':>10}")
    print("-" * 96)
    rows = []
    for k in a.shapes:
        t = GammaTarget(k)
        ref = t.sample(a.n, generator=torch.Generator().manual_seed(123))
        ref2 = t.sample(a.n, generator=torch.Generator().manual_seed(999))
        floor = w1(ref, ref2)
        xg = run(t, a.steps, "gaussian", n=a.n, pred_noise=a.pred_noise)
        wg = w1(xg, ref)
        wm = w1(run(t, a.steps, "gaussian-mono", n=a.n, pred_noise=a.pred_noise), ref)
        # best kappa for the gamma family, so the comparison is not a hyperparameter artefact
        best = None
        for kap in a.kappas:
            t.kappa = kap
            wk = w1(run(t, a.steps, "gamma", n=a.n, pred_noise=a.pred_noise), ref)
            if best is None or wk < best[0]:
                best = (wk, kap)
        wb, kbest = best
        xb = None
        eps = 1e-3 * t.mean
        atom_g = float((xg <= eps).double().mean())
        atom_t = float((ref <= eps).double().mean())
        beh = "diverges" if k < 1 else ("finite" if k == 1 else "vanishes")
        rows.append({"k": k, "w1_gauss_free": wg, "w1_gauss_mono": wm,
                     "w1_gamma": wb, "best_kappa": kbest, "floor": floor,
                     "atom_gauss": atom_g})
        print(f"{k:>8.1f}{beh:>15}{wg:>16.5f}{wm:>16.5f}{wb:>11.5f}"
              f"{wm/wg:>10.2f}{wb/wm:>10.2f}")
    print("\nmono/free isolates monotonicity: same driver, same noise, same support")
    print("handling, steps merely forbidden to decrease. gamma/mono is what remains")
    print("once monotonicity is accounted for.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
