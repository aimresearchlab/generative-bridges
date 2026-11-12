"""Does respecting the [0,1] support of a ratio mask actually buy anything?

Measuring that a Gaussian bridge leaves the feasible set is not enough: if
projecting back is free, the violation does not matter. This tests the
consequence directly, on the empirical distribution of ideal ratio masks from
real speech.

Setup. Treat one time-frequency bin at a time, so the target law Psi is the
empirical mask marginal and everything is one-dimensional. The source is the
all-ones mask (suppress nothing), which is where an enhancement system starts.
With Psi on a grid, E[Y | xi_s] is available in closed form by Bayes for both
bridge families, so no network is involved and every residual error belongs to
the transport and its projection.

    Gaussian   xi_s = (1-s) + s y + gamma sqrt(s(1-s)) eps        support R
    Beta       xi_s = 1 - (1-y) B,  B ~ Beta(kappa s, kappa(1-s)) support [0,1]

The Beta family is the gamma-bridge transition of the Levy note; a convex
combination of two points in [0,1] cannot leave [0,1], so the constraint holds
by construction rather than by projection.

Three samplers are compared against Psi in 1-Wasserstein distance:
  gaussian-clip-each   clip to [0,1] after every step, as a real sampler must
  gaussian-clip-end    clip only the final sample
  beta                 no clipping anywhere
"""

from __future__ import annotations

import argparse, glob, json, math, os, wave

import torch


# ----------------------------------------------------------------- data

def ideal_masks(root="data/speech", n_clips=24, snr_db=5.0, sr=16000):
    import torchaudio
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))[:n_clips]
    if not files:
        raise RuntimeError(f"no wavs under {root}")
    out = []
    for f in files:
        with wave.open(f, "rb") as wf:
            fs, raw = wf.getframerate(), wf.readframes(wf.getnframes())
        w = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        if fs != sr:
            w = torchaudio.functional.resample(w, fs, sr)
        win = torch.hann_window(512)
        Sc = torch.stft(w, 512, 128, window=win, return_complex=True)
        n = torch.randn_like(w) * (w.pow(2).mean().sqrt() * 10 ** (-snr_db / 20))
        Sn = torch.stft(w + n, 512, 128, window=win, return_complex=True)
        out.append((Sc.abs() / Sn.abs().clamp(min=1e-8)).clamp(0, 1).flatten())
    return torch.cat(out).double()


def w1(a, b):
    n = min(len(a), len(b))
    return float((a.sort().values[:n] - b.sort().values[:n]).abs().mean())


# ------------------------------------------------- exact conditional means

class MaskTarget:
    """Psi on a grid, with E[Y | xi_s] in closed form for both bridge families."""

    def __init__(self, samples, n_grid=256, kappa=8.0, gamma=0.3162277660168379):
        edges = torch.linspace(0, 1, n_grid + 1, dtype=torch.float64)
        self.y = ((edges[:-1] + edges[1:]) / 2)
        w = torch.histc(samples.float(), bins=n_grid, min=0, max=1).double()
        self.w = w / w.sum()
        self.kappa, self.gamma = kappa, gamma
        self.lgam = torch.lgamma

    def sample(self, n, generator=None):
        i = torch.multinomial(self.w, n, replacement=True, generator=generator)
        return self.y[i]

    # --- Gaussian family
    def cond_mean_gauss(self, xi, s):
        if s <= 0:
            return torch.full_like(xi, float((self.w * self.y).sum()))
        v = self.gamma ** 2 * s * (1 - s) + 1e-12
        mu = (1 - s) + s * self.y                       # source is the all-ones mask
        logp = -0.5 * (xi[:, None] - mu[None, :]) ** 2 / v + torch.log(self.w + 1e-300)
        pi = torch.softmax(logp, dim=1)
        return (pi * self.y[None, :]).sum(1)

    # --- transformed-Brownian family:  Z_t = g(W_t), g = logistic.
    # Definition 2.1 asks only that the driver be cadlag, so g(W) qualifies. Since
    # g is a bijection, conditioning Z on its endpoints is conditioning W on their
    # preimages, and the bridge is the image of the Gaussian bridge under g. The
    # support is exact and the paths are NOT monotone, which is what the Beta
    # family gives up.
    def cond_mean_logit(self, xi, s, x0=1.0 - 1e-3):
        if s <= 0:
            return torch.full_like(xi, float((self.w * self.y).sum()))
        eps = 1e-6
        yl = torch.logit(self.y.clamp(eps, 1 - eps))
        xl = math.log(x0 / (1 - x0))
        el = torch.logit(xi.clamp(eps, 1 - eps))
        v = self.gamma ** 2 * s * (1 - s) + 1e-12
        mu = (1 - s) * xl + s * yl
        logp = -0.5 * (el[:, None] - mu[None, :]) ** 2 / v + torch.log(self.w + 1e-300)
        pi = torch.softmax(logp, dim=1)
        return (pi * yl[None, :]).sum(1)      # returns E[g^-1(Y) | xi], as required

    # --- Beta family:  xi_s = 1 - (1-y) B,  B ~ Beta(kappa s, kappa(1-s))
    def cond_mean_beta(self, xi, s):
        # xi_0 = x carries no information about Y, so the posterior is the prior.
        # The density is degenerate there: B = (1-xi)/(1-y) = 0 for every y, and
        # the surviving gap^{-a} factor would spuriously prefer y near 1.
        if s <= 0:
            return torch.full_like(xi, float((self.w * self.y).sum()))
        a, b = self.kappa * s, self.kappa * (1 - s)
        gap = (1 - self.y).clamp(min=1e-9)
        B = ((1 - xi)[:, None] / gap[None, :]).clamp(1e-9, 1 - 1e-9)
        logpdf = ((a - 1) * torch.log(B) + (b - 1) * torch.log1p(-B)
                  + self.lgam(torch.tensor(a + b)) - self.lgam(torch.tensor(a))
                  - self.lgam(torch.tensor(b)) - torch.log(gap)[None, :])
        # states outside [1-gap, 1] are impossible under this family
        logpdf = torch.where(((1 - xi)[:, None] <= gap[None, :]), logpdf,
                             torch.full_like(logpdf, -1e30))
        logp = logpdf + torch.log(self.w + 1e-300)[None, :]
        pi = torch.softmax(logp, dim=1)
        return (pi * self.y[None, :]).sum(1)


# ------------------------------------------------------------- samplers

def run(target, n_steps, mode, n=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    xi = torch.full((n,), 1.0 - 1e-3 if mode == "logit" else 1.0,
                    dtype=torch.float64)   # all-ones mask; logit needs an interior start
    grid = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64)
    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i + 1])
        d, oms = u - s, 1 - s
        if mode.startswith("gaussian"):
            yhat = target.cond_mean_gauss(xi, s)
            mean = xi + (yhat - xi) * (d / oms)
            var = target.gamma ** 2 * d * (1 - u) / oms
            xi = mean if (i == n_steps - 1 or var <= 0) else \
                mean + math.sqrt(var) * torch.randn(n, dtype=torch.float64, generator=g)
            if mode == "gaussian-clip-each":
                xi = xi.clamp(0, 1)
        elif mode == "logit":
            # exact Gaussian bridge posterior, taken in g-inverse space
            eps = 1e-6
            yl = target.cond_mean_logit(xi, s)
            el = torch.logit(xi.clamp(eps, 1 - eps))
            mean = el + (yl - el) * (d / oms)
            var = target.gamma ** 2 * d * (1 - u) / oms
            el = mean if (i == n_steps - 1 or var <= 0) else \
                mean + math.sqrt(var) * torch.randn(n, dtype=torch.float64, generator=g)
            xi = torch.sigmoid(el)
        else:
            yhat = target.cond_mean_beta(xi, s)
            a, b = target.kappa * d, target.kappa * (1 - u)
            if i == n_steps - 1 or b <= 0:
                xi = yhat
            else:
                B = torch._standard_gamma(torch.full((n,), a, dtype=torch.float64))
                C = torch._standard_gamma(torch.full((n,), b, dtype=torch.float64))
                xi = xi + (yhat - xi) * (B / (B + C).clamp(min=1e-30))
    return xi.clamp(0, 1) if mode == "gaussian-clip-end" else xi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--kappa", type=float, default=8.0)
    p.add_argument("--out", default="runs/mask_bridge_1d.json")
    a = p.parse_args()

    m = ideal_masks()
    tgt = MaskTarget(m, kappa=a.kappa)
    ref = tgt.sample(a.n, generator=torch.Generator().manual_seed(123))
    ref2 = tgt.sample(a.n, generator=torch.Generator().manual_seed(999))
    floor = w1(ref, ref2)
    print(f"ideal ratio masks from {m.numel():,} real TF bins; "
          f"{100*float(((m<0.05)|(m>0.95)).float().mean()):.0f}% of mass within 0.05 of a boundary")
    print(f"1-Wasserstein floor between two draws of n={a.n}: {floor:.5f}\n")

    modes = ["gaussian-clip-each", "gaussian-clip-end", "beta", "logit"]
    print(f"{'steps':>6}" + "".join(f"{x:>20}" for x in modes))
    print("-" * (6 + 20 * len(modes)))
    rows = []
    for ns in a.steps:
        vals = {}
        for mode in modes:
            x = run(tgt, ns, mode, n=a.n)
            vals[mode] = w1(x, ref)
        rows.append({"n_steps": ns, **vals, "floor": floor})
        print(f"{ns:>6}" + "".join(f"{vals[k]:>20.5f}" for k in modes))
    best = min(rows, key=lambda r: r["beta"])
    g_best = min(r["gaussian-clip-each"] for r in rows)
    print(f"\nbest Beta {best['beta']:.5f} vs best Gaussian(clip-each) {g_best:.5f}"
          f"   ratio {g_best/best['beta']:.2f}x   (floor {floor:.5f})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
