"""Wright-Fisher bridge on [0,1], and a test against the real mask distribution.

Why this driver
---------------
The mask experiments isolated the cost of MONOTONICITY, not of support: a
Gaussian bridge forced merely not to decrease was 8-15x worse, far more than the
2-3x by which the gamma bridge trailed a free Gaussian. So the family worth
testing is one that is bounded on [0,1] and NOT monotone. Wright-Fisher is the
canonical such diffusion.

It sits inside Definition 2.1 of the manuscript -- a continuous, hence cadlag,
process -- but outside Section 4: its diffusion coefficient sqrt(x(1-x)) is
state dependent, so it has neither homogeneous nor independent increments and
none of the Levy machinery reaches it.

Avoiding the series expansion
-----------------------------
The WF *diffusion* has a transition density only as an infinite mixture over a
death process. The WF *model* it is the limit of does not: with population size
M, one generation is a binomial resampling

    P(i -> j) = Binom(j; M, i/M),

an exact, finite transition matrix. The bridge is then the exact Doob
h-transform of that chain,

    P(xi_u = j | xi_s = i, xi_1 = y)  =  P^a[i,j] P^b[j,y] / P^{a+b}[i,y],

with a and b the generation counts on either side. Everything below is exact for
the chain, which is the object WF is the diffusion limit of.

Both boundaries are absorbing, so the terminal law carries atoms at 0 and 1 --
matching mask data, where 77% of mass lies within 0.05 of a boundary. The
absolute-continuity requirement Psi << mu_{Z_T} of Definition 2.1 therefore
holds, which it would not for a WF with strong mutation, where the boundaries
are inaccessible.
"""

from __future__ import annotations

import argparse, glob, json, math, os, wave

import torch


# ------------------------------------------------------------------ chain

class WFChain:
    """Discrete Wright-Fisher on {0, 1/M, ..., 1} with binomial resampling."""

    def __init__(self, M=128, generations=256, dtype=torch.float64):
        self.M, self.G = M, generations
        self.x = torch.arange(M + 1, dtype=dtype) / M
        j = torch.arange(M + 1, dtype=dtype)
        logC = (torch.lgamma(torch.tensor(M + 1.0, dtype=dtype))
                - torch.lgamma(j + 1) - torch.lgamma(M - j + 1))
        p = self.x.clamp(1e-12, 1 - 1e-12)
        logP = (logC[None, :] + j[None, :] * torch.log(p)[:, None]
                + (M - j)[None, :] * torch.log1p(-p)[:, None])
        P = torch.exp(logP)
        P[0] = 0.0; P[0, 0] = 1.0                      # 0 is absorbing
        P[M] = 0.0; P[M, M] = 1.0                      # 1 is absorbing
        self.P = P / P.sum(1, keepdim=True)
        # powers P^k for k = 0..G
        self.pow = [torch.eye(M + 1, dtype=dtype)]
        for _ in range(self.G):
            self.pow.append(self.pow[-1] @ self.P)

    def gen(self, s):
        """generations elapsed by normalised time s"""
        return int(round(float(s) * self.G))


# ------------------------------------------------------------------ target

def ideal_masks(root="data/speech", n_clips=24, snr_db=5.0, sr=16000):
    import torchaudio
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))[:n_clips]
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


# ------------------------------------------------------------------ bridges

class WFBridge:
    def __init__(self, chain: WFChain, psi: torch.Tensor, i0: int):
        self.c, self.psi, self.i0 = chain, psi / psi.sum(), i0

    def cond_mean(self, idx, s):
        """E[Y | xi_s] for the bridge started at state i0, exact for the chain."""
        c = self.c
        a, b = c.gen(s), c.G - c.gen(s)
        if a == 0:
            return torch.full((len(idx),), float((self.psi * c.x).sum()),
                              dtype=torch.float64)
        num = c.pow[a][self.i0][:, None] * c.pow[b]            # p(xi_s=i | Y=y) up to Z
        post = num[idx] * self.psi[None, :]
        post = post / post.sum(1, keepdim=True).clamp(min=1e-300)
        return (post * c.x[None, :]).sum(1)

    def step(self, idx, yhat_idx, s, u, generator=None):
        """Sample xi_u | xi_s, pinned at yhat -- the exact h-transform kernel."""
        c = self.c
        a = c.gen(u) - c.gen(s)
        b = c.G - c.gen(u)
        if a <= 0:
            return idx
        fwd = c.pow[a][idx]                                    # (n, M+1)
        back = c.pow[b][:, yhat_idx].T if b > 0 else \
            torch.nn.functional.one_hot(yhat_idx, c.M + 1).double()
        w = (fwd * back).clamp(min=0)
        tot = w.sum(1, keepdim=True)
        # Both boundaries absorb, so a path already at 0 or 1 cannot reach a pin
        # that disagrees with it and the h-transform is undefined there. Those
        # states stay put, which is what absorption means; the alternative would
        # be to move a path the chain forbids from moving.
        dead = (tot.squeeze(1) <= 0)
        w = torch.where(dead[:, None], torch.zeros_like(w), w / tot.clamp(min=1e-300))
        w[dead] = 0.0
        w[dead, :] = torch.nn.functional.one_hot(idx[dead], c.M + 1).double()
        return torch.multinomial(w, 1, generator=generator).squeeze(1)


def run_wf(chain, psi, n_steps, n=200_000, seed=0, x0=0.5, pred_noise=0.0):
    g = torch.Generator().manual_seed(seed)
    br = WFBridge(chain, psi, i0=int(round(x0 * chain.M)))
    idx = torch.full((n,), br.i0, dtype=torch.long)
    grid = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64)
    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i + 1])
        yh = br.cond_mean(idx, s)
        if pred_noise:
            yh = yh + pred_noise * torch.randn(n, dtype=torch.float64, generator=g)
        yi = (yh.clamp(0, 1) * chain.M).round().long().clamp(0, chain.M)
        idx = yi if i == n_steps - 1 else br.step(idx, yi, s, u, generator=g)
    return chain.x[idx]


def run_gauss(psi, y_grid, n_steps, gamma=0.3162277660168379, n=200_000, seed=0,
              x0=0.5, pred_noise=0.0, clip_each=False):
    g = torch.Generator().manual_seed(seed)
    xi = torch.full((n,), x0, dtype=torch.float64)
    grid = torch.linspace(0, 1, n_steps + 1, dtype=torch.float64)
    mean_y = float((psi * y_grid).sum())
    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i + 1])
        d, oms = u - s, 1 - s
        if s <= 0:
            yh = torch.full((n,), mean_y, dtype=torch.float64)
        else:
            v = gamma ** 2 * s * (1 - s) + 1e-12
            mu = (1 - s) * x0 + s * y_grid
            lp = -0.5 * (xi[:, None] - mu[None, :]) ** 2 / v + torch.log(psi + 1e-300)
            yh = (torch.softmax(lp, 1) * y_grid[None, :]).sum(1)
        if pred_noise:
            yh = yh + pred_noise * torch.randn(n, dtype=torch.float64, generator=g)
        mean = xi + (yh - xi) * (d / oms)
        var = gamma ** 2 * d * (1 - u) / oms
        xi = mean if (i == n_steps - 1 or var <= 0) else \
            mean + math.sqrt(var) * torch.randn(n, dtype=torch.float64, generator=g)
        if clip_each:
            xi = xi.clamp(0, 1)
    return xi.clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--generations", type=int, default=256)
    p.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    p.add_argument("--n", type=int, default=100_000)
    p.add_argument("--pred_noise", type=float, default=0.05)
    p.add_argument("--out", default="runs/wf_bridge.json")
    a = p.parse_args()

    m = ideal_masks()
    chain = WFChain(M=a.M, generations=a.generations)
    psi = torch.histc(m.float(), bins=a.M + 1, min=0, max=1).double()
    psi = psi / psi.sum()
    ref_i = torch.multinomial(psi, a.n, replacement=True,
                              generator=torch.Generator().manual_seed(123))
    ref = chain.x[ref_i]
    ref2 = chain.x[torch.multinomial(psi, a.n, replacement=True,
                                     generator=torch.Generator().manual_seed(999))]
    floor = w1(ref, ref2)
    print(f"real ideal ratio masks, {m.numel():,} TF bins; "
          f"{100*float(((m<0.05)|(m>0.95)).double().mean()):.0f}% within 0.05 of a boundary")
    print(f"WF chain M={a.M}, {a.generations} generations, both boundaries absorbing")
    print(f"prediction noise {a.pred_noise}, W1 floor {floor:.5f}\n")

    print(f"{'steps':>6}{'gauss clip-end':>17}{'gauss clip-each':>18}{'Wright-Fisher':>16}{'WF/gauss':>10}")
    print("-" * 67)
    rows = []
    for ns in a.steps:
        wg = w1(run_gauss(psi, chain.x, ns, n=a.n, pred_noise=a.pred_noise), ref)
        wc = w1(run_gauss(psi, chain.x, ns, n=a.n, pred_noise=a.pred_noise,
                          clip_each=True), ref)
        ww = w1(run_wf(chain, psi, ns, n=a.n, pred_noise=a.pred_noise), ref)
        rows.append({"n_steps": ns, "gauss_clip_end": wg, "gauss_clip_each": wc,
                     "wright_fisher": ww, "floor": floor})
        print(f"{ns:>6}{wg:>17.5f}{wc:>18.5f}{ww:>16.5f}{ww/wg:>10.2f}")
    print("\nWF/gauss < 1 means the bounded, non-monotone driver wins.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
