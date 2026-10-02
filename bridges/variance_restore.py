"""Restore the posterior variance the point-mass step discards, and measure it.

The exact bridge step is a mixture over the posterior pi_s(dy) of the terminal
value. The `bridge` sampler replaces pi_s by a point mass at its mean, and the
law of total variance says what that costs:

    Var[xi_u | xi_s] = gamma^2 (u-s)(1-u)/(1-s) + ((u-s)/(1-s))^2 Var[Y | xi_s]

the second term being dropped. Proposition 2.13 is about exactly that term. It
is a supermartingale in the Loewner order, it vanishes at the terminal time, and
at s = 0 it equals Var[Y], since xi_0 is independent of Y. Those two boundary
values are what the estimated profile is checked against below.

No new network is needed: the training objective already estimates the quantity,
because E||y - f(xi_s,s)||^2 -> tr Var[Y|xi_s]/n once f attains the L^2
projection. This script measures that profile on held-out data, feeds an
isotropic version back into the sampler, and reports what changes.

The prediction is differential. `bridge` makes the point-mass approximation and
should gain coverage; `euler` discretises the SDE whose drift is already the
conditional mean, makes no such approximation, and should be unaffected. A
correction that moved both equally would indicate something other than the
mechanism claimed here.
"""

from __future__ import annotations

import argparse, json, os

import numpy as np
import torch

from . import data as D
from .evaluate import generate, load_model, make_scheme
from .fid import InceptionStats, activations, frechet_distance, real_stats, stats_from_tensor
from .metrics import precision_recall


@torch.no_grad()
def variance_profile(model, bridge, dataset, device, n_grid=41, n_samples=2048,
                     batch=256, seed=11):
    """tr Var[Y | xi_s] / n against s, estimated by the regression loss."""
    y_all = D.load(dataset, root="./data", split="test")
    g = torch.Generator().manual_seed(seed)
    Y = y_all[torch.randperm(len(y_all), generator=g)[:n_samples]].to(device)
    X = torch.randn(Y.shape, generator=g).to(device)
    E = torch.randn(Y.shape, generator=g).to(device)
    ss, vs = [], []
    for s in np.linspace(0.0, 1.0, n_grid):
        s_t = torch.full((Y.shape[0],), float(s), device=device)
        xi = bridge.sample_marginal(X, Y, s_t, noise=E)
        tot = 0.0
        for i in range(0, len(xi), batch):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yh = bridge.endpoint(model, xi[i:i + batch], float(s))
            tot += float(((yh.float() - Y[i:i + batch]) ** 2).sum())
        ss.append(float(s)); vs.append(tot / Y.numel())
    return np.array(ss), np.array(vs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/cifar10_drift_g0.3162277660168379/ckpt_040000.pt")
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--steps", type=int, nargs="+", default=[10, 50, 200])
    p.add_argument("--scale", type=float, default=1.0,
                   help="multiplier on the restored variance; 0 reproduces the "
                        "point-mass sampler, 1 restores the estimate in full")
    p.add_argument("--variants", nargs="+",
                   default=["bridge", "bridge+var", "euler", "euler+var"],
                   help="euler+var is a control: the Euler sampler makes no "
                        "point-mass approximation, so restoring the variance "
                        "must leave it unchanged.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--out", default="runs/variance_restore.json")
    a = p.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, targs, step = load_model(a.ckpt, dev)
    dataset = targs["dataset"]
    bridge, _ = make_scheme(targs)
    if a.compile:
        model = torch.compile(model)

    ss, vs = variance_profile(model, bridge, dataset, dev)
    real = D.load(dataset, root="./data", split="train")
    var_y = float(real.var())
    print(f"variance profile: v(0) = {vs[0]:.5f} against Var[Y] = {var_y:.5f} "
          f"(ratio {vs[0]/var_y:.3f}); v(1) = {vs[-1]:.2e}", flush=True)
    print(f"monotone decreasing: {bool(np.all(np.diff(vs) <= 1e-6))}", flush=True)

    def profile(s):
        return a.scale * float(np.interp(s, ss, vs))

    stats = InceptionStats(dev)
    mu_r, sig_r = real_stats(dataset, "train", dev, stats=stats)
    real_act = activations(real, dev, batch_size=250, stats=stats, max_n=a.n_samples)

    rows = []
    for n in a.steps:
        for tag in a.variants:
            sampler = tag.replace("+var", "")
            vp = profile if tag.endswith("+var") else None
            g = torch.Generator(device=dev).manual_seed(0)
            out = []
            for i in range(0, a.n_samples, a.batch_size):
                b = min(a.batch_size, a.n_samples - i)
                x0 = torch.randn(b, D.channels(dataset), D.IMAGE_SIZE, D.IMAGE_SIZE,
                                 device=dev, generator=g)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out.append(bridge.sample(model, x0.shape, dev, n_steps=n,
                                             sampler=sampler, eta=1.0, generator=g,
                                             x_init=x0, var_profile=vp).float().cpu())
            x = torch.cat(out)
            mu, sig = stats_from_tensor(x, dev, batch_size=250, stats=stats)
            fa = activations(x, dev, batch_size=250, stats=stats, max_n=a.n_samples)
            r = {"sampler": tag, "n_steps": n,
                 "fid_train": frechet_distance(mu, sig, mu_r, sig_r),
                 "pixel_std": float(x.std()), "real_std": float(real.std())}
            r.update(precision_recall(real_act, fa))
            rows.append(r)
            print(f"  {tag:>12} {n:>4} steps: FID {r['fid_train']:6.2f}  "
                  f"P {r['precision']:.3f}  R {r['recall']:.3f}  "
                  f"cov {r['coverage']:.3f}  std {r['pixel_std']:.4f}", flush=True)
            del fa, x
            torch.cuda.empty_cache()

    json.dump({"profile_s": ss.tolist(), "profile_v": vs.tolist(),
               "var_y": var_y, "scale": a.scale, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
