"""Per-time diagnostics for a trained bridge model.

Two quantities, both evaluated on held-out data.

1. `loss(s)` -- the regression loss  E || y - f(xi_s, s) ||^2  as a function of
   bridge time.  Its shape is informative on its own: at s -> 0 the state xi_0 = x
   is independent of y, so the best possible predictor is the dataset mean and
   the loss floor is Var[Y]; at s -> 1 the state pins down y and the loss -> 0.

2. `time_sensitivity(s)` -- how much the network's output actually depends on
   the time input.  Holding the state xi fixed, the time argument is swept
   across [0, 1] and the variation of the prediction is measured:

       sens(xi) = std_s''[ f(xi, s'') ]  /  std_pixels[ f(xi, s) ]

   A model that resolves bridge time has sens = O(1).  A model whose time
   embedding has collapsed is time-blind and gives sens ~ 0: it returns nearly
   the same endpoint estimate wherever on the bridge you tell it that it is.

The second is the direct measurement of the time-embedding hypothesis: feeding
t on a [0, 0.1) scale into a sinusoidal code built for [0, 1000) makes every
channel the same near-linear function of t.
"""

from __future__ import annotations

import argparse, json

import torch

from .bridge import BridgeConfig, GaussianBridge
from . import data as D
from .evaluate import load_model


@torch.no_grad()
def profile(ckpt: str, device="cuda", n: int = 2048, n_s: int = 21, seed: int = 0,
            n_probe: int = 17):
    model, targs, step = load_model(ckpt, device, use_ema=True)
    bridge = GaussianBridge(BridgeConfig(gamma=targs["gamma"], t_scale=targs["t_scale"]))
    y = D.load(targs["dataset"], split="test")[:n].to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(y.shape, device=device, generator=g)

    data_var = float(y.var())
    rows = []
    for s_val in torch.linspace(0.0, 1.0, n_s).tolist():
        s = torch.full((y.shape[0],), s_val, device=device)
        xi = bridge.sample_marginal(x, y, s)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xi, s * bridge.cfg.t_scale).float()
            # sweep the time argument with the state held fixed
            probes = torch.linspace(0.0, 1.0, n_probe, device=device)
            outs = torch.stack([
                model(xi, torch.full((y.shape[0],), float(p), device=device) * bridge.cfg.t_scale).float()
                for p in probes
            ])
        mse = float(((y - pred) ** 2).mean())
        sens = float(outs.std(0).mean() / (pred.std() + 1e-12))
        rows.append({"s": s_val, "mse": mse, "mse_rel_to_var": mse / data_var,
                     "time_sensitivity": sens})
    return {"ckpt": ckpt, "step": step, "dataset": targs["dataset"],
            "t_scale": targs["t_scale"], "gamma": targs["gamma"],
            "data_var": data_var, "profile": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    allr = []
    for c in a.ckpts:
        r = profile(c, n=a.n)
        allr.append(r)
        print(f"\n{c}  (step {r['step']}, t_scale {r['t_scale']}, Var[Y]={r['data_var']:.4f})")
        print(f"{'s':>6}{'MSE':>11}{'MSE/Var[Y]':>13}{'time-sens':>12}")
        print("-" * 42)
        for row in r["profile"]:
            print(f"{row['s']:>6.2f}{row['mse']:>11.5f}{row['mse_rel_to_var']:>13.4f}"
                  f"{row['time_sensitivity']:>12.4f}")
        ts = [w["time_sensitivity"] for w in r["profile"]]
        print(f"mean time-sensitivity: {sum(ts)/len(ts):.4f}")
    if a.out:
        json.dump(allr, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
