"""Score the SAME samples through both evaluation pipelines.

Holds the model, the sampler, the sample count and the random seed fixed, and
varies only the measurement. If FID drifts with step count under the original
pipeline but not under the clean one, the published degradation is an artefact
of the measurement rather than a property of the method.
"""

from __future__ import annotations

import argparse, json

import torch

from . import data as D
from .bridge import BridgeConfig, GaussianBridge
from .evaluate import generate, load_model
from .fid import InceptionStats, frechet_distance, real_stats, stats_from_tensor
from .legacy_fid import legacy_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, nargs="+", default=[2, 10, 100, 1000])
    p.add_argument("--sampler", default="euler_det")
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/pipeline_ab.json")
    a = p.parse_args()

    device = "cuda"
    model, targs, step = load_model(a.ckpt, device, use_ema=True)
    dataset = targs["dataset"]
    ch = D.channels(dataset)
    bridge = GaussianBridge(BridgeConfig(gamma=targs["gamma"], t_scale=targs["t_scale"]))
    stats = InceptionStats(device)

    # clean reference: full train split, single clamped affine, no per-image rescale
    mu_c, sig_c = real_stats(dataset, "train", device, stats=stats)
    # legacy reference: the original min-max'd its reals too
    real = D.load(dataset, split="train")[: a.n_samples]
    mu_l, sig_l = legacy_stats(real, device, ch, real=True, stats=stats)
    print(f"{a.ckpt}  sampler={a.sampler}  n={a.n_samples}", flush=True)
    print(f"{'NFE':>6}{'clean FID':>12}{'legacy FID':>13}{'clean delta':>13}{'legacy delta':>14}",
          flush=True)

    rows, prev_c, prev_l = [], None, None
    for n_steps in sorted(a.steps):
        x = generate(model, bridge, a.n_samples, dataset, device, n_steps,
                     a.sampler, 1.0, a.batch_size, a.seed)
        fc = frechet_distance(*stats_from_tensor(x, device, batch_size=250, stats=stats),
                              mu_c, sig_c)
        fl = frechet_distance(*legacy_stats(x, device, ch, real=False, stats=stats),
                              mu_l, sig_l)
        dc = "" if prev_c is None else f"{fc - prev_c:+.3f}"
        dl = "" if prev_l is None else f"{fl - prev_l:+.3f}"
        print(f"{n_steps:>6}{fc:>12.3f}{fl:>13.3f}{dc:>13}{dl:>14}", flush=True)
        rows.append({"n_steps": n_steps, "fid_clean": fc, "fid_legacy": fl,
                     "sampler": a.sampler, "n_samples": a.n_samples,
                     "t_scale": targs["t_scale"], "ckpt": a.ckpt})
        prev_c, prev_l = fc, fl
        json.dump(rows, open(a.out, "w"), indent=2)

    print(f"\nwrote {a.out}")
    cl = [r["fid_clean"] for r in rows]
    lg = [r["fid_legacy"] for r in rows]
    print(f"clean : {'improves throughout' if cl == sorted(cl, reverse=True) else 'NON-MONOTONE'}"
          f"   {cl[0]:.2f} -> {cl[-1]:.2f}")
    print(f"legacy: {'improves throughout' if lg == sorted(lg, reverse=True) else 'NON-MONOTONE'}"
          f"   {lg[0]:.2f} -> {lg[-1]:.2f}")


if __name__ == "__main__":
    main()
