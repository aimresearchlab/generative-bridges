"""Sample from a trained bridge and measure FID as a function of sampling steps.

This is the experiment the referees asked about: does quality improve
monotonically as the discretisation is refined?  The sweep is over
(sampler, n_steps) so that the effect of the *sampler* can be separated from
the effect of the *model*.

    euler_det   deterministic Euler  -- what the published numbers used
    euler       Euler-Maruyama       -- Algorithm 2 of the paper
    bridge      exact bridge posterior

Network function evaluations (NFE) equal n_steps for every sampler, so the
columns are directly comparable at equal cost.
"""

from __future__ import annotations

import argparse, itertools, json, os, time

import numpy as np
import torch

from .bridge import BridgeConfig, GaussianBridge
from . import data as D
from .fid import InceptionStats, frechet_distance, real_stats, stats_from_tensor
from .unet import build_unet


def load_model(ckpt_path: str, device, use_ema: bool = True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]
    model = build_unet(args["dataset"],
                       learn_sigma=args.get("method") == "improved_ddpm")
    model.load_state_dict(ck["ema" if use_ema else "model"])
    return model.to(device).eval(), args, ck["step"]


def make_scheme(targs):
    """Return (scheme, samplers) for either a bridge or a flow-matching checkpoint."""
    if targs.get("method") == "flow_matching":
        from .flowmatch import FlowMatching
        return FlowMatching(t_scale=targs["t_scale"]), ["flow_euler"]
    if targs.get("method") == "improved_ddpm":
        from .improved_ddpm import ImprovedDiffusion
        d = ImprovedDiffusion(targs["timesteps"], targs["schedule"], "cuda")
        return d, ["improved_ancestral"]
    if targs.get("method") == "ddpm":
        from .ddpm_baseline import Diffusion
        d = Diffusion(targs["timesteps"], targs["schedule"], "cuda")
        return d, ["ddim", "ddpm_ancestral"]   # default; an explicit --samplers wins
    return GaussianBridge(BridgeConfig(gamma=targs["gamma"], t_scale=targs["t_scale"])), None


@torch.no_grad()
def generate(model, bridge, n: int, dataset: str, device, n_steps: int,
             sampler: str, eta: float, batch_size: int, seed: int,
             amp: bool = True) -> torch.Tensor:
    c = D.channels(dataset)
    out = torch.empty(n, c, D.IMAGE_SIZE, D.IMAGE_SIZE, dtype=torch.float32)
    g = torch.Generator(device=device).manual_seed(seed)
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        x0 = torch.randn(b, c, D.IMAGE_SIZE, D.IMAGE_SIZE, device=device, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            if sampler == "improved_ancestral":
                s = bridge.sample(model, (b, c, D.IMAGE_SIZE, D.IMAGE_SIZE), device,
                                  n_steps=n_steps, generator=g, x_init=x0)
            elif sampler in ("ddim", "ddpm_ancestral"):
                s = bridge.sample(model, (b, c, D.IMAGE_SIZE, D.IMAGE_SIZE), device,
                                  n_steps=n_steps,
                                  eta=0.0 if sampler == "ddim" else 1.0,
                                  generator=g, x_init=x0)
            elif sampler == "flow_euler":
                s = bridge.sample(model, (b, c, D.IMAGE_SIZE, D.IMAGE_SIZE), device,
                                  n_steps=n_steps, generator=g, x_init=x0)
            else:
                s = bridge.sample(model, (b, c, D.IMAGE_SIZE, D.IMAGE_SIZE), device,
                                  n_steps=n_steps, sampler=sampler, eta=eta,
                                  generator=g, x_init=x0)
        out[done:done + b] = s.float().cpu()
        done += b
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="json file for results (default: next to ckpt)")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--samplers", nargs="+", default=["bridge", "euler", "euler_det"])
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw_weights", action="store_true", help="use non-EMA weights")
    p.add_argument("--save_grid", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    # did the caller name samplers, or are we on the default list?
    args.samplers_explicit = any(a.startswith("--samplers") for a in __import__("sys").argv)

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, targs, step = load_model(args.ckpt, device, use_ema=not args.raw_weights)
    dataset = targs["dataset"]
    bridge, forced = make_scheme(targs)
    if forced is not None and not args.samplers_explicit:
        args.samplers = forced
    print(f"loaded {args.ckpt} (step {step}, dataset {dataset}, "
          f"method {targs.get('method', 'bridge')}, gamma {targs.get('gamma', 0.0):.4f}, "
          f"t_scale {targs['t_scale']})", flush=True)
    if args.compile:
        model = torch.compile(model)

    stats = InceptionStats(device)
    mu_tr, sig_tr = real_stats(dataset, "train", device, stats=stats)
    mu_te, sig_te = real_stats(dataset, "test", device, stats=stats)
    print("real statistics ready", flush=True)

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt),
                                        f"fid_sweep_step{step}.json")
    results = []
    if os.path.exists(out_path):
        results = json.load(open(out_path))
    done = {(r["sampler"], r["n_steps"]) for r in results}

    # step-major: all samplers at each step count, cheapest first, so a usable
    # comparison exists long before the expensive tail finishes
    for n_steps, sampler in itertools.product(sorted(args.steps), args.samplers):
        if (sampler, n_steps) in done:
            continue
        t0 = time.time()
        x = generate(model, bridge, args.n_samples, dataset, device, n_steps,
                     sampler, args.eta, args.batch_size, args.seed)
        gen_s = time.time() - t0
        mu, sig = stats_from_tensor(x, device, batch_size=250, stats=stats)
        rec = {
            "sampler": sampler, "n_steps": n_steps, "nfe": n_steps,
            "fid_train": frechet_distance(mu, sig, mu_tr, sig_tr),
            "fid_test": frechet_distance(mu, sig, mu_te, sig_te),
            "gen_seconds": gen_s,
            "imgs_per_s": args.n_samples / gen_s,
            "n_samples": args.n_samples, "eta": args.eta, "ckpt_step": step,
        }
        results.append(rec)
        json.dump(results, open(out_path, "w"), indent=2)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in rec.items()}), flush=True)

        if args.save_grid:
            from torchvision.utils import save_image
            gdir = os.path.join(os.path.dirname(out_path), "grids")
            os.makedirs(gdir, exist_ok=True)
            save_image((x[:64].clamp(-1, 1) + 1) / 2,
                       os.path.join(gdir, f"{sampler}_{n_steps:04d}.png"), nrow=8)

    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
