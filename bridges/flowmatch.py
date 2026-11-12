"""Conditional flow matching, at matched budget, as a modern low-step baseline.

Why this is the right control
-----------------------------
Written in normalised time, the bridge marginal is

    xi_s = (1-s) x + s y + gamma sqrt(s(1-s)) eps

and flow matching with the linear (rectified) path is the same interpolant at
gamma = 0:

    z_s  = (1-s) x + s y

The two also share a regression target up to a change of variable: the bridge
predicts the endpoint yhat ~= E[Y | xi_s], flow matching predicts the velocity
v ~= E[Y - X | z_s], and along the linear path

    v = (yhat - z_s) / (1 - s).

So with the *same* UNet, optimiser, EMA, batch size and step count, this
isolates exactly two things: the bridge's stochastic latent (gamma > 0) and its
endpoint parameterisation.  Any difference in the few-step regime is
attributable to those and not to architecture or training budget.

Sampling is Euler on  dz/ds = v(z_s, s)  from s = 0 to 1, so NFE = n_steps and
the cost is directly comparable to the bridge samplers.
"""

from __future__ import annotations

import argparse, json, os, time

import torch

from . import data as D
from .data import InfiniteSampler, load
from .ema import EMA
from .unet import build_unet

T_SCALE = 1000.0  # same time-embedding convention as the bridge


class FlowMatching:
    """Linear-path conditional flow matching (a.k.a. rectified flow / CondOT)."""

    def __init__(self, t_scale: float = T_SCALE):
        self.t_scale = t_scale

    def training_loss(self, model, y, x=None):
        if x is None:
            x = torch.randn_like(y)                    # independent coupling
        s = torch.rand(y.shape[0], device=y.device, dtype=y.dtype)
        s_ = s.view(-1, *([1] * (y.ndim - 1)))
        z = (1.0 - s_) * x + s_ * y
        target = y - x                                  # the conditional velocity
        pred = model(z, s * self.t_scale)
        return torch.nn.functional.mse_loss(pred, target), {}

    @torch.no_grad()
    def sample(self, model, shape, device, n_steps=50, generator=None, x_init=None):
        z = x_init if x_init is not None else torch.randn(shape, device=device, generator=generator)
        grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=torch.float64)
        for i in range(n_steps):
            s = float(grid[i])
            d = float(grid[i + 1]) - s
            t_in = torch.full((z.shape[0],), s * self.t_scale, device=device, dtype=z.dtype)
            z = z + model(z, t_in) * d
        return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out", default="./runs/mnist_flow")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="matched to bridges.train; AdamW would otherwise default to 0.01")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(vars(args)); cfg["method"] = "flow_matching"; cfg["t_scale"] = T_SCALE
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    y_all = load(args.dataset, root=args.data_root, split="train")
    sampler = InfiniteSampler(y_all, args.batch_size, device, seed=args.seed)
    model = build_unet(args.dataset).to(device)
    print(f"flow matching | {args.dataset} | {sum(q.numel() for q in model.parameters())/1e6:.1f}M params",
          flush=True)
    ema = EMA(model, decay=args.ema_decay, warmup=True)
    if args.compile:
        model = torch.compile(model)

    fm = FlowMatching()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))
    t0, running, seen = time.time(), 0.0, 0

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))
        y = sampler()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = fm.training_loss(model, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model._orig_mod if hasattr(model, "_orig_mod") else model, step)

        running += loss.item(); seen += 1
        if step % args.log_every == 0:
            rec = {"step": step, "loss": running / seen, "grad_norm": float(gn),
                   "it_per_s": step / (time.time() - t0)}
            print(json.dumps(rec), flush=True)
            with open(os.path.join(args.out, "train_log.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            running, seen = 0.0, 0

        if step % args.ckpt_every == 0 or step == args.steps:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model": raw.state_dict(), "ema": ema.state_dict(),
                        "args": cfg}, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            print(f"saved ckpt_{step:06d}.pt", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
