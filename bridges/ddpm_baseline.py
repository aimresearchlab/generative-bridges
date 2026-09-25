"""DDPM / improved-DDPM baseline with DDIM sampling, at matched budget.

Why this exists
---------------
The original comparison sampled its DDPM baselines at k steps by *rebuilding the
beta schedule with k entries* (`sample_ddpm.py` constructs
`DiffusionModel(args.n_steps, ...)`, and `DiffusionModel.__init__` sets
`beta = linspace(1e-4, 0.02, T)`).  At k = 2 that is a forward process with
alpha_bar_T ~= 0.98 -- one which barely noises the data at all -- driven by a
network trained against the 1000-step schedule.  That is a schedule mismatch,
not a fast sampler, and it is why the published few-step DDPM numbers are ~400
FID.

The correct few-step diffusion baseline is *step respacing*: train on the full
schedule, then sample on a subsequence of timesteps.  DDIM does exactly this and
is the standard reference for low-NFE diffusion.  Both the linear (Ho et al.)
and cosine (Nichol & Dhariwal) schedules are provided so the "simple" and
"improved" baselines differ in the way the papers intend.

Shares the UNet, optimiser, EMA, batch size, step count and FID path with the
bridge and flow-matching runs, so the only difference is the generative process.
"""

from __future__ import annotations

import argparse, json, math, os, time

import torch

from . import data as D
from .data import InfiniteSampler, load
from .ema import EMA
from .unet import build_unet

T_SCALE = 1.0  # DDPM feeds integer timesteps directly; the embedding wants O(1e3)


def make_betas(schedule: str, n: int) -> torch.Tensor:
    if schedule == "linear":
        # Ho et al., stated for T=1000 and rescaled so other T keep the same SNR path
        scale = 1000.0 / n
        return torch.linspace(scale * 1e-4, scale * 0.02, n, dtype=torch.float64)
    if schedule == "cosine":
        # Nichol & Dhariwal
        s = 0.008
        t = torch.linspace(0, n, n + 1, dtype=torch.float64) / n
        ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        return torch.clip(1 - ab[1:] / ab[:-1], 0, 0.999)
    raise ValueError(schedule)


class Diffusion:
    """epsilon-prediction DDPM with ancestral and DDIM samplers."""

    def __init__(self, n_timesteps: int, schedule: str, device):
        self.T = n_timesteps
        self.betas = make_betas(schedule, n_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def training_loss(self, model, y):
        b = y.shape[0]
        t = torch.randint(0, self.T, (b,), device=y.device)
        ab = self.abar[t].view(-1, 1, 1, 1).to(y.dtype)
        eps = torch.randn_like(y)
        x_t = ab.sqrt() * y + (1 - ab).sqrt() * eps
        pred = model(x_t, t.to(y.dtype))
        return torch.nn.functional.mse_loss(pred, eps), {}

    @torch.no_grad()
    def sample(self, model, shape, device, n_steps=50, eta=0.0, generator=None,
               x_init=None, clip=True):
        """DDIM on a respaced subsequence. eta=0 deterministic, eta=1 ancestral."""
        # evenly spaced subsequence of the TRAINED schedule -- this is the fix
        idx = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long).flip(0).tolist()
        x = x_init if x_init is not None else torch.randn(shape, device=device,
                                                          generator=generator)
        for i, t in enumerate(idx):
            ab_t = self.abar[t]
            t_prev = idx[i + 1] if i + 1 < len(idx) else None
            ab_prev = self.abar[t_prev] if t_prev is not None else torch.tensor(
                1.0, device=device, dtype=self.abar.dtype)
            tt = torch.full((x.shape[0],), float(t), device=device, dtype=x.dtype)
            eps = model(x, tt).to(torch.float64)
            xf = x.to(torch.float64)
            x0 = (xf - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip:
                x0 = x0.clamp(-1, 1)
            sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt
            if eta > 0 and t_prev is not None:
                x = x + sigma * torch.randn(x.shape, device=device, dtype=x.dtype,
                                            generator=generator)
            x = x.to(shape and torch.float32)
        return x.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out", default="./runs/mnist_ddpm")
    p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
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
    cfg = dict(vars(args))
    cfg["method"] = "ddpm"
    cfg["t_scale"] = T_SCALE
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    y_all = load(args.dataset, root=args.data_root, split="train")
    sampler = InfiniteSampler(y_all, args.batch_size, device, seed=args.seed)
    model = build_unet(args.dataset).to(device)
    print(f"ddpm ({args.schedule}) | {args.dataset} | "
          f"{sum(q.numel() for q in model.parameters())/1e6:.1f}M params", flush=True)
    ema = EMA(model, decay=args.ema_decay, warmup=True)
    if args.compile:
        model = torch.compile(model)

    diff = Diffusion(args.timesteps, args.schedule, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    t0, running, seen = time.time(), 0.0, 0

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))
        y = sampler()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = diff.training_loss(model, y)
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
