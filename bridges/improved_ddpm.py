"""Improved DDPM (Nichol & Dhariwal, ICML 2021) -- the baseline the original
comparison actually ran, and which was missing from ours.

Why this is separate from ddpm_baseline.py
------------------------------------------
That module is DDPM with a cosine schedule but a *fixed* reverse variance. The
defining feature of improved DDPM is a **learned** reverse variance trained with
a hybrid objective, and that is precisely what buys good samples at few steps --
it is the paper's headline claim. A cosine schedule alone is a weakened
stand-in, so comparing against it would repeat the weak-baseline objection we
are levelling at the original manuscript.

The original's own configuration confirms this is what it ran: its README passes
`MODEL_FLAGS="--image_size 28 --learn_sigma True"` to OpenAI's improved-diffusion
scripts. (Its Section 3 cites this baseline as [13], Song et al.'s score-SDE
paper, but describes and runs [13a], Nichol & Dhariwal -- a citation slip worth
fixing in the revision.)

What is implemented
-------------------
* the network emits 2C channels: the noise estimate and a per-pixel coefficient
  v, mapped to a variance interpolated in log space between the two natural
  bounds, Sigma = exp(v log beta_t + (1-v) log beta-tilde_t);
* the hybrid loss L_simple + lambda * L_vlb with lambda = 0.001, the VLB term
  taking a stop-gradient through the mean so it trains only the variance;
* ancestral sampling that uses the learned variance -- the sampler the method
  exists to enable.

Shares the UNet, optimiser, EMA, batch size, step count and FID path with every
other arm.
"""

from __future__ import annotations

import argparse, json, math, os, time

import torch

from . import data as D
from .data import InfiniteSampler, load
from .ddpm_baseline import make_betas
from .ema import EMA
from .unet import build_unet

T_SCALE = 1.0


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def approx_std_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """Log-likelihood of 8-bit x under a Gaussian, integrated over each bin."""
    centered = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered + 1.0 / 255.0)
    min_in = inv_stdv * (centered - 1.0 / 255.0)
    cdf_plus = approx_std_normal_cdf(plus_in)
    cdf_min = approx_std_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(cdf_delta.clamp(min=1e-12))))


class ImprovedDiffusion:
    def __init__(self, n_timesteps: int, schedule: str, device):
        self.T = n_timesteps
        b = make_betas(schedule, n_timesteps).to(device)
        self.betas = b
        self.alphas = 1.0 - b
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.abar_prev = torch.cat([torch.ones(1, dtype=b.dtype, device=device),
                                    self.abar[:-1]])
        # posterior q(x_{t-1} | x_t, x_0)
        self.post_var = b * (1.0 - self.abar_prev) / (1.0 - self.abar)
        self.post_logvar_clipped = torch.log(
            torch.cat([self.post_var[1:2], self.post_var[1:]]))
        self.post_c0 = b * self.abar_prev.sqrt() / (1.0 - self.abar)
        self.post_ct = (1.0 - self.abar_prev) * self.alphas.sqrt() / (1.0 - self.abar)

    def _g(self, arr, t, shape):
        return arr[t].view(-1, *([1] * (len(shape) - 1))).to(torch.float32)

    def q_posterior(self, x0, xt, t):
        mean = self._g(self.post_c0, t, xt.shape) * x0 + \
               self._g(self.post_ct, t, xt.shape) * xt
        return mean, self._g(self.post_logvar_clipped, t, xt.shape)

    def p_mean_logvar(self, model_out, xt, t, clip=True):
        """Split the 2C output into (eps, v) and form the reverse Gaussian."""
        c = xt.shape[1]
        eps, v = model_out[:, :c], model_out[:, c:]
        # v in [-1,1] -> fraction in [0,1], interpolate the two bounds in log space
        frac = (v.float() + 1.0) / 2.0
        max_log = self._g(torch.log(self.betas), t, xt.shape)
        min_log = self._g(self.post_logvar_clipped, t, xt.shape)
        logvar = frac * max_log + (1.0 - frac) * min_log

        ab = self._g(self.abar, t, xt.shape)
        x0 = (xt - (1.0 - ab).sqrt() * eps.float()) / ab.sqrt()
        if clip:
            x0 = x0.clamp(-1, 1)
        mean, _ = self.q_posterior(x0, xt, t)
        return mean, logvar, x0, eps

    def training_loss(self, model, y, lam: float = 1e-3):
        b = y.shape[0]
        t = torch.randint(0, self.T, (b,), device=y.device)
        ab = self._g(self.abar, t, y.shape)
        noise = torch.randn_like(y)
        xt = ab.sqrt() * y + (1.0 - ab).sqrt() * noise
        out = model(xt, t.to(y.dtype)).float()

        c = y.shape[1]
        eps_pred = out[:, :c]
        l_simple = torch.nn.functional.mse_loss(eps_pred, noise)

        # VLB term: freeze the mean so this gradient trains only the variance
        frozen = torch.cat([eps_pred.detach(), out[:, c:]], dim=1)
        mean_p, logvar_p, _, _ = self.p_mean_logvar(frozen, xt, t)
        mean_q, logvar_q = self.q_posterior(y, xt, t)
        kl = normal_kl(mean_q, logvar_q, mean_p, logvar_p).flatten(1).mean(1) / math.log(2.0)
        nll = -discretized_gaussian_log_likelihood(
            y, mean_p, 0.5 * logvar_p).flatten(1).mean(1) / math.log(2.0)
        l_vlb = torch.where(t == 0, nll, kl).mean()

        return l_simple + lam * l_vlb, {"l_simple": float(l_simple.detach()),
                                        "l_vlb": float(l_vlb.detach())}

    @torch.no_grad()
    def sample(self, model, shape, device, n_steps=50, generator=None, x_init=None,
               clip=True, **kw):
        """Ancestral sampling on a respaced subsequence, using the learned variance."""
        idx = torch.linspace(0, self.T - 1, n_steps, dtype=torch.long).flip(0).tolist()
        x = x_init if x_init is not None else torch.randn(shape, device=device,
                                                          generator=generator)
        for i, t in enumerate(idx):
            tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x, tt.to(x.dtype))
            mean, logvar, _, _ = self.p_mean_logvar(out.float(), x.float(), tt, clip=clip)
            if i == len(idx) - 1:
                x = mean
            else:
                z = torch.randn(x.shape, device=device, dtype=mean.dtype,
                                generator=generator)
                x = mean + (0.5 * logvar).exp() * z
        return x.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out", default="./runs/mnist_improved")
    p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--vlb_weight", type=float, default=1e-3)
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
    cfg = dict(vars(args)); cfg["method"] = "improved_ddpm"; cfg["t_scale"] = T_SCALE
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    y_all = load(args.dataset, root=args.data_root, split="train")
    sampler = InfiniteSampler(y_all, args.batch_size, device, seed=args.seed)
    model = build_unet(args.dataset, learn_sigma=True).to(device)
    print(f"improved ddpm ({args.schedule}, learned sigma) | {args.dataset} | "
          f"{sum(q.numel() for q in model.parameters())/1e6:.1f}M params", flush=True)
    ema = EMA(model, decay=args.ema_decay, warmup=True)
    if args.compile:
        model = torch.compile(model)

    diff = ImprovedDiffusion(args.timesteps, args.schedule, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    t0, run, seen = time.time(), {"loss": 0.0, "l_simple": 0.0, "l_vlb": 0.0}, 0

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))
        y = sampler()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts = diff.training_loss(model, y, lam=args.vlb_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model._orig_mod if hasattr(model, "_orig_mod") else model, step)

        run["loss"] += loss.item(); run["l_simple"] += parts["l_simple"]
        run["l_vlb"] += parts["l_vlb"]; seen += 1
        if step % args.log_every == 0:
            rec = {"step": step, **{k: v / seen for k, v in run.items()},
                   "grad_norm": float(gn), "it_per_s": step / (time.time() - t0)}
            print(json.dumps(rec), flush=True)
            with open(os.path.join(args.out, "train_log.jsonl"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            run = {k: 0.0 for k in run}; seen = 0

        if step % args.ckpt_every == 0 or step == args.steps:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model": raw.state_dict(), "ema": ema.state_dict(),
                        "args": cfg}, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            print(f"saved ckpt_{step:06d}.pt", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
