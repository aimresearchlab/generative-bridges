"""Training loop for Gaussian random-bridge generative models."""

from __future__ import annotations

import argparse, json, os, time

import torch

from .bridge import BridgeConfig, GaussianBridge
from .data import InfiniteSampler, channels, load
from .ema import EMA
from .unet import build_unet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out", default="./runs/mnist_bridge")
    p.add_argument("--steps", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--gamma", type=float, default=0.3162277660168379,
                   help="bridge noise scale; gamma^2 = sigma^2 * T (published: sigma=1, T=0.1)")
    p.add_argument("--param", default="endpoint", choices=["endpoint", "drift"],
                   help="what the network regresses: the endpoint E[Y|xi_s], or the "
                        "SDE drift (E[Y|xi_s]-xi_s)/(1-s) that the sampler actually "
                        "integrates. At gamma=0 the drift objective is conditional "
                        "flow matching.")
    p.add_argument("--s_max", type=float, default=0.999,
                   help="upper limit of the s grid for drift training")
    p.add_argument("--t_scale", type=float, default=1000.0,
                   help="scale applied to s in [0,1] before the sinusoidal time embedding. "
                        "Set to the manuscript's T (e.g. 0.1) to reproduce its time conditioning, "
                        "over which range the sinusoidal embedding is near-linear.")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_every", type=int, default=10000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    y_all = load(args.dataset, root=args.data_root, split="train")
    sampler = InfiniteSampler(y_all, args.batch_size, device, seed=args.seed)
    print(f"{args.dataset}: {len(y_all)} train images, {tuple(y_all.shape[1:])}", flush=True)

    model = build_unet(args.dataset).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"model: {n_par/1e6:.1f}M params", flush=True)
    ema = EMA(model, decay=args.ema_decay, warmup=True)
    if args.compile:
        model = torch.compile(model)

    bridge = GaussianBridge(BridgeConfig(gamma=args.gamma, t_scale=args.t_scale,
                                     param=args.param, s_max=args.s_max))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.999))

    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float32
    log_path = os.path.join(args.out, "train_log.jsonl")
    t0 = time.time()
    running, seen = 0.0, 0

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:                       # linear warmup, then constant
            g["lr"] = args.lr * min(1.0, step / max(1, args.warmup))

        y = sampler()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp != "fp32"):
            loss, _ = bridge.training_loss(model, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model._orig_mod if hasattr(model, "_orig_mod") else model, step)

        running += loss.item(); seen += 1
        if step % args.log_every == 0:
            el = time.time() - t0
            rec = {"step": step, "loss": running / seen, "grad_norm": float(gnorm),
                   "it_per_s": step / el, "elapsed_s": el}
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            running, seen = 0.0, 0

        if step % args.ckpt_every == 0 or step == args.steps:
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model": raw.state_dict(), "ema": ema.state_dict(),
                        "args": vars(args)},
                       os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            print(f"saved ckpt_{step:06d}.pt", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
