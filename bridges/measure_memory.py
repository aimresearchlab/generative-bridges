"""Peak GPU memory for training and for sampling, per method.

Associate editor point 4 asks for memory consumption alongside the other cost
figures. Measured rather than argued: torch.cuda.max_memory_allocated around one
training step and around one sampling batch, after a warm-up so that allocator
caching and any compiled kernels are already resident.

The expectation is that the rows agree, since every method uses the same UNet at
the same batch size and differs only in what it regresses. Improved denoising
diffusion is the exception: a learned reverse variance doubles the output
channels. Reporting the numbers settles that rather than asserting it.
"""

from __future__ import annotations

import argparse, json

import torch

from . import data as D
from .bridge import BridgeConfig, GaussianBridge
from .flowmatch import FlowMatching
from .improved_ddpm import ImprovedDiffusion
from .unet import build_unet

MB = 1024 ** 2


def peak(fn, device="cuda"):
    fn()                                              # warm up allocator / kernels
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--sample_batch", type=int, default=500)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--out", default="runs/memory.json")
    a = p.parse_args()
    dev, c = "cuda", D.channels(a.dataset)
    y = D.load(a.dataset, split="train")[:a.batch_size].to(dev)
    shape = (a.sample_batch, c, D.IMAGE_SIZE, D.IMAGE_SIZE)

    rows = []
    for name, learn_sigma, make in [
        ("Bridge (drift)", False, lambda: GaussianBridge(BridgeConfig(gamma=0.3162277660168379, param="drift"))),
        ("Flow matching", False, lambda: FlowMatching()),
        ("Improved DDPM", True, lambda: ImprovedDiffusion(1000, "cosine", "cuda")),
    ]:
        model = build_unet(a.dataset, learn_sigma=learn_sigma).to(dev)
        sch = make()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

        def train_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = sch.training_loss(model, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        def sample():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(sch, GaussianBridge):
                    sch.sample(model, shape, dev, n_steps=a.n_steps, sampler="euler")
                elif isinstance(sch, FlowMatching):
                    sch.sample(model, shape, dev, n_steps=a.n_steps)
                else:
                    sch.sample(model, shape, dev, n_steps=a.n_steps)

        tr, sa = peak(train_step), peak(sample)
        rows.append({"method": name,
                     "params_M": round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
                     "train_peak_MB": round(tr, 1), "train_batch": a.batch_size,
                     "sample_peak_MB": round(sa, 1), "sample_batch": a.sample_batch,
                     "sample_steps": a.n_steps})
        print(json.dumps(rows[-1]), flush=True)
        del model, opt, sch
        torch.cuda.empty_cache()

    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
