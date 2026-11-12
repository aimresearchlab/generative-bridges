"""Exponential moving average of model weights.

The original training scripts kept no EMA and sampled from the raw weights.
Every diffusion baseline in the comparison (improved-diffusion in particular)
samples from an EMA copy, so this was also a confound in the published table.
"""

from __future__ import annotations

import copy
import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmup: int = 0):
        self.decay = decay
        self.warmup = warmup
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int):
        # ramp the decay in so early steps are not dominated by the random init
        d = min(self.decay, (1 + step) / (10 + step)) if self.warmup else self.decay
        for pe, pm in zip(self.ema.parameters(), model.parameters()):
            pe.lerp_(pm.detach(), 1.0 - d)
        for be, bm in zip(self.ema.buffers(), model.buffers()):
            be.copy_(bm)

    def state_dict(self):
        return self.ema.state_dict()
