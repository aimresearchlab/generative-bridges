"""Replica of the original evaluation pipeline, to test whether it manufactures
step-dependent FID drift.

Neither the deterministic sampler nor the degenerate time conditioning produces
absolute degradation with sampling steps in the clean pipeline; both merely
converge to a worse level. The published curve degrades (MNIST 19.34 -> 22.01 ->
23.55 across 10 -> 100 -> 1000 steps). So the remaining candidate is the
measurement itself.

The original path, from sample_bridges.py and compute_metrics.py:

  1. `save_image`: if the float array falls outside [0,1], rescale it by its own
     min and max -- a per-image contrast stretch.
  2. `plt.imsave(..., cmap='gray')`: matplotlib normalises AGAIN by the array's
     own min/max, then writes RGBA.
  3. read the PNG back, keep the first `image_channels` channels.
  4. `process_image_auto`: per-image min-max a THIRD time, scale to uint8.
  Reals go through `process_image`, which is also per-image min-max.

Per-image contrast normalisation is not measurement-neutral. Sample contrast
varies systematically with step count -- a sampler that under-disperses produces
progressively lower-contrast images as the grid refines -- and stretching each
image to full range converts that contrast drift into a change in the Inception
features. That is a mechanism by which FID can move with step count for reasons
that have nothing to do with sample quality.

This module reproduces that path exactly so the same samples can be scored both
ways. `legacy_process` is the fake path, `legacy_process_real` the real path.
"""

from __future__ import annotations

import io

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
except Exception:  # pragma: no cover
    plt = None


def _save_image_replica(image_np: np.ndarray) -> np.ndarray:
    """save_image() + plt.imsave() + read-back, without touching the disk."""
    # step 1: the original's own rescale, applied only when out of [0,1]
    if image_np.dtype in (np.float32, np.float64):
        if image_np.max() > 1.0 or image_np.min() < 0.0:
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    buf = io.BytesIO()
    # step 2: matplotlib normalises again, per image, and writes RGBA
    if image_np.ndim == 3 and image_np.shape[2] == 1:
        plt.imsave(buf, image_np[:, :, 0], cmap="gray", format="png")
    else:
        plt.imsave(buf, image_np, format="png")
    buf.seek(0)
    # step 3: read back
    return np.array(Image.open(buf))


def legacy_process(x: torch.Tensor, channels: int) -> torch.Tensor:
    """Generated samples in [-1,1] -> uint8 RGB, via the original's exact path."""
    out = torch.empty(x.shape[0], 3, x.shape[2], x.shape[3], dtype=torch.uint8)
    for i in range(x.shape[0]):
        arr = x[i].cpu().permute(1, 2, 0).numpy()          # CHW -> HWC, as the original
        rgba = _save_image_replica(arr)
        img = rgba[:, :, :channels]                        # keep first `channels`
        t = torch.from_numpy(np.ascontiguousarray(img))
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        t = t.permute(2, 0, 1).float() / 255.0
        # step 4: process_image_auto -- per-image min-max a third time
        t = (t - t.min()) / (t.max() - t.min() + 1e-12)
        t = (t * 255).clamp(0, 255).to(torch.uint8)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        out[i] = t
    return out


def legacy_process_real(x: torch.Tensor) -> torch.Tensor:
    """Real images -> uint8 RGB via compute_metrics.process_image (per-image min-max)."""
    out = torch.empty(x.shape[0], 3, x.shape[2], x.shape[3], dtype=torch.uint8)
    for i in range(x.shape[0]):
        t = x[i]
        t = (t - t.min()) / (t.max() - t.min() + 1e-12)
        t = (t * 255).clamp(0, 255).to(torch.uint8)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        out[i] = t
    return out


@torch.no_grad()
def legacy_stats(x: torch.Tensor, device, channels: int, real: bool,
                 batch_size: int = 100, stats=None):
    from .fid import InceptionStats
    stats = stats or InceptionStats(device)
    stats.reset()
    for i in range(0, len(x), batch_size):
        chunk = x[i:i + batch_size]
        u8 = legacy_process_real(chunk) if real else legacy_process(chunk, channels)
        stats.update(u8.to(device))
    return stats.finalize()
