"""FID with an explicit, reproducible protocol.

What is fixed relative to the original `compute_metrics.py`:

1. No PNG round-trip.  Samples go straight from the model to uint8.  The old
   path was  float -> min-max rescale -> matplotlib.imsave (which rescales
   *again*, per image) -> RGBA PNG -> read back -> min-max a third time.
2. No per-image contrast normalisation of either reals or fakes.  A single
   affine map [-1,1] -> [0,255] with clamping is used for both.
3. Reals and fakes are on the same grid (32x32) and go through identical
   preprocessing.
4. Real statistics are computed once over the *full* split and cached, rather
   than over the first `max_images` entries in dataset order.

The Inception feature extractor was built for natural colour images; for MNIST
it is used only for continuity with the literature, and a LeNet-feature
distance on the same samples is reported alongside it (see `mnist_fd`).
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3

from . import data as D


def _sqrtm_newton(a: np.ndarray, iters: int = 100, eps: float = 1e-12) -> np.ndarray:
    """Matrix square root by Denman-Beavers iteration (scipy-free, float64)."""
    norm = np.linalg.norm(a)
    y = a / norm
    z = np.eye(a.shape[0], dtype=np.float64)
    for _ in range(iters):
        y_next = 0.5 * (y + np.linalg.inv(z))
        z = 0.5 * (z + np.linalg.inv(y))
        if np.linalg.norm(y_next - y) < eps * np.linalg.norm(y):
            y = y_next
            break
        y = y_next
    return y * np.sqrt(norm)


def frechet_distance(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    covmean = _sqrtm_newton(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


class InceptionStats:
    """Streaming mean/covariance of pool3 features over uint8 RGB batches."""

    def __init__(self, device, feature: int = 2048):
        self.net = NoTrainInceptionV3(name="inception-v3-compat",
                                      features_list=[str(feature)]).to(device).eval()
        self.device = device
        self.dim = feature
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
        self.cov = torch.zeros(self.dim, self.dim, dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, x_uint8: torch.Tensor):
        f = self.net(x_uint8.to(self.device)).double()
        self.n += f.shape[0]
        self.sum += f.sum(0)
        self.cov += f.T @ f

    def finalize(self):
        mu = self.sum / self.n
        cov = (self.cov - self.n * torch.outer(mu, mu)) / (self.n - 1)
        return mu.cpu().numpy(), cov.cpu().numpy()


@torch.no_grad()
def stats_from_tensor(x: torch.Tensor, device, batch_size: int = 128, stats=None):
    """x: float NCHW in [-1,1]. Returns (mu, sigma)."""
    stats = stats or InceptionStats(device)
    stats.reset()
    for i in range(0, len(x), batch_size):
        stats.update(D.to_uint8(x[i:i + batch_size]))
    return stats.finalize()


def real_stats(dataset: str, split: str, device, root: str = "./data",
               cache_dir: str = "./fid_cache", batch_size: int = 128, stats=None):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset}_{split}_{D.IMAGE_SIZE}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mu"], z["sigma"]
    x = D.load(dataset, root=root, split=split)
    mu, sigma = stats_from_tensor(x, device, batch_size=batch_size, stats=stats)
    np.savez(path, mu=mu, sigma=sigma, n=len(x))
    return mu, sigma
