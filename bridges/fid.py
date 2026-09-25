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


def _psd_sqrt(a: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition, negatives clamped."""
    w, v = np.linalg.eigh((a + a.T) / 2.0)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def frechet_distance(mu1, s1, mu2, s2) -> float:
    """||mu1-mu2||^2 + tr(S1) + tr(S2) - 2 tr((S1 S2)^{1/2}).

    tr((S1 S2)^{1/2}) is evaluated as sum_i sqrt(lambda_i(M)) with
    M = S1^{1/2} S2 S1^{1/2}, which is symmetric PSD and similar to S1 S2.  This
    is stable for rank-deficient or ill-conditioned covariances -- the case
    whenever the sample count is not comfortably above the 2048 feature
    dimension.  A Denman-Beavers iteration inverts its iterates and diverges
    there, which can return a negative "distance".
    """
    mu1, mu2 = np.asarray(mu1, dtype=np.float64), np.asarray(mu2, dtype=np.float64)
    s1, s2 = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    a = _psd_sqrt(s1)
    m = a @ s2 @ a
    lam = np.clip(np.linalg.eigvalsh((m + m.T) / 2.0), 0.0, None)
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.sqrt(lam).sum())


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
