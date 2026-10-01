"""Sample-quality measures that separate fidelity from coverage.

FID collapses fidelity and diversity into one scalar, so a model that produces
clean samples of half the modes and one that produces mediocre samples of all of
them can score alike.  Reviewer 1 asks whether the generated samples show mode
collapse or missing detail, which FID alone cannot answer.

Implemented here, all from the same Inception-v3 pool3 activations the FID path
already extracts, so adding them costs no extra sampling:

  precision / recall   Kynkaanniemi et al., NeurIPS 2019.  Build a k-NN manifold
                       for each set: a point is "in" the other set's manifold if
                       it lies within that point's distance to its own k-th
                       nearest neighbour.  Precision is the fraction of generated
                       samples inside the real manifold (fidelity); recall is the
                       fraction of real samples inside the generated manifold
                       (coverage).  Mode dropping shows as recall falling while
                       precision holds.

  density / coverage   Naeem et al., ICML 2020.  Same construction, made robust
                       to outliers: density counts how many real neighbourhoods
                       a generated point falls into rather than whether it falls
                       into any, and coverage asks what fraction of real points
                       have SOME generated point inside their neighbourhood.

  KID                  Binkowski et al., ICLR 2018.  Unbiased MMD with a
                       polynomial kernel.  Unlike FID it has no estimator bias
                       at finite n, so it is the honest measure at small sample
                       counts where the FID floor is large.
"""

from __future__ import annotations

import torch


def _pairwise(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Euclidean distances, in chunks so (50k, 50k) does not have to be resident."""
    out = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)
    for i in range(0, a.shape[0], chunk):
        out[i:i + chunk] = torch.cdist(a[i:i + chunk], b)
    return out


def _kth_radius(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest OTHER point in the same set."""
    d = _pairwise(x, x)
    d.fill_diagonal_(float("inf"))
    return d.kthvalue(k, dim=1).values


@torch.no_grad()
def precision_recall(real: torch.Tensor, fake: torch.Tensor, k: int = 3) -> dict:
    """Kynkaanniemi precision/recall and Naeem density/coverage, in one pass."""
    real, fake = real.double(), fake.double()
    r_real = _kth_radius(real, k)
    r_fake = _kth_radius(fake, k)
    d_fr = _pairwise(fake, real)                       # (n_fake, n_real)

    inside_real = (d_fr <= r_real[None, :])            # fake j inside real i's ball
    precision = inside_real.any(dim=1).double().mean()
    density = inside_real.double().sum(dim=1).mean() / k
    coverage = inside_real.any(dim=0).double().mean()  # real i has SOME fake inside

    d_rf = d_fr.T                                      # (n_real, n_fake)
    recall = (d_rf <= r_fake[None, :]).any(dim=1).double().mean()
    return {"precision": float(precision), "recall": float(recall),
            "density": float(density), "coverage": float(coverage), "k": k}


@torch.no_grad()
def kid(real: torch.Tensor, fake: torch.Tensor, n_subsets: int = 100,
        subset_size: int = 1000, seed: int = 0) -> dict:
    """Unbiased MMD^2 with the cubic polynomial kernel, averaged over subsets."""
    real, fake = real.double(), fake.double()
    d = real.shape[1]
    g = torch.Generator(device=real.device).manual_seed(seed)
    m = min(subset_size, real.shape[0], fake.shape[0])

    def poly(a, b):
        return (a @ b.T / d + 1.0) ** 3

    vals = []
    for _ in range(n_subsets):
        r = real[torch.randperm(real.shape[0], generator=g, device=real.device)[:m]]
        f = fake[torch.randperm(fake.shape[0], generator=g, device=fake.device)[:m]]
        krr, kff, krf = poly(r, r), poly(f, f), poly(r, f)
        krr.fill_diagonal_(0); kff.fill_diagonal_(0)
        vals.append((krr.sum() + kff.sum()) / (m * (m - 1)) - 2 * krf.mean())
    v = torch.stack(vals)
    return {"kid": float(v.mean()), "kid_std": float(v.std()),
            "n_subsets": n_subsets, "subset_size": m}
