"""Dataset loading with a single, explicit convention for pixel scaling.

Convention
----------
Everything downstream works with float tensors in [-1, 1] at a 32x32 grid, and
converts back to uint8 [0, 255] exactly once, at the boundary with FID.

Three things are easy to vary by accident across a pipeline and each is a
silent distribution shift between training, sampling and evaluation: the pixel
range, the grid size, and whether a per-image normalisation is applied on the
way to the metric.  Routing generated images through an image writer that
rescales each one independently introduces the third without it being visible in
any configuration file.  This module fixes one scale and one grid, and the
conversion to uint8 happens exactly once, at the boundary with FID.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

IMAGE_SIZE = 32


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """uint8 HWC or HW -> float NCHW in [-1, 1] at IMAGE_SIZE, bilinear, antialiased."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if t.ndim == 3:            # N H W        (MNIST)
        t = t.unsqueeze(1)
    elif t.ndim == 4:          # N H W C      (CIFAR-10)
        t = t.permute(0, 3, 1, 2)
    t = t.float().div_(255.0)
    if t.shape[-1] != IMAGE_SIZE:
        t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                          align_corners=False, antialias=True)
    return t.mul_(2.0).sub_(1.0).clamp_(-1.0, 1.0)


def load(dataset: str, root: str = "./data", split: str = "train") -> torch.Tensor:
    """Return the whole split as one float tensor in [-1, 1], NCHW, 32x32."""
    train = split == "train"
    if dataset == "mnist":
        d = datasets.MNIST(root=root, train=train, download=False)
        arr = d.data.numpy()
    elif dataset == "cifar10":
        d = datasets.CIFAR10(root=root, train=train, download=False)
        arr = d.data
    else:
        raise ValueError(dataset)
    return _to_tensor(arr)


def labels(dataset: str, root: str = "./data", split: str = "train") -> torch.Tensor:
    """Class labels for the split, in the same order as `load`."""
    train = split == "train"
    if dataset == "mnist":
        return datasets.MNIST(root=root, train=train, download=False).targets.long()
    if dataset == "cifar10":
        return torch.tensor(datasets.CIFAR10(root=root, train=train,
                                             download=False).targets).long()
    raise ValueError(dataset)


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] uint8, 3-channel. Clamped, never per-image rescaled.

    Per-image min-max rescaling (as the original save_image/process_image did)
    silently applies a contrast stretch to every sample before it is measured,
    which changes FID in an uncontrolled, sample-dependent way.
    """
    x = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def channels(dataset: str) -> int:
    return 1 if dataset == "mnist" else 3


class InfiniteSampler:
    """Shuffled, epoch-free batches from an in-memory tensor (both datasets fit)."""

    def __init__(self, data: torch.Tensor, batch_size: int, device, seed: int = 0):
        self.data = data.to(device, non_blocking=True)
        self.bs = batch_size
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self._perm, self._i = None, 0

    def __call__(self) -> torch.Tensor:
        if self._perm is None or self._i + self.bs > len(self._perm):
            self._perm = torch.randperm(len(self.data), generator=self.g)
            self._i = 0
        idx = self._perm[self._i:self._i + self.bs]
        self._i += self.bs
        return self.data[idx.to(self.data.device)]
