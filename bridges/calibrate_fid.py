"""What does FID = 0 look like at a given sample count?

FID is biased upward at finite n and the bias is large at the sample counts that
fit in a sweep budget.  Measuring the distance between two *disjoint real*
subsets gives the achievable floor, which is the reference any model number has
to be read against.  Run this before choosing n for a sweep.
"""
from __future__ import annotations

import argparse

from . import data as D
from .fid import InceptionStats, frechet_distance, stats_from_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--counts", type=int, nargs="+", default=[2000, 5000, 10000, 25000])
    a = p.parse_args()

    dev = "cuda"
    st = InceptionStats(dev)
    x = D.load(a.dataset, split="train")
    print(f"{a.dataset}: FID between two disjoint real subsets (the achievable floor)")
    for n in a.counts:
        if 2 * n > len(x):
            print(f"  n={n:>6}: skipped (need {2*n} images, have {len(x)})")
            continue
        lo = stats_from_tensor(x[:n], dev, batch_size=250, stats=st)
        hi = stats_from_tensor(x[n:2 * n], dev, batch_size=250, stats=st)
        print(f"  n={n:>6}: {frechet_distance(*lo, *hi):8.3f}")


if __name__ == "__main__":
    main()
