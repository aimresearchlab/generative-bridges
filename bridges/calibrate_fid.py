"""What does FID = 0 look like at a given sample count?

FID is biased upward at finite n and the bias is large at the sample counts that
fit in a sweep budget, so every model number has to be read against a floor.

The floor has to use the SAME comparator as the model number, or it is not a
floor.  A sweep entry compares n generated images against the statistics of the
*full* reference split, so its floor is n real held-out images against that same
full split -- not n against n, which carries finite-sample error on both sides
and overstates the floor by a large factor.  Both are reported: `--mode holdout`
(default, the right one) and `--mode disjoint` (n vs n, kept because earlier
runs quote it).

The held-out images come from the test split, so the number also contains
whatever genuine train/test shift the dataset has.  That shift is common to
every model row, and on these datasets it is the smaller of the two terms.
"""
from __future__ import annotations

import argparse

from . import data as D
from .fid import InceptionStats, frechet_distance, stats_from_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--counts", type=int, nargs="+", default=[2000, 5000, 10000, 25000])
    p.add_argument("--mode", default="holdout", choices=["holdout", "disjoint", "both"])
    a = p.parse_args()

    dev = "cuda"
    st = InceptionStats(dev)
    x = D.load(a.dataset, split="train")
    te = D.load(a.dataset, split="test")
    print(f"{a.dataset}: achievable FID floors ({len(x)} train, {len(te)} test)")
    ref = stats_from_tensor(x, dev, batch_size=250, stats=st)     # the sweep's comparator
    print(f"{'n':>8}{'holdout vs full train':>24}{'n vs n disjoint':>18}")
    for n in a.counts:
        row = f"  n={n:>5}"
        if a.mode in ("holdout", "both"):
            k = min(n, len(te))
            h = stats_from_tensor(te[:k], dev, batch_size=250, stats=st)
            tag = f"{frechet_distance(*h, *ref):.3f}" + ("" if k == n else f" (n={k})")
            row += f"{tag:>24}"
        else:
            row += f"{'-':>24}"
        if a.mode in ("disjoint", "both") and 2 * n <= len(x):
            lo = stats_from_tensor(x[:n], dev, batch_size=250, stats=st)
            hi = stats_from_tensor(x[n:2 * n], dev, batch_size=250, stats=st)
            row += f"{frechet_distance(*lo, *hi):>18.3f}"
        else:
            row += f"{'-':>18}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
