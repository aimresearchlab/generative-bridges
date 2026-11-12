"""Sample grids for the appendix, assembled from the grids the sweeps write.

Every panel is the first 64 samples of a run. The sampler draws its initial
noise from one seeded generator in fixed batch order, so the first 64 are the
same images whatever n_samples the sweep asked for, and panels from a 2000-
sample run and a 50000-sample run are directly comparable. Nothing here
generates anything; it only arranges PNGs that already exist.
"""
from __future__ import annotations

import argparse, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

R = "runs"
BR_D = "cifar10_drift_g0.3162277660168379"      # headline: drift objective
BR_E = "cifar10_g0316_t1000"                    # same gamma, endpoint objective
MN = "mnist_g0316_t1000"


def panel(ax, path, title=None, sub=None):
    if not os.path.exists(path):
        ax.text(.5, .5, "not yet run", ha="center", va="center", fontsize=7,
                color="0.5"); ax.set_xticks([]); ax.set_yticks([])
        if title: ax.set_title(title, fontsize=9)
        return False
    ax.imshow(mpimg.imread(path), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(.4); sp.set_color("0.6")
    if title:
        ax.set_title(title, fontsize=9, pad=3)
    if sub:
        ax.set_xlabel(sub, fontsize=7.5, labelpad=2)
    return True


def fig_steps(tag, run, sampler, steps, fids, out, bare):
    fig, axes = plt.subplots(1, len(steps), figsize=(2.05 * len(steps), 2.45))
    for ax, n in zip(axes, steps):
        panel(ax, f"{R}/{run}/grids/{sampler}_{n:04d}.png",
              title=f"{n} NFE", sub=(f"FID {fids[n]}" if n in fids else None))
    if not bare:
        fig.suptitle(f"{tag}: samples against sampling budget", fontsize=10)
    fig.tight_layout()
    save(fig, out)


def fig_models(rows, out, bare, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(2.05 * len(rows), 2.45))
    for ax, (lab, path, sub) in zip(axes, rows):
        panel(ax, path, title=lab, sub=sub)
    if not bare:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, out)


def save(fig, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bare", action="store_true", help="no baked-in title; LaTeX supplies one")
    a = p.parse_args()
    d = f"{R}/figures"

    # 1. the corrected bridge across the sampling budget, CIFAR-10, n=50,000
    fid50k = {1: "386.0", 2: "168.4", 5: "44.0", 10: "19.2", 20: "12.3",
              50: "9.8", 100: "9.2", 200: "8.9", 1000: "9.2"}
    fig_steps("CIFAR-10, bridge $\\gamma=0.3162$, drift objective", BR_D, "euler",
              [2, 5, 10, 50, 200], fid50k, f"{d}/samples_steps_cifar", a.bare)

    # 2. what the objective correction looks like, matched gamma and budget
    # Both FIDs quoted at n=2000, the sample count at which BOTH were measured.
    # The drift model's n=50,000 figure is 9.83 and the endpoint model has no
    # n=50,000 measurement, so pairing 42.1 with 9.83 would compare across two
    # different floors (14.60 against roughly 0.5) and overstate the gap. The
    # images are unaffected: both grids are the first 64 samples of one seeded
    # generator, so they are the same draws at either sample count.
    #
    # Both panels use the SAME sampler (euler).  Showing the endpoint model
    # under its own best sampler (`bridge`, FID 42.14) against the drift model
    # under euler would vary the objective and the sampler at once, which is
    # not the ablation this figure claims to show.  Matched on euler the pair
    # is 38.93 -> 23.23, and the difference is the objective alone.
    fig_models([("Endpoint objective", f"{R}/{BR_E}/grids/euler_0050.png", "FID 38.9"),
                ("Drift objective",    f"{R}/{BR_D}/grids/euler_0050.png",  "FID 23.2"),
                ("Real data",          f"{R}/figures/real_cifar.png",       None)],
               f"{d}/samples_objective_cifar", a.bare,
               "CIFAR-10 at 50 NFE, $\\gamma=0.3162$, same architecture and budget")

    # 3. the comparison at matched budget
    fig_models([("Bridge (drift)",  f"{R}/{BR_D}/grids/euler_0050.png", None),
                ("Flow matching",   f"{R}/cifar10_flow/grids/flow_euler_0050.png", None),
                ("Improved DDPM",   f"{R}/cifar10_improved/grids/improved_ancestral_0050.png", None),
                ("DDPM linear",     f"{R}/cifar10_ddpm_linear/grids/ddpm_ancestral_0050.png", None)],
               f"{d}/samples_models_cifar", a.bare,
               "CIFAR-10 at 50 NFE, one architecture, one budget")

    # 4. MNIST across the budget
    fig_steps("MNIST, bridge $\\gamma=0.3162$", MN, "euler",
              [2, 5, 10, 50, 1000], {},   # MNIST euler FIDs predate the NFE
                                          # accounting fix; re-measure before
                                          # attaching numbers to these panels
              f"{d}/samples_steps_mnist", a.bare)


if __name__ == "__main__":
    main()
