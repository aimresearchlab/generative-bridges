"""The headline figure: FID against network evaluations, all methods, matched budget.

Form: change over a quantitative x with several series -> line chart, log-log,
because a power-law decay is then a straight line and a plateau a flat one,
which is the distinction the figure exists to show.

Colour: categorical, reference-palette slots in fixed order, never cycled. Line
charts use the adjacent pairlist, which that order clears in both modes for all
eight slots. With more than four series a legend carries identity (direct labels
would collide), plus a marked measurement floor as a recessive reference rule.

The figure is written to be read standalone on a phone, so the caption is baked
into the image rather than living only in surrounding prose.
"""

from __future__ import annotations

import argparse, json, os, textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# (label, run dir, sampler, palette slot, linestyle)
SERIES = [
    ("Bridge — exact posterior",      "mnist_g0316_t1000", "bridge",     0, "-"),
    ("Bridge — Euler–Maruyama",       "mnist_g0316_t1000", "euler",      1, "-"),
    ("Bridge — deterministic Euler",  "mnist_g0316_t1000", "euler_det",  2, "-"),
    ("Flow matching",                 "mnist_flow",        "flow_euler", 3, "-"),
    ("DDPM cosine + DDIM",            "mnist_ddpm_cosine", "ddim",       6, (0, (5, 2))),
    ("DDPM linear + DDIM",            "mnist_ddpm_linear", "ddim",       7, (0, (5, 2))),
]


def load(root, run_dir, sampler, n_samples, pattern="fid_n50k*.json"):
    import glob
    vals = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, run_dir, pattern))):
        try:
            rows = json.load(open(p))
        except Exception:
            continue
        for r in rows:
            if r.get("sampler") == sampler and r.get("n_samples") == n_samples:
                vals[r["n_steps"]].append(r["fid_train"])
    return {k: sum(v) / len(v) for k, v in sorted(vals.items())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    p.add_argument("--n_samples", type=int, default=50000)
    p.add_argument("--floor", type=float, default=0.15)
    p.add_argument("--out", default="runs/figures/mnist_headline.png")
    p.add_argument("--caption", default=None)
    p.add_argument("--provisional", nargs="*", default=[],
                   help="series to mark with square markers as not yet trusted")
    p.add_argument("--compact", action="store_true",
                   help="no baked-in caption or title; for embedding where the "
                        "surrounding document supplies them")
    a = p.parse_args()

    data = [(lbl, load(a.root, d, s, a.n_samples), slot, ls)
            for lbl, d, s, slot, ls in SERIES]
    data = [t for t in data if t[1]]
    if not data:
        print("no data"); return

    if a.compact:
        fig = plt.figure(figsize=(9.2, 3.7), facecolor=SURFACE)
        ax = fig.add_axes([0.068, 0.155, 0.915, 0.815])
    else:
        fig = plt.figure(figsize=(8.6, 9.3), facecolor=SURFACE)
        ax = fig.add_axes([0.105, 0.40, 0.865, 0.475])
    ax.set_facecolor(SURFACE)

    ax.axhline(a.floor, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"measurement floor  ({a.floor:.2f})", xy=(0.99, a.floor),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=8, color=INK_MUTED, zorder=2)

    for lbl, series, slot, ls in data:
        xs = sorted(series)
        ys = [series[x] for x in xs]
        mark = "o" if lbl not in a.provisional else "s"
        ax.plot(xs, ys, color=PALETTE[slot], lw=2.0, ls=ls, marker=mark, ms=5.2,
                mec=SURFACE, mew=1.1, label=lbl, zorder=3)

    allx = sorted({k for _, s, _, _ in data for k in s})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(allx); ax.set_xticklabels([str(k) for k in allx], fontsize=9)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=9))
    ax.set_xlabel("Network evaluations per sample  (= sampling steps)",
                  fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("FID   (lower is better)", fontsize=10, color=INK_SECONDARY)
    ax.grid(True, which="major", color=INK_MUTED, alpha=0.22, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=INK_MUTED, alpha=0.09, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.margins(x=0.05)

    leg = ax.legend(frameon=False, fontsize=9, loc="upper right",
                    handlelength=2.4, labelspacing=0.45)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    if a.compact:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        fig.savefig(a.out, dpi=200, facecolor=SURFACE)
        print(f"wrote {a.out} (compact)")
        return

    fig.text(0.105, 0.978, "Sampling efficiency on MNIST at matched training budget",
             fontsize=14, color=INK_PRIMARY, ha="left", va="top")
    sub = ("Every method uses the same 7.28M-parameter UNet, AdamW 2e-4, EMA 0.9999,\n"
           "batch 128 and 40,000 training steps, and is scored by identical FID code.")
    fig.text(0.105, 0.950, sub, fontsize=9.5, color=INK_SECONDARY,
             ha="left", va="top", linespacing=1.5)

    cap = a.caption or (
        "FID against network evaluations per generated sample, on MNIST. Six schemes "
        "trained under an identical budget and measured through identical code; only "
        "the generative process and its sampler differ. 50,000 samples per point against "
        "the full 60,000-image training split. The dashed rule at 0.15 is the achievable "
        "floor, measured between disjoint real subsets of the same size. "
        "The bridge leads at 10-20 evaluations and flow matching from 50 onward, both "
        "well below the diffusion baselines throughout. Deterministic Euler (green) is "
        "the sampler the original implementation evaluated: it converges to about three "
        "times the floor excess of the correct samplers, which is the reported plateau. "
        "The DDPM curves turn upward after 100 evaluations. That is a documented failure "
        "mode rather than an implementation artefact: clipping beta at 0.999 gives the "
        "cosine schedule an effectively zero terminal SNR (alpha-bar 2.4e-9), where "
        "epsilon-prediction is degenerate and the x0 reconstruction amplifies error "
        "20,000-fold (Lin et al., WACV 2024). It affects deterministic DDIM only - the "
        "ancestral sampler is flat over the same range - and two independent timestep "
        "grids reproduce it. The remedy, v-prediction, is not applied here."
    )
    fig.text(0.105, 0.335, "\n".join(textwrap.wrap(cap, 106)),
             fontsize=8.2, color=INK_SECONDARY, ha="left", va="top", linespacing=1.6)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
