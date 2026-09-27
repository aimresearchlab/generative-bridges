"""Every measured curve, faceted into two panels.

Why facet
---------
There are nine measured (method, sampler) curves, ten once improved+DDIM lands,
against an eight-slot categorical palette. Cycling hues would give two different
methods the same colour, so the series are split into two panels instead: the
bridge's own samplers, and the baselines. Colour is assigned from one global
entity -> slot table, so a given curve keeps its hue wherever it appears.

Both panels share the y-scale and the measurement-floor rule, so a value in one
can be read against a value in the other.

Folded out of the figure (still in the table): DDPM cosine ancestral and DDPM
linear ancestral. They are dominated at every step count, and the
ancestral-versus-DDIM contrast they would illustrate is already carried by the
improved-DDPM pair, which is the stronger model.
"""

from __future__ import annotations

import argparse, glob, json, os, textwrap
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

# one global entity -> (run dir, sampler, palette slot, linestyle)
PANELS = [
    ("Bridge samplers (ours)", [
        ("Exact posterior",       "mnist_g0316_t1000", "bridge",     0, "-"),
        ("Euler-Maruyama",        "mnist_g0316_t1000", "euler",      1, "-"),
        ("Deterministic Euler",   "mnist_g0316_t1000", "euler_det",  2, "-"),
    ]),
    ("Baselines at matched budget", [
        ("Flow matching",         "mnist_flow",        "flow_euler", 3, "-"),
        ("Improved DDPM (anc.)",  "mnist_improved",    "improved_ancestral", 4, "-"),
        ("Improved DDPM + DDIM",  "mnist_improved",    "improved_ddim",      5, (0, (5, 2))),
        ("DDPM cosine + DDIM",    "mnist_ddpm_cosine", "ddim",       6, (0, (5, 2))),
        ("DDPM linear + DDIM",    "mnist_ddpm_linear", "ddim",       7, (0, (5, 2))),
    ]),
]


def load(root, run_dir, sampler, n_samples):
    vals = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, run_dir, "fid_n50k*.json"))):
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
    p.add_argument("--out", default="runs/figures/mnist_all_curves.png")
    a = p.parse_args()

    data = []
    for title, rows in PANELS:
        got = [(lbl, load(a.root, d, s, a.n_samples), slot, ls)
               for lbl, d, s, slot, ls in rows]
        data.append((title, [g for g in got if g[1]]))

    allv = [v for _, rows in data for _, sr, _, _ in rows for v in sr.values()]
    allx = sorted({k for _, rows in data for _, sr, _, _ in rows for k in sr})
    ylo, yhi = min(min(allv), a.floor) * 0.72, max(allv) * 1.5

    fig = plt.figure(figsize=(8.8, 10.4), facecolor=SURFACE)
    axes = [fig.add_axes([0.10, 0.600, 0.875, 0.260]),
            fig.add_axes([0.10, 0.283, 0.875, 0.260])]

    for ax, (title, rows) in zip(axes, data):
        ax.set_facecolor(SURFACE)
        ax.axhline(a.floor, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for lbl, sr, slot, ls in rows:
            xs = sorted(sr)
            ax.plot(xs, [sr[x] for x in xs], color=PALETTE[slot], lw=2.0, ls=ls,
                    marker="o", ms=5.0, mec=SURFACE, mew=1.1, label=lbl, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(allx); ax.set_xticklabels([str(k) for k in allx], fontsize=8.5)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=8))
        ax.set_ylabel("FID  (lower is better)", fontsize=9.5, color=INK_SECONDARY)
        ax.grid(True, which="major", color=INK_MUTED, alpha=0.22, lw=0.7, zorder=0)
        ax.grid(True, which="minor", color=INK_MUTED, alpha=0.09, lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(INK_MUTED); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        ax.set_title(title, fontsize=10.5, color=INK_PRIMARY, loc="left", pad=7)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right",
                        handlelength=2.3, labelspacing=0.35)
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    axes[0].annotate(f"measurement floor ({a.floor:.2f})", xy=(0.99, a.floor),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=7.5, color=INK_MUTED)
    axes[-1].set_xlabel("Network evaluations per sample  (= sampling steps)",
                        fontsize=9.5, color=INK_SECONDARY)

    fig.text(0.10, 0.982, "MNIST: every measured sampler at matched training budget",
             fontsize=13.5, color=INK_PRIMARY, ha="left", va="top")
    fig.text(0.10, 0.950,
             "One 7.28M-parameter UNet, AdamW 2e-4, EMA 0.9999, batch 128, 40,000 steps,\n"
             "identical FID code. Only the generative process and its sampler differ.",
             fontsize=9.5, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)

    cap = (
        "FID against network evaluations per generated sample, 50,000 samples per point "
        "against the full 60,000-image training split. Panels share the y-scale, so values "
        "can be read across them; the dashed rule is the achievable floor measured between "
        "disjoint real subsets of the same size. "
        "Top: the bridge's three samplers. Deterministic Euler is the one the original "
        "implementation evaluated, and it converges to roughly three times the floor excess "
        "of the two correct samplers - the reported step-count plateau. "
        "Bottom: the baselines. Both DDIM curves collapse beyond 100 evaluations, a "
        "documented consequence of an effectively zero terminal signal-to-noise ratio under "
        "epsilon-prediction (Lin et al., WACV 2024); the ancestral samplers, which re-inject "
        "noise at that step, do not. Improved DDPM reaches 1.93 at 1000 evaluations against "
        "the 1.90 reported in the original manuscript, which is the check that this baseline "
        "is faithful rather than weakened. "
        "Two dominated rows, DDPM cosine and linear with ancestral sampling, are kept in the "
        "table but left out here: nine curves exceed an eight-slot palette, and giving two "
        "methods the same colour would be worse than omitting the two that carry no argument."
    )
    fig.text(0.10, 0.225, "\n".join(textwrap.wrap(cap, 108)),
             fontsize=8.2, color=INK_SECONDARY, ha="left", va="top", linespacing=1.6)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=190, facecolor=SURFACE)
    print(f"wrote {a.out}  ({sum(len(r) for _, r in data)} curves)")


if __name__ == "__main__":
    main()
