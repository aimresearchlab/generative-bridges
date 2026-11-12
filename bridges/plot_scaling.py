"""FID against sampling steps, across methods and samplers.

Form
----
The question is how quality changes as the discretisation is refined, for
several generative processes -- change over a quantitative x with multiple
series, so a line chart. Both axes are logarithmic: NFE spans 1..1000, and on
log-log a power-law decay is a straight line while a plateau is a flat one,
which is precisely the distinction under test.

Colour
------
Categorical (series identity), taken from the reference palette's fixed slot
order and never cycled. Line charts use the adjacent pairlist, which that order
clears in both modes for all eight slots. A series beyond the eighth would fold
into "other" rather than get a generated hue.

Accessibility
-------------
A legend is always present. With four or fewer series each line is also directly
labelled at its right-hand end, staggered so near-equal endpoints do not
overprint, so identity is never carried by colour alone. Text wears ink tokens,
never the series colour. The measurement floor is drawn as a recessive reference
rule, not as a series.
"""

from __future__ import annotations

import argparse, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"

# reference palette, fixed slot order -- assigned in order of first appearance
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SHORT = {
    "bridge": "exact post.",
    "euler": "Euler-M.",
    "euler_det": "det. Euler",
    "flow_euler": "flow match.",
    "ddim": "DDIM",
    "ddpm_ancestral": "ancestral",
}

PRETTY = {
    "bridge": "bridge (exact posterior)",
    "euler": "bridge (Euler-Maruyama)",
    "euler_det": "bridge (deterministic Euler)",
    "flow_euler": "flow matching (Euler ODE)",
    "ddim": "DDPM + DDIM",
    "ddpm_ancestral": "DDPM ancestral",
}


def load_series(paths, labels=None):
    """-> ({series_name: {nfe: fid}}, {series_name: short_label})."""
    out, short = defaultdict(dict), {}
    for i, path in enumerate(paths):
        tag = (labels[i] if labels and i < len(labels) else None)
        for r in json.load(open(path)):
            name = PRETTY.get(r["sampler"], r["sampler"])
            sh = SHORT.get(r["sampler"], r["sampler"])
            if tag:
                name, sh = f"{tag}: {name}", f"{tag}: {sh}"
            out[name][r["n_steps"]] = r["fid_train"]
            short[name] = sh
    return dict(out), short


def plot(series, out, title=None, floor=None, floor_label=None, ylabel=None,
         short=None):
    short = short or {}
    fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    names = list(series.keys())
    if len(names) > len(PALETTE):
        raise ValueError(f"{len(names)} series exceeds the {len(PALETTE)}-slot palette; "
                         "fold the tail into 'other' or facet instead of cycling hues")

    if floor is not None:
        ax.axhline(floor, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        # right-aligned: the legend sits lower-left, so this cannot collide with it
        ax.annotate(floor_label or f"measurement floor ({floor:.2f})",
                    xy=(0.985, floor), xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=8, color=INK_MUTED, va="bottom", zorder=2)

    ends = []
    for slot, name in enumerate(names):
        xs = sorted(series[name])
        ys = [series[name][x] for x in xs]
        ax.plot(xs, ys, color=PALETTE[slot], lw=2.0, marker="o", ms=5.5,
                mec=SURFACE, mew=1.2, label=name, zorder=3)
        ends.append((ys[-1], xs[-1], name, PALETTE[slot]))

    # direct labels for <= 4 series, staggered against overprinting
    if len(names) <= 4:
        ends.sort()
        dy, prev = {}, None
        for yv, xv, nm, _ in ends:
            dy[nm] = 0 if prev is None or yv / prev >= 1.18 else dy[prev_nm] + 10
            prev, prev_nm = yv, nm
        for yv, xv, nm, _ in ends:
            ax.annotate(short.get(nm, nm), (xv, yv), textcoords="offset points",
                        xytext=(8, dy[nm]), va="center", fontsize=8.5,
                        color=INK_SECONDARY, zorder=4)

    allx = sorted({k for v in series.values() for k in v})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(allx)
    ax.set_xticklabels([str(k) for k in allx], fontsize=8.5)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=9))
    ax.set_xlabel("sampling steps  (= network evaluations)", fontsize=9.5,
                  color=INK_SECONDARY)
    ax.set_ylabel(ylabel or "FID vs train split  (lower is better)", fontsize=9.5,
                  color=INK_SECONDARY)
    ax.grid(True, which="major", color=INK_MUTED, alpha=0.22, lw=0.7, zorder=0)
    ax.grid(True, which="minor", color=INK_MUTED, alpha=0.10, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
    ax.margins(x=0.22 if len(names) <= 4 else 0.06)

    # upper right is the empty quadrant: curves fall left-to-right, and the
    # direct labels sit at the right edge low down, so nothing can collide
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.88)
        fig.suptitle(title, fontsize=11.5, color=INK_PRIMARY, x=0.012, ha="left", y=0.975)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}  ({len(names)} series, {len(allx)} step counts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="runs/scaling.png")
    p.add_argument("--title", default=None)
    p.add_argument("--labels", nargs="*", default=None,
                   help="one prefix per input file, for cross-run comparisons")
    p.add_argument("--floor", type=float, default=None)
    a = p.parse_args()
    paths = [x for x in a.paths if os.path.exists(x)]
    ser, sh = load_series(paths, a.labels)
    plot(ser, a.out, a.title, floor=a.floor, short=sh)
