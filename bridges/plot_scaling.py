"""FID against sampling steps, per sampler.

Form: the question is whether quality improves as the discretisation is refined,
for three sampler variants -- a change-over-a-quantitative-x comparison, so a
line chart. Both axes are logarithmic: steps span 1..1000, and on log-log a
power-law decay is a straight line while a plateau is a flat one, which is
exactly the distinction being tested.

Colour: categorical (series identity), slots 1-3 of the reference palette in
fixed order. That three-slot set is the documented all-pairs-validated subset in
both modes. Slot 3 (aqua) sits below 3:1 on the light surface, so the relief
rule applies and every series carries a direct label in addition to the legend;
identity is never carried by colour alone.
"""

from __future__ import annotations

import argparse, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SERIES = {"bridge": "#2a78d6", "euler": "#eb6834", "euler_det": "#1baf7a"}
LABEL = {
    "bridge": "bridge (exact posterior)",
    "euler": "euler-maruyama (Alg. 2)",
    "euler_det": "deterministic euler (as published)",
}


def load(path):
    by = defaultdict(dict)
    for r in json.load(open(path)):
        by[r["sampler"]][r["n_steps"]] = r["fid_train"]
    return by


def plot(paths, out, title=None):
    n = len(paths)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.9), squeeze=False,
                             facecolor=SURFACE)
    for ax, path in zip(axes[0], paths):
        by = load(path)
        ax.set_facecolor(SURFACE)
        for name in ("bridge", "euler", "euler_det"):        # fixed order, never cycled
            if name not in by:
                continue
            xs = sorted(by[name])
            ys = [by[name][x] for x in xs]
            ax.plot(xs, ys, color=SERIES[name], lw=2.0, marker="o", ms=5.5,
                    mec=SURFACE, mew=1.2, label=LABEL[name], zorder=3, clip_on=False)
            # direct label at the right-hand end (relief rule; identity never colour-alone)
            ax.annotate(name, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=8.5,
                        color=INK_SECONDARY, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("sampling steps  (= network evaluations)", fontsize=9.5,
                      color=INK_SECONDARY)
        ax.set_ylabel("FID vs train split  (lower is better)", fontsize=9.5,
                      color=INK_SECONDARY)
        sub = os.path.basename(path).replace("fid_sweep_", "").replace(".json", "")
        ax.set_title(sub, fontsize=10.5, color=INK_PRIMARY, loc="left", pad=9)
        ax.grid(True, which="major", color=INK_MUTED, alpha=0.22, lw=0.7, zorder=0)
        ax.grid(True, which="minor", color=INK_MUTED, alpha=0.10, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK_MUTED); ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8.5)
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)       # text wears ink tokens, not series colour
        ax.margins(x=0.13)

    if title:
        fig.suptitle(title, fontsize=12, color=INK_PRIMARY, x=0.005, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1.0))
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="runs/scaling.png")
    p.add_argument("--title", default=None)
    a = p.parse_args()
    plot([x for x in a.paths if os.path.exists(x)], a.out, a.title)
