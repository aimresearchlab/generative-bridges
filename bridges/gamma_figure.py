"""FID against sampling steps for several values of the bridge noise scale.

gamma^2 = sigma^2 T is the method's only free parameter. The manuscript varies
it without saying so: Section 3 reports T = 0.1 "after preliminary parameter
tuning" while Section 2.4 defers sigma to future work, but training and sampling
depend on the two only through gamma, so those are one knob and one tuning.

The 1-D study (bridges/sanity_1d.py) predicts the shape. With the conditional
expectation exact, the deterministic sampler's residual 1-Wasserstein error at
2000 steps rose monotonically with gamma -- 1.2x the Monte-Carlo floor at
gamma = 0.032, 6.9x at 0.316, 41.9x at 1.0, 114x at 2.0, the terminal standard
deviation collapsing from 1.758 to 0.775 -- while the correct samplers sat at
the floor for every gamma. The missing score term is (gamma^2/2) grad log p_s,
so its bias should scale with gamma and vanish as gamma -> 0, where the scheme
degenerates to a deterministic interpolant.
"""
from __future__ import annotations
import argparse, glob, json, os, re, textwrap
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mticker

SURFACE="#fcfcfb"; INK_P="#0b0b0b"; INK_S="#52514e"; INK_M="#8a8985"
PALETTE=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]
PUBLISHED = 0.3162


def collect(root, n_samples):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "mnist_g*"))):
        f = os.path.join(d, "fid_gamma_n10k.json")
        if not os.path.exists(f):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        g = float(cfg.get("gamma", PUBLISHED))
        by = defaultdict(dict)
        for r in json.load(open(f)):
            if r.get("n_samples") == n_samples:
                by[r["sampler"]][r["n_steps"]] = r["fid_train"]
        if by:
            out[g] = by
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--floor", type=float, default=0.70)
    p.add_argument("--out", default="runs/figures/mnist_gamma.png")
    a = p.parse_args()
    data = collect(a.root, a.n_samples)
    if not data:
        print("no gamma sweep data yet"); return

    fig = plt.figure(figsize=(9.4, 7.4), facecolor=SURFACE)
    axes = [fig.add_axes([0.085, 0.435, 0.415, 0.42]),
            fig.add_axes([0.565, 0.435, 0.415, 0.42])]
    allv = [v for by in data.values() for sr in by.values() for v in sr.values()]
    lo, hi = min(min(allv), a.floor)*0.75, max(allv)*1.4

    for ax, samp, title in zip(axes, ("euler", "euler_det"),
                               ("Euler-Maruyama (correct sampler)",
                                "Deterministic Euler (biased sampler)")):
        ax.set_facecolor(SURFACE)
        ax.axhline(a.floor, color=INK_M, lw=1.0, ls=(0,(4,3)), zorder=1)
        for i,(g, by) in enumerate(data.items()):
            if samp not in by: continue
            xs = sorted(by[samp]); ys=[by[samp][x] for x in xs]
            lbl = f"$\\gamma$ = {g:g}" + ("  (published)" if abs(g-PUBLISHED)<1e-3 else "")
            ax.plot(xs, ys, color=PALETTE[i%8], lw=2.2 if abs(g-PUBLISHED)<1e-3 else 1.8,
                    marker="o", ms=4.8, mec=SURFACE, mew=1.0, label=lbl, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lo,hi)
        xs_all = sorted({k for by in data.values() for sr in by.values() for k in sr})
        ax.set_xticks(xs_all); ax.set_xticklabels([str(k) for k in xs_all], fontsize=8)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel("network evaluations", fontsize=9.5, color=INK_S)
        ax.set_title(title, fontsize=10, color=INK_P, loc="left", pad=7)
        ax.grid(True, which="major", color=INK_M, alpha=0.22, lw=0.7, zorder=0)
        for sp in ("top","right"): ax.spines[sp].set_visible(False)
        for sp in ("left","bottom"): ax.spines[sp].set_color(INK_M); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK_S, labelsize=8.5)
        leg=ax.legend(frameon=False, fontsize=8.2, loc="upper right")
        for t in leg.get_texts(): t.set_color(INK_S)
    axes[0].set_ylabel("FID  (lower is better)", fontsize=9.5, color=INK_S)

    fig.text(0.085, 0.965, r"Sensitivity to the bridge noise scale $\gamma^2=\sigma^2 T$",
             fontsize=13.5, color=INK_P, ha="left", va="top")
    fig.text(0.085, 0.925,
             "One UNet, budget and evaluation throughout; only the driving process's scale changes.",
             fontsize=9.5, color=INK_S, ha="left", va="top")
    cap = (
        "gamma is the method's only free parameter. The manuscript tunes it without saying so: "
        "Section 3 reports T = 0.1 after preliminary tuning while Section 2.4 defers sigma to "
        "future work, but training and sampling depend on the two only through gamma^2 = sigma^2 T, "
        "so these are one knob and one tuning. "
        "The prediction comes from the one-dimensional study, where the conditional expectation is "
        "exact and sampler error is isolated: at 2000 steps the deterministic sampler's residual "
        "1-Wasserstein error rose monotonically with gamma -- 1.2x the Monte-Carlo floor at 0.032, "
        "6.9x at 0.316, 41.9x at 1.0 and 114x at 2.0, where the terminal standard deviation "
        "collapsed from 1.758 to 0.775 -- while the correct samplers stayed at the floor for every "
        "gamma. The term the deterministic map omits is (gamma^2/2) grad log p_s, so its bias "
        "should scale with gamma and vanish as gamma -> 0. Right panel tests that on images; left "
        "panel shows what the correct sampler does with the same models."
    )
    fig.text(0.085, 0.345, "\n".join(textwrap.wrap(cap, 112)), fontsize=8.2,
             color=INK_S, ha="left", va="top", linespacing=1.6)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=190, facecolor=SURFACE)
    print(f"wrote {a.out}  ({len(data)} gamma values)")


if __name__ == "__main__":
    main()
