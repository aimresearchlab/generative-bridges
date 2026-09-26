"""Build the paper's tables from the sweep JSONs.

Emits LaTeX (booktabs) and Markdown for the same content, so the numbers in the
manuscript and the numbers in the repo cannot drift apart: both are generated
from the JSON the evaluation wrote.

Covers what the review asked for beyond raw FID -- network evaluations,
wall-clock, throughput -- and reports mean +/- sd wherever repeated sampling
seeds exist rather than a single unqualified number.
"""

from __future__ import annotations

import argparse, glob, json, os, statistics as st
from collections import defaultdict

# display name -> (run directory, sweep-file glob)
RUNS = [
    ("Bridge (exact posterior)",   "mnist_g0316_t1000", "bridge"),
    ("Bridge (Euler-Maruyama)",    "mnist_g0316_t1000", "euler"),
    ("Bridge (deterministic Euler)", "mnist_g0316_t1000", "euler_det"),
    ("Flow matching",              "mnist_flow",        "flow_euler"),
    ("DDPM cosine + DDIM",         "mnist_ddpm_cosine", "ddim"),
    ("DDPM cosine ancestral",      "mnist_ddpm_cosine", "ddpm_ancestral"),
    ("DDPM linear + DDIM",         "mnist_ddpm_linear", "ddim"),
    ("DDPM linear ancestral",      "mnist_ddpm_linear", "ddpm_ancestral"),
]


def collect(root: str, run_dir: str, sampler: str, n_samples: int | None = None,
            pattern: str = "fid_*.json"):
    """-> {nfe: (fid_train_mean, sd, fid_test_mean, n_seeds, imgs_per_s)}

    `n_samples` filters strictly: FID is only comparable at a fixed sample
    count, so mixing sweeps run at different n into one mean would be wrong.
    """
    tr, te, sp = defaultdict(list), defaultdict(list), {}
    for path in sorted(glob.glob(os.path.join(root, run_dir, pattern))):
        try:
            rows = json.load(open(path))
        except Exception:
            continue
        for r in rows:
            if r.get("sampler") != sampler:
                continue
            if n_samples is not None and r.get("n_samples") != n_samples:
                continue
            tr[r["n_steps"]].append(r["fid_train"])
            te[r["n_steps"]].append(r["fid_test"])
            sp[r["n_steps"]] = r.get("imgs_per_s")
    out = {}
    for n in sorted(tr):
        out[n] = (st.mean(tr[n]), st.stdev(tr[n]) if len(tr[n]) > 1 else None,
                  st.mean(te[n]), len(tr[n]), sp.get(n))
    return out


def fmt(v, sd, prec=2):
    if v is None:
        return "--"
    return f"{v:.{prec}f}" if sd is None else f"{v:.{prec}f} $\\pm$ {sd:.2f}"


def fmt_md(v, sd, prec=2):
    if v is None:
        return "--"
    return f"{v:.{prec}f}" if sd is None else f"{v:.{prec}f} ±{sd:.2f}"


def build(root, steps, runs, dataset, n_samples, floor, pattern="fid_*.json"):
    data = {name: collect(root, d, s, n_samples=n_samples, pattern=pattern)
            for name, d, s in runs}
    data = {k: v for k, v in data.items() if v}
    if not data:
        return None, None

    hdr = " & ".join(str(s) for s in steps)
    tex = [
        r"\begin{table}[t]",
        r"\caption{FID on %s against the number of sampling steps, which equals the "
        r"number of network evaluations. All rows share one UNet, optimiser, EMA "
        r"decay, batch size and training budget, and are measured through an "
        r"identical FID implementation; only the generative process and its sampler "
        r"differ. %d generated samples per entry; the achievable floor, measured "
        r"between disjoint real subsets of the same size, is %.2f. Values are "
        r"mean $\pm$ sd over sampling seeds where repeats were run.}"
        % (dataset, n_samples, floor),
        r"\label{tab:fid-%s}" % dataset.lower().replace("-", ""),
        r"\centering", r"\small",
        r"\begin{tabular}{l" + "c" * len(steps) + "}", r"\toprule",
        r"Model & \multicolumn{%d}{c}{Sampling steps (NFE)} \\" % len(steps),
        r"\cmidrule(lr){2-%d}" % (len(steps) + 1),
        " & " + hdr + r" \\", r"\midrule",
    ]
    md = ["| Model | " + " | ".join(str(s) for s in steps) + " |",
          "|---|" + "---|" * len(steps)]

    # best per column, for bolding
    best = {}
    for s in steps:
        vals = [(d[s][0], n) for n, d in data.items() if s in d]
        if vals:
            best[s] = min(vals)[1]

    for name, d in data.items():
        cells, cells_md = [], []
        for s in steps:
            if s not in d:
                cells.append("--"); cells_md.append("--"); continue
            v, sd, _, _, _ = d[s]
            c, c_md = fmt(v, sd), fmt_md(v, sd)
            if best.get(s) == name:
                c, c_md = r"\textbf{%s}" % c, f"**{c_md}**"
            cells.append(c); cells_md.append(c_md)
        tex.append(f"{name} & " + " & ".join(cells) + r" \\")
        md.append(f"| {name} | " + " | ".join(cells_md) + " |")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    # cost table
    tex += ["", r"\begin{table}[t]",
            r"\caption{Sampling cost on %s. Throughput is images per second for the "
            r"whole batch on one GB10; NFE is exactly the number of sampling steps "
            r"for every method, so the columns are comparable at equal cost.}" % dataset,
            r"\centering", r"\small",
            r"\begin{tabular}{l" + "c" * len(steps) + "}", r"\toprule",
            r"Model (images/s) & " + hdr + r" \\", r"\midrule"]
    md += ["", "| Model (images/s) | " + " | ".join(str(s) for s in steps) + " |",
           "|---|" + "---|" * len(steps)]
    for name, d in data.items():
        cells = [f"{d[s][4]:.0f}" if s in d and d[s][4] else "--" for s in steps]
        tex.append(f"{name} & " + " & ".join(cells) + r" \\")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    return "\n".join(tex), "\n".join(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    p.add_argument("--dataset", default="MNIST")
    p.add_argument("--prefix", default="mnist", help="run-directory prefix")
    p.add_argument("--steps", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 50, 100, 200, 1000])
    p.add_argument("--n_samples", type=int, default=5000)
    p.add_argument("--floor", type=float, default=1.50)
    p.add_argument("--out_tex", default="runs/tables.tex")
    p.add_argument("--out_md", default="runs/tables.md")
    p.add_argument("--pattern", default="fid_*.json",
                   help="which sweep files to read, e.g. fid_n50k*.json")
    a = p.parse_args()

    runs = [(n, d.replace("mnist", a.prefix), s) for n, d, s in RUNS]
    tex, md = build(a.root, a.steps, runs, a.dataset, a.n_samples, a.floor,
                    pattern=a.pattern)
    if tex is None:
        print("no sweep data found"); return
    open(a.out_tex, "w").write(tex + "\n")
    open(a.out_md, "w").write(md + "\n")
    print(md)
    print(f"\nwrote {a.out_tex} and {a.out_md}")


if __name__ == "__main__":
    main()
