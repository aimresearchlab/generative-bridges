"""Render FID sweeps as markdown tables."""

from __future__ import annotations

import argparse, json, os
from collections import defaultdict


def table(path: str) -> str:
    rows = json.load(open(path))
    by = defaultdict(dict)
    for r in rows:
        by[r["sampler"]][r["n_steps"]] = r
    steps = sorted({r["n_steps"] for r in rows})
    samplers = [s for s in ("bridge", "euler", "euler_det") if s in by] + \
               [s for s in by if s not in ("bridge", "euler", "euler_det")]

    out = [f"### `{os.path.basename(path)}`  (n={rows[0]['n_samples']}, ckpt step {rows[0]['ckpt_step']})",
           "",
           "FID vs train split; NFE = sampling steps.",
           "",
           "| sampler | " + " | ".join(str(s) for s in steps) + " | best |",
           "|---|" + "---|" * (len(steps) + 1)]
    for s in samplers:
        vals = []
        best = min((by[s][k]["fid_train"] for k in steps if k in by[s]), default=float("nan"))
        for k in steps:
            if k not in by[s]:
                vals.append("-")
                continue
            v = by[s][k]["fid_train"]
            vals.append(f"**{v:.2f}**" if v == best else f"{v:.2f}")
        out.append(f"| `{s}` | " + " | ".join(vals) + f" | {best:.2f} |")

    out += ["", "Monotonicity (FID at max steps minus FID at best steps; >0 means it got worse):", ""]
    for s in samplers:
        ks = [k for k in steps if k in by[s]]
        if not ks:
            continue
        last, best = by[s][ks[-1]]["fid_train"], min(by[s][k]["fid_train"] for k in ks)
        arg = min(ks, key=lambda k: by[s][k]["fid_train"])
        out.append(f"- `{s}`: best {best:.2f} @ {arg} steps, {last:.2f} @ {ks[-1]} steps "
                   f"(**{last - best:+.2f}**)")

    out += ["", "Throughput:", "", "| sampler | steps | imgs/s | s per 10k |", "|---|---|---|---|"]
    for s in samplers:
        for k in steps:
            if k in by[s]:
                r = by[s][k]
                out.append(f"| `{s}` | {k} | {r['imgs_per_s']:.0f} | "
                           f"{10000 / r['imgs_per_s']:.1f} |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    txt = "\n\n".join(table(x) for x in a.paths if os.path.exists(x))
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
