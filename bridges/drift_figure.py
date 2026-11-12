"""Velocity-field error along the bridge, for the endpoint- and velocity-trained
networks on both datasets.  Axes follow the manuscript's time convention t in [0,T]."""
import json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--bare", action="store_true")
p.add_argument("--out", default="runs/figures/drift_gap")
a = p.parse_args()

LBL = {"bridge g0.05": r"endpoint, $\gamma=0.05$",
       "bridge g0.1": r"endpoint, $\gamma=0.1$",
       "bridge g0.316": r"endpoint, $\gamma=0.3162$",
       "flow": r"velocity-trained"}

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for k, (f, title) in enumerate([("runs/mse_by_s_cifar.json", "CIFAR-10"),
                                ("runs/mse_by_s_mnist.json", "MNIST")]):
    rows = json.load(open(f))
    s = [r["s"] for r in rows]
    keys = [q for q in rows[0] if q != "s"]
    for q, c, m in zip(keys, ["C0", "C1", "C2"], ["o", "s", "^"]):
        y = [r[q] / (1 - r["s"]) ** 2 for r in rows]
        ax[k].plot(s, y, marker=m, color=c, ms=4, label=LBL.get(q, q))
    ax[k].set_yscale("log")
    ax[k].set_xlabel(r"$t/T$")
    ax[k].set_title(title, fontsize=10)
    ax[k].grid(alpha=.3, which="both", lw=.4)
ax[0].set_ylabel(r"velocity MSE  $\|\hat{u}-u\|^{2}$")
ax[0].legend(fontsize=8, frameon=False)
if not a.bare:
    fig.suptitle("Error in the velocity field the sampler integrates", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=160, bbox_inches="tight")
print(f"wrote {a.out}.png/.pdf")
