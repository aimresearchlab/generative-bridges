"""Where the endpoint-trained bridge loses the flow-matching net: the terminal region."""
import json, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--bare", action="store_true")
p.add_argument("--out", default="runs/figures/drift_gap")
a = p.parse_args()

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for k, (f, title) in enumerate([("runs/mse_by_s_cifar.json", "CIFAR-10"),
                                ("runs/mse_by_s_mnist.json", "MNIST")]):
    rows = json.load(open(f))
    s = [r["s"] for r in rows]
    keys = [q for q in rows[0] if q != "s"]
    for q, c, m in zip(keys, ["C0", "C1", "C2"], ["o", "s", "^"]):
        y = [r[q] / (1 - r["s"]) ** 2 for r in rows]
        ax[k].plot(s, y, marker=m, color=c, ms=4, label=q)
    ax[k].set_yscale("log"); ax[k].set_xlabel("bridge time $s$")
    ax[k].set_title(title, fontsize=10)
    ax[k].grid(alpha=.3, which="both", lw=.4)
ax[0].set_ylabel(r"drift MSE  $\|\hat b - b\|^2$")
ax[0].legend(fontsize=8, frameon=False)
if not a.bare:
    fig.suptitle("Error in the drift the sampler integrates, along the path", fontsize=11)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}", dpi=160, bbox_inches="tight")
print(f"wrote {a.out}.png/.pdf")
