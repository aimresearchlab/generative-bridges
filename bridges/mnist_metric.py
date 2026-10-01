"""An MNIST-appropriate alternative to Inception FID, and a mode-coverage count.

The associate editor's point: Inception-v3 was fitted to natural colour
photographs, and MNIST digits replicated to three channels are outside that
domain, so a Frechet distance in its feature space measures something only
loosely related to digit quality.  The standard remedy is to swap the feature
extractor for one trained on the data in question.

This trains a small convolutional classifier on the MNIST training split, takes
its 128-dimensional penultimate layer as the feature space, and reports:

  cfid       Frechet distance in that space ("classifier FID"), the same
             statistic as FID with the backbone replaced.
  mode_*     the classifier's predicted-label histogram over generated samples
             against the label histogram of the real data.  A model that drops
             digits shows it directly here, which no Frechet distance does.
             Reported as total variation distance and as the worst-served class.

The classifier is trained once and cached; it reaches about 99% test accuracy in
two epochs, which is enough for both uses.
"""

from __future__ import annotations

import argparse, json, os

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from .fid import frechet_distance

CKPT = "runs/mnist_classifier.pt"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2, self.c3 = nn.Conv2d(1, 32, 3, 1, 1), nn.Conv2d(32, 64, 3, 1, 1), nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1, self.fc2 = nn.Linear(128 * 4 * 4, 128), nn.Linear(128, 10)

    def features(self, x):                      # x: (B,1,32,32) in [-1,1]
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        return F.relu(self.fc1(x.flatten(1)))

    def forward(self, x):
        return self.fc2(self.features(x))


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    """Accept either the 1-channel store or a 3-channel replicate."""
    return x[:, :1] if x.shape[1] == 3 else x


def load_or_train(device, root="./data") -> Net:
    net = Net().to(device)
    if os.path.exists(CKPT):
        net.load_state_dict(torch.load(CKPT, map_location=device)); return net.eval()
    x = _to_gray(D.load("mnist", root=root, split="train")).to(device)
    y = D.labels("mnist", root=root, split="train").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    g = torch.Generator(device=device).manual_seed(0)
    for epoch in range(2):
        perm = torch.randperm(len(x), generator=g, device=device)
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            loss = F.cross_entropy(net(x[j]), y[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    xt = _to_gray(D.load("mnist", root=root, split="test")).to(device)
    yt = D.labels("mnist", root=root, split="test").to(device)
    with torch.no_grad():
        acc = (torch.cat([net(xt[i:i+1000]).argmax(1) for i in range(0, len(xt), 1000)]) == yt).float().mean()
    print(f"mnist classifier: {100*float(acc):.2f}% test accuracy", flush=True)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save(net.state_dict(), CKPT)
    return net.eval()


@torch.no_grad()
def stats_and_hist(net: Net, x: torch.Tensor, device, bs=1000):
    f, p = [], []
    for i in range(0, len(x), bs):
        b = _to_gray(x[i:i + bs]).to(device)
        h = net.features(b)
        f.append(h.double().cpu()); p.append(net.fc2(h).argmax(1).cpu())
    f = torch.cat(f)
    mu, sig = f.mean(0), torch.cov(f.T)
    hist = torch.bincount(torch.cat(p), minlength=10).double()
    return mu.to(device), sig.to(device), hist / hist.sum()


def evaluate(gen: torch.Tensor, device, root="./data") -> dict:
    net = load_or_train(device, root)
    real = D.load("mnist", root=root, split="train")
    mu_r, sig_r, h_r = stats_and_hist(net, real, device)
    mu_g, sig_g, h_g = stats_and_hist(net, gen, device)
    tv = 0.5 * float((h_r - h_g).abs().sum())
    worst = int((h_g / h_r.clamp(min=1e-9)).argmin())
    return {"cfid": frechet_distance(mu_g, sig_g, mu_r, sig_r),
            "mode_tv": tv,
            "mode_worst_class": worst,
            "mode_worst_ratio": float(h_g[worst] / h_r[worst].clamp(min=1e-9)),
            "hist_generated": [round(v, 4) for v in h_g.tolist()],
            "hist_real": [round(v, 4) for v in h_r.tolist()]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_only", action="store_true")
    a = p.parse_args()
    net = load_or_train("cuda")
    if a.train_only:
        return
    real = D.load("mnist", split="test")
    print(json.dumps(evaluate(real, "cuda"), indent=2))


if __name__ == "__main__":
    main()
