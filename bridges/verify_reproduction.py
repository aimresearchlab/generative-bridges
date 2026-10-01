"""Is the re-implementation's `euler_det` arm bit-faithful to the original sampler?

The attribution claim ("the published numbers are explained by the deterministic
sampler plus the manuscript's time conditioning") only holds if the reproduction arm
really is the original scheme. This drives both implementations with the same
network and the same initial state and compares trajectories.

Correspondence being tested, with t = sT and gamma^2 = sigma^2 T:
    original grid   t in [0, T-eps],  delta = (T-eps)/n,  drift (f-xi)/(T-t)*delta
    ours            s in [0, 1-eps'], d = (1-eps')/n,     drift (f-xi)/(1-s)*d
    model input     original passes raw t; ours passes s*t_scale with t_scale = T
"""
import math, torch, torch.nn as nn
from simple_bridge import SimpleBrownianBridgeDiffusion
from bridges.bridge import GaussianBridge, BridgeConfig

torch.manual_seed(0)
dev = 'cuda'

class Toy(nn.Module):
    """Small deterministic net that genuinely depends on both x and t."""
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(1, 1, 3, padding=1)
    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1).float()
        return torch.tanh(self.c(x)) * (1 + 2 * t) + 0.3 * t

net = Toy().to(dev).eval()
T = 0.1
gamma = math.sqrt(T)          # sigma = 1 in the original

for n_steps in (2, 10, 100, 1000):
    x0 = torch.randn(16, 1, 32, 32, device=dev)

    orig = SimpleBrownianBridgeDiffusion(T, net, dev, trainY=None, deterministic=True)
    with torch.no_grad():
        # replicate the original loop exactly, but from our fixed x0
        t_values = torch.linspace(0, T, steps=n_steps + 1, device=dev) * (T - 1e-7) / T
        delta = (T - 1e-7) / n_steps
        xi = x0.clone()
        for t in t_values[:-1]:
            t_list = torch.stack([t for _ in range(16)])
            xi = xi + (net(xi, t_list) - xi) / (T - t) * delta
        xi_orig = xi

    ours = GaussianBridge(BridgeConfig(gamma=gamma, t_scale=T))
    with torch.no_grad():
        xi_ours = ours.sample(net, x0.shape, dev, n_steps=n_steps, sampler='euler_det',
                              x_init=x0.clone(), return_endpoint=False)

    d = (xi_orig - xi_ours).abs().max().item()
    rel = d / xi_orig.abs().max().item()
    print(f"n_steps={n_steps:>5}: max|orig - ours| = {d:.3e}  (rel {rel:.3e})")

# and the deliberate deviation: reporting the endpoint estimate vs the raw state
print()
for n_steps in (2, 10, 100):
    x0 = torch.randn(16, 1, 32, 32, device=dev)
    ours = GaussianBridge(BridgeConfig(gamma=gamma, t_scale=T))
    with torch.no_grad():
        a = ours.sample(net, x0.shape, dev, n_steps=n_steps, sampler='euler',
                        x_init=x0.clone(), return_endpoint=False,
                        generator=torch.Generator(device=dev).manual_seed(1))
        b = ours.sample(net, x0.shape, dev, n_steps=n_steps, sampler='euler',
                        x_init=x0.clone(), return_endpoint=True,
                        generator=torch.Generator(device=dev).manual_seed(1))
    print(f"stochastic euler, n_steps={n_steps:>4}: std of raw state {a.std():.4f} vs "
          f"endpoint estimate {b.std():.4f}  (residual bridge noise sd ~ "
          f"{gamma*math.sqrt(1.0/n_steps):.4f})")
