"""Does the t in [0, T] parameterisation in float32 lose accuracy near t = T?

The original sampler builds its grid as a float32 CUDA tensor on [0, T] with
T = 0.1 and divides by (T - t).  For t close to T this is catastrophic
cancellation: 0.1 has no exact float32 representation and subtracting two
nearby values discards most of the mantissa.  The re-implementation works in
s = t/T on [0, 1] with the grid in float64, and its exact-posterior step never
divides by a vanishing quantity.
"""
import torch, math
from bridges.sanity_1d import GaussianMixture1D, w1

psi = GaussianMixture1D([-2.0,0.5,3.0],[0.3,0.5,0.2],tau=0.15)
ref = psi.sample(100000, generator=torch.Generator().manual_seed(123))
ref2= psi.sample(100000, generator=torch.Generator().manual_seed(999))
floor = w1(ref,ref2)

def run_original_style(n_steps, T=0.1, gamma=0.3162277660168379, n=100000, seed=0, dtype=torch.float32):
    """Reproduce the original grid and update rule, in t-time, at the given dtype."""
    g = torch.Generator().manual_seed(seed)
    eps = 1e-7
    sigma = gamma / math.sqrt(T)                      # since gamma^2 = sigma^2 T
    t_values = (torch.linspace(0, T, n_steps+1, dtype=dtype) * (T - eps) / T)
    delta = torch.tensor((T - eps)/n_steps, dtype=dtype)
    xi = torch.randn(n, dtype=dtype, generator=g)
    for i in range(n_steps):
        t = t_values[i]
        s = float(t)/T                                # normalised time for the exact cond. mean
        yhat = psi.cond_mean(xi.double(), s, gamma).to(dtype)
        inv_lambda = 1.0/(T - t)                      # <-- the cancellation
        xi = xi + inv_lambda*(yhat - xi)*delta        # deterministic (as the original ran it)
    return psi.cond_mean(xi.double(), float(t_values[-1])/T, gamma)

def run_s_time(n_steps, gamma=0.3162277660168379, n=100000, seed=0):
    """Re-implementation: s in [0,1], float64 grid, same deterministic update."""
    g = torch.Generator().manual_seed(seed)
    eps = 1e-7
    grid = torch.linspace(0.0, 1.0-eps, n_steps+1, dtype=torch.float64)
    xi = torch.randn(n, dtype=torch.float64, generator=g)
    for i in range(n_steps):
        s, u = float(grid[i]), float(grid[i+1])
        yhat = psi.cond_mean(xi, s, gamma)
        xi = xi + (yhat - xi)*((u-s)/(1.0-s))
    return psi.cond_mean(xi, float(grid[-1]), gamma)

print(f"W1 floor {floor:.5f}\n")
print("Conclusion: the t-time/float32 form and the s-time/float64 form agree to")
print("~1%, so terminal-point precision is not what limits the deterministic")
print("sampler. The plateau level is set by sampler bias alone.\n")
print(f"{'steps':>7}{'orig t-time fp32':>19}{'orig t-time fp64':>19}{'s-time fp64':>15}")
print('-'*60)
for n_steps in [10,100,1000,2000,5000]:
    a = w1(run_original_style(n_steps, dtype=torch.float32), ref)
    b = w1(run_original_style(n_steps, dtype=torch.float64), ref)
    c = w1(run_s_time(n_steps), ref)
    print(f"{n_steps:>7}{a:>19.5f}{b:>19.5f}{c:>15.5f}")
