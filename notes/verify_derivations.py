"""Numerical check of Propositions 2 and 3 in levy_bridge_sampling.tex.

Each proposition is checked against the *defining* construction: simulate the
driving process's independent increments, condition on the total increment, and
compare the conditional law of the first increment with the claimed Beta /
Binomial law. The identities under test are never used to generate the data.
"""
import torch

torch.manual_seed(0)
N = 4_000_000
DT = torch.float64


def two_sample_ks(a, b):
    a, b = a.sort().values.contiguous(), b.sort().values.contiguous()
    grid = torch.cat([a, b]).sort().values
    return float((torch.searchsorted(a, grid).double() / len(a)
                  - torch.searchsorted(b, grid).double() / len(b)).abs().max())


def gamma(shape, rate, n):
    return torch.distributions.Gamma(torch.tensor(shape, dtype=DT),
                                     torch.tensor(rate, dtype=DT)).sample((n,))


print("=" * 78)
print("Proposition 2:  xi_t = xi_s + (y - xi_s) * Beta(kappa(t-s), kappa(1-t))")
print("=" * 78)
print(f"{'kappa':>6}{'s':>6}{'t':>6}{'KS':>10}{'floor':>9}{'mean':>10}{'(t-s)/(1-s)':>13}"
      f"{'var':>11}{'var(thy)':>11}")
print("-" * 78)
for kappa in (3.0, 20.0):
    for (s, t) in [(0.0, 0.3), (0.2, 0.5), (0.5, 0.9), (0.9, 0.95)]:
        a, b = kappa * (t - s), kappa * (1.0 - t)
        A, B = gamma(a, kappa, N), gamma(b, kappa, N)
        frac = A / (A + B)
        ref = torch.distributions.Beta(torch.tensor(a, dtype=DT),
                                       torch.tensor(b, dtype=DT)).sample((N,))
        ref2 = torch.distributions.Beta(torch.tensor(a, dtype=DT),
                                        torch.tensor(b, dtype=DT)).sample((N,))
        print(f"{kappa:>6.0f}{s:>6.2f}{t:>6.2f}{two_sample_ks(frac, ref):>10.5f}"
              f"{two_sample_ks(ref, ref2):>9.5f}{float(frac.mean()):>10.5f}"
              f"{(t - s) / (1 - s):>13.5f}{float(frac.var()):>11.6f}"
              f"{a * b / ((a + b) ** 2 * (a + b + 1)):>11.6f}")

print("\nNear-terminal degeneracy (kappa(1-t) << 1): Beta concentrates so hard at 1")
print("that (xi_t - xi_s)/(y - xi_s) rounds to exactly 1 in floating point.")
for kappa, t in [(3.0, 0.99), (3.0, 0.999)]:
    a, b = kappa * t, kappa * (1 - t)
    A, B = gamma(a, kappa, N), gamma(b, kappa, N)
    frac = A / (A + B)
    print(f"  kappa={kappa}, t={t}: Beta({a:.3f},{b:.4f}) -> frac==1.0 exactly in "
          f"{float((frac == 1).double().mean()) * 100:.1f}% of draws "
          f"(mean {float(frac.mean()):.6f} vs theory {a/(a+b):.6f})")
print("  This is correct behaviour for the sampler -- it lands on the endpoint -- but")
print("  an implementation should special-case the final step rather than rely on it.")

print()
print("=" * 78)
print("Proposition 3:  xi_t = xi_s + Binomial(y - xi_s, (t-s)/(1-s))")
print("=" * 78)
print(f"{'lambda':>7}{'s':>6}{'t':>6}{'m':>4}{'kept':>10}{'TVD':>9}{'mean':>10}"
      f"{'m*p':>10}{'var':>10}{'m*p(1-p)':>11}")
print("-" * 78)
for lam in (7.0, 30.0):
    for (s, t) in [(0.0, 0.3), (0.2, 0.5), (0.5, 0.9), (0.9, 0.95)]:
        m = int(round(lam * (1.0 - s)))          # test at the mode of the total
        n1 = torch.poisson(torch.full((N,), lam * (t - s), dtype=DT))
        n2 = torch.poisson(torch.full((N,), lam * (1.0 - t), dtype=DT))
        j = n1[(n1 + n2) == m]
        p = (t - s) / (1.0 - s)
        binom = torch.distributions.Binomial(m, torch.tensor(p, dtype=DT))
        tvd = 0.5 * sum(abs(float((j == k).double().mean())
                            - float(binom.log_prob(torch.tensor(float(k), dtype=DT)).exp()))
                        for k in range(m + 1))
        print(f"{lam:>7.0f}{s:>6.2f}{t:>6.2f}{m:>4}{len(j):>10}{tvd:>9.5f}"
              f"{float(j.mean()):>10.4f}{m * p:>10.4f}{float(j.var()):>10.4f}{m * p * (1 - p):>11.4f}")

print("\nKS and TVD at the 1e-3 level are the Monte-Carlo floors at these sample sizes.")
