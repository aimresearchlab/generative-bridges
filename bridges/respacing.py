"""Correct posterior coefficients for a respaced diffusion schedule.

What this module is for
------------------
An ancestral sampler that visits a subsequence t_0 > t_1 > ... > t_{n-1} of the
training timesteps must take the jump t_i -> t_{i+1} in one move. Using the
schedule's own per-timestep posterior at t_i instead takes the step
t_i -> t_i - 1, which for a coarse grid is a tiny move followed by a full
variance re-injection. The chain then never denoises: with n = 2 on a cosine
schedule the first step is

    mean = 1.6e-3 * x0_hat + 3.2e-2 * x_t,   then + 1.0 * z

which is indistinguishable from fresh noise, and the run scores at the
noise-level FID of ~418 regardless of how well the model was trained.

This is what OpenAI's improved-diffusion does with `SpacedDiffusion`: it derives
a NEW beta sequence from the subsequence of alpha-bar values, and everything
downstream uses that. The formulation below is equivalent and avoids rebuilding
the object per step count.

For a jump from s to u (u the next grid point, or the data end), with
ab_s = alpha-bar[s] and ab_u = alpha-bar[u] (1 at the data end):

    beta_eff = 1 - ab_s / ab_u
    var      = beta_eff (1 - ab_u) / (1 - ab_s)
    c0       = beta_eff sqrt(ab_u)      / (1 - ab_s)      weight on x0_hat
    ct       = (1 - ab_u) sqrt(ab_s/ab_u) / (1 - ab_s)    weight on x_s

At the final jump ab_u = 1, so c0 = 1, ct = 0 and var = 0: the sampler lands
exactly on its own endpoint prediction, which is the behaviour a last step must
have. DDIM was already correct because it references ab_prev at the next grid
point directly, so only the ancestral path needs it.
"""

from __future__ import annotations

import torch


def respaced_posterior(abar: torch.Tensor, s: int, u: int | None):
    """Posterior coefficients for the jump s -> u. `u=None` means the data end.

    Returns (c0, ct, var, beta_eff) as 0-dim tensors in abar's dtype.
    """
    ab_s = abar[s]
    one = torch.tensor(1.0, dtype=abar.dtype, device=abar.device)
    ab_u = one if u is None else abar[u]
    beta_eff = (one - ab_s / ab_u).clamp(min=0.0, max=1.0)
    denom = (one - ab_s).clamp(min=1e-20)
    var = beta_eff * (one - ab_u) / denom
    c0 = beta_eff * ab_u.sqrt() / denom
    ct = (one - ab_u) * (ab_s / ab_u).sqrt() / denom
    return c0, ct, var, beta_eff


def uniform_grid(T: int, n_steps: int) -> list[int]:
    """Descending subsequence of [0, T-1] with n_steps entries, always ending at 0.

    n_steps == 1 gives [T-1] rather than linspace's degenerate [0]: a one-step
    sampler must start from pure noise, not from the data end.
    """
    if n_steps <= 1:
        return [T - 1]
    return torch.linspace(0, T - 1, n_steps).round().long().flip(0).tolist()
