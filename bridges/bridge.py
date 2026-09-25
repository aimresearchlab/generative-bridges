"""Gaussian random-bridges as stochastic transports.

Derivation (from first principles, following Cor. 2.8 / Prop. 2.10 of the paper
but reparameterised so that the free parameters are identifiable).

Setup
-----
Source  x ~ Phi   (here: N(0, I), the "noise" end)
Target  y ~ Psi   (here: image data in [-1, 1])
Driver  Z_t = x + sigma W_t  on t in [0, T].

The Phi-initialised Gaussian random bridge to Psi has, conditional on the
endpoints (x, y),

    xi_t | (x, y) ~ N( x + (t/T)(y - x),  sigma^2 * t(T - t)/T * I ).

Reparameterisation
------------------
Put s = t/T in [0, 1] and gamma^2 := sigma^2 * T.  Then

    xi_s = (1 - s) x + s y + gamma * sqrt(s(1 - s)) * eps,    eps ~ N(0, I).

*** (sigma, T) is a redundant parameterisation: training and sampling depend on
them only through the single scalar gamma^2 = sigma^2 * T. ***

The original code fixes sigma = 1 and tunes T, so the published "T = 0.1"
is the statement gamma = sqrt(0.1) ~= 0.3162.  We expose `gamma` directly.

Learning target
---------------
The network regresses the L^2 best estimate of the endpoint,

    f_theta(xi_s, s) ~= E[Y | xi_s],        loss = E || y - f_theta(xi_s, s) ||^2

with s ~ U[0, 1).  This is an endpoint ("x0") prediction, not a score and not a
noise prediction.

Sampling
--------
In s-time the non-anticipative SDE is

    d xi_s = (E[Y | xi_s] - xi_s) / (1 - s) ds + gamma dB_s.                (*)

Three samplers are provided:

`euler`         Euler-Maruyama on (*).  This is Algorithm 2 of the paper.

`euler_det`     (*) with the diffusion term deleted.  This is what the original
                repository actually ran at evaluation time
                (sample_bridges.py passes deterministic=True).  It is BIASED:
                deleting gamma dB_s without adding the score correction
                -(gamma^2/2) grad log p_s does NOT give the probability-flow
                ODE of (*), so its Delta -> 0 limit is not Psi.  Kept only to
                reproduce the published numbers.

`bridge`        Exact Brownian-bridge posterior step (recommended).  Given the
                current state xi_s and the predicted endpoint yhat, the law of
                xi_u for s < u <= 1 under the bridge pinned at (s, xi_s) and
                (1, yhat) is exactly Gaussian:

                    mean = xi_s + (u - s)/(1 - s) * (yhat - xi_s)
                    var  = gamma^2 * (u - s)(1 - u) / (1 - s)

                The variance vanishes at u = 1, so the terminal point is hit
                exactly and no epsilon offset / singularity handling is needed.
                A stochasticity knob `eta` in [0, 1] scales the injected
                standard deviation (eta = 0 recovers a deterministic bridge
                flow, eta = 1 the full posterior).  To first order in
                (u - s) this agrees with `euler`, so it is the exact version of
                the same dynamics.

Note on `bridge`: replacing the posterior pi_s(dy) by a point mass at its mean
yhat mirrors DDPM/DDIM ancestral sampling and under-disperses by
Var[Y | xi_s].  Prop. 2.13 of the paper describes exactly that quantity; a
learned variance head could restore it.  See `GaussianBridge.step` for the hook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class BridgeConfig:
    gamma: float = 0.3162277660168379  # sqrt(0.1): matches published sigma=1, T=0.1
    t_scale: float = 1000.0            # s in [0,1] is scaled by this before the
                                       # sinusoidal time embedding, so the
                                       # embedding sees the dynamic range it was
                                       # designed for (see unet.timestep_embedding)


class GaussianBridge:
    """Training marginals and samplers for a Phi-initialised Gaussian bridge."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.gamma = float(cfg.gamma)

    # ---------------------------------------------------------------- training

    def sample_marginal(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw xi_s ~ N((1-s)x + sy, gamma^2 s(1-s) I), exactly and in one shot.

        x, y : (B, C, H, W)   s : (B,)
        """
        s_ = s.view(-1, *([1] * (x.ndim - 1)))
        if noise is None:
            noise = torch.randn_like(x)
        std = self.gamma * torch.sqrt(torch.clamp(s_ * (1.0 - s_), min=0.0))
        return (1.0 - s_) * x + s_ * y + std * noise

    def training_loss(self, model, y: torch.Tensor, x: torch.Tensor | None = None):
        """MSE between the true endpoint y and the network's estimate of E[Y|xi_s]."""
        b = y.shape[0]
        if x is None:
            x = torch.randn_like(y)  # independent coupling Gamma = Phi (x) Psi
        s = torch.rand(b, device=y.device, dtype=y.dtype)
        xi = self.sample_marginal(x, y, s)
        pred = model(xi, s * self.cfg.t_scale)
        return torch.nn.functional.mse_loss(pred, y), {"s": s, "xi": xi, "pred": pred}

    # ---------------------------------------------------------------- sampling

    @staticmethod
    def timesteps(n_steps: int, device, eps: float = 0.0) -> torch.Tensor:
        """Uniform grid 0 = s_0 < ... < s_n = 1 - eps."""
        return torch.linspace(0.0, 1.0 - eps, n_steps + 1, device=device, dtype=torch.float64)

    def step(
        self,
        xi: torch.Tensor,
        yhat: torch.Tensor,
        s: float,
        u: float,
        sampler: str,
        eta: float = 1.0,
        generator: torch.Generator | None = None,
        var_yhat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance the state from time s to time u using endpoint estimate yhat."""
        d = u - s
        one_minus_s = 1.0 - s

        if sampler in ("euler", "euler_det"):
            out = xi + (yhat - xi) * (d / one_minus_s)
            if sampler == "euler" and eta > 0.0:
                std = eta * self.gamma * math.sqrt(d)
                out = out + std * self._randn(xi, generator)
            return out

        if sampler == "bridge":
            # exact Brownian-bridge posterior on [s, 1] pinned at (s, xi), (1, yhat)
            mean = xi + (yhat - xi) * (d / one_minus_s)
            var = self.gamma**2 * d * (1.0 - u) / one_minus_s
            if var_yhat is not None:
                # optional: restore the Var[Y | xi_s] the point-mass approximation drops
                var = var + var_yhat * (d / one_minus_s) ** 2
            if eta > 0.0 and var > 0.0:
                std = eta * math.sqrt(var)
                mean = mean + std * self._randn(xi, generator)
            return mean

        raise ValueError(f"unknown sampler: {sampler!r}")

    @staticmethod
    def _randn(ref: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(ref)
        return torch.randn(ref.shape, device=ref.device, dtype=ref.dtype, generator=generator)

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        device,
        n_steps: int = 50,
        sampler: str = "bridge",
        eta: float = 1.0,
        eps: float = 0.0,
        return_endpoint: bool = True,
        generator: torch.Generator | None = None,
        x_init: torch.Tensor | None = None,
        progress=None,
    ) -> torch.Tensor:
        """Generate samples. Returns a tensor in the data's own scale ([-1, 1])."""
        if sampler == "euler_det" and eps <= 0.0:
            # the Euler drift coefficient d/(1-s) is singular at s = 1; the original
            # code offsets the grid by eps. Keep that behaviour for faithfulness.
            eps = 1e-7

        if x_init is None:
            x_init = self._randn(torch.empty(shape, device=device), generator)
        xi = x_init
        grid = self.timesteps(n_steps, device, eps=eps)

        it = range(n_steps)
        if progress is not None:
            it = progress(it)

        yhat = None
        for i in it:
            s = float(grid[i])
            u = float(grid[i + 1])
            t_in = torch.full((xi.shape[0],), s * self.cfg.t_scale, device=device, dtype=xi.dtype)
            yhat = model(xi, t_in)
            last = i == n_steps - 1
            xi = self.step(
                xi, yhat, s, u, sampler=sampler,
                eta=0.0 if (last and sampler == "bridge") else eta,
                generator=generator,
            )

        if return_endpoint and yhat is not None:
            # xi_1 = Y by construction, and the network's estimate of it is yhat.
            # For `bridge` with u = 1 the step already returns exactly yhat.
            if sampler in ("euler", "euler_det"):
                # the Euler grid stops at 1 - eps, leaving residual bridge noise on
                # xi; report the endpoint estimate instead, which is the quantity
                # the scheme is defined to produce.
                s_last = float(grid[-1])
                t_in = torch.full((xi.shape[0],), s_last * self.cfg.t_scale,
                                  device=device, dtype=xi.dtype)
                return model(xi, t_in)
        return xi
