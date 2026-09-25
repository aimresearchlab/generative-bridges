"""A compact, auditable UNet for 32x32 images.

Standard ADM/DDPM topology (GroupNorm + SiLU residual blocks, scale-shift time
conditioning, self-attention at chosen resolutions, skip connections), written
out in full so the architecture can be checked against the text of the paper
rather than inherited from a vendored file.

Time conditioning
-----------------
`timestep_embedding` is the usual sinusoidal code with max_period = 10^4.  It
resolves inputs spanning O(10^3); fed inputs spanning O(10^-1) every channel
collapses to the same near-linear function of t and the embedding degenerates
to rank one.  Callers must therefore pass t on a ~[0, 1000] scale.  The bridge
code does this via BridgeConfig.t_scale; see bridges/bridge.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def zero_module(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        p.detach().zero_()
    return m


def norm(ch: int) -> nn.Module:
    return nn.GroupNorm(32 if ch % 32 == 0 else min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.in_layers = nn.Sequential(norm(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        # scale-shift ("FiLM") conditioning: emb -> (scale, shift)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, 2 * out_ch))
        self.out_layers = nn.Sequential(
            norm(out_ch), nn.SiLU(), nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_layers[0](h) * (1 + scale) + shift
        h = self.out_layers[3](self.out_layers[2](self.out_layers[1](h)))
        return self.skip(x) + h


class Attention(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4):
        super().__init__()
        assert ch % n_heads == 0, (ch, n_heads)
        self.n_heads = n_heads
        self.norm = norm(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(y).reshape(b, self.n_heads, 3 * (c // self.n_heads), h * w).chunk(3, dim=2)
        # (b, heads, hw, dim) for scaled_dot_product_attention
        q, k, v = (z.transpose(-1, -2).contiguous() for z in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h * w)
        return x + self.proj(out).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    model_channels: int = 128
    channel_mult: tuple = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)  # spatial sizes at which to attend
    num_heads: int = 4
    dropout: float = 0.0
    image_size: int = 32


class UNet(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.model_channels
        emb_ch = ch * 4
        self.time_embed = nn.Sequential(nn.Linear(ch, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch))

        self.input_blocks = nn.ModuleList([nn.ModuleList([nn.Conv2d(cfg.in_channels, ch, 3, padding=1)])])
        skip_channels = [ch]
        cur, res = ch, cfg.image_size
        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers = [ResBlock(cur, ch * mult, emb_ch, cfg.dropout)]
                cur = ch * mult
                if res in cfg.attention_resolutions:
                    layers.append(Attention(cur, cfg.num_heads))
                self.input_blocks.append(nn.ModuleList(layers))
                skip_channels.append(cur)
            if level != len(cfg.channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(cur)]))
                skip_channels.append(cur)
                res //= 2

        self.middle = nn.ModuleList([
            ResBlock(cur, cur, emb_ch, cfg.dropout),
            Attention(cur, cfg.num_heads),
            ResBlock(cur, cur, emb_ch, cfg.dropout),
        ])

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                layers = [ResBlock(cur + skip_channels.pop(), ch * mult, emb_ch, cfg.dropout)]
                cur = ch * mult
                if res in cfg.attention_resolutions:
                    layers.append(Attention(cur, cfg.num_heads))
                if level != 0 and i == cfg.num_res_blocks:
                    layers.append(Upsample(cur))
                    res *= 2
                self.output_blocks.append(nn.ModuleList(layers))

        self.out = nn.Sequential(norm(cur), nn.SiLU(),
                                 zero_module(nn.Conv2d(cur, cfg.out_channels, 3, padding=1)))

    def forward(self, x, t):
        emb = self.time_embed(timestep_embedding(t, self.cfg.model_channels))
        hs, h = [], x
        for blocks in self.input_blocks:
            for m in blocks:
                h = m(h, emb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.middle:
            h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        for blocks in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for m in blocks:
                h = m(h, emb) if isinstance(m, ResBlock) else m(h)
        return self.out(h)


def build_unet(dataset: str) -> UNet:
    """Architectures as described in the paper's Section 3, at a single 32x32 grid."""
    if dataset == "mnist":
        cfg = UNetConfig(in_channels=1, out_channels=1, model_channels=64,
                         channel_mult=(1, 2, 2), num_res_blocks=2,
                         attention_resolutions=(16, 8), num_heads=4, image_size=32)
    elif dataset == "cifar10":
        # num_res_blocks=3 matches the vendored CIFAR config (52.5M params); with 2
        # this net is 39.6M and capacity becomes a confound in any comparison
        # against the original numbers.
        cfg = UNetConfig(in_channels=3, out_channels=3, model_channels=128,
                         channel_mult=(1, 2, 2, 2), num_res_blocks=3,
                         attention_resolutions=(16, 8), num_heads=4, image_size=32)
    else:
        raise ValueError(dataset)
    return UNet(cfg)
