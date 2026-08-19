"""Small MLP backbones with EDM preconditioning."""
import math
import torch
import torch.nn as nn


class FourierEmb(nn.Module):
    def __init__(self, dim, scale=16.0):
        super().__init__()
        self.register_buffer("freq", torch.randn(dim // 2) * scale)

    def forward(self, t):
        a = 2 * math.pi * t[:, None] * self.freq[None, :]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class MLP(nn.Module):
    """Conditional MLP: (x, c) -> R^d, with FiLM-style noise conditioning."""

    def __init__(self, d, width=256, depth=4, emb=128):
        super().__init__()
        self.emb = nn.Sequential(FourierEmb(emb), nn.Linear(emb, emb), nn.SiLU(),
                                 nn.Linear(emb, emb))
        self.inp = nn.Linear(d, width)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict(dict(
                film=nn.Linear(emb, 2 * width),
                fc1=nn.Linear(width, width),
                fc2=nn.Linear(width, width),
                norm=nn.LayerNorm(width))))
        self.out = nn.Linear(width, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, c):
        e = self.emb(c)
        h = self.inp(x)
        for b in self.blocks:
            g, s = b["film"](e).chunk(2, dim=-1)
            r = b["norm"](h) * (1 + g) + s
            r = b["fc2"](torch.nn.functional.silu(b["fc1"](r)))
            h = h + r
        return self.out(h)


class EDMPrecond(nn.Module):
    """Wraps an MLP as a denoiser D(x;sigma); exposes .score(x, sigma)."""

    def __init__(self, d, sigma_data=1.0, **kw):
        super().__init__()
        self.d, self.sigma_data = d, sigma_data
        self.net = MLP(d, **kw)

    def _c(self, sigma):
        sd = self.sigma_data
        s2 = sigma ** 2 + sd ** 2
        return (sd ** 2 / s2, sigma * sd / s2.sqrt(), 1.0 / s2.sqrt(), sigma.log() / 4)

    def forward(self, x, sigma):
        sigma = sigma.reshape(-1) if torch.is_tensor(sigma) else \
            torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
        cs, co, ci, cn = self._c(sigma)
        F = self.net(ci[:, None] * x, cn)
        return cs[:, None] * x + co[:, None] * F

    def score(self, x, sigma):
        sig = sigma if torch.is_tensor(sigma) else torch.tensor(sigma)
        sig = sig.reshape(-1).to(x.device, x.dtype)
        if sig.numel() == 1:
            sig = sig.expand(x.shape[0])
        return (self.forward(x, sig) - x) / sig[:, None] ** 2
