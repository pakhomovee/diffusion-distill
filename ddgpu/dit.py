"""Self-contained DiT (Peebles & Xie) with adaLN-Zero, for latent diffusion.

Kept dependency-free (torch only) so parameter counts and memory estimates in
RUNPLAN.md are derived from this exact module rather than quoted from a paper.
Weights are load-compatible with facebook/DiT-* checkpoints at the tensor-shape
level for the standard configs (see DIT_CONFIGS).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

DIT_CONFIGS = {           # name: (depth, hidden, heads)
    "DiT-T/2":  ( 2,   64,  4),   # smoke tests only
    "DiT-S/2":  (12,  384,  6),
    "DiT-B/2":  (12,  768, 12),
    "DiT-L/2":  (24, 1024, 16),
    "DiT-XL/2": (28, 1152, 16),
}


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, t):
        half = self.freq_dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        # do not reach into self.mlp[0].weight -- the layer may be wrapped
        # (e.g. by LoRALinear), which has no .weight of its own
        return self.mlp(torch.cat([a.cos(), a.sin()], dim=-1).to(self._dtype()))

    def _dtype(self):
        return next(self.mlp.parameters()).dtype


class LabelEmbedder(nn.Module):
    """Class embedding with a learned null token for classifier-free guidance."""

    def __init__(self, n_classes, hidden, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, hidden)
        self.n_classes, self.dropout = n_classes, dropout

    def forward(self, y, train=True, force_drop=None):
        if force_drop is not None:
            y = torch.where(force_drop, self.n_classes, y)
        elif train and self.dropout > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout
            y = torch.where(drop, self.n_classes, y)
        return self.emb(y)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"),
                                 nn.Linear(h, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        sa, ga, aa, sm, gm, am = self.ada(c).chunk(6, dim=1)
        x = x + aa.unsqueeze(1) * self.attn(modulate(self.n1(x), sa, ga))
        x = x + am.unsqueeze(1) * self.mlp(modulate(self.n2(x), sm, gm))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.lin = nn.Linear(dim, patch * patch * out_ch)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        for m in (self.lin, self.ada[1]):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, c):
        s, g = self.ada(c).chunk(2, dim=1)
        return self.lin(modulate(self.norm(x), s, g))


def sincos_pos_embed(dim, grid):
    """2D sin-cos positional embedding, (grid*grid, dim)."""
    def emb1d(d, pos):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float64) / (d / 2.0))
        out = pos.reshape(-1)[:, None].double() * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)
    g = torch.arange(grid, dtype=torch.float64)
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    return torch.cat([emb1d(dim // 2, gx), emb1d(dim // 2, gy)], dim=1).float()


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_ch=4, hidden=1152, depth=28,
                 heads=16, mlp_ratio=4.0, n_classes=1000, class_dropout=0.1,
                 learn_sigma=False, grad_ckpt=False):
        super().__init__()
        self.in_ch, self.patch, self.grid = in_ch, patch_size, input_size // patch_size
        self.out_ch = in_ch * (2 if learn_sigma else 1)
        self.grad_ckpt = grad_ckpt
        self.x_embed = nn.Conv2d(in_ch, hidden, patch_size, patch_size)
        self.t_embed = TimestepEmbedder(hidden)
        self.y_embed = LabelEmbedder(n_classes, hidden, class_dropout)
        self.register_buffer("pos", sincos_pos_embed(hidden, self.grid)[None], persistent=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, patch_size, self.out_ch)

    def unpatchify(self, x):
        B, p, c, g = x.shape[0], self.patch, self.out_ch, self.grid
        x = x.reshape(B, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(B, c, g * p, g * p)

    @property
    def trunk_dims(self):
        """(token width, conditioning width) for a discriminator head.

        Reported rather than introspected: `DMD2Trainer` used to read
        `final.lin.in_features`, which only happens to be right because a DiT's
        token and conditioning widths coincide. A UNet's do not.
        """
        return self.final.lin.in_features, self.final.lin.in_features

    def trunk(self, x, t, y, force_drop=None):
        """Shared body. Returns (tokens, conditioning) so a discriminator head
        can reuse the critic's features instead of paying for its own network --
        which is what DMD2 does."""
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.t_embed(t) + self.y_embed(y, self.training, force_drop)
        for b in self.blocks:
            if self.grad_ckpt and self.training:
                h = torch.utils.checkpoint.checkpoint(b, h, c, use_reentrant=False)
            else:
                h = b(h, c)
        return h, c

    def forward(self, x, t, y, force_drop=None):
        h, c = self.trunk(x, t, y, force_drop)
        return self.unpatchify(self.final(h, c))


def make_dit(name="DiT-XL/2", input_size=32, **kw):
    depth, hidden, heads = DIT_CONFIGS[name]
    patch = int(name.split("/")[1])
    return DiT(input_size=input_size, patch_size=patch, hidden=hidden,
               depth=depth, heads=heads, **kw)
