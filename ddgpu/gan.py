"""DMD2's auxiliary GAN term.

Included so the BASELINE is the real DMD2 and not a weakened DMD. Track A's
claim is that this term can be replaced by a closed-form, hyperparameter-free
real-data score; that claim is only worth anything if the thing it replaces is
implemented properly and tuned.

Following DMD2: the discriminator is a small head on the FAKE-SCORE network's
own features, not a separate network, so it costs almost no extra memory. It
sees noised real samples and noised student samples at the same sigma.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANHead(nn.Module):
    """DMD2's discriminator: a head on the CRITIC's own features.

    `cond_dim` defaults to `hidden` because a DiT's token width and conditioning
    width are the same. A UNet's are not -- `ddpm-cifar10-32` has 256-channel
    mid-block features and a 512-wide time embedding -- so the two are separate
    parameters and the adapter reports both via `trunk_dims`.
    """

    def __init__(self, hidden, cond_dim=None):
        super().__init__()
        cond_dim = hidden if cond_dim is None else cond_dim
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden))
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, tokens, cond):
        s, g = self.ada(cond).chunk(2, dim=-1)
        h = self.norm(tokens) * (1 + g.unsqueeze(1)) + s.unsqueeze(1)
        return self.out(h.mean(1)).squeeze(-1)          # (B,) logit


def d_loss(logit_real, logit_fake):
    """Hinge loss for the discriminator."""
    return (F.relu(1 - logit_real).mean() + F.relu(1 + logit_fake).mean()) * 0.5


def g_loss(logit_fake):
    """Non-saturating generator loss."""
    return -logit_fake.mean()


class ConvGANHead(nn.Module):
    """Standalone discriminator, for backbones with no exposed token trunk.

    `GANHead` reuses the critic's DiT features, which is what DMD2 does and what
    makes the term nearly free. The cheap tier's teachers are UNets (diffusers
    DDPM, NVlabs EDM) and REPA SiTs loaded through their own code -- none of
    which expose a `trunk()` returning (tokens, conditioning), and hooking their
    mid-blocks would couple us to three third-party layer layouts.

    So on those backbones the discriminator is a small standalone conv net on
    the noised sample, conditioned on log-sigma. This DEVIATES from DMD2's
    design and the deviation is stated rather than hidden -- but it does not
    weaken the comparison, because the baseline and the method arms use the
    identical head, and the method arm's whole point is that it uses no head at
    all. What it does mean: the cheap tier's absolute baseline FID is not
    directly comparable to DMD2's published number, only to our own baseline.
    """

    def __init__(self, in_ch, res=32, width=64, depth=4, cond_dim=128):
        super().__init__()
        # Each block halves the spatial size, so the usable depth is bounded by
        # the input resolution: 4 stride-2 blocks on an 8x8 input runs out of
        # pixels and raises inside the third conv. Clamp rather than let the
        # caller discover it at step 1 of a run.
        depth = max(1, min(depth, int(math.log2(max(res, 2))) - 1))
        self.cond = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(),
                                  nn.Linear(cond_dim, cond_dim))
        chs, layers = in_ch, []
        for i in range(depth):
            out = min(width * 2 ** i, 512)
            layers.append(nn.Conv2d(chs, out, 4, 2, 1))
            chs = out
        self.convs = nn.ModuleList(layers)
        self.films = nn.ModuleList([nn.Linear(cond_dim, 2 * c.out_channels)
                                    for c in layers])
        for f in self.films:
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)
        self.out = nn.Conv2d(chs, 1, 3, 1, 1)

    def forward(self, x, sigma):
        c = self.cond(sigma.reshape(-1, 1).log().clamp(-10, 10) / 4)
        h = x
        for conv, film in zip(self.convs, self.films):
            h = conv(h)
            s, g = film(c).chunk(2, dim=-1)
            h = F.silu(h * (1 + g[:, :, None, None]) + s[:, :, None, None])
        return self.out(h).mean(dim=(1, 2, 3))          # (B,) logit
