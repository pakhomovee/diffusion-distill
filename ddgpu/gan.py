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


def _groups(c, min_per_group=4):
    """Largest GroupNorm group count that divides c, capped at DMD2's 32.

    Also requires at least `min_per_group` channels per group, which is not
    fussiness. The second conv lands on 1x1, so GroupNorm there normalises over
    (channels_per_group * 1 * 1) values -- and with ONE channel per group that
    is a single number, whose normalisation is identically zero. The head then
    emits a constant and no gradient reaches the critic at all: caught with a
    16-channel bottleneck, where both logits came back 0.0822 and the gradient
    norm into the UNet was exactly 0.

    DMD2 never hits this (768 channels / 32 groups = 24 per group), and neither
    does ddpm-cifar10-32 (256 / 32 = 8), but a small model does.
    """
    return next((g for g in (32, 16, 8, 4, 2, 1)
                 if c % g == 0 and c // g >= min_per_group), 1)


class GANHead(nn.Module):
    """DMD2's discriminator: strided convs on the CRITIC's bottleneck.

    Mirrors `main/edm/edm_guidance.py`'s `cls_pred_branch`, which on
    ImageNet-64 is

        Conv2d(768 -> 768, k4 s2 p1)   8x8 -> 4x4
        GroupNorm(32), SiLU
        Conv2d(768 -> 768, k4 s4 p0)   4x4 -> 1x1
        GroupNorm(32), SiLU
        Conv2d(768 -> 1,   k1)

    generalised over the bottleneck's spatial size, because ddpm-cifar10-32's
    bottleneck is 4x4x256 rather than 8x8x768. The second conv's kernel and
    stride are both `spatial // 2`, so it always lands on 1x1 -- the same
    two-strided-convs-then-pointwise shape DMD2 uses.

    Deliberately UNCONDITIONED. DMD2's head takes only the bottleneck; the noise
    level is already in those features because the critic's own forward was
    given it. The previous version applied adaLN from the time embedding, which
    the reference does not.

    And it keeps the spatial extent. The previous version did `out(h.mean(1))`,
    mean-pooling the tokens before classifying -- so a global shift in the
    feature average was enough to satisfy it, which is what the degenerate run's
    channel statistics looked like. See DMD2_DIFF.md.
    """

    def __init__(self, in_ch, spatial):
        super().__init__()
        s = int(spatial)
        if s < 2:
            raise ValueError(f"bottleneck must be at least 2x2, got {s}")
        layers = [nn.Conv2d(in_ch, in_ch, 4, 2, 1),
                  nn.GroupNorm(_groups(in_ch), in_ch), nn.SiLU()]
        mid = s // 2
        if mid > 1:                       # collapse whatever is left to 1x1
            layers += [nn.Conv2d(in_ch, in_ch, mid, mid, 0),
                       nn.GroupNorm(_groups(in_ch), in_ch), nn.SiLU()]
        layers.append(nn.Conv2d(in_ch, 1, 1, 1, 0))
        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        """feat: (B, C, H, W) bottleneck -> (B,) logit."""
        return self.net(feat).flatten(1).mean(1)


def d_loss(logit_real, logit_fake):
    """Softplus (logistic) discriminator loss, as DMD2 uses.

    Was hinge. DMD2's `compute_guidance_clean_cls_loss` is
    `softplus(pred_on_fake) + softplus(-pred_on_real)`, and matching it removes
    one difference from the reference. Uninformative value is 2*log(2) = 1.386,
    where hinge's was 1.0.
    """
    return F.softplus(logit_fake).mean() + F.softplus(-logit_real).mean()


def g_loss(logit_fake):
    """Non-saturating generator loss, BOUNDED.

    Was `-logit_fake.mean()`, which is unbounded: the generator is rewarded
    without limit for driving the logit up, and nothing stops it walking off
    into whatever direction the discriminator happens to score highly. DMD2 uses
    `softplus(-pred_on_fake)`, which saturates once the discriminator is fooled
    and the gradient vanishes. See DMD2_DIFF.md -- with a mean-pooling head,
    the unbounded version is a plausible cause of the degenerate near-black
    samples in LOG ENTRY 015's successor run.
    """
    return F.softplus(-logit_fake).mean()


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
