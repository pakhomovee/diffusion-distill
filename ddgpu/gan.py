"""DMD2's auxiliary GAN term.

Included so the BASELINE is the real DMD2 and not a weakened DMD. Track A's
claim is that this term can be replaced by a closed-form, hyperparameter-free
real-data score; that claim is only worth anything if the thing it replaces is
implemented properly and tuned.

Following DMD2: the discriminator is a small head on the FAKE-SCORE network's
own features, not a separate network, so it costs almost no extra memory. It
sees noised real samples and noised student samples at the same sigma.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GANHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
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
