"""Closed-form discrepancies against the standard Gaussian.

This is where idea #1 (Stein/kernel methods) actually belongs. info.txt's worry
about KSD is that it degrades in high dimension -- true when the target is an
unknown data distribution whose score must be supplied by a network. Against
N(0, I) three of those problems vanish:

  * the score is EXACTLY -z; there is no teacher error to propagate;
  * the median heuristic is not a heuristic: E||z-z'||^2 = 2d exactly, so the
    bandwidth is known a priori and does not need re-tuning per dimension;
  * the mean-embedding integrals are analytic, so MMD needs no second sample
    and its estimator variance drops accordingly.

What does NOT vanish: noise space has the same dimension as data space, so these
are still d-dimensional two-sample problems. The gain is conditioning, not
dimension. Stated plainly so the claim is not oversold.
"""
import math
import torch


def _pdist2(a, b):
    return torch.cdist(a, b).pow(2)


def bandwidth(d, mult=1.0):
    """h^2 = mult * d. With h^2 ~ d the analytic constants below are O(1) in any
    dimension; with h^2 = O(1) they underflow like (h^2/(1+h^2))^{d/2}."""
    return mult * d


def mmd2_to_gaussian(z, h2=None, unbiased=True):
    """MMD^2(q, N(0,I)) with a Gaussian kernel, using ANALYTIC mean embeddings.

    k(z,z') = exp(-||z-z'||^2 / (2 h^2))
    mu(z)   = E_{w~N} k(z,w) = (h^2/(1+h^2))^{d/2} exp(-||z||^2 / (2(1+h^2)))
    C       = E_{v,w~N} k(v,w) = (h^2/(h^2+2))^{d/2}
    """
    z = z.reshape(z.shape[0], -1).float()
    B, d = z.shape
    h2 = bandwidth(d) if h2 is None else h2
    K = torch.exp(-_pdist2(z, z) / (2 * h2))
    if unbiased:
        t1 = (K.sum() - K.diagonal().sum()) / (B * (B - 1))
    else:
        t1 = K.mean()
    log_a = 0.5 * d * math.log(h2 / (1 + h2))
    mu = torch.exp(log_a - z.pow(2).sum(-1) / (2 * (1 + h2)))
    C = math.exp(0.5 * d * math.log(h2 / (h2 + 2)))
    return t1 - 2 * mu.mean() + C


def ksd2_to_gaussian(z, h2=None, unbiased=True):
    """Kernel Stein discrepancy to N(0,I), fully closed form.

    With s(z) = -z and a Gaussian kernel,
        u(z,z') = k(z,z') * [ z.z' - ||z-z'||^2/h^2 + d/h^2 - ||z-z'||^2/h^4 ]
    The U-statistic form gives O(1/B^2) gradient variance rather than O(1/B),
    which is the main reason to prefer it over a plain MMD term here.
    """
    z = z.reshape(z.shape[0], -1).float()
    B, d = z.shape
    h2 = bandwidth(d) if h2 is None else h2
    D2 = _pdist2(z, z)
    K = torch.exp(-D2 / (2 * h2))
    U = K * (z @ z.T - D2 / h2 + d / h2 - D2 / h2 ** 2)
    if unbiased:
        return (U.sum() - U.diagonal().sum()) / (B * (B - 1))
    return U.mean()


def gaussian_moment_penalty(z):
    """Cheap O(Bd) companion term: matches mean and per-coordinate second moment.

    Kernel discrepancies in d >> B are weak against low-order moment error -- the
    one failure mode we can rule out for free. Not a substitute for MMD/KSD; a
    guard rail underneath it.
    """
    z = z.reshape(z.shape[0], -1).float()
    m = z.mean(0)
    v = z.var(0, unbiased=False)
    return m.pow(2).mean() + (v - 1).pow(2).mean()


def sliced_gaussian_ks(z, n_slices=128):
    """Sliced 1-D Gaussianity check via the Cramer-von-Mises statistic.

    Diagnostic, not a training loss: detects directions along which E#p_data is
    non-Gaussian, which is exactly the collusion signature Track B must rule out.
    """
    z = z.reshape(z.shape[0], -1).float()
    B, d = z.shape
    V = torch.randn(d, n_slices, device=z.device)
    V = V / V.norm(dim=0, keepdim=True)
    p = (z @ V).sort(dim=0).values                       # (B, n_slices)
    cdf = 0.5 * (1 + torch.erf(p / math.sqrt(2)))
    emp = (torch.arange(1, B + 1, device=z.device).float() - 0.5)[:, None] / B
    return ((cdf - emp) ** 2).mean(0)                    # (n_slices,)
