"""Closed-form Gaussian-mixture target with an exactly known noised score.

Convention: EDM/VE. The forward process is  x_sigma = x_0 + sigma * eps,
eps ~ N(0, I).  If p_0 is a GMM, then p_sigma is a GMM with the same weights
and means and covariances Sigma_k + sigma^2 I, so the score is available in
closed form at every noise level.  That is the whole point of using this
target: it gives us the ground-truth s(x, sigma) against which a learned
teacher's bias and a minibatch estimator's variance can both be measured.

Component covariances share one random orthogonal basis R and have power-law
diagonal spectra, so the target sits near a low-dimensional manifold the way
image latents do rather than filling the ambient space.
"""
import math
import torch


class GMM:
    def __init__(self, d, K=8, spec_decay=1.0, rank_frac=0.25, mean_scale=1.0,
                 seed=0, device="cpu", dtype=torch.float32):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.d, self.K, self.device, self.dtype = d, K, device, dtype

        # shared orthonormal basis
        A = torch.randn(d, d, generator=g, dtype=torch.float64)
        R, _ = torch.linalg.qr(A)
        self.R = R.to(dtype)                                        # (d,d)

        # power-law spectrum, truncated to an effective rank
        idx = torch.arange(1, d + 1, dtype=torch.float64)
        lam = idx.pow(-spec_decay)
        r = max(1, int(rank_frac * d))
        lam[r:] *= 1e-2                                             # soft manifold
        lam = lam / lam.max()
        # per-component scale jitter
        scale = (0.5 + torch.rand(K, 1, generator=g, dtype=torch.float64))
        self.var = (lam[None, :] * scale).to(dtype)                 # (K,d) eigenvalues

        self.mu = (mean_scale * torch.randn(K, d, generator=g, dtype=torch.float64)
                   / math.sqrt(d) * math.sqrt(d)).to(dtype)         # (K,d)
        self.logw = torch.zeros(K, dtype=dtype)                     # uniform weights

        for n in ("R", "var", "mu", "logw"):
            setattr(self, n, getattr(self, n).to(device))
        # overall data std, used to set sensible sigma ranges
        self.data_std = float(self.sample(4096, seed=12345).std())

    # ---- sampling -------------------------------------------------------
    def sample(self, n, seed=None):
        g = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        k = torch.randint(0, self.K, (n,), generator=g)
        eps = torch.randn(n, self.d, generator=g, dtype=self.dtype)
        z = eps * self.var[k].sqrt().cpu()
        x = self.mu[k].cpu() + z @ self.R.cpu().T
        return x.to(self.device)

    # ---- exact noised score ---------------------------------------------
    def _quad_and_logdet(self, x, sigma):
        """Return per-component (Mahalanobis quad, logdet, whitened residual).

        x: (B,d)  sigma: (B,) or scalar.  Works in the rotated basis where the
        covariance Sigma_k + sigma^2 I is diagonal.
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=x.dtype)
        sigma = sigma.reshape(-1, 1, 1)
        y = x @ self.R                                              # (B,d) rotated
        m = self.mu @ self.R                                        # (K,d) rotated
        r = y[:, None, :] - m[None, :, :]                           # (B,K,d)
        v = self.var[None, :, :] + sigma ** 2                       # (B,K,d)
        quad = (r * r / v).sum(-1)                                  # (B,K)
        logdet = torch.log(v).sum(-1)                               # (B,K)
        return r, v, quad, logdet

    def log_prob(self, x, sigma):
        _, _, quad, logdet = self._quad_and_logdet(x, sigma)
        lp = self.logw[None, :] - 0.5 * (quad + logdet + self.d * math.log(2 * math.pi))
        return torch.logsumexp(lp, dim=1) - torch.logsumexp(self.logw, 0)

    def score(self, x, sigma):
        """Exact grad_x log p_sigma(x).  Shape (B,d)."""
        r, v, quad, logdet = self._quad_and_logdet(x, sigma)
        lp = self.logw[None, :] - 0.5 * (quad + logdet)
        gam = torch.softmax(lp, dim=1)                              # (B,K) posteriors
        g_rot = -(gam[:, :, None] * (r / v)).sum(1)                 # (B,d) in rotated basis
        return g_rot @ self.R.T
