"""EDM preconditioning and schedules for latent diffusion, shared by all tracks."""
import torch


class EDMWrapper(torch.nn.Module):
    """Wraps a raw DiT as an EDM denoiser D(x, sigma, y) and exposes .score().

    Karras et al. preconditioning; the DiT sees c_in*x and a log-sigma
    conditioning signal, which is what makes one network usable across the
    entire noise range without a hand-built timestep discretisation.
    """

    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net, self.sigma_data = net, sigma_data

    def _coef(self, sigma):
        sd = self.sigma_data
        s2 = sigma ** 2 + sd ** 2
        return (sd ** 2 / s2, sigma * sd / s2.sqrt(), 1.0 / s2.sqrt(),
                sigma.log() / 4)

    def forward(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        cs, co, ci, cn = self._coef(sigma)
        v = lambda t: t.reshape(-1, 1, 1, 1)
        F = self.net(v(ci) * x, cn, y, **kw)
        return v(cs) * x + v(co) * F

    def score(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        D = self.forward(x, sigma, y, **kw)
        return (D - x) / sigma.reshape(-1, 1, 1, 1) ** 2

    def cfg_score(self, x, sigma, y, scale=1.0, **kw):
        """Classifier-free-guided score. scale=1 is unguided."""
        if scale == 1.0:
            return self.score(x, sigma, y, **kw)
        sigma = _as_batch(sigma, x)
        drop = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        c = self.score(x, sigma, y, force_drop=drop, **kw)
        u = self.score(x, sigma, y, force_drop=~drop, **kw)
        return u + scale * (c - u)


def _as_batch(sigma, x):
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    sigma = sigma.reshape(-1).to(x.device, x.dtype)
    return sigma.expand(x.shape[0]) if sigma.numel() == 1 else sigma


def sample_sigma(n, P_mean=-1.2, P_std=1.2, device="cpu"):
    return (torch.randn(n, device=device) * P_std + P_mean).exp()


def dsm_weight(sigma, sigma_data):
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


def edm_sigmas(n, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    i = torch.arange(n, device=device, dtype=torch.float64)
    a, b = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    s = (a + i / max(n - 1, 1) * (b - a)) ** rho
    return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)]).float()
