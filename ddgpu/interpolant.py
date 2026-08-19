"""Linear stochastic-interpolant (SiT / flow-matching) preconditioning, in sigma.

Third preconditioning family, alongside `edm.EDMWrapper` (Karras) and
`vp.VPPrecond` (DDPM/DiT). This one covers SiT and REPA checkpoints, which
predict a *velocity* along a linear interpolant rather than eps or x0.

The convention, verified against REPA's own `loss.py:SILoss`:

    x_t = alpha_t * x0 + sigma_t * eps,   alpha_t = 1 - t,  sigma_t = t,  t in [0,1]
    target v = d_alpha_t * x0 + d_sigma_t * eps = eps - x0
    the network receives t RAW in [0,1] -- not scaled by 1000 like DiT

Change of variables into our sigma coordinates (x = x0 + sigma*eps):

    x_t = (1-t)(x0 + (t/(1-t)) eps)  =>  sigma = t/(1-t),  t = sigma/(1+sigma)

so `x_t = (1-t) x`, i.e. `c_in = 1-t = 1/(1+sigma)`. REPA's own lognormal
weighting uses exactly `time_input = sigma/(1+sigma)`, which is the same map --
that agreement is the reason to trust this rather than a guess.

Recovering the score. From `v = eps - x0` and `x_t = (1-t)x0 + t*eps`:

    eps = x_t + (1-t) v        (substitute and simplify)
    x0  = x_t - t v

`x` and `x_t` are deterministically related, so `E[eps|x] = E[eps|x_t]` and

    score(x, sigma) = -E[eps|x]/sigma = -(x + v_hat) / (sigma (1 + sigma))
    D(x, sigma)     = x + sigma^2 score = x/(1+sigma) - sigma*v_hat/(1+sigma)

i.e. `c_skip = 1/(1+sigma)`, `c_out = -sigma/(1+sigma)`, `F = v_hat`.

Two limits worth checking by hand, because they catch sign errors that otherwise
only show up as bad samples:

    sigma -> 0 :  x + v_hat = (x0 + sigma*eps) + (eps - x0) = (1+sigma) eps
                  so score -> -eps/sigma, and D -> x.            (correct)
    sigma -> oo:  D -> -v_hat -> E[x0].                          (correct)

Both are asserted in tests/test_teachers.py against an oracle velocity field.
"""
import torch
import torch.nn as nn


class LinearInterpolant:
    """sigma <-> t for the linear path. Closed form in both directions."""

    #: t is confined to [t_eps, 1 - t_eps]; t=1 is sigma=inf and t=0 is sigma=0,
    #: both of which the network was never trained at and neither of which is
    #: representable. 1e-4 corresponds to sigma in [1e-4, 1e4].
    t_eps = 1e-4

    def t_of_sigma(self, sigma):
        return (sigma / (1.0 + sigma)).clamp(self.t_eps, 1.0 - self.t_eps)

    def sigma_of_t(self, t):
        t = torch.as_tensor(t).clamp(self.t_eps, 1.0 - self.t_eps)
        return t / (1.0 - t)

    @property
    def sigma_max(self):
        return (1.0 - self.t_eps) / self.t_eps          # 9999.0

    @property
    def sigma_min(self):
        return self.t_eps / (1.0 - self.t_eps)

    def student_sigmas(self, n_steps, sigma_max=None):
        """Backward-Euler grid, evenly spaced in t -- the variable the model was
        trained uniformly over (`SILoss` samples `t ~ U[0,1]`). Spacing evenly in
        sigma instead would concentrate every step in the high-noise tail."""
        hi = (self.t_of_sigma(torch.tensor(float(sigma_max)))
              if sigma_max else torch.tensor(1.0 - self.t_eps))
        ts = torch.linspace(float(hi), 0.0, n_steps + 1)[:n_steps]
        s = self.sigma_of_t(ts)
        return torch.cat([s, torch.zeros(1)])


class InterpolantPrecond(nn.Module):
    """Wraps a velocity-predicting SiT as a sigma-parameterised EDM denoiser.

    API-identical to `EDMWrapper` and `VPPrecond`, so DMD2Trainer, the lambda
    estimator and the GAN head are unchanged.

    `tuple_out=True` handles SiT's `forward -> (x, zs)` return: REPA models emit
    the projector features alongside the prediction, and silently taking the
    tuple would produce a TypeError deep inside the loss rather than here.
    """

    def __init__(self, net, path=None, out_ch=None, sigma_data=1.0,
                 tuple_out=True, t_scale=1.0):
        super().__init__()
        self.net = net
        self.path = path or LinearInterpolant()
        self.out_ch, self.sigma_data = out_ch, sigma_data
        self.tuple_out = tuple_out
        # t_scale exists ONLY as an escape hatch. REPA passes t raw in [0,1]
        # (loss.py: `model(model_input, time_input.flatten())`), so 1.0 is
        # correct for SiT. A checkpoint trained with a x1000 convention would
        # need 1000.0 -- `teachers.validate_teacher` detects which by measuring
        # the denoising loss, rather than leaving it to be guessed.
        self.t_scale = t_scale

    def _coef(self, sigma):
        """(c_skip, c_out, c_in, c_noise), matching the EDMWrapper contract."""
        one_plus = 1.0 + sigma
        t = self.path.t_of_sigma(sigma)
        return (1.0 / one_plus, -sigma / one_plus, 1.0 / one_plus,
                (t * self.t_scale).to(sigma.dtype))

    def _v(self, x, sigma, y, **kw):
        cs, co, ci, cn = self._coef(sigma)
        v = lambda t: t.reshape(-1, 1, 1, 1)
        out = self.net(v(ci) * x, cn, y, **kw)
        if self.tuple_out and isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, :self.out_ch] if self.out_ch else out

    def forward(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        cs, co, ci, cn = self._coef(sigma)
        v = lambda t: t.reshape(-1, 1, 1, 1)
        return v(cs) * x + v(co) * self._v(x, sigma, y, **kw)

    def score(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        s = sigma.reshape(-1, 1, 1, 1)
        return -(x + self._v(x, sigma, y, **kw)) / (s * (1.0 + s))

    def cfg_score(self, x, sigma, y, scale=1.0, **kw):
        if scale == 1.0:
            return self.score(x, sigma, y, **kw)
        sigma = _as_batch(sigma, x)
        keep = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        c = self.score(x, sigma, y, force_drop=keep, **kw)
        u = self.score(x, sigma, y, force_drop=~keep, **kw)
        return u + scale * (c - u)


def _as_batch(sigma, x):
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    sigma = sigma.reshape(-1).to(x.device, x.dtype)
    return sigma.expand(x.shape[0]) if sigma.numel() == 1 else sigma
