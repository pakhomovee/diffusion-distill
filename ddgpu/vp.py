"""VP (linear-beta latent-diffusion) preconditioning, in sigma coordinates.

Why this file exists
--------------------
Everything else in `ddgpu/` speaks EDM: a denoiser `D(x, sigma, y)` on
`x = x0 + sigma * eps`, and a score `s = (D - x) / sigma^2`. But the only
publicly released latent-diffusion teachers of the right architecture --
facebook's `DiT-XL-2-256x256.pt` and `DiT-XL-2-512x512.pt` -- are **VP**
models: they predict `eps` from `x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps`
with a *discrete* timestep `t in {0..999}` and a linear beta schedule.

Those are two different parameterisations of the same object, and the
difference is not cosmetic:

  * the network input is scaled differently (`c_in` is `sqrt(abar)`, not
    `1/sqrt(sigma^2+sigma_d^2)`),
  * the noise conditioning is a timestep index, not `log(sigma)/4`,
  * the output is `eps`, not a skip-corrected `x0`.

So loading a public checkpoint into `EDMWrapper` and calling it would produce
garbage, and -- worse -- `init_from_teacher=true` would hand the student those
same weights under the wrong preconditioning, which looks like a training
failure rather than a bug. `VPPrecond` closes that gap: it exposes the *exact*
same `(forward, score, cfg_score, _coef)` surface as `EDMWrapper`, so the
trainers, the lambda estimator and the GAN head are unchanged, while the
network underneath stays in its native convention and the checkpoint loads
bit-exactly.

The change of variables (Karras et al. 2022, App. C)
----------------------------------------------------
With `x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps` and `sigma(t)^2 =
(1-abar_t)/abar_t`, writing `x = x0 + sigma eps` gives `x_t = sqrt(abar_t) x`.
Hence

    eps_hat = net(sqrt(abar) * x, t(sigma), y)
    score   = -eps_hat / sigma
    D       = x + sigma^2 * score = x - sigma * eps_hat

i.e. `c_skip = 1`, `c_out = -sigma`, `c_in = sqrt(abar) = 1/sqrt(1+sigma^2)`,
`c_noise = t(sigma)`. `t(sigma)` is obtained by inverting the (monotone) sigma
table with linear interpolation in `log sigma`, so `sigma` stays continuous --
the sinusoidal timestep embedding is defined for non-integer `t`.
"""
import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
class VPSchedule:
    """Linear-beta DDPM schedule, and the sigma <-> t map it induces.

    Defaults reproduce `guided_diffusion.get_named_beta_schedule("linear", T)`
    as used by DiT: `betas = linspace(1e-4 * 1000/T, 2e-2 * 1000/T, T)`.
    """

    def __init__(self, n_timestep=1000, beta_start=1e-4, beta_end=2e-2,
                 device="cpu"):
        scale = 1000.0 / n_timestep
        betas = torch.linspace(beta_start * scale, beta_end * scale, n_timestep,
                               dtype=torch.float64)
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.T = n_timestep
        self.abar = abar.to(device)
        self.sigmas = ((1.0 - abar) / abar).sqrt().to(device)      # (T,), increasing
        self.log_sigmas = self.sigmas.log()
        self.t_index = torch.arange(n_timestep, dtype=torch.float64, device=device)

    @property
    def sigma_min(self):
        return float(self.sigmas[0])

    @property
    def sigma_max(self):
        return float(self.sigmas[-1])

    def to(self, device):
        for k in ("abar", "sigmas", "log_sigmas", "t_index"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def t_of_sigma(self, sigma):
        """Continuous timestep for a sigma, by linear interpolation in log-sigma.

        Clamped to [0, T-1]: a sigma outside the schedule (e.g. sigma=0 at the
        last sampler step) maps to the nearest end rather than raising, which is
        what every sampler in the field does.
        """
        ls = sigma.double().clamp_min(1e-20).log()
        tab = self.log_sigmas.to(ls.device)
        i = torch.searchsorted(tab, ls.contiguous()).clamp(1, self.T - 1)
        lo, hi = tab[i - 1], tab[i]
        w = ((ls - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
        return ((i - 1).double() + w).clamp(0, self.T - 1)

    def sigma_of_t(self, t):
        """Inverse of `t_of_sigma`, for building samplers in timestep space."""
        t = torch.as_tensor(t, dtype=torch.float64,
                            device=self.sigmas.device).clamp(0, self.T - 1)
        i0 = t.floor().long().clamp(0, self.T - 2)
        w = (t - i0.double())
        ls = self.log_sigmas[i0] * (1 - w) + self.log_sigmas[i0 + 1] * w
        return ls.exp().float()

    def student_sigmas(self, n_steps, sigma_max=None):
        """Backward-Euler sampling grid for an n-step student, in sigma space.

        DMD2's few-step ImageNet student uses the evenly spaced timesteps
        {999, 749, 499, 249} for n_steps=4 (and {999} for one step), which is
        what this reproduces -- *not* an EDM rho=7 grid, because the student was
        initialised from a VP teacher and its useful noise levels are the ones
        that teacher was trained on. Returns n_steps+1 values ending at 0.

        Signature matches `interpolant.LinearInterpolant.student_sigmas` so the
        trainer and the sampler can call either without branching.
        """
        t_max = (self.T - 1 if sigma_max is None
                 else float(self.t_of_sigma(torch.tensor([float(sigma_max)]))[0]))
        ts = [t_max - i * (t_max + 1) / n_steps for i in range(n_steps)]
        s = self.sigma_of_t(torch.tensor(ts, dtype=torch.float64))
        return torch.cat([s, torch.zeros(1, device=s.device, dtype=s.dtype)])


# ---------------------------------------------------------------------------
# Preconditioner
# ---------------------------------------------------------------------------
class VPPrecond(nn.Module):
    """Wraps a raw eps-predicting DiT as a sigma-parameterised EDM denoiser.

    API-identical to `edm.EDMWrapper` so nothing downstream needs to branch.
    `out_ch` slices off the learned-variance channels of a `learn_sigma=True`
    checkpoint (DiT-XL/2 predicts 8 channels: eps, then the variance
    interpolation logits, which distillation never uses).
    """

    def __init__(self, net, schedule=None, out_ch=None, sigma_data=1.0):
        super().__init__()
        self.net = net
        self.sch = schedule or VPSchedule()
        self.out_ch = out_ch
        self.sigma_data = sigma_data          # only used for DSM loss weighting

    def _apply(self, fn):                      # keep the schedule on the module's device
        out = super()._apply(fn)
        try:
            self.sch.to(next(self.net.parameters()).device)
        except StopIteration:
            pass
        return out

    def _coef(self, sigma):
        """(c_skip, c_out, c_in, c_noise), matching EDMWrapper's contract."""
        c_in = 1.0 / (1.0 + sigma ** 2).sqrt()
        c_noise = self.sch.t_of_sigma(sigma).to(sigma.dtype)
        return torch.ones_like(sigma), -sigma, c_in, c_noise

    def _eps(self, x, sigma, y, **kw):
        cs, co, ci, cn = self._coef(sigma)
        v = lambda t: t.reshape(-1, 1, 1, 1)
        out = self.net(v(ci) * x, cn, y, **kw)
        return out[:, :self.out_ch] if self.out_ch else out

    # `D = x - sigma * eps` is a CATASTROPHIC CANCELLATION when sigma is large:
    # both terms are O(sigma) and the answer is only O(sigma_data), so an error
    # `d` in eps lands in D multiplied by sigma. Under bf16 autocast the network
    # returns eps with ulp 2^-7, and at CIFAR's sigma_max=157.4 with
    # sigma_data=0.5 that is rounding noise of std 0.26 against a 0.5 signal --
    # SNR 2, an image buried in speckle. A one-step student generates at
    # sigma_max on EVERY sample, so it cannot express a clean image at all.
    #
    # This is specific to the VP eps-parameterisation. EDM's c_out is
    # sigma*sigma_data/sqrt(sigma^2+sigma_data^2) -> sigma_data and the
    # interpolant's is -sigma/(1+sigma) -> -1; both keep the network's
    # contribution O(1), which is what that preconditioning is FOR. Here c_out
    # is an unnormalised -sigma, so the guard below has to do the same job.
    #
    # `score` needs no guard: it DIVIDES by sigma and shrinks the same error.
    FP32_MARGIN = 0.05        # tolerated rounding noise, as a fraction of sigma_data

    def _fp32_needed(self, sigma):
        """Would low-precision eps put more than FP32_MARGIN*sigma_data into D?"""
        ulp = torch.finfo(torch.bfloat16).eps      # the worst autocast may hand us
        return float(sigma.max()) * ulp > self.FP32_MARGIN * self.sigma_data

    def forward(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        if self._fp32_needed(sigma):
            # A no-op when the caller was not autocasting in the first place.
            with torch.autocast(x.device.type, enabled=False):
                eps = self._eps(x.float(), sigma.float(), y, **kw)
        else:
            eps = self._eps(x, sigma, y, **kw)
        return x - sigma.reshape(-1, 1, 1, 1) * eps

    def score(self, x, sigma, y, **kw):
        sigma = _as_batch(sigma, x)
        return -self._eps(x, sigma, y, **kw) / sigma.reshape(-1, 1, 1, 1)

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


# ---------------------------------------------------------------------------
# Noise-level sampling
# ---------------------------------------------------------------------------
def make_sigma_sampler(cfg, schedule=None):
    """Returns `f(n, device) -> sigma`, chosen by `cfg['sigma_dist']`.

    'lognormal'    : EDM's `exp(N(P_mean, P_std))`. Calibrated for sigma_data
                     ~0.5; the default for EDM-preconditioned runs.
    'interp_uniform_t' : uniform over the linear-interpolant time t, which is
                     what SiT/flow-matching models are trained with.
    'vp_uniform_t' : uniform over an integer timestep window, mapped to sigma.
                     This is what DMD2 does, and it is the right choice for a VP
                     teacher: it puts mass where that teacher was actually
                     trained, over the schedule's full 0.006..157 sigma range,
                     rather than over the 0.03..3 band a lognormal would give.
    """
    dist = cfg.get("sigma_dist", "lognormal")
    if dist == "lognormal":
        P_mean, P_std = cfg["P_mean"], cfg["P_std"]
        return lambda n, device: (torch.randn(n, device=device) * P_std + P_mean).exp()
    if dist == "interp_uniform_t":
        # SiT / flow matching trains with t ~ U[0,1] on the linear interpolant,
        # so this is the matched choice there. `t_lo`/`t_hi` trim the ends, which
        # are sigma=0 and sigma=inf and representable at neither.
        assert schedule is not None, "interp_uniform_t needs a LinearInterpolant"
        lo, hi = float(cfg.get("t_lo", 0.02)), float(cfg.get("t_hi", 0.98))

        def f(n, device):
            t = torch.rand(n, device=device) * (hi - lo) + lo
            return (t / (1.0 - t)).to(device)
        return f
    if dist == "vp_uniform_t":
        assert schedule is not None, "vp_uniform_t needs a VPSchedule"
        lo = int(cfg.get("t_min", 20))
        hi = int(cfg.get("t_max", schedule.T - 21))

        def f(n, device):
            t = torch.randint(lo, hi + 1, (n,), device=device)
            return schedule.sigma_of_t(t.double()).to(device)
        return f
    raise ValueError(f"unknown sigma_dist {dist!r}")
