"""DMD2 trainer, with Track A's doubly-robust real-score term as a drop-in.

Baseline (mode='teacher' + gan_weight>0) reproduces DMD2: a one/few-step student
trained by distribution matching against a frozen teacher, with a fake-score
critic and an auxiliary GAN term on real data.

Track A (mode='robust', gan_weight=0) replaces that GAN term with a closed-form
nonparametric real-data score, fused with the teacher at an ONLINE-ESTIMATED
weight lambda(sigma). No discriminator, no extra network, one fewer
hyperparameter.
"""
import math
import torch
import torch.nn.functional as F

from .edm import EDMWrapper, dsm_weight, edm_sigmas
from .vp import make_sigma_sampler
from .robust import LambdaEstimator, LambdaProbe, robust_real_score
from .gan import GANHead, ConvGANHead, d_loss, g_loss


class DMD2Trainer:
    def __init__(self, student, critic, teacher, cfg, device="cuda", schedule=None):
        """student/critic/teacher: sigma-parameterised denoisers (EDMWrapper or
        VPPrecond -- the trainer does not care which). cfg: dict-like.
        schedule: VPSchedule when the teacher is a VP checkpoint, else None."""
        self.G, self.mu, self.T = student, critic, teacher
        self.c, self.dev, self.sch = cfg, device, schedule
        self.sigma_data = cfg["sigma_data"]
        # Training noise levels. A VP teacher is only trained on its own
        # schedule, so `vp_uniform_t` (DMD2's choice) puts mass where the
        # teacher is valid; EDM's lognormal would concentrate in 0.03..3 and
        # never visit the top three quarters of a 0.01..157 sigma range.
        self.sample_sigma = make_sigma_sampler(cfg, schedule)
        self.gen_sigmas = self._student_grid()
        # The gate is expressed as a MULTIPLE of sigma_data, not an absolute
        # sigma, because that is the unit it was measured in. exp01 sweep: the
        # fused estimator is never worse than the better single estimator for
        # any gate in [0.7, 1.6]*sigma_data, and is catastrophic below 0.5.
        # 1.0 sits in the middle of the safe band. See LOG.log ENTRY 008.
        gate_mult = cfg.get("lam_gate_mult", 1.0)
        self.lam_mode = cfg.get("lam_estimator", "dsm")
        self.est = LambdaEstimator(n_bins=cfg.get("lam_bins", 32),
                                   ema=cfg.get("lam_ema", 0.999),
                                   mode=self.lam_mode,
                                   min_count=cfg.get("lam_min_count", 256),
                                   gate_below=(None if gate_mult is None
                                               else gate_mult * self.sigma_data),
                                   device=device)
        # The gate exists only for the legacy ratio estimator. The DSM estimator
        # does not need one -- that is the point of it.
        self.lam_gate = None if self.lam_mode == "dsm" else self.est.gate_below
        self.opt_G = torch.optim.AdamW(self.G.parameters(), lr=cfg["lr_g"],
                                       betas=(0.0, 0.999), weight_decay=0.01)
        # DMD2's auxiliary discriminator: a head on the critic's own features.
        # gan_weight = 0 turns it off, which is the Track A configuration.
        # Two discriminator flavours. DMD2 reuses the critic's own token trunk,
        # which our DiT exposes as `.trunk()`. Third-party backbones (diffusers
        # UNets, NVlabs EDM, REPA SiT) do not, so they get a small standalone
        # conv head instead -- see gan.ConvGANHead for why that is honest rather
        # than a shortcut.
        self.gan, self.gan_kind = None, None
        if cfg.get("gan_weight", 0.0) > 0:
            base = getattr(_raw(critic), "net", None)
            # `trunk_dims` rather than `final.lin.in_features`: the old probe
            # only worked on our DiT, where the token and conditioning widths
            # happen to be equal, so every third-party backbone fell through to
            # ConvGANHead -- and LOG ENTRY 015 measured that substitute
            # collapsing the baseline. Anything that reports both widths now
            # gets DMD2's actual head on the critic's own features.
            if hasattr(base, "trunk") and hasattr(base, "trunk_dims"):
                self.gan = GANHead(*base.trunk_dims).to(device)
                self.gan_kind = "trunk"
            else:
                self.gan = ConvGANHead(cfg["shape"][0], res=cfg["shape"][-1]).to(device)
                self.gan_kind = "conv"
        params_D = list(self.mu.parameters()) + (
            list(self.gan.parameters()) if self.gan else [])
        self.opt_D = torch.optim.AdamW(params_D, lr=cfg["lr_d"],
                                       betas=(0.0, 0.999), weight_decay=0.01)
        # Diagnostic only -- never touches the training lambda. See FINDINGS 3.1.
        self.probe = LambdaProbe(n_bins=cfg.get("probe_bins", 16),
                                 sigma_max=max(2.0 * cfg["sigma_max"], 200.0),
                                 device=device)
        self.step_i = 0

    def _student_grid(self):
        """Backward-Euler sigma grid for a few-step student (None if one-step)."""
        n = self.c["n_student_steps"]
        if n <= 1:
            return None
        if self.sch is not None:
            return self.sch.student_sigmas(n, sigma_max=self.c["sigma_max"]).to(self.dev)
        return edm_sigmas(n, sigma_max=self.c["sigma_max"], device=self.dev)

    def _disc(self, x, sigma, y):
        """Discriminator logit."""
        if self.gan_kind == "conv":
            return self.gan(x, sigma)
        net = _raw(self.mu)
        cs, co, ci, cn = net._coef(sigma)
        tok, cond = net.net.trunk(ci.reshape(-1, 1, 1, 1) * x, cn, y)
        return self.gan(tok, cond)

    # ---------------- student ------------------------------------------
    def generate(self, n, y, z=None, grad=True):
        """One-step (or few-step backward-Euler) student sample.

        For the few-step student, only ONE randomly chosen step carries gradient
        and the rest run under no_grad. That is what DMD2 does, and it is also
        what DDP requires: multiple grad-tracked forwards of the same module
        before a single backward trips the reducer's "marked ready twice" check.
        """
        smax = self.c["sigma_max"]
        x = (torch.randn(n, *self.c["shape"], device=self.dev) if z is None else z) * smax
        if self.gen_sigmas is None:
            return self.G(x, torch.full((n,), smax, device=self.dev), y)
        sig = self.gen_sigmas
        k = int(torch.randint(0, len(sig) - 1, ()).item()) if grad else -1
        for i in range(len(sig) - 1):
            if grad and i == k:
                x0 = self.G(x, sig[i].expand(n), y)
            else:
                with torch.no_grad():
                    x0 = self.G(x, sig[i].expand(n), y)
                x0 = x0.detach()
            x = x0 + sig[i + 1] * torch.randn_like(x0) if sig[i + 1] > 0 else x0
        return x

    # ---------------- distribution-matching gradient --------------------
    def dm_loss(self, x0, y, real_batch):
        n = x0.shape[0]
        sigma = self.sample_sigma(n, self.dev)
        xt = x0 + sigma.reshape(-1, 1, 1, 1) * torch.randn_like(x0)

        with torch.no_grad():
            s_real, lam = robust_real_score(
                self.T, xt, sigma, y, real_batch, self.est,
                cfg=self.c["cfg_scale"], mode=self.c["mode"],
                fixed_lam=self.c.get("fixed_lam"),
                gather=self.c.get("gather_real", True),
                gate=self.lam_gate)
            s_fake = _raw(self.mu).score(xt, sigma, y)
            v = sigma.reshape(-1, 1, 1, 1) ** 2
            D_real, D_fake = xt + v * s_real, xt + v * s_fake
            # DMD2's scale-free gradient: difference of denoiser predictions,
            # normalised by the teacher's own reconstruction magnitude so the
            # loss scale does not depend on sigma.
            grad = (D_fake - D_real)
            norm = (x0.detach() - D_real).abs().mean(dim=(1, 2, 3), keepdim=True)
            grad = grad / norm.clamp_min(1e-4)
            grad = torch.nan_to_num(grad)
        # surrogate whose gradient wrt theta equals `grad`
        loss = 0.5 * F.mse_loss(x0, (x0 - grad).detach())
        return loss, dict(lam=lam.mean().item(), sigma=sigma.mean().item())

    # ---------------- critic (fake score) -------------------------------
    def critic_loss(self, x0):
        n = x0.shape[0]
        y = torch.randint(0, self.c["n_classes"], (n,), device=self.dev)
        sigma = self.sample_sigma(n, self.dev)
        xt = x0 + sigma.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
        D = self.mu(xt, sigma, y)
        w = dsm_weight(sigma, self.sigma_data).reshape(-1, 1, 1, 1)
        return (w * (D - x0) ** 2).mean()

    # ---------------- one optimisation step -----------------------------
    def step(self, real_batch, real_y):
        c, n = self.c, real_batch.shape[0]
        logs = {}
        # --- critic updates (two-timescale) ---
        for _ in range(c["d_steps"]):
            with torch.no_grad():
                yg = torch.randint(0, c["n_classes"], (n,), device=self.dev)
                xg = self.generate(n, yg, grad=False)
            ld = self.critic_loss(xg)
            if self.gan is not None:
                sg = self.sample_sigma(n, self.dev)
                v = sg.reshape(-1, 1, 1, 1)
                xr_n = real_batch + v * torch.randn_like(real_batch)
                xg_n = xg + v * torch.randn_like(xg)
                lgan = d_loss(self._disc(xr_n, sg, real_y), self._disc(xg_n, sg, yg))
                ld = ld + c["gan_weight"] * lgan
                logs["loss_gan_d"] = lgan.item()
            self.opt_D.zero_grad(set_to_none=True)
            ld.backward()
            torch.nn.utils.clip_grad_norm_(self.mu.parameters(), c["clip"])
            self.opt_D.step()
            logs["loss_d"] = ld.item()
        # --- lambda calibration on held-out real data ---
        if (c["mode"] == "robust" and self.lam_mode == "dsm"
                and self.step_i % c.get("lam_calib_every", 4) == 0):
            from .robust import all_gather_batch
            rb = all_gather_batch(real_batch) if c.get("gather_real", True) else real_batch
            ry = all_gather_batch(real_y.reshape(-1, 1, 1, 1).float()).reshape(-1).long() \
                if c.get("gather_real", True) else real_y
            sc = self.sample_sigma(rb.shape[0], self.dev)
            self.est.calibrate(self.T, rb, ry, sc, cfg=c["cfg_scale"])

        # --- generator update ---
        yg = torch.randint(0, c["n_classes"], (n,), device=self.dev)
        xg = self.generate(n, yg)
        lg, info = self.dm_loss(xg, yg, real_batch)
        if self.gan is not None:
            sg = self.sample_sigma(n, self.dev)
            xg_n = xg + sg.reshape(-1, 1, 1, 1) * torch.randn_like(xg)
            lga = g_loss(self._disc(xg_n, sg, yg))
            # How hard is the GAN term ACTUALLY pulling? Not answerable from the
            # loss values: the DM loss averages over B*C*H*W and the GAN loss
            # over B, so a unit of GAN loss carries C*H*W = 3072x more gradient
            # on CIFAR. That factor is exact, but the ratio also depends on
            # |d logit / d x|, which is a property of the discriminator and has
            # to be measured. Taken w.r.t. the SAMPLES rather than the
            # parameters: it is the same ratio, needs no second parameter
            # backward, and cannot upset DDP's reducer.
            if self.step_i % c.get("diag_every", 500) == 0:
                gd = torch.autograd.grad(lg, xg, retain_graph=True)[0].norm()
                gg = torch.autograd.grad(c["gan_weight"] * lga, xg,
                                         retain_graph=True)[0].norm()
                info["gnorm_dm"] = gd.item()
                info["gnorm_gan"] = gg.item()
                # >1 means the adversary outweighs distribution matching.
                info["gan_pull"] = (gg / gd.clamp_min(1e-12)).item()
            lg = lg + c["gan_weight"] * lga
            info["loss_gan_g"] = lga.item()
        self.opt_G.zero_grad(set_to_none=True)
        lg.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), c["clip"])
        self.opt_G.step()
        logs.update(loss_g=lg.item(), **info)
        self.step_i += 1
        return logs


    # ---------------- lambda drift diagnostic ---------------------------
    @torch.no_grad()
    def probe_lambda(self, real_batch, real_y, n_repeat=4):
        """Measure the ratio-lambda statistic at real AND student samples.

        FINDINGS 1.2 predicts the two curves start apart -- lower at student
        samples, where the teacher is off-manifold and the real batch is worth
        more -- and converge as the student lands on the manifold. That drift is
        the mechanism claim; this is the measurement of it.

        Both legs share the same sigmas, the same real batch and the same
        teacher call pattern, so the ratio estimator's known slack is common to
        them and the DIFFERENCE is still meaningful even though neither number
        is lambda*. Costs `2 * n_repeat` teacher forwards, run every
        `probe_every` steps only.
        """
        from .robust import all_gather_batch
        c = self.c
        rb = all_gather_batch(real_batch) if c.get("gather_real", True) else real_batch
        n = real_batch.shape[0]
        out = {}
        for name, src_y in (("real", real_y), ("student", None)):
            self.probe.reset()
            for _ in range(n_repeat):
                y = (src_y if src_y is not None else
                     torch.randint(0, c["n_classes"], (n,), device=self.dev))
                x0 = real_batch if src_y is not None else self.generate(n, y, grad=False)
                sig = self.sample_sigma(n, self.dev)
                xt = x0 + sig.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
                self.probe.observe(self.T, xt, y, sig, rb, cfg=c["cfg_scale"])
            s, lam, vrel, cnt = self.probe.curve()
            out[name] = dict(sigma=s.tolist(), lam_ratio=lam.tolist(),
                             var_rel=vrel.tolist(), count=cnt.tolist())
        return out


def _raw(m):
    """Unwrap DDP."""
    return m.module if hasattr(m, "module") else m
