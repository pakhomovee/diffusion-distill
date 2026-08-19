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

from .edm import EDMWrapper, sample_sigma, dsm_weight, edm_sigmas
from .robust import LambdaEstimator, robust_real_score
from .gan import GANHead, d_loss, g_loss


class DMD2Trainer:
    def __init__(self, student, critic, teacher, cfg, device="cuda"):
        """student/critic/teacher: EDMWrapper-wrapped DiTs. cfg: dict-like."""
        self.G, self.mu, self.T = student, critic, teacher
        self.c, self.dev = cfg, device
        self.sigma_data = cfg["sigma_data"]
        self.gen_sigmas = edm_sigmas(cfg["n_student_steps"], device=device) \
            if cfg["n_student_steps"] > 1 else None
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
        self.gan = None
        if cfg.get("gan_weight", 0.0) > 0:
            hid = _raw(critic).net.final.lin.in_features
            self.gan = GANHead(hid).to(device)
        params_D = list(self.mu.parameters()) + (
            list(self.gan.parameters()) if self.gan else [])
        self.opt_D = torch.optim.AdamW(params_D, lr=cfg["lr_d"],
                                       betas=(0.0, 0.999), weight_decay=0.01)
        self.step_i = 0

    def _disc(self, x, sigma, y):
        """Discriminator logit, reusing the critic trunk."""
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
        sigma = sample_sigma(n, self.c["P_mean"], self.c["P_std"], self.dev)
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
        sigma = sample_sigma(n, self.c["P_mean"], self.c["P_std"], self.dev)
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
                sg = sample_sigma(n, c["P_mean"], c["P_std"], self.dev)
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
            sc = sample_sigma(rb.shape[0], c["P_mean"], c["P_std"], self.dev)
            self.est.calibrate(self.T, rb, ry, sc, cfg=c["cfg_scale"])

        # --- generator update ---
        yg = torch.randint(0, c["n_classes"], (n,), device=self.dev)
        xg = self.generate(n, yg)
        lg, info = self.dm_loss(xg, yg, real_batch)
        if self.gan is not None:
            sg = sample_sigma(n, c["P_mean"], c["P_std"], self.dev)
            xg_n = xg + sg.reshape(-1, 1, 1, 1) * torch.randn_like(xg)
            lga = g_loss(self._disc(xg_n, sg, yg))
            lg = lg + c["gan_weight"] * lga
            info["loss_gan_g"] = lga.item()
        self.opt_G.zero_grad(set_to_none=True)
        lg.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), c["clip"])
        self.opt_G.step()
        logs.update(loss_g=lg.item(), **info)
        self.step_i += 1
        return logs


def _raw(m):
    """Unwrap DDP."""
    return m.module if hasattr(m, "module") else m
