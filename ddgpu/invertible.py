"""TRACK B -- invertible / noise-space distillation.

Idea (info.txt #2): stop matching an unknown p_data in 16k dims. Learn a
bijection instead, and do the distribution matching in noise space where the
target is exactly N(0, I).

    G_theta : Z -> X   one-step generator (initialised from the teacher)
    E_phi   : X -> Z   encoder, the approximate inverse

Correctness argument, stated so the failure mode is visible:
    if  E # p_data = N(0,I)   AND   G(E(x)) = x  for p_data-a.e. x,
    then  G # N(0,I) = G # (E # p_data) = p_data.
So the two constraints TOGETHER are sufficient. Neither alone is. The failure is
that both hold only approximately and the errors compensate -- "collusion":
E maps onto a proper subspace, G is only sensible on that subspace, the cycle
loss is happy, and samples from the full N(0,I) are garbage.

Three defences, in increasing order of how much I trust them:
 1. TEACHER ANCHOR (strongest). The teacher's probability-flow ODE IS an exact
    bijection Phi: Z -> X. So there exist ground-truth pairs (z, Phi(z)),
    computable offline. G regresses onto Phi and E onto Phi^{-1}. This turns an
    ill-posed joint constraint into two supervised regressions plus a
    distribution-matching correction, and it is what makes collusion expensive.
 2. FULL-RANK GAUSSIANITY. Per exp/02, the cheap O(Bd) moment penalty is the
    only term with reliable power against the low-rank/collapsed-direction
    perturbation, which is precisely the collusion signature. MMD catches scale
    error, KSD catches correlation. They are used TOGETHER because their blind
    spots are complementary -- that is a measured result, not a guess.
 3. PRIOR-SIDE CYCLE. E(G(z)) ~= z for z ~ N(0,I) drawn from the FULL prior,
    not just from encoded data, so the generator is pinned everywhere the prior
    has mass.

Note on cost: E REPLACES the fake-score critic, it does not add to it. Track B
holds the same two trainable networks as DMD2 (G, E vs G, mu) plus the frozen
teacher, so its VRAM footprint matches the DMD2 baseline. It is critic-free in
the sense that matters -- no adversary, no inner-loop two-timescale schedule.
"""
import torch
import torch.nn.functional as F

from .edm import EDMWrapper, edm_sigmas
from .gauss_disc import (mmd2_to_gaussian, ksd2_to_gaussian,
                         gaussian_moment_penalty, sliced_gaussian_ks)


# ---------------------------------------------------------------------------
# Teacher anchor: exact (z, x) pairs from the deterministic PF-ODE
# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_ode(teacher, x, y, n_steps=32, cfg=1.0, reverse=False):
    """Deterministic Heun integration of the EDM probability-flow ODE.

        dx/dsigma = (x - D(x, sigma)) / sigma

    reverse=False: x is noise at sigma_max  -> returns data at sigma_min
    reverse=True : x is data at sigma_min   -> returns noise at sigma_max
    Both directions share the discretisation, so Phi^{-1}(Phi(z)) = z up to
    integrator error. That is what makes these pairs usable as supervision.

    Note we stop at sigma_min rather than 0: the drift is undefined at sigma=0,
    and an inversion that starts there is not defined either.
    """
    sig = edm_sigmas(n_steps, device=x.device)[:-1]      # drop the trailing 0
    if reverse:
        sig = sig.flip(0)

    def denoise(u, s):
        sb = s.expand(u.shape[0])
        if cfg == 1.0:
            return teacher(u, sb, y)
        # CFG in denoiser space is equivalent to CFG on the score
        sc = teacher.cfg_score(u, sb, y, scale=cfg)
        return u + sb.reshape(-1, 1, 1, 1) ** 2 * sc

    for i in range(len(sig) - 1):
        s, s1 = sig[i], sig[i + 1]
        d = (x - denoise(x, s)) / s
        xn = x + (s1 - s) * d
        d2 = (xn - denoise(xn, s1)) / s1
        x = x + (s1 - s) * 0.5 * (d + d2)
    return x


class AnchorCache:
    """Offline (z, x) pairs from the teacher ODE, generated once and reused.

    Generating these is the one genuinely expensive preprocessing step in Track
    B: n_steps teacher forwards per pair. See RUNPLAN.md for its GPU cost. They
    are noise-free supervision and are worth far more per FLOP than an extra
    training step.
    """

    def __init__(self, path=None, device="cuda"):
        self.z = self.x = self.y = None
        self.device = device
        if path:
            d = torch.load(path, map_location="cpu")
            self.z, self.x, self.y = d["z"], d["x"], d["y"]

    @torch.no_grad()
    def build(self, teacher, n, shape, n_classes, sigma_max, n_steps=32,
              batch=64, cfg=1.0, device="cuda"):
        zs, xs, ys = [], [], []
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, *shape, device=device) * sigma_max
            y = torch.randint(0, n_classes, (b,), device=device)
            x = teacher_ode(teacher, z, y, n_steps, cfg)
            zs.append(z.cpu()); xs.append(x.cpu()); ys.append(y.cpu())
        self.z, self.x, self.y = torch.cat(zs), torch.cat(xs), torch.cat(ys)
        return self

    def sample(self, n):
        i = torch.randint(0, self.z.shape[0], (n,))
        return (self.z[i].to(self.device), self.x[i].to(self.device),
                self.y[i].to(self.device))

    def save(self, path):
        torch.save(dict(z=self.z, x=self.x, y=self.y), path)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class InvertibleTrainer:
    """G and E trained jointly. No critic, no discriminator, no inner loop."""

    def __init__(self, G, E, teacher, cfg, anchors=None, device="cuda"):
        self.G, self.E, self.T = G, E, teacher
        self.c, self.dev, self.anchors = cfg, device, anchors
        self.smax = cfg["sigma_max"]
        self.opt = torch.optim.AdamW(
            list(G.parameters()) + list(E.parameters()),
            lr=cfg["lr_g"], betas=(0.9, 0.999), weight_decay=0.01)
        self.step_i = 0

    # ---- the two maps -------------------------------------------------
    def gen(self, z, y):
        """G: noise -> data. Reuses the EDM parameterisation at sigma_max."""
        n = z.shape[0]
        return self.G(z, torch.full((n,), self.smax, device=self.dev), y)

    def enc(self, x, y):
        """E: data -> noise, returned at the SAME scale G consumes (sigma_max).

        E is a raw DiT, deliberately NOT EDM-preconditioned: the preconditioning
        is built to emit clean data (output scale ~sigma_data), whereas E must
        emit a code of scale sigma_max. Feeding it through EDMWrapper would put
        a factor of ~160 in the wrong place at init. Instead E outputs a unit-
        scale code which we rescale here.
        """
        n = x.shape[0]
        c = torch.zeros(n, device=self.dev)          # constant conditioning
        xin = x / self.c["sigma_data"]
        # SKIP CONNECTION, and it is load-bearing. DiT zero-initialises its final
        # layer -- standard, and correct for a denoiser because the EDM skip
        # carries the signal. E has no such skip, so a zero-init E outputs
        # EXACTLY ZERO, which is the fully-collapsed encoder: the precise
        # degenerate solution Track B is supposed to avoid, handed to the
        # optimiser as its starting point. With this skip, E starts as a
        # whitening map (z has std sigma_max when x has std sigma_data) and
        # learns the residual.
        return (xin + self.E(xin, c, y)) * self.smax

    # ---- losses -------------------------------------------------------
    def gaussianity(self, z, gather=True):
        """Discrepancy of E#p_data from N(0,I). Terms have complementary blind
        spots (exp/02): moment catches rank collapse, MMD catches scale, KSD
        catches correlation."""
        zc = z.reshape(z.shape[0], -1) / self.smax
        if gather and torch.distributed.is_available() and torch.distributed.is_initialized():
            zc = _all_gather(zc)                      # bigger effective batch, cheap
        c = self.c
        terms = dict(
            mmd=mmd2_to_gaussian(zc) * c["w_mmd"],
            ksd=ksd2_to_gaussian(zc) * c["w_ksd"],
            mom=gaussian_moment_penalty(zc) * c["w_moment"],
        )
        return sum(terms.values()), {k: float(v) for k, v in terms.items()}

    def step(self, real_x, real_y):
        """NOTE on DDP: G and E are each invoked more than once per backward
        (reconstruction leg, cycle leg, anchor leg). DDP's default reducer marks
        a parameter ready on the first autograd hook and errors on the second,
        so these models MUST be constructed with static_graph=True -- which
        ddgpu.train.wrap_ddp does for track B. Without it this step raises
        "Expected to mark a variable ready only once" on the second leg."""
        c, n = self.c, real_x.shape[0]
        logs = {}

        # --- leg 1: data -> noise -> data (reconstruction + Gaussianity) ---
        z_enc = self.enc(real_x, real_y)
        l_gauss, gterms = self.gaussianity(z_enc)
        x_rec = self.gen(z_enc, real_y)
        l_rec = F.mse_loss(x_rec, real_x)

        # --- leg 2: noise -> data -> noise (prior-side cycle) ---
        z = torch.randn(n, *c["shape"], device=self.dev) * self.smax
        y = torch.randint(0, c["n_classes"], (n,), device=self.dev)
        x_gen = self.gen(z, y)
        z_cyc = self.enc(x_gen, y)
        l_cyc = F.mse_loss(z_cyc, z) / self.smax ** 2

        # --- leg 3: teacher anchor (supervised, both directions) ---
        l_anc_g = l_anc_e = torch.zeros((), device=self.dev)
        if self.anchors is not None and c["w_anchor"] > 0:
            za, xa, ya = self.anchors.sample(min(n, c["anchor_batch"]))
            l_anc_g = F.mse_loss(self.gen(za, ya), xa)
            l_anc_e = F.mse_loss(self.enc(xa, ya), za) / self.smax ** 2

        loss = (c["w_rec"] * l_rec + l_gauss + c["w_cyc"] * l_cyc
                + c["w_anchor"] * (l_anc_g + l_anc_e))
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.G.parameters()) + list(self.E.parameters()), c["clip"])
        self.opt.step()
        self.step_i += 1

        logs.update(loss=loss.item(), rec=l_rec.item(), cyc=l_cyc.item(),
                    anc_g=float(l_anc_g.detach()), anc_e=float(l_anc_e.detach()), **gterms)
        return logs

    # ---- diagnostics --------------------------------------------------
    @torch.no_grad()
    def diagnose(self, real_x, real_y, n_slices=128):
        """Collusion audit. Run periodically; these are the numbers that decide
        whether Track B is working or quietly degenerating."""
        z = self.enc(real_x, real_y).reshape(real_x.shape[0], -1) / self.smax
        # effective rank of the encoded distribution -- rank collapse is the
        # collusion signature and shows up here before it shows up in FID
        s = torch.linalg.svdvals(z - z.mean(0, keepdim=True))
        p = s / s.sum()
        eff_rank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
        max_rank = min(z.shape)          # not d: a batch of B can span at most B dims
        cvm = sliced_gaussian_ks(z, n_slices)
        # round-trip errors in both directions
        zr = torch.randn_like(z).reshape(real_x.shape) * self.smax
        yr = torch.randint(0, self.c["n_classes"], (real_x.shape[0],), device=self.dev)
        rt_z = F.mse_loss(self.enc(self.gen(zr, yr), yr), zr).item() / self.smax ** 2
        rt_x = F.mse_loss(self.gen(self.enc(real_x, real_y), real_y), real_x).item()
        return dict(eff_rank=eff_rank, eff_rank_frac=eff_rank / max_rank,
                    cvm_mean=float(cvm.mean()), cvm_max=float(cvm.max()),
                    roundtrip_z=rt_z, roundtrip_x=rt_x,
                    z_std=float(z.std()), z_mean_abs=float(z.mean(0).abs().mean()))


def _all_gather(t):
    import torch.distributed as dist
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    out[dist.get_rank()] = t                      # keep local grad path
    return torch.cat(out, 0)
