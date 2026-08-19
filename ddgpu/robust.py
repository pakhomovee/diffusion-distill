"""TRACK A -- doubly-robust score fusion.

Replaces DMD2's hand-weighted real-data adversarial term with a variance-optimal
convex combination of two estimators of the real score s(x_sigma):

    A = teacher score (CFG'd)      deterministic given x -> zero variance, bias b_A
    B = minibatch empirical score  the exact score of the real batch convolved
                                   with N(0, sigma^2 I). No discriminator, no
                                   extra network, closed form.

    s_hat = lam(sigma) * A + (1 - lam(sigma)) * B

The weight is ESTIMATED ONLINE, not tuned:

    lam_hat = V_B / E||A - B||^2,   V_B from a half-batch split, ||.||^2 = ||b_A-b_B||^2 + V_B

See LOG.log ENTRY 001: on a target with a closed-form score, lam_hat tracks the
exact MSE-optimal lam* to within 0.05 for sigma > 1.5*sigma_data, and under-shoots
at low sigma (where it should be 1.0) because it drops the <b_B, u> term. Hence
the low-noise gate below, which is not a fudge: at small sigma the empirical
score provably collapses to nearest-neighbour (variance ~ sigma^-4) and the
teacher must win.
"""
import torch


def empirical_score(x, x0, sigma, chunk=256):
    """Score of the real batch convolved with N(0, sigma^2 I), evaluated at x.

    x:  (B, ...) student samples.  x0: (N, ...) real samples.  sigma: (B,).
    Cost is one (B,N) distance matrix -- no network, negligible FLOPs next to a
    DiT forward. Never materialises (B,N,d).
    """
    Bsz = x.shape[0]
    xf, x0f = x.reshape(Bsz, -1), x0.reshape(x0.shape[0], -1)
    sig = sigma.reshape(-1, 1)
    out = torch.empty_like(xf)
    for i in range(0, Bsz, chunk):
        xb, sb = xf[i:i + chunk], sig[i:i + chunk]
        d2 = torch.cdist(xb.float(), x0f.float()).pow(2)
        w = torch.softmax(-d2 / (2 * sb.float() ** 2), dim=1)
        out[i:i + chunk] = ((w @ x0f.float() - xb.float()) / sb.float() ** 2).to(xf.dtype)
    return out.reshape_as(x)


@torch.no_grad()
def dsm_lambda_terms(teacher, real_x, real_y, sigma, cfg=1.0, lam_grid=None):
    """The DSM calibration statistic, in one place.

    Called by BOTH `LambdaEstimator.calibrate` (online, during training) and
    `exp/10_lambda_real.py` (offline, to make the figure). One implementation,
    so the plotted lambda and the trained lambda cannot drift apart -- which
    would be an easy and completely invisible way to publish a curve that does
    not describe the run it is attached to.

    Returns per-sample `num`, `den` (so `lambda = sum(num)/sum(den)`), the
    sigmas, and -- when `lam_grid` is given -- the held-out denoising loss
    `E||lam*A + (1-lam)*B - g||^2` at each lambda on the grid. That loss is the
    quantity the method actually minimises, and it needs no ground truth, so a
    lambda estimated on one split and evaluated on another is a FALSIFIABLE
    claim rather than a plot.

    `real_x` is SPLIT in half: the first half supplies calibration points, the
    second supplies the empirical score. They must be disjoint or B has seen the
    very sample it is being scored against.
    """
    m = real_x.shape[0] // 2
    if m < 2:
        return None
    C, D = real_x[:m], real_x[m:]
    yC = real_y[:m]
    sig = sigma[:m] if sigma.numel() >= m else sigma[:1].expand(m)
    eps = torch.randn_like(C)
    v = sig.reshape(-1, 1, 1, 1)
    xt = C + v * eps
    g = -eps / v
    A = teacher.cfg_score(xt, sig, yC, scale=cfg)
    B = empirical_score(xt, D, sig)
    f = lambda t: t.reshape(t.shape[0], -1)
    fA, fB, fg = f(A), f(B), f(g)
    dAB = fA - fB
    out = dict(num=(dAB * (fg - fB)).sum(-1), den=(dAB * dAB).sum(-1), sigma=sig)
    if lam_grid is not None:
        out["loss"] = torch.stack([((l * fA + (1 - l) * fB - fg) ** 2).sum(-1)
                                   for l in lam_grid])          # (n_lam, m)
        out["lam_grid"] = lam_grid
    return out


class LambdaEstimator:
    """Online estimate of lambda*(sigma), bucketed in log-sigma with EMA.

    Keeps running means of the numerator (V_B) and denominator (E||A-B||^2) per
    noise bucket, so the curve is read off the run rather than swept.
    """

    def __init__(self, n_bins=32, sigma_min=0.002, sigma_max=80.0, ema=0.99,
                 gate_below=None, mode="dsm", min_count=256, device="cpu"):
        self.n, self.ema, self.mode, self.min_count = n_bins, ema, mode, min_count
        self.lo, self.hi = torch.log(torch.tensor(sigma_min)), torch.log(torch.tensor(sigma_max))
        self.num = torch.zeros(n_bins, device=device)
        self.den = torch.zeros(n_bins, device=device)
        self.cnt = torch.zeros(n_bins, device=device)
        self.dsm_num = torch.zeros(n_bins, device=device)
        self.dsm_den = torch.zeros(n_bins, device=device)
        self.dsm_cnt = torch.zeros(n_bins, device=device)
        self.gate_below = gate_below      # only used by mode="ratio"

    def bucket(self, sigma):
        u = (sigma.log() - self.lo.to(sigma)) / (self.hi - self.lo).to(sigma)
        return (u * self.n).long().clamp(0, self.n - 1)

    @torch.no_grad()
    def update(self, A, B_full, B1, B2, sigma):
        """A, B_*: (B, ...) scores. B1/B2 are the two half-batch estimates."""
        f = lambda t: t.reshape(t.shape[0], -1)
        v = (f(B1) - f(B2)).pow(2).sum(-1) / 4.0          # unbiased V_B estimate
        d = (f(A) - f(B_full)).pow(2).sum(-1)             # ||b_A-b_B||^2 + V_B
        b = self.bucket(sigma)
        for src, dst in ((v, self.num), (d, self.den)):
            dst.mul_(self.ema).index_add_(0, b, src.float() * (1 - self.ema))
        self.cnt.index_add_(0, b, torch.ones_like(v))

    @torch.no_grad()
    def calibrate(self, teacher, real_x, real_y, sigma, cfg=1.0):
        """DSM calibration -- the estimator that actually transfers.

        On held-out real data an unbiased estimate of the true score is free:
        x_t = x_0 + sigma*eps gives g = -eps/sigma with E[g | x_t] = s(x_t).
        Since A is deterministic given x_t and B depends only on the training
        batch, both are conditionally independent of the noise in g, so

            E|| lam*A + (1-lam)*B - g ||^2 = MSE(lam) + const(sigma)

        with const independent of lam. Minimising the held-out denoising loss
        therefore minimises the TRUE score MSE exactly, and the minimiser is

            lam = E<A-B, g-B> / E||A-B||^2

        No true score, no gate, no assumption about the empirical score's bias,
        and nothing calibrated per dimension. This replaces the sigma gate,
        which exp04 showed fails at d=512 (3x worse than baseline, 21x in its
        worst bucket) because the crossover it stands in for moves with
        dimension and with teacher quality. See LOG.log ENTRY 011.

        real_x is SPLIT: half supplies the calibration points, the other half
        supplies the empirical score. They must be disjoint or B has seen the
        very sample it is being scored against, and memorises it.
        """
        t = dsm_lambda_terms(teacher, real_x, real_y, sigma, cfg=cfg)
        if t is None:
            return
        num, den, sig = t["num"], t["den"], t["sigma"]
        b = self.bucket(sig)
        self.dsm_num.mul_(self.ema).index_add_(0, b, num.float() * (1 - self.ema))
        self.dsm_den.mul_(self.ema).index_add_(0, b, den.float() * (1 - self.ema))
        self.dsm_cnt.index_add_(0, b, torch.ones_like(num))

    @torch.no_grad()
    def lam(self, sigma):
        if self.mode == "dsm":
            b = self.bucket(sigma)
            num, den = self.dsm_num[b], self.dsm_den[b]
            # fall back to the teacher in buckets with too little calibration
            l = torch.where((den > 1e-12) & (self.dsm_cnt[b] >= self.min_count),
                            num / den.clamp_min(1e-12), torch.ones_like(den))
            return l.clamp(0.0, 1.0)
        b = self.bucket(sigma)
        num, den = self.num[b], self.den[b]
        l = torch.where(den > 1e-12, num / den.clamp_min(1e-12),
                        torch.ones_like(den))
        l = l.clamp(0.0, 1.0)
        if self.gate_below is not None:
            l = torch.where(sigma < self.gate_below, torch.ones_like(l), l)
        return l

    def curve(self):
        s = torch.exp(self.lo + (torch.arange(self.n) + 0.5) / self.n * (self.hi - self.lo))
        cnt = (self.dsm_cnt if self.mode == "dsm" else self.cnt).cpu()
        return s, self.lam(s.to(self.num.device)).cpu(), cnt


@torch.no_grad()
def all_gather_batch(t):
    """Concatenate the real batch across ranks.

    V_B scales like 1/N, so the empirical score's usefulness is set by the
    number of real samples it sees -- and per-device that is just the micro
    batch (32-128). Gathering costs one all-gather of a (B,d) tensor, ~2 MB at
    B=128/d=4096, which is nothing next to a DiT forward, and multiplies N by
    the world size. This is the cheapest quality lever in Track A.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return t
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, 0)


@torch.no_grad()
def robust_real_score(teacher, x, sigma, y, real_batch, est, cfg=1.0,
                      mode="robust", fixed_lam=None, gather=True, gate=None):
    """The drop-in replacement for DMD2's teacher-score term.

    mode: 'teacher'  -> plain DMD2 (lambda = 1), the baseline
          'data'     -> lambda = 0, empirical score only (ablation)
          'fixed'    -> hand-set lambda, the hyperparameter this paper removes
          'robust'   -> online lambda_hat(sigma)
    Returns (score, lambda_used).

    Below `gate` the empirical score is not merely down-weighted, it is not
    COMPUTED. Two reasons, and both matter:
      * correctness -- lambda* = 1 there (LOG ENTRY 001), so B contributes
        nothing but numerical noise;
      * precision -- cdist forms ||a||^2 + ||b||^2 - 2a.b, which at d=16384 with
        O(1) entries carries ~1e-3 absolute error. Divided by 2*sigma^2 with
        sigma=0.01 that is an O(10) perturbation of the softmax logits, i.e. the
        nearest-neighbour weights become essentially arbitrary. Skipping is
        cheaper AND more correct than computing a number we would discard.
    Typically ~40% of a batch falls below the gate at the default P_mean, so this
    is also a real saving.
    """
    A = teacher.cfg_score(x, sigma, y, scale=cfg)
    if mode == "teacher":
        return A, torch.ones_like(sigma)
    if gather:
        real_batch = all_gather_batch(real_batch)

    active = torch.ones_like(sigma, dtype=torch.bool) if gate is None else (sigma >= gate)
    if not active.any():
        return A, torch.ones_like(sigma)

    idx = active.nonzero(as_tuple=True)[0]
    xa, sa = x[idx], sigma[idx]
    h = real_batch.shape[0] // 2
    Bf = empirical_score(xa, real_batch, sa)
    B1 = empirical_score(xa, real_batch[:h], sa)
    B2 = empirical_score(xa, real_batch[h:], sa)
    est.update(A[idx], Bf, B1, B2, sa)

    if mode == "data":
        lam_a = torch.zeros_like(sa)
    elif mode == "fixed":
        lam_a = torch.full_like(sa, float(fixed_lam))
    else:
        lam_a = est.lam(sa)

    lam = torch.ones_like(sigma)
    lam[idx] = lam_a
    out = A.clone()
    v = lam_a.reshape(-1, 1, 1, 1)
    out[idx] = v * A[idx] + (1 - v) * Bf
    return out, lam


class LambdaProbe:
    """Diagnostic-only lambda statistic, evaluable at ARBITRARY points.

    FINDINGS.md 1.2 argued that DMD2's real-data term corrects the teacher *off
    the data manifold* rather than on it, and 3.1 turned that into a falsifiable
    prediction: lambda should drift toward 1 at student samples as the student
    converges onto the manifold. Confirming it is what separates "we tuned a
    weight automatically" from "we identified what the real-data term does".

    The shipped estimator (`LambdaEstimator.calibrate`, mode='dsm') cannot make
    that measurement: it needs `g = -eps/sigma` on *held-out real* data to be an
    unbiased draw of the true score, and there is no such quantity at a student
    sample. So this probe uses the ratio statistic instead,

        lam_ratio = V_B / E||A - B||^2,

    which needs no ground truth and is therefore computable anywhere. Two
    caveats, both load-bearing when reading the plot:

      * it is a LOWER BOUND on lambda*, because it drops the <b_B, u> term
        (LOG.log ENTRY 001), and the bound is loosest at small sigma;
      * it is therefore NOT comparable to the lambda used in training.

    What it *is* comparable to is itself, at the same step, at a different point
    set. Real-vs-student at matched sigma and matched batch, tracked over
    training, is the drift measurement -- and every term that makes the bound
    loose is common to both legs.

    Never feeds training. Accumulates plain means, not an EMA, and is reset
    between probes so each logged curve is a snapshot rather than a smear.
    """

    def __init__(self, n_bins=16, sigma_min=0.002, sigma_max=200.0, device="cpu"):
        self.n = n_bins
        self.lo = torch.log(torch.tensor(sigma_min))
        self.hi = torch.log(torch.tensor(sigma_max))
        self.dev = device
        self.reset()

    def reset(self):
        z = lambda: torch.zeros(self.n, device=self.dev)
        self.vb, self.d2, self.sn, self.cnt = z(), z(), z(), z()

    def bucket(self, sigma):
        u = (sigma.log() - self.lo.to(sigma)) / (self.hi - self.lo).to(sigma)
        return (u * self.n).long().clamp(0, self.n - 1)

    @torch.no_grad()
    def observe(self, teacher, x, y, sigma, real_batch, cfg=1.0):
        """Accumulate the statistic at points `x` (already at noise `sigma`)."""
        h = real_batch.shape[0] // 2
        if h < 2:
            return
        A = teacher.cfg_score(x, sigma, y, scale=cfg)
        Bf = empirical_score(x, real_batch, sigma)
        B1 = empirical_score(x, real_batch[:h], sigma)
        B2 = empirical_score(x, real_batch[h:], sigma)
        f = lambda t: t.reshape(t.shape[0], -1).float()
        v = (f(B1) - f(B2)).pow(2).sum(-1) / 4.0
        d = (f(A) - f(Bf)).pow(2).sum(-1)
        s = f(A).pow(2).sum(-1)
        b = self.bucket(sigma)
        for src, dst in ((v, self.vb), (d, self.d2), (s, self.sn)):
            dst.index_add_(0, b, src)
        self.cnt.index_add_(0, b, torch.ones_like(v))

    @torch.no_grad()
    def curve(self, min_count=8):
        """(sigma, lam_ratio, var_B/||A||^2, count), NaN where under-sampled."""
        s = torch.exp(self.lo + (torch.arange(self.n) + 0.5) / self.n * (self.hi - self.lo))
        ok = self.cnt >= min_count
        nan = torch.full_like(self.vb, float("nan"))
        lam = torch.where(ok, self.vb / self.d2.clamp_min(1e-20), nan).clamp(0.0, 1.0)
        vrel = torch.where(ok, self.vb / self.sn.clamp_min(1e-20), nan)
        return s, lam.cpu(), vrel.cpu(), self.cnt.cpu()
