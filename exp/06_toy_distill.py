"""EXP 06 -- end-to-end distillation on a target with a CLOSED-FORM everything.

exp01 showed lambda*(sigma) has a shape opposite to the folklore, and that a
variance-optimal fusion cuts score MSE ~2x at moderate noise. That is a claim
about an ESTIMATOR. It does not follow that a distilled student gets better.
This closes the loop: run the actual DMD2 loop, with mode as the only variable,
and measure the student against the true distribution.

Why a GMM rather than images, given the conclusion has to transfer:
  * mode coverage is EXACT here. Each student sample has a closed-form posterior
    over the K components, so "did the student drop a mode" is a number, not an
    impression. FID cannot do this; precision/recall only approximates it.
  * the true score is known, so a failure can be attributed to the method rather
    than to a mis-trained teacher.
  * it runs on 2 CPU cores, so the GPU budget is spent on the questions this
    CANNOT answer (dimension, architecture, real data), not on this one.

Reported per config:
  energy_dist  : energy distance to true samples. A proper metric, no bandwidth.
  logp         : mean log p_true(x_student). Fidelity. Mode-seeking raises it.
  mode_kl / modes_covered : MODE COLLAPSE, measured exactly from closed-form
                 component posteriors. This is what a single FID hides.
  precision/recall : kNN-manifold, for continuity with the image literature.
"""
import argparse, copy, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dd.gmm import GMM
from dd.nets import EDMPrecond
from dd.diffusion import train_teacher, sample_sigma_train, dsm_loss
from dd.estimators import empirical_score

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGMA_MAX = 80.0


# ---------------------------------------------------------------- lambda ----
class Lam:
    """Same online estimator as ddgpu/robust.py, in the toy harness."""

    def __init__(self, n_bins=24, smin=0.002, smax=SIGMA_MAX, ema=0.99, gate=None):
        self.n, self.ema, self.gate = n_bins, ema, gate
        self.lo, self.hi = math.log(smin), math.log(smax)
        self.num = torch.zeros(n_bins); self.den = torch.zeros(n_bins)

    def _b(self, sig):
        u = (sig.log() - self.lo) / (self.hi - self.lo)
        return (u * self.n).long().clamp(0, self.n - 1)

    def update(self, A, Bf, B1, B2, sig):
        v = ((B1 - B2) ** 2).sum(-1) / 4
        d = ((A - Bf) ** 2).sum(-1)
        b = self._b(sig)
        self.num.mul_(self.ema).index_add_(0, b, v * (1 - self.ema))
        self.den.mul_(self.ema).index_add_(0, b, d * (1 - self.ema))

    def lam(self, sig):
        b = self._b(sig)
        l = torch.where(self.den[b] > 1e-12, self.num[b] / self.den[b].clamp_min(1e-12),
                        torch.ones(len(sig))).clamp(0, 1)
        if self.gate is not None:
            l = torch.where(sig < self.gate, torch.ones_like(l), l)
        return l


# ------------------------------------------------------------- distillation --
def distill(gm, teacher, train_set, mode, steps, bs, d, gate, lr=2e-4,
            fixed_lam=0.5, seed=0, log_every=500):
    torch.manual_seed(seed)
    G = copy.deepcopy(teacher); mu = copy.deepcopy(teacher)
    for m in (G, mu):
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.999))
    oD = torch.optim.Adam(mu.parameters(), lr=lr, betas=(0.0, 0.999))
    est = Lam(gate=gate)
    ds = lambda n: train_set[torch.randint(0, train_set.shape[0], (n,))]
    hist = []

    def gen(n):
        z = torch.randn(n, d) * SIGMA_MAX
        return G(z, torch.full((n,), SIGMA_MAX))

    for it in range(steps):
        # --- critic ---
        with torch.no_grad():
            xg = gen(bs)
        sig = sample_sigma_train(bs)
        xt = xg + sig[:, None] * torch.randn_like(xg)
        D = mu(xt, sig)
        w = (sig ** 2 + gm.data_std ** 2) / (sig * gm.data_std) ** 2
        ld = (w[:, None] * (D - xg) ** 2).mean()
        oD.zero_grad(set_to_none=True); ld.backward()
        torch.nn.utils.clip_grad_norm_(mu.parameters(), 1.0); oD.step()

        # --- generator ---
        xg = gen(bs)
        sig = sample_sigma_train(bs)
        xt = xg + sig[:, None] * torch.randn_like(xg)
        with torch.no_grad():
            A = teacher.score(xt, sig)
            if mode == "teacher":
                s_real, lam = A, torch.ones(bs)
            else:
                act = (sig >= gate) if gate else torch.ones(bs, dtype=torch.bool)
                s_real, lam = A.clone(), torch.ones(bs)
                if act.any():
                    i = act.nonzero(as_tuple=True)[0]
                    x0r = ds(bs); h = bs // 2
                    Bf = empirical_score(xt[i], x0r, sig[i])
                    est.update(A[i], Bf, empirical_score(xt[i], x0r[:h], sig[i]),
                               empirical_score(xt[i], x0r[h:], sig[i]), sig[i])
                    la = (torch.zeros(len(i)) if mode == "data" else
                          torch.full((len(i),), fixed_lam) if mode == "fixed" else
                          est.lam(sig[i]))
                    s_real[i] = la[:, None] * A[i] + (1 - la[:, None]) * Bf
                    lam[i] = la
            s_fake = mu.score(xt, sig)
            v = sig[:, None] ** 2
            D_real, D_fake = xt + v * s_real, xt + v * s_fake
            grad = (D_fake - D_real) / (xg.detach() - D_real).abs().mean(1, keepdim=True).clamp_min(1e-4)
            grad = torch.nan_to_num(grad).clamp(-50, 50)
        lg = 0.5 * ((xg - (xg - grad).detach()) ** 2).mean()
        oG.zero_grad(set_to_none=True); lg.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0); oG.step()

        if (it + 1) % log_every == 0:
            hist.append(dict(it=it + 1, loss_d=ld.item(), loss_g=lg.item(),
                             lam=lam.mean().item()))
            print(f"    it {it+1:5d} ld {ld.item():8.3f} lg {lg.item():9.5f} "
                  f"lam {lam.mean().item():.3f}", flush=True)
    G.eval()
    return G, est, hist


# ------------------------------------------------------------------ metrics --
def energy_distance(x, y, m=2000):
    x, y = x[:m], y[:m]
    dxy = torch.cdist(x, y).mean()
    dxx = torch.cdist(x, x).mean()
    dyy = torch.cdist(y, y).mean()
    return float(2 * dxy - dxx - dyy)


def mode_coverage(gm, x, floor_mult=1.0):
    """Exact mode coverage from closed-form component posteriors.

    Returns (kl, n_covered, min_share).

    NOTE on the floor: an earlier version clamped the histogram at 1e-12, which
    made the KL saturate at a constant (22.0977 for K=8) the moment ANY component
    got zero mass -- so every collapsed run reported the identical number and the
    metric carried no information. The floor is now 1/n, the smallest share a
    sample of size n can resolve, which keeps the KL finite AND comparable.
    `n_covered` (components holding at least half their expected share) is the
    robust companion: it degrades gracefully and is what to read first.
    """
    r, v, quad, logdet = gm._quad_and_logdet(x, 1e-3)
    lp = gm.logw[None, :] - 0.5 * (quad + logdet)
    hist = torch.softmax(lp, 1).mean(0)
    tw = torch.softmax(gm.logw, 0)
    floor = floor_mult / x.shape[0]
    h = hist.clamp_min(floor)
    h = h / h.sum()
    kl = float((tw * (tw.log() - h.log())).sum())
    n_cov = int((hist >= 0.5 * tw).sum())
    return kl, n_cov, float((hist / tw).min())


def knn_pr(real, fake, k=3):
    def rad(z):
        return torch.cdist(z, z).kthvalue(k + 1, dim=1).values
    rr, fr = rad(real), rad(fake)
    prec = float((torch.cdist(fake, real) <= rr[None, :]).any(1).float().mean())
    rec = float((torch.cdist(real, fake) <= fr[None, :]).any(1).float().mean())
    return prec, rec


def evaluate(gm, G, d, n=4000):
    with torch.no_grad():
        z = torch.randn(n, d) * SIGMA_MAX
        xg = G(z, torch.full((n,), SIGMA_MAX))
    xr = gm.sample(n, seed=4242)
    prec, rec = knn_pr(xr[:2000], xg[:2000])
    kl, ncov, mshare = mode_coverage(gm, xg)
    return dict(energy_dist=energy_distance(xg, xr),
                logp=float(gm.log_prob(xg, 1e-3).mean()),
                logp_real=float(gm.log_prob(xr, 1e-3).mean()),
                mode_kl=kl, modes_covered=ncov, min_mode_share=mshare,
                precision=prec, recall=rec,
                std_ratio=float(xg.std() / xr.std()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--teacher_steps", type=int, default=6000)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--modes", nargs="+", default=["teacher", "robust", "fixed", "data"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()

    gm = GMM(d=a.d, K=8, seed=0)
    train_set = gm.sample(a.n_train, seed=1)
    tag = f"d{a.d}_s{a.teacher_steps}_w256x4_seed0"
    path = f"{ROOT}/ckpt/teacher_{tag}.pt"
    teacher = EDMPrecond(a.d, sigma_data=gm.data_std, width=256, depth=4)
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path))
    else:
        print("training teacher ...", flush=True)
        net = EDMPrecond(a.d, sigma_data=gm.data_std, width=256, depth=4)
        teacher, _ = train_teacher(net, lambda n: train_set[torch.randint(0, a.n_train, (n,))],
                                   steps=a.teacher_steps, bs=512, lr=2e-3,
                                   sigma_data=gm.data_std, log_every=2000)
        torch.save(teacher.state_dict(), path)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    gate = 0.5 * gm.data_std
    out = []
    for mode in a.modes:
        for seed in a.seeds:
            print(f"\n=== mode={mode} seed={seed} d={a.d} ===", flush=True)
            t0 = time.time()
            G, est, hist = distill(gm, teacher, train_set, mode, a.steps, a.bs,
                                   a.d, gate, seed=seed)
            m = evaluate(gm, G, a.d)
            m.update(mode=mode, seed=seed, d=a.d, steps=a.steps,
                     secs=round(time.time() - t0, 1))
            if mode == "robust":
                sb = torch.exp(torch.tensor(est.lo) + (torch.arange(est.n) + .5)
                               / est.n * (est.hi - est.lo))
                m["lam_curve"] = dict(sigma=sb.tolist(), lam=est.lam(sb).tolist())
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in m.items() if k != "lam_curve"}, flush=True)
            out.append(m)
            json.dump(out, open(f"{ROOT}/results/exp06_toy_distill_d{a.d}.json", "w"), indent=1)
    # teacher reference
    print("\n=== reference: teacher's own few-step samples ===", flush=True)
    from dd.diffusion import heun_sample
    xr = gm.sample(4000, seed=4242)
    for ns in (1, 8, 32):
        xt = heun_sample(teacher, 4000, a.d, n_steps=ns)
        pr, rc = knn_pr(xr[:2000], xt[:2000])
        kl, ncov, ms = mode_coverage(gm, xt)
        ref = dict(mode=f"teacher_heun{ns}", d=a.d, energy_dist=energy_distance(xt, xr),
                   logp=float(gm.log_prob(xt, 1e-3).mean()),
                   logp_real=float(gm.log_prob(xr, 1e-3).mean()),
                   mode_kl=kl, modes_covered=ncov, min_mode_share=ms,
                   precision=pr, recall=rc, std_ratio=float(xt.std() / xr.std()))
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ref.items()},
              flush=True)
        out.append(ref)
    json.dump(out, open(f"{ROOT}/results/exp06_toy_distill_d{a.d}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
