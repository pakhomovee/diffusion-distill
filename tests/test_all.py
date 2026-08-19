"""Invariant tests for the parts where a silent error would corrupt a result.

Not coverage for its own sake. Each test guards a specific way a wrong number
could look plausible and survive all the way into a plot.

  python tests/test_all.py
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

torch.set_num_threads(2)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------------
def t_gmm_score():
    """Closed-form noised score must equal autograd of the closed-form log-prob.
    If this drifts, every lambda* number in exp01/exp04 is wrong."""
    from dd.gmm import GMM
    gm = GMM(d=16, K=6, seed=0)
    x = gm.sample(64)
    for s in (0.05, 0.5, 5.0):
        xs = (x + s * torch.randn_like(x)).requires_grad_(True)
        g, = torch.autograd.grad(gm.log_prob(xs, s).sum(), xs)
        err = ((g - gm.score(xs.detach(), s)).norm() / g.norm()).item()
        check(f"GMM score == d/dx log_prob (sigma={s})", err < 1e-4, f"rel err {err:.1e}")


def t_empirical_score():
    """The minibatch empirical score must equal the exact score of the noised
    empirical measure. A softmax/temperature slip here would bias Track A
    everywhere without ever raising."""
    from dd.estimators import empirical_score
    torch.manual_seed(0)
    x0 = torch.randn(32, 8)
    x = (torch.randn(16, 8)).requires_grad_(True)
    for s in (0.3, 1.0, 3.0):
        lp = torch.logsumexp(-((x[:, None] - x0[None]) ** 2).sum(-1) / (2 * s ** 2), dim=1)
        g, = torch.autograd.grad(lp.sum(), x, retain_graph=True)
        e = empirical_score(x.detach(), x0, s)
        err = ((g - e).norm() / g.norm()).item()
        check(f"empirical_score == exact KDE score (sigma={s})", err < 1e-4, f"rel err {err:.1e}")


def t_lambda_recovers_known():
    """In a controlled setup where b_B = 0 by construction, the online estimator
    lam_hat = V_B/E||A-B||^2 must recover the exact lam*."""
    from dd.estimators import lambda_diagnostics
    from dd.gmm import GMM
    gm = GMM(d=8, K=4, seed=0)
    pool = gm.sample(20000, seed=3)
    x = gm.sample(256, seed=9) + 1.5 * torch.randn(256, 8)
    # teacher = exact score plus a fixed bias -> b_A known, b_B ~ 0 for large N
    # batch_n is chosen so V_B is comparable to the teacher's bias -- otherwise
    # lam* pins at 0 or 1 and the test passes without discriminating anything.
    # scale chosen so ||b_A||^2 ~ V_B at this batch_n, which is what puts lam*
    # in the interior; a mismatched scale pins it at 0 or 1 and the test stops
    # discriminating (it did, at 0.25 -> lam*=0.03).
    bias = 0.045 * torch.randn(8)
    r = lambda_diagnostics(
        x, 1.5, teacher_score=lambda a, b: gm.score(a, b) + bias,
        data_sampler=lambda n: pool[torch.randint(0, len(pool), (n,))],
        true_score=lambda a, b: gm.score(a, b), n_batches=64, batch_n=48)
    check("lam* is interior (test is discriminating)",
          0.05 < r["lam_star"] < 0.95, f"lam*={r['lam_star']:.3f}")
    check("lam_hat tracks lam* when b_B is small",
          abs(r["lam_hat"] - r["lam_star"]) < 0.15,
          f"lam*={r['lam_star']:.3f} lam^={r['lam_hat']:.3f}")
    check("fused MSE <= both single estimators",
          r["mse_star"] <= min(r["mse_teacher"], r["mse_data"]) * 1.001,
          f"{r['mse_star']:.3e} vs {min(r['mse_teacher'], r['mse_data']):.3e}")


def t_edm_score():
    """score(x,sigma) must be exactly (D(x,sigma) - x)/sigma^2."""
    from ddgpu.dit import make_dit
    from ddgpu.edm import EDMWrapper
    m = EDMWrapper(make_dit("DiT-T/2", input_size=8, in_ch=4, n_classes=10), 0.5)
    x = torch.randn(4, 4, 8, 8); s = torch.rand(4) + 0.1
    y = torch.randint(0, 10, (4,))
    with torch.no_grad():
        D = m(x, s, y); sc = m.score(x, s, y)
    err = (sc - (D - x) / s.reshape(-1, 1, 1, 1) ** 2).abs().max().item()
    check("EDM score == (D-x)/sigma^2", err < 1e-5, f"max abs {err:.1e}")


def t_ode_roundtrip():
    """Phi^{-1}(Phi(z)) ~= z. Track B's anchor pairs are supervision; if the
    integrator is not self-inverse the anchors teach E the wrong map."""
    from ddgpu.dit import make_dit
    from ddgpu.edm import EDMWrapper
    from ddgpu.invertible import teacher_ode
    torch.manual_seed(0)
    T = EDMWrapper(make_dit("DiT-T/2", input_size=8, in_ch=4, n_classes=10), 0.5).eval()
    y = torch.randint(0, 10, (4,))
    from ddgpu.edm import edm_sigmas
    z = torch.randn(4, 4, 8, 8) * float(edm_sigmas(32)[0])
    x = teacher_ode(T, z, y, n_steps=32)
    z2 = teacher_ode(T, x, y, n_steps=32, reverse=True)
    rel = ((z2 - z).norm() / z.norm()).item()
    check("teacher ODE round-trip z -> x -> z", rel < 0.05, f"rel err {rel:.3f}")


def t_gauss_disc():
    """MMD/KSD to N(0,I) must be ~0 under the null and clearly positive under a
    mean shift, at ALL dimensions -- the bandwidth h^2=d is what makes this hold."""
    from ddgpu.gauss_disc import mmd2_to_gaussian, ksd2_to_gaussian
    torch.manual_seed(0)
    for d in (64, 1024, 16384):
        z = torch.randn(256, d)
        zs = z + 0.15
        m0, m1 = mmd2_to_gaussian(z).item(), mmd2_to_gaussian(zs).item()
        k0, k1 = ksd2_to_gaussian(z).item(), ksd2_to_gaussian(zs).item()
        check(f"MMD null~0 << shift (d={d})", abs(m0) < 0.1 * m1, f"{m0:.2e} vs {m1:.2e}")
        check(f"KSD null~0 << shift (d={d})", abs(k0) < 0.1 * k1, f"{k0:.2e} vs {k1:.2e}")


def t_dit_params():
    """Guard against an architecture edit silently changing model size, which
    would invalidate every memory number in RUNPLAN.md."""
    from ddgpu.memcalc import dit_params
    want = {"DiT-S/2": 32.86, "DiT-B/2": 130.30, "DiT-L/2": 457.82, "DiT-XL/2": 674.82}
    for n, w in want.items():
        got = dit_params(n, 32) / 1e6
        check(f"{n} params == {w}M", abs(got - w) < 0.02, f"got {got:.2f}M")


def t_lora_isolation():
    """Perturbing one adapter must not move the others or the base."""
    from ddgpu.dit import make_dit
    from ddgpu.lora import AdapterSet
    torch.manual_seed(0)
    m = make_dit("DiT-T/2", input_size=8, in_ch=4, n_classes=10)
    a = AdapterSet(m, ["student", "critic"], rank=8)
    x = torch.randn(2, 4, 8, 8); t = torch.rand(2) + .1; y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        with a.use(None): base1 = m(x, t, y).clone()
        with a.use("critic"): crit1 = m(x, t, y).clone()
        with a.use("student"):
            for v in a.adapters.values():
                v.B.data.normal_(0, 0.1)
            stu = m(x, t, y).clone()
        with a.use("critic"): crit2 = m(x, t, y).clone()
        with a.use(None): base2 = m(x, t, y).clone()
    check("LoRA: base unaffected by adapter edit", torch.allclose(base1, base2))
    check("LoRA: sibling adapter unaffected", torch.allclose(crit1, crit2))
    check("LoRA: edited adapter did change output", not torch.allclose(stu, base1))


def t_gate_skips():
    """Below the gate, robust_real_score must return the teacher score exactly
    and must not touch the estimator."""
    from ddgpu.dit import make_dit
    from ddgpu.edm import EDMWrapper
    from ddgpu.robust import robust_real_score, LambdaEstimator
    T = EDMWrapper(make_dit("DiT-T/2", input_size=8, in_ch=4, n_classes=10), 0.5).eval()
    est = LambdaEstimator(gate_below=0.5)
    x = torch.randn(8, 4, 8, 8); y = torch.randint(0, 10, (8,))
    real = torch.randn(16, 4, 8, 8)
    s = torch.full((8,), 0.01)
    with torch.no_grad():
        out, lam = robust_real_score(T, x, s, y, real, est, mode="robust",
                                     gather=False, gate=0.5)
        ref = T.cfg_score(x, s, y, scale=1.0)
    check("below gate: score == teacher exactly", torch.allclose(out, ref))
    check("below gate: lambda == 1", bool((lam == 1).all()))
    check("below gate: estimator untouched", float(est.den.abs().sum()) == 0.0)


def t_dsm_lambda():
    """The DSM-calibrated lambda must recover the MSE-optimal lambda without any
    ground truth. This is the estimator that replaced the sigma gate, so it gets
    the strictest test in the file: it is checked against the EXACT lambda* on a
    target with a closed-form score, at two dimensions."""
    from dd.gmm import GMM
    from dd.estimators import empirical_score
    torch.manual_seed(0)
    for d in (16, 64):
        gm = GMM(d=d, K=4, seed=0)
        pool = gm.sample(20000, seed=3)
        held = gm.sample(512, seed=99)
        # a teacher with a KNOWN bias, so lam* is controlled rather than incidental
        bias = 0.08 * torch.randn(d)
        tscore = lambda x, s: gm.score(x, s) + bias
        sig = 1.2 * gm.data_std
        eps = torch.randn_like(held)
        x = held + sig * eps
        ghat = -eps / sig
        A = tscore(x, sig)
        num = den = 0.0
        Bs = []
        for _ in range(24):
            x0 = pool[torch.randint(0, len(pool), (256,))]
            B = empirical_score(x, x0, sig); Bs.append(B)
            num += ((A - B) * (ghat - B)).sum(-1).mean().item()
            den += ((A - B) ** 2).sum(-1).mean().item()
        lam_dsm = max(0.0, min(1.0, num / den))
        Bst = torch.stack(Bs); EB = Bst.mean(0)
        V = ((Bst - EB) ** 2).sum(-1).mean().item()
        s_true = gm.score(x, sig)
        bA, bB = A - s_true, EB - s_true
        u = bA - bB
        lam_star = (V - (bB * u).sum(-1).mean().item()) / ((u ** 2).sum(-1).mean().item() + V)
        check(f"lam_dsm recovers lam* with NO ground truth (d={d})",
              abs(lam_dsm - lam_star) < 0.10,
              f"lam*={lam_star:.3f} lam_dsm={lam_dsm:.3f}")


def t_dsm_estimator_wiring():
    """The trainer's LambdaEstimator must (a) fall back to the teacher in
    under-sampled buckets and (b) produce a curve once calibrated."""
    from ddgpu.dit import make_dit
    from ddgpu.edm import EDMWrapper
    from ddgpu.robust import LambdaEstimator
    torch.manual_seed(0)
    T = EDMWrapper(make_dit("DiT-T/2", input_size=8, in_ch=4, n_classes=10), 0.5).eval()
    est = LambdaEstimator(n_bins=16, mode="dsm", min_count=4, ema=0.9)
    sig = torch.full((8,), 1.0)
    check("uncalibrated bucket falls back to lambda=1",
          bool((est.lam(sig) == 1).all()))
    for _ in range(20):
        x = torch.randn(16, 4, 8, 8) * 0.5
        y = torch.randint(0, 10, (16,))
        est.calibrate(T, x, y, torch.full((16,), 1.0))
    _, l, cnt = est.curve()
    check("calibration fills buckets", int(cnt.sum()) > 0, f"total {int(cnt.sum())}")
    check("calibrated lambda stays in [0,1]", bool(((l >= 0) & (l <= 1)).all()))


if __name__ == "__main__":
    for fn in [t_gmm_score, t_empirical_score, t_lambda_recovers_known, t_dsm_lambda,
               t_dsm_estimator_wiring, t_edm_score,
               t_ode_roundtrip, t_gauss_disc, t_dit_params, t_lora_isolation,
               t_gate_skips]:
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            FAIL.append(fn.__name__)
    print(f"\n{'='*60}\n{'ALL PASS' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
