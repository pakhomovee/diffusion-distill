r"""EXP 08 -- lambda from a held-out denoising loss. No gate, no ground truth.

exp04/exp07 established: the doubly-robust idea is sound at every dimension
(oracle 0.85-0.91 mean), but both the online rule and the sigma-gate fail to
transfer, because the crossover they stand in for moves with dimension and with
teacher quality. Richardson extrapolation in N does not rescue it -- the KDE
bias scales like N^{-1/d}, i.e. it is essentially FLAT in high dimension, which
is the curse of dimensionality landing exactly on the correction term.

The right construction avoids estimating b_B at all.

On HELD-OUT real data we can form an unbiased estimate of the true score for
free: x_t = x_0 + sigma*eps with x_0 held out gives g_hat = -eps/sigma, and
E[g_hat | x_t] = s(x_t) exactly. Then for s_lam = lam*A + (1-lam)*B,

    E|| s_lam - g_hat ||^2  =  MSE(lam)  +  E||g_hat - s||^2
                                             \_ independent of lam _/

because A is deterministic given x_t and B depends only on the training batch,
so both are conditionally independent of the noise in g_hat. Minimising the
held-out denoising loss therefore minimises the true MSE EXACTLY, and the
minimiser is closed-form:

    lam_dsm = E<A - B, g_hat - B> / E||A - B||^2

No true score. No gate. No assumption about b_B. Nothing dimension-specific.

Caveat, stated up front: this is calibrated at points from the forward process on
held-out REAL data, whereas distillation queries the score at STUDENT samples,
where exp04 showed lambda* is different. So this is a calibration curve, not a
per-sample rule. Whether that is good enough is measured below.
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dd.gmm import GMM
from dd.nets import EDMPrecond
from dd.estimators import empirical_score

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def run(d, sigmas, N=256, n_batches=16, n_eval=384, seed=0):
    gm = GMM(d=d, K=8, seed=0)
    train = gm.sample(50000, seed=1)
    t = EDMPrecond(d, sigma_data=gm.data_std, width=256, depth=4)
    t.load_state_dict(torch.load(f"{ROOT}/ckpt/teacher_d{d}_s6000_w256x4_seed0.pt")); t.eval()
    ds = lambda n: train[torch.randint(0, 50000, (n,))]
    held = gm.sample(n_eval, seed=999)
    out = []
    print(f"\n=== d={d} (sigma_data={gm.data_std:.3f}, N={N}) ===", flush=True)
    print(f"{'sig/sd':>8} {'lam*':>6} {'lam_dsm':>8} {'lam^':>6} {'gate':>6} | "
          f"{'g*':>6} {'g_dsm':>7} {'g^':>8} {'g_gate':>8}", flush=True)
    for s in sigmas:
        eps = torch.randn(n_eval, d)
        x = held + s * eps
        ghat = -eps / s                       # unbiased estimate of s(x), given x
        A = t.score(x, s)
        strue = gm.score(x, s)
        num_d = den = numv = 0.0
        Bs = []
        h = N // 2
        for _ in range(n_batches):
            x0 = ds(N)
            B = empirical_score(x, x0, s); Bs.append(B)
            num_d += ((A - B) * (ghat - B)).sum(-1).mean().item()
            den += ((A - B) ** 2).sum(-1).mean().item()
            numv += ((empirical_score(x, x0[:h], s)
                      - empirical_score(x, x0[h:], s)) ** 2).sum(-1).mean().item() / 4
        lam_dsm = max(0.0, min(1.0, num_d / max(den / n_batches, 1e-30) / n_batches))
        lam_hat = (numv / n_batches) / max(den / n_batches, 1e-30)
        Bst = torch.stack(Bs); EB = Bst.mean(0)
        V_B = ((Bst - EB) ** 2).sum(-1).mean().item()
        bA, bB = A - strue, EB - strue
        u = bA - bB
        lam_star = (V_B - (bB * u).sum(-1).mean().item()) / max((u ** 2).sum(-1).mean().item() + V_B, 1e-30)
        mse = lambda l: l * l * (bA ** 2).sum(-1).mean().item() \
            + (1 - l) ** 2 * ((bB ** 2).sum(-1).mean().item() + V_B)
        gate = 1.0 if s / gm.data_std < 1.0 else lam_hat
        b = min(mse(1.0), mse(0.0))
        r = dict(d=d, sigma=s, data_std=gm.data_std, N=N, lam_star=lam_star,
                 lam_dsm=lam_dsm, lam_hat=lam_hat, lam_gate=gate,
                 g_star=mse(lam_star)/b, g_dsm=mse(lam_dsm)/b,
                 g_hat=mse(lam_hat)/b, g_gate=mse(gate)/b)
        out.append(r)
        print(f"{s/gm.data_std:8.3f} {lam_star:6.3f} {lam_dsm:8.3f} {lam_hat:6.3f} {gate:6.3f} | "
              f"{r['g_star']:6.3f} {r['g_dsm']:7.3f} {r['g_hat']:8.2f} {r['g_gate']:8.2f}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--n_eval", type=int, default=384)
    ap.add_argument("--n_sigma", type=int, default=11)
    a = ap.parse_args()
    sig = [float(x) for x in torch.logspace(math.log10(0.02), math.log10(20.0), a.n_sigma)]
    allr = []
    for d in a.dims:
        allr += run(d, sig, a.N, a.n_batches, a.n_eval)
        json.dump(allr, open(f"{ROOT}/results/exp08_dsm_lambda.json", "w"), indent=1)
    print("\n=== summary: mean / worst MSE ratio (lower is better; 1.0 = no better than best single) ===")
    print(f"{'d':>6} " + "".join(f"{k:>20}" for k in ["oracle lam*","lam_dsm","lam_hat","sigma gate"]))
    for d in a.dims:
        v = [r for r in allr if r["d"] == d]
        line = f"{d:6d} "
        for k in ["g_star", "g_dsm", "g_hat", "g_gate"]:
            g = [r[k] for r in v]
            line += f"{sum(g)/len(g):8.3f} /{max(g):9.2f}".rjust(20)
        print(line)
