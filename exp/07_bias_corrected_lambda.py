"""EXP 07 -- a dimension-transferable lambda estimator.

exp04 exposed the failure that matters: the shipped sigma-gate is calibrated in
units of sigma_data, but the quantity it is standing in for is the crossover
between the batch estimator's VARIANCE and the teacher's BIAS. That crossover
moves with dimension -- sigma/sigma_data ~ 10 at d=8, ~0.15 at d=512 -- so a
fixed gate is 3x WORSE than baseline at d=512 and 21x worse in the worst bucket.
A method that needs re-tuning per latent dimension is not usable.

The fix is to stop gating and estimate the term the online rule drops.

    lambda* = (V_B - <b_B, u>) / (||u||^2 + V_B),   u = b_A - b_B
    lambda_hat drops <b_B, u> because b_B needs the true score.

But b_B is the bias of a KDE-type estimator with N samples, and it VANISHES as
N -> infinity. So its size is visible in how the estimate MOVES with N:

    Delta := B_{N/2} - B_N   has   E[Delta] = b_B(N/2) - b_B(N)
    under b_B(N) ~ c N^-alpha,     b_B(N) = Delta / (2^alpha - 1)

with alpha=1 giving simply b_B ~ Delta. And u is directly estimable as A - B_N.
Both halves are ALREADY computed for the half-batch variance estimate, so the
correction is free. Two independent real batches are used for the two factors so
their noise does not correlate.

    lambda_rich = (V_B - <Delta_P, A - B_Q>) / E||A - B||^2

Tested here against the exact lambda* across the dimension ladder.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dd.gmm import GMM
from dd.nets import EDMPrecond
from dd.estimators import empirical_score

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def compare(x, sigma, teacher, gm, ds, n_batches, N, alpha=1.0):
    A = teacher.score(x, sigma)
    s = gm.score(x, sigma)
    h = N // 2
    Bs, num_h, den, num_r = [], 0.0, 0.0, 0.0
    for _ in range(n_batches):
        # two INDEPENDENT real batches: P for the bias probe, Q for u
        xp, xq = ds(N), ds(N)
        Bp, Bq = empirical_score(x, xp, sigma), empirical_score(x, xq, sigma)
        Bs.append(Bq)
        P1, P2 = empirical_score(x, xp[:h], sigma), empirical_score(x, xp[h:], sigma)
        num_h += ((P1 - P2) ** 2).sum(-1).mean().item() / 4.0          # V_B
        den += ((A - Bq) ** 2).sum(-1).mean().item()                   # ||u||^2 + V_B
        delta = (0.5 * (P1 + P2) - Bp) / (2 ** alpha - 1)              # ~ b_B
        num_r += (delta * (A - Bq)).sum(-1).mean().item()              # <b_B, u>
    num_h /= n_batches; den /= n_batches; num_r /= n_batches

    Bst = torch.stack(Bs); EB = Bst.mean(0)
    V_B = ((Bst - EB) ** 2).sum(-1).mean().item()
    b_A, b_B = A - s, EB - s
    u = b_A - b_B
    lam_star = (V_B - (b_B * u).sum(-1).mean().item()) / max((u ** 2).sum(-1).mean().item() + V_B, 1e-30)
    lam_hat = num_h / max(den, 1e-30)
    lam_rich = max(0.0, min(1.0, (num_h - num_r) / max(den, 1e-30)))

    def mse(l):
        return l * l * (b_A ** 2).sum(-1).mean().item() \
             + (1 - l) ** 2 * ((b_B ** 2).sum(-1).mean().item() + V_B)
    gate = 1.0 if sigma / gm.data_std < 1.0 else lam_hat
    b = min(mse(1.0), mse(0.0))
    return dict(sigma=sigma, lam_star=lam_star, lam_hat=lam_hat, lam_rich=lam_rich,
                lam_gate=gate, g_star=mse(lam_star) / b, g_hat=mse(lam_hat) / b,
                g_rich=mse(lam_rich) / b, g_gate=mse(gate) / b,
                true_bB2=(b_B ** 2).sum(-1).mean().item(),
                est_bB_ip=num_r, var_B=V_B, data_std=gm.data_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 128, 512])
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--n_eval", type=int, default=384)
    ap.add_argument("--n_sigma", type=int, default=11)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    out = []
    for d in a.dims:
        gm = GMM(d=d, K=8, seed=0)
        train_set = gm.sample(50000, seed=1)
        path = f"{ROOT}/ckpt/teacher_d{d}_s6000_w256x4_seed0.pt"
        t = EDMPrecond(d, sigma_data=gm.data_std, width=256, depth=4)
        t.load_state_dict(torch.load(path)); t.eval()
        ds = lambda n: train_set[torch.randint(0, 50000, (n,))]
        held = gm.sample(a.n_eval, seed=999)
        print(f"\n=== d={d}  (sigma_data={gm.data_std:.3f}) ===", flush=True)
        print(f"{'sig/sd':>8} {'lam*':>6} {'lam^':>6} {'rich':>6} {'gate':>6} | "
              f"{'g*':>6} {'g^':>7} {'g_rich':>7} {'g_gate':>7}", flush=True)
        for sg in torch.logspace(math.log10(0.02), math.log10(20.0), a.n_sigma):
            s = float(sg)
            x = held + s * torch.randn_like(held)
            r = compare(x, s, t, gm, ds, a.n_batches, a.N, a.alpha)
            r["d"] = d; out.append(r)
            print(f"{s/gm.data_std:8.3f} {r['lam_star']:6.3f} {r['lam_hat']:6.3f} "
                  f"{r['lam_rich']:6.3f} {r['lam_gate']:6.3f} | {r['g_star']:6.3f} "
                  f"{r['g_hat']:7.2f} {r['g_rich']:7.3f} {r['g_gate']:7.2f}", flush=True)
        json.dump(out, open(f"{ROOT}/results/exp07_bias_corrected.json", "w"), indent=1)
    print("\n=== summary (mean / worst MSE ratio over all cells) ===")
    for k in ["g_star", "g_hat", "g_rich", "g_gate"]:
        for d in a.dims:
            v = [r[k] for r in out if r["d"] == d]
            print(f"  {k:8s} d={d:4d}  mean {sum(v)/len(v):8.3f}  worst {max(v):8.2f}")


if __name__ == "__main__":
    main()
