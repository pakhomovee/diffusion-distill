"""EXP 04 -- does the lambda*(sigma) result survive what exp01 held fixed?

exp01 measured lambda* at one dimension, one teacher, one batch size, and at
ON-distribution evaluation points. Those are the three obvious objections.

  (1) DIMENSION LADDER d in {8, 32, 128, 512}. The empirical score's variance
      grows with d and so does the teacher's bias. Which grows faster decides
      whether exp01's ordering is a low-dimensional artefact.
  (2) OFF-DISTRIBUTION eval points. In real distillation x is a STUDENT sample,
      not a noised real one. A student that has not converged puts mass where the
      real batch has none -- exactly where a softmax over real samples degenerates.
      Simulated as: shrunk (mode-collapsed), shifted (biased), blurred (smoothed).
  (3) BATCH SIZE N in {64, 256, 1024}. V_B ~ 1/N, so lambda* must move with N.

Also validates the GATE from LOG ENTRY 001 FINDING 3: lambda_hat under-shoots at
low sigma because it drops <b_B,u>. Gating lambda to 1 below a threshold should
recover the lost MSE. Measured here against the exact lambda*.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dd.gmm import GMM
from dd.nets import EDMPrecond
from dd.diffusion import train_teacher
from dd.estimators import empirical_score

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/ckpt", exist_ok=True)


def get_teacher(d, gm, train_set, steps, width, depth, seed):
    tag = f"d{d}_s{steps}_w{width}x{depth}_seed{seed}"
    path = f"{ROOT}/ckpt/teacher_{tag}.pt"
    m = EDMPrecond(d, sigma_data=gm.data_std, width=width, depth=depth)
    if os.path.exists(path):
        m.load_state_dict(torch.load(path))
        return m.eval()
    print(f"  training teacher {tag} ...", flush=True)
    net = EDMPrecond(d, sigma_data=gm.data_std, width=width, depth=depth)
    sampler = lambda n: train_set[torch.randint(0, train_set.shape[0], (n,))]
    t0 = time.time()
    ema, _ = train_teacher(net, sampler, steps=steps, bs=512, lr=2e-3,
                           sigma_data=gm.data_std, log_every=steps, verbose=False)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    torch.save(ema.state_dict(), path)
    return ema.eval()


def eval_points(kind, gm, held, sigma, n):
    """x at which the score is estimated, mimicking different student states."""
    if kind == "ondist":
        x0 = held
    elif kind == "shrunk":                      # mode-collapsed student
        x0 = held * 0.7 + held.mean(0, keepdim=True) * 0.3
    elif kind == "shifted":                     # systematically biased student
        x0 = held + 0.3 * gm.data_std
    elif kind == "blurred":                     # over-smoothed student
        x0 = held + 0.3 * gm.data_std * torch.randn_like(held)
    else:
        raise ValueError(kind)
    return x0 + sigma * torch.randn_like(x0)


@torch.no_grad()
def diagnostics(x, sigma, teacher, gm, data_sampler, n_batches, batch_n, gate):
    A = teacher.score(x, sigma)
    s = gm.score(x, sigma)
    Bs, num_h, den_h = [], 0.0, 0.0
    h = batch_n // 2
    for _ in range(n_batches):
        x0 = data_sampler(batch_n)
        Bf = empirical_score(x, x0, sigma)
        Bs.append(Bf)
        num_h += ((empirical_score(x, x0[:h], sigma)
                   - empirical_score(x, x0[h:], sigma)) ** 2).sum(-1).mean().item() / 4
        den_h += ((A - Bf) ** 2).sum(-1).mean().item()
    num_h /= n_batches; den_h /= n_batches
    Bst = torch.stack(Bs); EB = Bst.mean(0)
    V_B = ((Bst - EB) ** 2).sum(-1).mean().item()
    b_A, b_B = A - s, EB - s
    u = b_A - b_B
    lam_star = (V_B - (b_B * u).sum(-1).mean().item()) / max((u ** 2).sum(-1).mean().item() + V_B, 1e-30)
    lam_hat = num_h / max(den_h, 1e-30)
    lam_gate = 1.0 if sigma < gate else lam_hat

    def mse(l):
        return ((l * b_A + (1 - l) * b_B) ** 2).sum(-1).mean().item() + (1 - l) ** 2 * V_B
    return dict(sigma=sigma, lam_star=lam_star, lam_hat=lam_hat, lam_gate=lam_gate,
                bias_A=(b_A ** 2).sum(-1).mean().item(),
                bias_B=(b_B ** 2).sum(-1).mean().item(), var_B=V_B,
                score_norm2=(s ** 2).sum(-1).mean().item(),
                mse_teacher=mse(1.0), mse_data=mse(0.0), mse_star=mse(lam_star),
                mse_hat=mse(lam_hat), mse_gate=mse(lam_gate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--kinds", nargs="+", default=["ondist", "shrunk", "shifted", "blurred"])
    ap.add_argument("--n_sigma", type=int, default=13)
    ap.add_argument("--n_eval", type=int, default=512)
    ap.add_argument("--n_batches", type=int, default=24)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n_train", type=int, default=50000)
    a = ap.parse_args()

    rows = []
    for d in a.dims:
        gm = GMM(d=d, K=8, seed=0)
        train_set = gm.sample(a.n_train, seed=1)
        teacher = get_teacher(d, gm, train_set, a.steps, 256, 4, 0)
        ds = lambda n: train_set[torch.randint(0, train_set.shape[0], (n,))]
        held = gm.sample(a.n_eval, seed=999)
        sigmas = torch.logspace(math.log10(0.02), math.log10(20.0), a.n_sigma)
        gate = 0.5 * gm.data_std
        for kind in a.kinds:
            for N in a.batches:
                if kind != "ondist" and N != 256:
                    continue                      # keep the grid affordable
                print(f"d={d} kind={kind} N={N}", flush=True)
                for sg in sigmas:
                    s = float(sg)
                    x = eval_points(kind, gm, held, s, a.n_eval)
                    r = diagnostics(x, s, teacher, gm, ds, a.n_batches, N, gate)
                    r.update(d=d, kind=kind, N=N, data_std=gm.data_std)
                    rows.append(r)
                    print(f"   sig={s:7.3f} lam*={r['lam_star']:.3f} "
                          f"lam^={r['lam_hat']:.3f} gate={r['lam_gate']:.3f} "
                          f"gain*={r['mse_star']/min(r['mse_teacher'],r['mse_data']):.3f} "
                          f"gain_gate={r['mse_gate']/min(r['mse_teacher'],r['mse_data']):.3f}",
                          flush=True)
                json.dump(rows, open(f"{ROOT}/results/exp04_lambda_ladder.json", "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
