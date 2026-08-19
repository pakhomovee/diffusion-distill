"""EXP 01 -- measure the doubly-robust weight lambda*(sigma).

Setting mirrors real distillation: a teacher trained by DSM on a FINITE dataset,
a minibatch empirical score drawn from that same dataset, and the POPULATION
score of the target as ground truth (we want the student to match the data
distribution, not the training set).

Outputs results/exp01_<tag>.json with per-sigma diagnostics.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from dd.gmm import GMM
from dd.nets import EDMPrecond
from dd.diffusion import train_teacher
from dd.estimators import lambda_diagnostics

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/results", exist_ok=True)
os.makedirs(f"{ROOT}/ckpt", exist_ok=True)


def get_teacher(d, gm, train_set, steps, width, depth, seed):
    tag = f"d{d}_s{steps}_w{width}x{depth}_seed{seed}"
    path = f"{ROOT}/ckpt/teacher_{tag}.pt"
    model = EDMPrecond(d, sigma_data=gm.data_std, width=width, depth=depth)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.eval()
    print(f"  training teacher {tag} ...", flush=True)
    net = EDMPrecond(d, sigma_data=gm.data_std, width=width, depth=depth)

    def sampler(n):
        i = torch.randint(0, train_set.shape[0], (n,))
        return train_set[i]

    t0 = time.time()
    ema, _ = train_teacher(net, sampler, steps=steps, bs=512, lr=2e-3,
                           sigma_data=gm.data_std, log_every=max(steps // 4, 1))
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    torch.save(ema.state_dict(), path)
    return ema.eval()


def run(d=32, K=8, n_train=50000, steps=6000, width=256, depth=4, seed=0,
        batch_n=256, n_eval=1024, n_batches=32, n_sigma=17, tag=None):
    gm = GMM(d=d, K=K, seed=seed)
    train_set = gm.sample(n_train, seed=seed + 1)
    teacher = get_teacher(d, gm, train_set, steps, width, depth, seed)

    def data_sampler(n):
        i = torch.randint(0, train_set.shape[0], (n,))
        return train_set[i]

    held = gm.sample(n_eval, seed=seed + 999)
    sigmas = torch.logspace(math.log10(0.01), math.log10(20.0), n_sigma)

    rows = []
    for sg in sigmas:
        s = float(sg)
        x = held + s * torch.randn_like(held)
        with torch.no_grad():
            r = lambda_diagnostics(
                x, s,
                teacher_score=lambda a, b: teacher.score(a, b),
                data_sampler=data_sampler,
                true_score=lambda a, b: gm.score(a, b),
                n_batches=n_batches, batch_n=batch_n)
        r["sigma"] = s
        rows.append(r)
        print(f"  sig={s:7.3f} lam*={r['lam_star']:.3f} lam^={r['lam_hat']:.3f} "
              f"bA={r['bias_A']:.3e} bB={r['bias_B']:.3e} VB={r['var_B']:.3e} "
              f"gain={r['mse_star']/min(r['mse_teacher'],r['mse_data']):.3f}", flush=True)

    tag = tag or f"d{d}_N{batch_n}_s{steps}"
    meta = dict(d=d, K=K, n_train=n_train, steps=steps, width=width, depth=depth,
                seed=seed, batch_n=batch_n, n_eval=n_eval, n_batches=n_batches,
                data_std=gm.data_std)
    out = f"{ROOT}/results/exp01_{tag}.json"
    json.dump(dict(meta=meta, rows=rows), open(out, "w"), indent=1)
    print("wrote", out, flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k, v in dict(d=32, K=8, n_train=50000, steps=6000, width=256, depth=4,
                     seed=0, batch_n=256, n_eval=1024, n_batches=32, n_sigma=17).items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--tag", type=str, default=None)
    a = p.parse_args()
    run(**vars(a))
