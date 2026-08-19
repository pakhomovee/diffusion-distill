"""PHASE A -- lambda(sigma) on real teachers, inference only. No training.

  python3 exp/10_lambda_real.py --teacher diffusers:google/ddpm-cifar10-32 \
      --data data/cifar10 --batches 40 --batch 512 --tag cifar10

This is the paper's Figure 1 and the cheapest high-information experiment in the
whole programme: it needs a pretrained teacher, real data, and forward passes.
No student, no critic, no optimiser. A whole dimension ladder costs a few
GPU-hours (FINDINGS.md 6).

Three things come out, and they are not equally strong. Read them in this order:

1. FALSIFIABLE -- the held-out gain.
   lambda is estimated on split 1 and then the denoising loss
   `E||lam*A + (1-lam)*B - g||^2` is evaluated on a FRESH split 2. Since
   `E||lam*A+(1-lam)*B-g||^2 = MSE(lam) + const(sigma)` with const independent of
   lambda, a reduction in that loss IS a reduction in true score MSE -- no ground
   truth needed. We report the loss at lambda_hat against lambda=1 (teacher
   alone, i.e. plain DMD2) and lambda=0 (data alone). If lambda_hat does not beat
   lambda=1 on held-out data, the method does not work at this dimension, and
   that is a real negative result available for a few GPU-hours instead of a
   week of distillation.

   Estimating on one split and testing on another is what makes this non-
   circular: lambda_hat is the closed-form argmin on split 1, so checking it
   against split 1 would prove nothing.

2. DESCRIPTIVE -- lambda_dsm(sigma).
   The shape FINDINGS.md 1.1 predicts (teacher-heavy at LOW noise, data-heavy at
   HIGH noise -- the opposite of the folklore) measured on a real image model for
   the first time. A result independent of whether the method improves FID.

3. COMPARATIVE ONLY -- the off-manifold panel.
   FINDINGS.md 1.2's mechanism claim is about STUDENT samples, and the DSM
   identity does not extend there: `g = -eps/sigma` is unbiased for the score of
   whatever distribution x0 was drawn from, so perturbing x0 measures lambda for
   the perturbed distribution, not the real one. So the off-manifold panel uses
   the ratio statistic (`LambdaProbe`), which needs no ground truth but is a
   LOWER BOUND on lambda*. Compare its two legs against each other; never read
   the levels, and never compare them to panel 2.

   Perturbations mirror exp04's rows: blur, shift, mode-collapse. They are a
   stand-in for a student, not a student. The real drift measurement needs a
   training run (`probe_every` in ddgpu/train.py).
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from ddgpu.teachers import load_teacher, validate_teacher
from ddgpu.data import build_dataset
from ddgpu.robust import dsm_lambda_terms, empirical_score, LambdaProbe


# ---------------------------------------------------------------------------
# Student proxies (exp04's rows, applied to real data)
# ---------------------------------------------------------------------------
def perturb(x, kind, strength=0.5):
    """Cheap stand-ins for the ways a student sits off the data manifold."""
    if kind == "real":
        return x
    if kind == "blur":                       # over-smoothed, the classic failure
        k = torch.tensor([1.0, 2.0, 1.0], device=x.device)
        k = (k[:, None] * k[None, :]) / 16.0
        k = k.expand(x.shape[1], 1, 3, 3)
        return torch.nn.functional.conv2d(x, k, padding=1, groups=x.shape[1])
    if kind == "shift":                      # systematic offset off the manifold
        return x + strength * x.std() * torch.randn(1, x.shape[1], 1, 1, device=x.device)
    if kind == "collapse":                   # mode collapse: every sample -> the mean
        mu = x.mean(0, keepdim=True)
        return mu + (1 - strength) * (x - mu)
    raise ValueError(kind)


PERTURBATIONS = ("real", "blur", "shift", "collapse")


# ---------------------------------------------------------------------------
def sigma_grid(lo, hi, n):
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n))


@torch.no_grad()
def measure(teacher, ds, meta, args, device):
    lam_grid = torch.linspace(0.0, 1.0, args.n_lam)
    sig_lo = max(args.sigma_min, meta["sigma_min"])
    sig_hi = min(args.sigma_max, meta["sigma_max"])
    sigmas = sigma_grid(sig_lo, sig_hi, args.n_sigma)
    n = len(ds)
    rng = np.random.default_rng(args.seed)

    def draw(k):
        idx = rng.choice(n, size=k, replace=False)
        xb = torch.stack([ds[int(i)][0] for i in idx]).to(device)
        yb = torch.tensor([ds[int(i)][1] for i in idx], device=device)
        return xb, yb

    rows = []
    for si, s in enumerate(sigmas):
        num = den = 0.0
        loss = torch.zeros(args.n_lam, dtype=torch.float64)
        cnt = 0
        for _ in range(args.batches):
            # split 1: estimate lambda.  split 2: evaluate it. Disjoint draws.
            x1, y1 = draw(2 * args.batch)
            sg = torch.full((x1.shape[0],), float(s), device=device)
            t1 = dsm_lambda_terms(teacher, x1, y1, sg, cfg=args.cfg)
            num += float(t1["num"].sum()); den += float(t1["den"].sum())

            x2, y2 = draw(2 * args.batch)
            t2 = dsm_lambda_terms(teacher, x2, y2, sg, cfg=args.cfg,
                                  lam_grid=lam_grid.to(device))
            loss += t2["loss"].sum(-1).double().cpu()
            cnt += t2["loss"].shape[1]

        lam_hat = float(np.clip(num / max(den, 1e-20), 0.0, 1.0))
        curve = (loss / max(cnt, 1)).numpy()
        j = int(np.argmin(curve))
        l_teacher = float(curve[-1])                      # lambda = 1
        l_data = float(curve[0])                          # lambda = 0
        l_hat = float(np.interp(lam_hat, lam_grid.numpy(), curve))
        rows.append(dict(
            sigma=float(s), sigma_rel=float(s) / meta.get("sigma_data", 1.0),
            lam_hat=lam_hat, lam_grid_argmin=float(lam_grid[j]),
            heldout_loss_at_lam=l_hat, heldout_loss_teacher=l_teacher,
            heldout_loss_data=l_data,
            # < 1 means the fusion beats the plain teacher on held-out data.
            # This is the number the method lives or dies by.
            gain_vs_teacher=l_hat / max(l_teacher, 1e-20),
            gain_best_possible=float(curve[j]) / max(l_teacher, 1e-20),
            n_calib=cnt, loss_curve=curve.tolist()))
        print(f"  sigma={float(s):9.4f}  lam_hat={lam_hat:.3f}  "
              f"grid_argmin={float(lam_grid[j]):.3f}  "
              f"gain_vs_teacher={rows[-1]['gain_vs_teacher']:.4f}", flush=True)
    return dict(lam_grid=lam_grid.tolist(), sigmas=sigmas.tolist(), rows=rows)


@torch.no_grad()
def offmanifold(teacher, ds, meta, args, device):
    """Ratio-statistic lambda at real vs perturbed points. COMPARATIVE ONLY."""
    n = len(ds)
    rng = np.random.default_rng(args.seed + 1)
    out = {}
    for kind in PERTURBATIONS:
        pr = LambdaProbe(n_bins=args.n_sigma,
                         sigma_min=max(args.sigma_min, meta["sigma_min"]),
                         sigma_max=min(args.sigma_max, meta["sigma_max"]),
                         device=device)
        for _ in range(args.batches):
            idx = rng.choice(n, size=2 * args.batch, replace=False)
            xb = torch.stack([ds[int(i)][0] for i in idx]).to(device)
            yb = torch.tensor([ds[int(i)][1] for i in idx], device=device)
            half = args.batch
            x0, real = perturb(xb[:half], kind), xb[half:]
            sg = torch.exp(torch.rand(half, device=device)
                           * math.log(min(args.sigma_max, meta["sigma_max"])
                                      / max(args.sigma_min, meta["sigma_min"]))
                           + math.log(max(args.sigma_min, meta["sigma_min"])))
            xt = x0 + sg.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
            pr.observe(teacher, xt, yb[:half], sg, real, cfg=args.cfg)
        s, lam, vrel, cnt = pr.curve(min_count=4)
        out[kind] = dict(sigma=s.tolist(), lam_ratio=lam.tolist(),
                         var_rel=vrel.tolist(), count=cnt.tolist())
        med = np.nanmedian(np.array(lam))
        print(f"  {kind:9s} median lam_ratio = {med:.3f}", flush=True)
    return out


# ---------------------------------------------------------------------------
def plot(res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["dsm"]["rows"]
    s = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].plot(s, [r["lam_hat"] for r in rows], "o-", lw=1.8, ms=3, color="tab:blue")
    ax[0].set_xscale("log"); ax[0].set_ylim(-0.03, 1.03)
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel(r"$\sigma$"); ax[0].set_ylabel(r"$\hat\lambda_{\rm dsm}(\sigma)$")
    ax[0].set_title("weight on the teacher")

    g = [r["gain_vs_teacher"] for r in rows]
    ax[1].plot(s, g, "o-", lw=1.8, ms=3, color="tab:green", label=r"at $\hat\lambda$")
    ax[1].plot(s, [r["gain_best_possible"] for r in rows], "--", lw=1.2,
               color="tab:gray", label="best on grid")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$\sigma$")
    ax[1].set_ylabel("held-out loss / teacher-only")
    ax[1].set_title("below 1.0 = fusion beats plain DMD2")
    ax[1].legend(fontsize=8)

    if "offmanifold" in res:
        for k, c in (("real", "tab:blue"), ("blur", "tab:orange"),
                     ("shift", "tab:purple"), ("collapse", "tab:red")):
            d = res["offmanifold"].get(k)
            if d:
                ax[2].plot(d["sigma"], d["lam_ratio"], lw=1.5, label=k, color=c)
        ax[2].set_xscale("log"); ax[2].set_xlabel(r"$\sigma$")
        ax[2].set_ylabel(r"$\hat\lambda_{\rm ratio}$")
        ax[2].set_title("off-manifold proxy (lower bound; compare legs only)")
        ax[2].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160)
    print(f"[fig] {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teacher", required=True, help="<family>:<path>, see ddgpu.teachers")
    p.add_argument("--data", required=True, help="prepared dataset dir")
    p.add_argument("--tag", default=None)
    p.add_argument("--batch", type=int, default=256,
                   help="per split; FINDINGS 1.4 wants the empirical score to see >=256")
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--n-sigma", type=int, default=12)
    p.add_argument("--n-lam", type=int, default=21)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-offmanifold", action="store_true")
    p.add_argument("--repa-dir", default=None)
    p.add_argument("--edm-repo", default=None)
    p.add_argument("--out", default="results/lambda_real")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {k: v for k, v in dict(repa_dir=a.repa_dir, edm_repo=a.edm_repo).items() if v}
    teacher, meta = load_teacher(a.teacher, device=dev, **kw)
    print(json.dumps(meta, indent=1, default=str))

    ds, resolved = build_dataset(dict(data=a.data, shape=meta["shape"],
                                      n_classes=meta["n_classes"]))
    meta["sigma_data"] = resolved.get("sigma_data", 1.0)

    xb = torch.stack([ds[i][0] for i in range(min(64, len(ds)))]).to(dev)
    yb = torch.tensor([ds[i][1] for i in range(min(64, len(ds)))]).to(dev)
    check = validate_teacher(teacher, xb, yb, device=dev)
    print("TEACHER-CHECK " + json.dumps(check))
    if check["verdict"] != "OK":
        print("!! teacher validation SUSPECT -- every number below is untrustworthy.\n"
              "   See ddgpu/teachers.py:validate_teacher for how to read it.")

    tag = a.tag or a.teacher.split(":")[-1].replace("/", "_")
    os.makedirs(a.out, exist_ok=True)
    print(f"\n== DSM lambda + held-out gain ({meta['dims']} dims) ==")
    res = dict(teacher=meta, check=check, args=vars(a),
               dsm=measure(teacher, ds, meta, a, dev))
    if not a.skip_offmanifold:
        print("\n== off-manifold proxy (ratio statistic, lower bound) ==")
        res["offmanifold"] = offmanifold(teacher, ds, meta, a, dev)

    jp = f"{a.out}/{tag}.json"
    json.dump(res, open(jp, "w"), indent=1, default=str)
    print(f"\n[json] {jp}")
    plot(res, f"{a.out}/{tag}.png",
         f"{meta['arch']} -- {meta['dims']} dims ({meta['space']}), "
         f"{meta['n_params'] / 1e6:.0f}M params")

    best = min(res["dsm"]["rows"], key=lambda r: r["gain_vs_teacher"])
    print(f"\nBEST held-out gain vs plain teacher: {best['gain_vs_teacher']:.4f} "
          f"at sigma={best['sigma']:.3f} (lambda={best['lam_hat']:.3f})")
    print("gain < 1.0 means the fusion reduces true score MSE. That is the claim.")


if __name__ == "__main__":
    main()
