"""Turn a run directory into the paper's figures.

  python3 exp/09_plot_run.py runs/imagenet256_robust [runs/imagenet256_dmd2 ...]

Produces, per run, into <run>/figures/:

  lambda_curve.png  lambda_dsm(sigma) at several checkpoints. This is FINDINGS.md
                    4's "Figure 1": the first measurement of the variance-optimal
                    teacher-versus-data weight on a real image model. It is a
                    result whether or not the method improves FID, because it is
                    what makes 1.1's claim about the field checkable rather than
                    synthetic.

  lambda_drift.png  the ratio statistic at real vs student samples over training.
                    1.2 predicts the student curve starts BELOW the real one (the
                    real batch is worth more where the teacher is off-manifold)
                    and drifts up toward it as the student lands on the manifold.
                    If it does, an annealed real-data weight falls out of the
                    theory rather than a sweep. If it does not, the off-manifold
                    explanation is wrong -- also a result, and much cheaper to
                    learn here than after the XL runs.

  loss_curves.png   whatever the trainer logged, for triage.

Reads only what the run already wrote; costs nothing and needs no GPU.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _steps(paths, pat):
    out = []
    for p in paths:
        b = os.path.basename(p)
        s = b[len(pat):-len(".json")]
        out.append((int(s) if s.isdigit() else 10 ** 9, p))
    return [p for _, p in sorted(out)]


def plot_lambda_curve(run, out):
    fs = _steps(glob.glob(f"{run}/lambda_curve_*.json"), "lambda_curve_")
    if not fs:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(fs):
        d = json.load(open(f))
        sig, lam, cnt = (np.array(d[k]) for k in ("sigma", "lam", "count"))
        # Buckets that never accumulated `lam_min_count` samples fall back to
        # lambda=1 by construction; plotting them would draw a flat line that
        # looks like a measurement and is not one.
        m = cnt >= 1
        ax.plot(sig[m], lam[m], color=cmap(i / max(len(fs) - 1, 1)),
                label=f"it {d.get('it', '?')}", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"$\lambda(\sigma)$  (weight on the teacher)")
    ax.set_ylim(-0.03, 1.03)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_title("Estimated variance-optimal teacher weight")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{out}/lambda_curve.png", dpi=160); plt.close(fig)
    return True


def plot_drift(run, out):
    fs = _steps(glob.glob(f"{run}/probe_*.json"), "probe_")
    if not fs:
        return False
    its, series = [], {"real": [], "student": []}
    curves = {}
    for f in fs:
        d = json.load(open(f))
        its.append(d["it"])
        for k in series:
            lam = np.array(d[k]["lam_ratio"], float)
            series[k].append(np.nanmedian(lam))
            curves.setdefault(k, []).append((d["it"], np.array(d[k]["sigma"]), lam))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, c in (("real", "tab:blue"), ("student", "tab:red")):
        axes[0].plot(its, series[k], "o-", color=c, label=f"{k} samples", lw=1.6, ms=3)
    axes[0].set_xlabel("step"); axes[0].set_ylabel(r"median $\hat\lambda_{ratio}$")
    axes[0].set_title(r"Does $\lambda$ drift toward 1 at student samples?")
    axes[0].legend(fontsize=8)

    # Same statistic resolved in sigma, first and last probe only.
    for k, c in (("real", "tab:blue"), ("student", "tab:red")):
        for (it, sig, lam), st in zip([curves[k][0], curves[k][-1]], (":", "-")):
            axes[1].plot(sig, lam, st, color=c, lw=1.5, label=f"{k} @ it {it}")
    axes[1].set_xscale("log"); axes[1].set_xlabel(r"$\sigma$")
    axes[1].set_ylabel(r"$\hat\lambda_{ratio}$")
    axes[1].set_title("first (dotted) vs last (solid) probe")
    axes[1].legend(fontsize=7)
    fig.suptitle("Diagnostic only: the ratio statistic is a LOWER BOUND on "
                 r"$\lambda^*$; compare the two legs, not the levels", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/lambda_drift.png", dpi=160); plt.close(fig)
    return True


def plot_losses(run, out):
    p = f"{run}/hist.json"
    if not os.path.exists(p):
        return False
    hist = json.load(open(p))
    keys = [k for k in ("loss_g", "loss_d", "loss", "lam", "rec", "cyc",
                        "diag_eff_rank_frac") if any(k in h for h in hist)]
    if not keys:
        return False
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3), squeeze=False)
    for ax, k in zip(axes[0], keys):
        xy = [(h["it"], h[k]) for h in hist if k in h and "it" in h]
        if not xy:
            continue
        x, y = zip(*xy)
        ax.plot(x, y, lw=1.0)
        ax.set_title(k, fontsize=9); ax.set_xlabel("step")
        if k.startswith("loss") and min(y) > 0:
            ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(f"{out}/loss_curves.png", dpi=140); plt.close(fig)
    return True


def main(runs):
    for run in runs:
        out = f"{run}/figures"
        os.makedirs(out, exist_ok=True)
        made = [n for n, ok in (("lambda_curve", plot_lambda_curve(run, out)),
                                ("lambda_drift", plot_drift(run, out)),
                                ("loss_curves", plot_losses(run, out))) if ok]
        print(f"{run}: {', '.join(made) if made else 'nothing to plot'} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
