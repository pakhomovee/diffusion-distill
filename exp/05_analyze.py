"""Turn results/*.json into the figures and tables the write-up needs."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, F = f"{ROOT}/results", f"{ROOT}/figures"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_lambda_curve():
    p = f"{R}/exp01_d32_N256_s6000.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p)); rows = d["rows"]; sd = d["meta"]["data_std"]
    s = np.array([r["sigma"] for r in rows]) / sd
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].semilogx(s, [r["lam_star"] for r in rows], "o-", label=r"$\lambda^*$ (exact)")
    ax[0].semilogx(s, [r["lam_hat"] for r in rows], "s--", label=r"$\hat\lambda$ (online)")
    ax[0].axhline(0.5, color="k", lw=0.5, ls=":")
    ax[0].set_xlabel(r"$\sigma/\sigma_{data}$"); ax[0].set_ylabel(r"$\lambda$")
    ax[0].set_title("teacher weight: high at LOW noise\n(folklore predicts the opposite)")
    ax[0].legend(); ax[0].set_ylim(-0.02, 1.05)

    ax[1].loglog(s, [r["bias_A"] / r["score_norm2"] for r in rows], "o-", label=r"teacher bias$^2$")
    ax[1].loglog(s, [r["bias_B"] / r["score_norm2"] for r in rows], "s-", label=r"batch bias$^2$")
    ax[1].loglog(s, [r["var_B"] / r["score_norm2"] for r in rows], "^-", label=r"batch variance")
    ax[1].set_xlabel(r"$\sigma/\sigma_{data}$"); ax[1].set_ylabel("relative to $\\|s\\|^2$")
    ax[1].set_title("why: the batch estimator memorises\nas $\\sigma\\to0$ (variance $\\sim\\sigma^{-4}$)")
    ax[1].legend()

    base = np.minimum([r["mse_teacher"] for r in rows], [r["mse_data"] for r in rows])
    # gate in units of sigma/sigma_data. Swept in LOG ENTRY 008: worst-case
    # ratio is 1.000 for any gate in [0.7,1.6]; 1.0 is the middle of that band.
    gate = 1.0
    mse_gate = []
    for r, ss in zip(rows, s):
        l = 1.0 if ss < gate else r["lam_hat"]
        mse_gate.append(l ** 2 * r["bias_A"]
                        + (1 - l) ** 2 * (r["bias_B"] + r["var_B"]))
    ax[2].loglog(s, np.array([r["mse_star"] for r in rows]) / base, "o-", label=r"$\lambda^*$ (exact)")
    ax[2].loglog(s, np.array([r["mse_hat"] for r in rows]) / base, "s--", label=r"$\hat\lambda$ (ungated)")
    ax[2].loglog(s, np.array(mse_gate) / base, "^:", label=r"$\hat\lambda$ + gate")
    ax[2].axhline(1.0, color="k", lw=0.8)
    ax[2].axvline(gate, color="gray", lw=0.8, ls="--")
    ax[2].set_xlabel(r"$\sigma/\sigma_{data}$"); ax[2].set_ylabel("MSE / best single estimator")
    ax[2].set_title("win at moderate noise; the ungated online\nrule is catastrophic below $\\sigma_{data}$")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{F}/fig1_lambda.png"); plt.close(fig)
    print("wrote fig1_lambda.png")


def fig_power():
    p = f"{R}/exp02_gauss_power.json"
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    perts = list(dict.fromkeys(r["perturb"] for r in rows))
    stats = list(dict.fromkeys(r["stat"] for r in rows))
    dims = sorted({r["d"] for r in rows})
    fig, axes = plt.subplots(1, len(perts), figsize=(2.6 * len(perts), 3.0), sharey=True)
    mark = dict(mmd="o-", ksd="s--", moment="^:")
    for ax, pt in zip(axes, perts):
        for st in stats:
            y = [next(r["power"] for r in rows if r["perturb"] == pt and r["stat"] == st and r["d"] == d)
                 for d in dims]
            ax.semilogx(dims, y, mark[st], base=2, label=st)
        ax.set_title(pt); ax.set_xlabel("latent dim $d$"); ax.set_ylim(-0.03, 1.05)
        ax.axhline(0.05, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel("detection power @ 5% FPR")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Can a Gaussianity term see the ways an encoder fails?  (B=256)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{F}/fig2_power_dim.png", bbox_inches="tight"); plt.close(fig)
    print("wrote fig2_power_dim.png")


def fig_power_batch():
    p = f"{R}/exp03_gauss_power_batch.json"
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    dims = sorted({r["d"] for r in rows})
    perts = list(dict.fromkeys(r["perturb"] for r in rows))
    stats = list(dict.fromkeys(r["stat"] for r in rows))
    Bs = sorted({r["B"] for r in rows})
    fig, axes = plt.subplots(len(dims), len(perts), figsize=(2.5 * len(perts), 2.6 * len(dims)),
                             sharey=True, squeeze=False)
    mark = dict(mmd="o-", ksd="s--", moment="^:")
    for i, d in enumerate(dims):
        for j, pt in enumerate(perts):
            ax = axes[i][j]
            for st in stats:
                y = [next((r["power"] for r in rows if r["perturb"] == pt and r["stat"] == st
                           and r["d"] == d and r["B"] == B), np.nan) for B in Bs]
                ax.semilogx(Bs, y, mark[st], base=2, label=st)
            ax.set_ylim(-0.03, 1.05); ax.axhline(0.05, color="k", lw=0.5, ls=":")
            if i == 0:
                ax.set_title(pt)
            if i == len(dims) - 1:
                ax.set_xlabel("batch $B$")
            if j == 0:
                ax.set_ylabel(f"d={d}\npower")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Does a bigger batch rescue the Gaussianity term?", y=1.0)
    fig.tight_layout(); fig.savefig(f"{F}/fig3_power_batch.png", bbox_inches="tight"); plt.close(fig)
    print("wrote fig3_power_batch.png")


def fig_ladder():
    p = f"{R}/exp04_lambda_ladder.json"
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    dims = sorted({r["d"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for d in dims:
        rr = [r for r in rows if r["d"] == d and r["kind"] == "ondist" and r["N"] == 256]
        if not rr:
            continue
        s = np.array([r["sigma"] for r in rr]) / rr[0]["data_std"]
        ax[0].semilogx(s, [r["lam_star"] for r in rr], "o-", label=f"d={d}")
        base = np.minimum([r["mse_teacher"] for r in rr], [r["mse_data"] for r in rr])
        ax[1].semilogx(s, np.array([r["mse_star"] for r in rr]) / base, "o-", label=f"d={d}")
    ax[0].set_title(r"$\lambda^*$ across the dimension ladder"); ax[0].legend()
    ax[0].set_xlabel(r"$\sigma/\sigma_{data}$"); ax[0].set_ylabel(r"$\lambda^*$")
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_ylim(0, 1.15)
    ax[1].set_title("MSE gain vs dimension"); ax[1].set_xlabel(r"$\sigma/\sigma_{data}$")
    ax[1].set_ylabel("MSE / best single"); ax[1].legend()
    d0 = dims[len(dims) // 2]
    for kind in ["ondist", "shrunk", "shifted", "blurred"]:
        rr = [r for r in rows if r["d"] == d0 and r["kind"] == kind and r["N"] == 256]
        if rr:
            s = np.array([r["sigma"] for r in rr]) / rr[0]["data_std"]
            ax[2].semilogx(s, [r["lam_star"] for r in rr], "o-", label=kind)
    ax[2].set_title(f"off-distribution eval points (d={d0})"); ax[2].legend()
    ax[2].set_xlabel(r"$\sigma/\sigma_{data}$"); ax[2].set_ylabel(r"$\lambda^*$")
    fig.tight_layout(); fig.savefig(f"{F}/fig4_ladder.png"); plt.close(fig)
    print("wrote fig4_ladder.png")


if __name__ == "__main__":
    fig_lambda_curve(); fig_power(); fig_power_batch(); fig_ladder()
