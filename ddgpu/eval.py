"""Evaluation. Built to make info.txt's two methodological warnings unavoidable:

  1. "FID at matched step count will mislead you -- compare at matched
     WALL-CLOCK." So the harness keys results on GPU-seconds consumed, and
     refuses to emit a comparison table unless the runs it is comparing are
     within a tolerance of each other in GPU-seconds.
  2. "always report precision/recall separately -- mode-seeking objectives buy
     FID with diversity, and that trade will hide inside a single number."
     So `evaluate` returns precision and recall alongside FID, and the table
     prints all three or none.
"""
import json, math, os
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Feature statistics
# ---------------------------------------------------------------------------
@torch.no_grad()
def inception_features(images, model, batch=64, device="cuda"):
    """images: uint8 (N,3,H,W) in [0,255]. Returns (N, 2048) pool3 features."""
    out = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(device).float() / 127.5 - 1.0
        out.append(model(x).squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out)


def fid_from_feats(f1, f2):
    from scipy import linalg
    m1, m2 = f1.mean(0).numpy(), f2.mean(0).numpy()
    c1 = np.cov(f1.numpy(), rowvar=False)
    c2 = np.cov(f2.numpy(), rowvar=False)
    cc, _ = linalg.sqrtm(c1.dot(c2), disp=False)
    if np.iscomplexobj(cc):
        cc = cc.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * cc))


def precision_recall(real, fake, k=3, batch=1024):
    """Kynkaanniemi et al. improved precision/recall.

    precision = fraction of FAKE samples inside the real manifold  -> fidelity
    recall    = fraction of REAL samples inside the fake manifold  -> coverage
    Reporting only FID hides a fidelity-for-coverage trade; these separate it.
    """
    def radii(x):
        r = torch.empty(len(x))
        for i in range(0, len(x), batch):
            d = torch.cdist(x[i:i + batch], x)
            r[i:i + batch] = d.kthvalue(k + 1, dim=1).values
        return r

    def frac_inside(q, ref, ref_r):
        n = 0
        for i in range(0, len(q), batch):
            d = torch.cdist(q[i:i + batch], ref)
            n += (d <= ref_r[None, :]).any(1).sum().item()
        return n / len(q)

    rr, fr = radii(real), radii(fake)
    return frac_inside(fake, real, rr), frac_inside(real, fake, fr)


# ---------------------------------------------------------------------------
# Matched-wall-clock comparison
# ---------------------------------------------------------------------------
class RunRecord:
    """One row of the comparison table."""

    def __init__(self, name, gpu_seconds, n_gpus, step, fid, precision, recall,
                 nfe, extra=None):
        self.__dict__.update(locals()); del self.self

    def as_dict(self):
        d = dict(self.__dict__)
        d["gpu_hours"] = self.gpu_seconds / 3600
        return d


def comparison_table(records, tol=0.15, require_pr=True):
    """Refuses to compare runs that did not consume comparable GPU time.

    tol: max relative spread in gpu_seconds across the compared runs.
    """
    gs = [r.gpu_seconds for r in records]
    spread = (max(gs) - min(gs)) / max(min(gs), 1e-9)
    if spread > tol:
        raise ValueError(
            f"Runs differ by {spread:.0%} in GPU-seconds (tolerance {tol:.0%}). "
            f"A matched-STEP comparison here would be misleading: methods with "
            f"different per-step cost are not comparable at equal steps. "
            f"Truncate the cheaper runs to matched GPU-seconds and re-evaluate. "
            f"gpu_hours = " + ", ".join(f"{r.name}:{r.gpu_seconds/3600:.1f}" for r in records))
    if require_pr and any(r.precision is None or r.recall is None for r in records):
        raise ValueError("precision/recall missing; FID alone hides the "
                         "fidelity-vs-coverage trade. Compute both or pass "
                         "require_pr=False deliberately.")
    w = max(len(r.name) for r in records) + 2
    lines = [f"{'run':{w}}{'GPU-h':>8}{'NFE':>5}{'FID':>9}{'Prec':>8}{'Rec':>8}"]
    for r in sorted(records, key=lambda r: r.fid):
        lines.append(f"{r.name:{w}}{r.gpu_seconds/3600:8.1f}{r.nfe:5d}"
                     f"{r.fid:9.2f}{r.precision:8.3f}{r.recall:8.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Track-B-specific evaluation: inversion quality
# ---------------------------------------------------------------------------
@torch.no_grad()
def inversion_metrics(G, E, real_x, real_y, sigma_max, n_edit_dirs=8):
    """The fallback-paper metrics: if Track B's distribution matching fails but
    the bijection works, this is the result that still stands.

      recon_mse    : ||G(E(x)) - x||^2, the reconstruction the editing
                     literature reports for DDIM inversion
      cycle_mse    : ||E(G(z)) - z||^2 on prior samples
      edit_linear  : does moving in z-space move smoothly in x-space? measured
                     as the correlation between step size in z and in x along
                     random directions -- a degenerate G fails this even when
                     recon_mse looks fine
    """
    z = E(real_x, real_y)
    recon = torch.nn.functional.mse_loss(G(z, real_y), real_x).item()
    zr = torch.randn_like(z) * sigma_max
    cyc = torch.nn.functional.mse_loss(E(G(zr, real_y), real_y), zr).item() / sigma_max ** 2
    cors = []
    for _ in range(n_edit_dirs):
        d = torch.randn_like(z)
        d = d / d.reshape(len(d), -1).norm(dim=1).reshape(-1, 1, 1, 1)
        steps = torch.linspace(0, 1, 6, device=z.device) * sigma_max
        dx = torch.stack([(G(z + s * d, real_y) - G(z, real_y))
                          .reshape(len(z), -1).norm(dim=1) for s in steps])
        cors.append(float(torch.corrcoef(torch.stack([steps, dx.mean(1)]))[0, 1]))
    return dict(recon_mse=recon, cycle_mse=cyc, edit_linearity=float(np.mean(cors)))
