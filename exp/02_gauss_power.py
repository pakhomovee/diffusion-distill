"""EXP 02 -- detection power of Gaussianity discrepancies vs dimension.

Track B's whole bet is that "match E#p_data to N(0,I)" is an easier problem than
"match G#N(0,I) to p_data". That is only true if the Gaussianity discrepancy can
actually DETECT the ways an encoder fails, at the dimension of a real latent
(4096 at 256px, 16384 at 512px, 65536 for SDXL).

So: for each discrepancy, dimension, batch size and perturbation type, estimate
the null distribution and the alternative distribution and report POWER at a 5%
false-positive rate. A discrepancy with low power against a perturbation cannot
prevent that perturbation during training -- that is the collusion failure mode.

Perturbations, chosen to be the realistic encoder failures:
  mean    : E#p has a small mean offset                (easy, sanity check)
  scale   : E#p is isotropically over/under-dispersed  (variance mis-calibration)
  lowrank : k directions collapsed, rest inflated to keep total variance
            -> THE collusion signature: encoder uses a subspace, generator
               ignores the rest, cycle loss is still satisfied
  mixture : two symmetric modes, EXACT same mean and covariance as N(0,I)
            -> the classic KSD/moment blind spot; only a kernel term can see it
  corr    : correlated coordinates, unit marginals
"""
import os, sys, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ddgpu.gauss_disc import mmd2_to_gaussian, ksd2_to_gaussian, gaussian_moment_penalty

torch.set_num_threads(2)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def perturb(kind, B, d, s, gen):
    z = torch.randn(B, d, generator=gen)
    if kind == "null":
        return z
    if kind == "mean":
        return z + s
    if kind == "scale":
        return z * (1 + s)
    if kind == "lowrank":
        k = max(1, int(s * d))                      # collapse a fraction s of dims
        z[:, :k] *= 0.0
        z[:, k:] *= math.sqrt(d / max(d - k, 1))    # keep total variance at d
        return z
    if kind == "mixture":
        # two modes at +-m e_1, variance of coord 0 shrunk so the covariance is
        # still exactly I -> mean and covariance are indistinguishable from N(0,I)
        m = s
        sgn = (torch.randint(0, 2, (B, 1), generator=gen).float() * 2 - 1)
        z[:, :1] = z[:, :1] * math.sqrt(max(1 - m * m, 1e-6)) + sgn * m
        return z
    if kind == "corr":
        w = torch.randn(B, 1, generator=gen)
        return (z + s * w) / math.sqrt(1 + s * s)
    raise ValueError(kind)


STATS = dict(mmd=mmd2_to_gaussian, ksd=ksd2_to_gaussian, moment=gaussian_moment_penalty)


_NULL_CACHE = {}


def null_dist(stat, B, d, reps, seed=0):
    """Null is shared across perturbations -- recomputing it per perturbation was
    the dominant cost and buys nothing."""
    key = (stat, B, d, reps, seed)
    if key not in _NULL_CACHE:
        gen = torch.Generator().manual_seed(seed)
        f = STATS[stat]
        _NULL_CACHE[key] = torch.tensor(
            [f(perturb("null", B, d, 0, gen)).item() for _ in range(reps)])
    return _NULL_CACHE[key]


def power(stat, kind, B, d, s, reps, seed=0):
    f = STATS[stat]
    null = null_dist(stat, B, d, reps, seed)
    gen = torch.Generator().manual_seed(seed + 7919)
    alt = torch.tensor([f(perturb(kind, B, d, s, gen)).item() for _ in range(reps)])
    thr = null.quantile(0.95)                        # 5% false positive rate
    return (alt > thr).float().mean().item(), null.mean().item(), alt.mean().item()


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    dims = [256, 1024, 4096, 16384]
    B = 256
    grid = [("mean", 0.05), ("scale", 0.02), ("lowrank", 0.10),
            ("mixture", 0.5), ("corr", 0.10)]
    rows = []
    hdr = f"{'perturb':>10} {'strength':>8} {'d':>6} | " + " ".join(f"{s:>8}" for s in STATS)
    print(hdr); print("-" * len(hdr))
    t0 = time.time()
    for kind, s in grid:
        for d in dims:
            p = {}
            for st in STATS:
                pw, n, a = power(st, kind, B, d, s, reps)
                p[st] = pw
                rows.append(dict(perturb=kind, strength=s, d=d, stat=st, power=pw,
                                 null_mean=n, alt_mean=a, B=B, reps=reps))
            print(f"{kind:>10} {s:8.3f} {d:6d} | " + " ".join(f"{p[st]:8.2f}" for st in STATS),
                  flush=True)
        print()
    print(f"[{time.time()-t0:.0f}s]")
    json.dump(rows, open(f"{ROOT}/results/exp02_gauss_power.json", "w"), indent=1)
