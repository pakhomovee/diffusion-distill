#!/usr/bin/env python3
"""What KIND of wrong is a sample grid? Answer it with statistics, not squinting.

  python3 scripts/sample_stats.py runs/cifar10_dmd2/samples_final.png
  python3 scripts/sample_stats.py samples.png --sigma-max 157.407 --tile 32

"Still looks noisy" describes at least four different failures that need
opposite fixes, and they are hard to tell apart by eye at 32x32:

  additive noise      flat power spectrum, excess high-frequency energy
  blurry/undertrained steep spectrum, DEFICIT of high-frequency energy
  layout/dimension    neighbouring pixels stop being correlated at all
  mode collapse       between-image variance collapses toward zero

This measures all four against real data, then -- for a VP student, where
`D = x - sigma*eps_hat` -- divides the image-space error by sigma_max to report
how accurate `eps_hat` actually is. That last number is the one that decides
what to do next, because it separates two causes that look identical on screen:

  * rounding. bf16's ulp alone puts ~39 levels of noise into D at sigma_max
    (RUNPLAN.md section 6). If the measured noise is near that, the fp32 guard
    in `vp.VPPrecond` is off or bypassed -- a bug, fix it.
  * learning. If the measured noise is far below the bf16 figure, eps_hat is
    simply not accurate enough yet. No bug; either train longer or stop
    amplifying the error by 157x, which is what `--sigma-max` reports on.

The amplification table is the cheap lever. Error in the image is linear in the
sigma the one-step student starts from, so halving sigma_max halves the grain
for free, where reaching the same place by training means driving eps_hat's
error down by the same factor.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.prepare import image_source          # noqa: E402


def tiles(path, tile):
    """Split a grid PNG back into the images it was assembled from."""
    g = np.array(Image.open(path).convert("RGB")).astype(np.float32)
    if g.shape[0] % tile or g.shape[1] % tile:
        raise SystemExit(f"{path} is {g.shape[1]}x{g.shape[0]}, not a whole "
                         f"number of {tile}x{tile} tiles; pass --tile")
    return np.stack([g[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
                     for r in range(g.shape[0] // tile)
                     for c in range(g.shape[1] // tile)])


def real_images(source, tile, n):
    ds = image_source(source, tile)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int)
    # Back to [0,255] to compare against a PNG on its own terms.
    return np.stack([((ds[int(i)][0] + 1) * 127.5).clamp(0, 255)
                     .permute(1, 2, 0).numpy() for i in idx]).astype(np.float32)


def hf_power(x):
    """Mean power in the top frequency ring -- where noise lives and images don't.

    PER CHANNEL, not on a greyscale reduction. Averaging the channels first
    would divide channel-independent noise by three before measuring it, and
    the reported "levels" would then be sqrt(3) too small -- verified by
    injecting a known std and reading it back.
    """
    F = np.fft.fftshift(np.abs(np.fft.fft2(x, axes=(1, 2))) ** 2,
                        axes=(1, 2)).mean((0, 3))
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.mgrid[:F.shape[0], :F.shape[1]]
    return F[np.hypot(yy - cy, xx - cx) >= F.shape[0] * 0.44].mean() / (
        x.shape[1] * x.shape[2])


def run_sigma_max(grid_path):
    """The sigma_max the run beside this grid was TRAINED at, or None.

    Not a convenience. The whole eps_hat calculation divides by this number, so
    a default that silently disagrees with the run reports the network as more
    accurate than it is -- a run trained at sigma_max=40 and read with the VP
    default of 157.4 comes out 3.9x too flattering, which is exactly the size of
    the effect being tested. `train.py` writes config.resolved.json into the run
    directory for this reason: every run records what it actually ran with.
    """
    d = os.path.dirname(os.path.abspath(grid_path))
    for _ in range(2):                      # the grid may sit one level down
        p = os.path.join(d, "config.resolved.json")
        if os.path.exists(p):
            try:
                v = json.load(open(p)).get("sigma_max")
            except (ValueError, OSError):
                return None
            return float(v) if isinstance(v, (int, float)) else None
        d = os.path.dirname(d)
    return None


def mean_delta_z(fake, real):
    """Per-channel mean difference, and how many standard errors it is.

    The uncertainty lives on BOTH sides, and at a typical grid size it is almost
    entirely the FAKE side: se over 5000 real images is ~0.5 levels, se over 64
    generated ones is ~4. Dividing by the real side alone -- which this did at
    first -- overstates the significance by an order of magnitude, turning a
    3-sigma colour cast into a reported 30-sigma one.

    Returns (delta, combined_se, se_fake, se_real), all per channel.
    """
    d = fake.mean((0, 1, 2)) - real.mean((0, 1, 2))
    se_f = fake.mean((1, 2)).std(0) / np.sqrt(len(fake))
    se_r = real.mean((1, 2)).std(0) / np.sqrt(len(real))
    return d, np.sqrt(se_f ** 2 + se_r ** 2), se_f, se_r


def hf_null(real, n, seed=0):
    """hf_power over random real subsets of the grid's own size.

    A 64-image grid estimates high-frequency power to about +-8.5%, so a bare
    "excess" of a few percent means nothing. This is the null it has to beat.
    """
    rng = np.random.default_rng(seed)
    reps = int(min(200, max(20, 20000 // max(n, 1))))
    return np.array([hf_power(real[rng.choice(len(real), n, replace=False)])
                     for _ in range(reps)])


def spectrum_slope(x):
    """log-log slope of the radial power spectrum. Natural ~ -2, white noise ~ 0."""
    grey = x.mean(-1)
    F = np.fft.fftshift(np.abs(np.fft.fft2(grey)) ** 2, axes=(1, 2)).mean(0)
    cy, cx = np.array(F.shape) // 2
    yy, xx = np.mgrid[:F.shape[0], :F.shape[1]]
    rad = np.hypot(yy - cy, xx - cx).astype(int)
    hi = max(4, int(F.shape[0] * 0.47))
    prof = np.array([F[rad == k].mean() for k in range(1, hi)])
    return float(np.polyfit(np.log(np.arange(1, hi)), np.log(prof), 1)[0])


def neighbour_corr(x):
    return float(np.corrcoef(x[:, :, :-1].ravel(), x[:, :, 1:].ravel())[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("grid", help="sample grid PNG (e.g. runs/<run>/samples_final.png)")
    p.add_argument("--tile", type=int, default=32, help="one image's size in the grid")
    p.add_argument("--source", default="cifar10-hf", help="real images to compare to")
    p.add_argument("--n-real", type=int, default=5000)
    p.add_argument("--sigma-max", type=float, default=None,
                   help="the sigma a one-step student starts from; default: read "
                        "from the run's config.resolved.json beside the grid")
    p.add_argument("--sigma-data", type=float, default=0.5)
    a = p.parse_args()

    fake = tiles(a.grid, a.tile)
    real = real_images(a.source, a.tile, a.n_real)
    print(f"[stats] {len(fake)} samples from {a.grid} vs {len(real)} real "
          f"from {a.source}")

    if a.sigma_max is None:
        a.sigma_max = run_sigma_max(a.grid)
        if a.sigma_max is None:
            a.sigma_max = 157.40728081040757
            print(f"[stats] no config.resolved.json beside the grid; assuming "
                  f"sigma_max={a.sigma_max:.3f} (the VP default). If the run "
                  "used another,\n        pass --sigma-max: every eps_hat "
                  "number below scales with it.")
        else:
            print(f"[stats] sigma_max={a.sigma_max:.3f}, from the run's "
                  "config.resolved.json")

    print("\n-- is the layout sane? (a dimension bug destroys these) --")
    nf, nr = neighbour_corr(fake), neighbour_corr(real)
    sf, sr = spectrum_slope(fake), spectrum_slope(real)
    print(f"   neighbour correlation  fake {nf:6.3f}   real {nr:6.3f}   "
          f"(white noise ~ 0.00)")
    print(f"   power-spectrum slope   fake {sf:6.2f}   real {sr:6.2f}   "
          f"(white noise ~ 0.00)")
    layout_ok = nf > 0.5 and sf < -1.0
    # Written out rather than inlined into the f-string: a multi-line expression
    # inside {} is PEP 701 syntax and a SyntaxError before Python 3.12.
    verdict = ("spatially coherent images" if layout_ok else
               "NOT IMAGES: suspect a reshape/permute/channel bug, not training")
    print("   => " + verdict)

    print("\n-- colour --")
    mf, mr = fake.mean((0, 1, 2)), real.mean((0, 1, 2))
    d, se, se_f, se_r = mean_delta_z(fake, real)
    print(f"   channel mean  fake {np.round(mf,1)}  real {np.round(mr,1)}")
    print(f"   se: fake(n={len(fake)}) {np.round(se_f,2)}  "
          f"real(n={len(real)}) {np.round(se_r,2)}  combined {np.round(se,2)}")
    print(f"   delta {np.round(d,1)} levels = {np.round(d/se,1)} standard errors")

    print("\n-- diversity (std across images of each image's mean) --")
    df, dr = fake.mean((1, 2)).std(0), real.mean((1, 2)).std(0)
    print(f"   fake {np.round(df,1)}   real {np.round(dr,1)}   "
          f"ratio {np.round(df/dr,2)}   (mode collapse -> 0)")

    print("\n-- high-frequency energy --")
    pf, pr = hf_power(fake), hf_power(real)
    null = hf_null(real, len(fake))
    lo, hi = np.percentile(null, [2.5, 97.5])
    z = (pf - null.mean()) / null.std()
    print(f"   top-ring power  fake {pf:9.2f}  real {pr:9.2f}  ratio {pf/pr:5.2f}x")
    print(f"   null from {len(null)} real subsets of {len(fake)}: "
          f"{null.mean():.2f} +- {null.std():.2f}, 95% [{lo:.2f}, {hi:.2f}]")
    print(f"   fake is z = {z:+.1f} against that null")
    if pf <= hi:
        if pf < lo:
            print("   => a DEFICIT of high frequency: blurry, i.e. undertrained,")
            print("      NOT noisy. Training longer helps; lowering sigma_max "
                  "will not.")
        else:
            print("   => within the null: no measurable excess noise at this "
                  "grid size.")
        return
    n255 = float(np.sqrt(pf - pr))
    c_lo, c_hi = float(np.sqrt(max(pf - hi, 0))), float(np.sqrt(max(pf - lo, 0)))
    unit = n255 / 127.5
    print(f"   excess as additive white noise: {n255:.2f} levels "
          f"[{c_lo:.2f}, {c_hi:.2f}] = {unit:.4f} in [-1,1]")

    err = unit / a.sigma_max
    print(f"\n-- what that implies about eps_hat (D = x - sigma*eps_hat) --")
    print(f"   implied error in eps_hat at sigma_max={a.sigma_max:.3f}: {err:.6f}")
    print("   the same noise, if it were pure rounding, would need:")
    for name, ulp in (("bf16", 2 ** -8), ("fp16", 2 ** -11),
                      ("tf32", 2 ** -11), ("fp32", 2 ** -24)):
        print(f"     {name:5} ulp {ulp:.2e} -> {ulp * a.sigma_max / 2 * 127.5:6.1f} "
              f"levels in D")
    bf16_levels = 2 ** -8 * a.sigma_max / 2 * 127.5
    if n255 > 0.5 * bf16_levels:
        print("   => consistent with bf16 rounding: the fp32 guard in vp.VPPrecond "
              "is\n      off or bypassed. That is a BUG -- fix it before reading "
              "anything else.")
    else:
        print("   => far below what rounding would cost, so this is eps_hat's own\n"
              "      error, not arithmetic. Not a bug: train longer, or amplify "
              "less.")

    print("\n-- amplification: the same eps_hat error at a lower start sigma --")
    seen = set()
    for s in (a.sigma_max, 80.0, 40.0, 20.0, 10.0):
        if s > a.sigma_max or round(s, 6) in seen:
            continue
        seen.add(round(s, 6))
        print(f"   sigma_max {s:8.3f} -> {s*err*127.5:5.1f} levels of grain, "
              f"SNR vs sigma_data {a.sigma_data/(s*err):6.1f}")
    print("   (error in the image is LINEAR in the sigma a one-step student starts\n"
          "    from, so halving sigma_max halves the grain without training at all;\n"
          "    `scripts/train.sh ... --set sigma_max=40` changes it. It is a\n"
          "    retrain, not a re-score: the generator is trained at this sigma.)")


if __name__ == "__main__":
    main()
