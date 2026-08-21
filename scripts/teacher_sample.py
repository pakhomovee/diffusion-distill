#!/usr/bin/env python3
"""Sample the TEACHER with a proper multi-step sampler. Establishes the ceiling.

  python3 scripts/teacher_sample.py --teacher diffusers:google/ddpm-cifar10-32
  python3 scripts/teacher_sample.py --teacher edm:ckpt/edm-cifar10-32x32-uncond-vp.pkl \
      --edm-repo /root/edm --steps 18

Run this BEFORE spending GPU-hours on another distillation programme. It
separates two possibilities that a bad student grid cannot:

  * the teacher draws clean CIFAR here -> the data path, the preconditioning
    wrapper, the decode and the grid are all fine, and whatever is wrong is in
    the distillation. Its FID is the ceiling the student is being measured
    against.
  * the teacher draws mush here -> nothing downstream can be better, and no
    amount of sigma_max or step-count tuning on the student will help. Go and
    read `ddgpu.teachers.validate_teacher`'s numbers instead.

Why the teacher can succeed at the very sigma where a one-step student cannot:
a multi-step sampler never needs `D(x, sigma_max)` to be *accurate*. It only
needs the direction `(x - D)/sigma`, and the next 50 steps correct whatever it
got wrong. The one-step student has no second chance -- its single evaluation
at sigma_max IS the image, which is why the sigma amplification (RUNPLAN.md
section 6) bites it and not this. A clean grid here alongside a grainy student
grid is therefore the EXPECTED result, and confirms the diagnosis rather than
contradicting it.

The sampler is Karras et al. Algorithm 1: deterministic 2nd-order Heun, on the
teacher's own sigma grid (`VPSchedule.student_sigmas` for VP teachers, the
rho=7 EDM grid otherwise) so the noise levels are ones it was trained on.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.teachers import load_teacher                 # noqa: E402
from ddgpu.edm import edm_sigmas                        # noqa: E402
from ddgpu.generate import _save_grid, decode           # noqa: E402


def sigma_grid(G, steps, sigma_max, device):
    """The teacher's own noise levels, not a generic grid."""
    sch = getattr(G, "sch", None) or getattr(G, "path", None)
    if sch is not None and hasattr(sch, "student_sigmas"):
        return sch.student_sigmas(steps, sigma_max=sigma_max).to(device).float()
    return edm_sigmas(steps, sigma_max=sigma_max, device=device)


@torch.no_grad()
def heun(G, n, shape, sigmas, device, gen, n_classes=1):
    """Karras Algorithm 1, deterministic (S_churn = 0)."""
    x = torch.randn(n, *shape, device=device, generator=gen) * float(sigmas[0])
    y = torch.randint(0, max(n_classes, 1), (n,), device=device, generator=gen)
    full = lambda s: torch.full((n,), float(s), device=device)
    for i in range(len(sigmas) - 1):
        s, s1 = float(sigmas[i]), float(sigmas[i + 1])
        d = (x - G(x, full(s), y)) / s
        x2 = x + (s1 - s) * d
        if s1 > 0:                                   # Heun correction
            d2 = (x2 - G(x2, full(s1), y)) / s1
            x2 = x + (s1 - s) * 0.5 * (d + d2)
        x = x2
        if (i + 1) % 10 == 0:
            print(f"[teacher] step {i+1}/{len(sigmas)-1}  sigma {s1:.3f}", flush=True)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teacher", default="diffusers:google/ddpm-cifar10-32")
    p.add_argument("--edm-repo", default=None)
    p.add_argument("--repa-dir", default=None)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=50, help="sampler steps (NFE ~ 2x)")
    p.add_argument("--n", type=int, default=64, help="images to draw")
    p.add_argument("--sigma-max", type=float, default=None,
                   help="default: the teacher's own")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="teacher_samples.png")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = {k: v for k, v in (("edm_repo", a.edm_repo), ("repa_dir", a.repa_dir)) if v}
    G, meta = load_teacher(a.teacher, device=dev, sigma_data=a.sigma_data, **kw)
    G.eval()
    shape = meta.get("shape") or [3, 32, 32]
    smax = a.sigma_max or float(meta.get("sigma_max", 80.0))
    print(f"[teacher] {a.teacher}  {meta.get('family')}  shape {tuple(shape)}  "
          f"sigma_max {smax:.3f}  n_classes {meta.get('n_classes')}")

    sig = sigma_grid(G, a.steps, smax, dev)
    print(f"[teacher] {len(sig)-1} steps, sigma {float(sig[0]):.2f} -> "
          f"{float(sig[-2]):.4f} -> 0")

    gen = torch.Generator(device=dev).manual_seed(a.seed)
    # No autocast: this is the reference, and at sigma_max a VP teacher needs
    # the precision for exactly the reason RUNPLAN section 6 gives.
    x = heun(G, a.n, list(shape), sig, dev, gen, int(meta.get("n_classes", 1) or 1))

    if meta.get("space") == "pixel" or shape[0] == 3:
        imgs = decode(x, None, 1.0).cpu()
    else:
        from diffusers import AutoencoderKL
        from ddgpu.prepare import LATENT_SCALE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
        imgs = decode(x, vae, LATENT_SCALE).cpu()

    _save_grid(imgs, a.out)
    print(f"[teacher] grid -> {a.out}")
    print("  Clean images here mean the data path and the teacher wrapper are "
          "fine\n  and the problem is downstream, in the distillation. Compare "
          "with:\n    python3 scripts/sample_stats.py " + a.out)


if __name__ == "__main__":
    main()
