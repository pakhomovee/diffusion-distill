"""How much noise does each compute precision put into the denoised output?

  python3 scripts/precision_probe.py --teacher diffusers:google/ddpm-cifar10-32
  python3 scripts/precision_probe.py --teacher edm:/ckpt/edm-imagenet-64.pkl --edm-repo /root/edm

Answers one question with a measurement instead of an argument: on THIS box,
with THIS teacher, which precision can compute `D` at sigma_max without burying
the image in rounding noise?

Why it is not obvious. A VP eps-model computes `D = x - sigma*eps_hat`, where
both terms are O(sigma) and the answer is only O(sigma_data). An error `d` in
`eps_hat` therefore arrives in `D` multiplied by sigma. At CIFAR's
sigma_max=157.4 with sigma_data=0.5, `eps_hat` must be accurate to
sigma_data/sigma_max = 0.32%; bf16's ulp is 0.78%, so bf16 alone puts noise of
std ~0.26 against a 0.5 signal. A one-step student generates at sigma_max on
every sample, so it cannot emit a clean image at all -- FID ~325, structure
buried in speckle, and both arms failing identically. See RUNPLAN.md 6.

None of that is about the GPU: bf16 has 8 mantissa bits on every card ever
made. What a newer card DOES change is whether the fp32 fallback is expensive,
and whether TF32 -- which rounds matmul INPUTS to 10 bits but keeps fp32
tensors, so the network's output is never quantised -- is accurate enough to
use instead. That last one genuinely depends on the network and is worth
measuring rather than reasoning about.

The reference is fp32 with TF32 explicitly off. `VPPrecond`'s own fp32 guard is
disabled while probing, otherwise it would (correctly) refuse to reproduce the
failure being measured.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.teachers import load_teacher          # noqa: E402


def _tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def _denoise(G, x, sigma, y, mode, dev):
    """One D(x;sigma) under the named precision mode."""
    _tf32(mode == "tf32")
    ctx = (torch.autocast(dev.type, torch.bfloat16) if mode == "bf16" else
           torch.autocast(dev.type, torch.float16) if mode == "fp16" else
           torch.autocast(dev.type, enabled=False))
    with torch.no_grad(), ctx:
        return G(x, sigma, y).float()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--teacher", default="diffusers:google/ddpm-cifar10-32")
    p.add_argument("--edm-repo", default=None)
    p.add_argument("--repa-dir", default=None)
    p.add_argument("--sigma-data", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--sigmas", default="",
                   help="comma-separated; default: a spread up to the teacher's sigma_max")
    p.add_argument("--reps", type=int, default=5, help="timed repeats per mode")
    p.add_argument("--tol", type=float, default=0.05,
                   help="usable if noise < tol * sigma_data (default 5%%)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        print(f"[probe] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    else:
        print("[probe] NO GPU -- autocast modes will not be representative")

    kw = {k: v for k, v in (("edm_repo", a.edm_repo), ("repa_dir", a.repa_dir)) if v}
    G, meta = load_teacher(a.teacher, device=dev, sigma_data=a.sigma_data, **kw)
    G.eval()
    smax = float(meta.get("sigma_max", 157.4))
    shape = meta.get("shape") or [3, 32, 32]
    ncls = int(meta.get("n_classes", 1) or 1)
    print(f"[probe] sigma_max={smax:.3f}  sigma_data={a.sigma_data}  shape={tuple(shape)}")

    sigmas = ([float(s) for s in a.sigmas.split(",")] if a.sigmas
              else [0.1, 1.0, 3.2, 10.0, 40.0, smax])

    # The guard exists to prevent exactly the failure being measured.
    old_margin = getattr(type(G), "FP32_MARGIN", None)
    if old_margin is not None:
        type(G).FP32_MARGIN = 1e9

    g = torch.Generator(device=dev).manual_seed(0)
    x0 = torch.randn(a.batch, *shape, device=dev, generator=g) * a.sigma_data
    y = torch.randint(0, ncls, (a.batch,), device=dev, generator=g)

    modes = ["fp32", "tf32", "fp16", "bf16"]
    print(f"\n{'sigma':>9}  " + "  ".join(f"{m:>18}" for m in modes[1:]))
    print(f"{'':>9}  " + "  ".join(f"{'noise/sigma_data':>18}" for _ in modes[1:]))
    verdict = {m: True for m in modes[1:]}
    for s in sigmas:
        xs = x0 + s * torch.randn(x0.shape, device=dev, generator=g)
        sig = torch.full((a.batch,), s, device=dev)
        ref = _denoise(G, xs, sig, y, "fp32", dev)
        cells = []
        for m in modes[1:]:
            err = (_denoise(G, xs, sig, y, m, dev) - ref).std().item() / a.sigma_data
            if err > a.tol:
                verdict[m] = False
            cells.append(f"{err:17.4f}{'!' if err > a.tol else ' '}")
        print(f"{s:9.2f}  " + "  ".join(cells))

    # Timing at sigma_max, which is where a one-step student always runs.
    print(f"\n{'mode':>6}  {'ms/forward':>11}  {'speedup':>8}   (batch {a.batch} at sigma_max)")
    sig = torch.full((a.batch,), smax, device=dev)
    base = None
    for m in modes:
        _denoise(G, x0, sig, y, m, dev)                      # warm up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(a.reps):
            _denoise(G, x0, sig, y, m, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / a.reps * 1e3
        base = base or dt
        print(f"{m:>6}  {dt:11.2f}  {base / dt:7.2f}x")

    if old_margin is not None:
        type(G).FP32_MARGIN = old_margin
    _tf32(False)

    ok = [m for m in modes[1:] if verdict[m]]
    print(f"\nusable at every sigma (noise < {a.tol:.0%} of sigma_data): "
          f"{', '.join(['fp32'] + ok)}")
    print("modes marked '!' put more rounding noise into D than the tolerance "
          "allows;\nat sigma_max that is what a one-step student emits as its image.")


if __name__ == "__main__":
    main()
