"""Empirical VRAM + throughput probe. RUN THIS FIRST ON THE GPU VM.

The activation-memory model in memcalc.py is analytic and therefore a guess with
error bars. This script measures the real peak allocation and real step time for
a given config on one card, so RUNPLAN.md's GPU counts can be corrected before
any long run is launched.

  python -m ddgpu.probe --arch DiT-B/2 --res 32 --method dmd2 --micro 64
  python -m ddgpu.probe --arch DiT-XL/2 --res 32 --method invertible --sweep
"""
import argparse, itertools, json, time
import torch

from ddgpu.dit import make_dit, DIT_CONFIGS

ROLES = {"dmd2": 2, "robust": 2, "invertible": 3}


def build(arch, res, n_trainable, opt, device, grad_ckpt=True):
    mk = lambda: make_dit(arch, input_size=res, grad_ckpt=grad_ckpt).to(device)
    train = [mk() for _ in range(n_trainable)]
    frozen = make_dit(arch, input_size=res).to(device, torch.bfloat16).eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    if opt == "adam8bit":
        import bitsandbytes as bnb
        opts = [bnb.optim.Adam8bit(m.parameters(), lr=1e-5) for m in train]
    else:
        opts = [torch.optim.AdamW(m.parameters(), lr=1e-5,
                                  foreach=True) for m in train]
    return train, frozen, opts


def probe(arch, res, method, micro, opt="adam_fp32", iters=6, device="cuda"):
    torch.cuda.reset_peak_memory_stats()
    n_tr = ROLES[method]
    train, frozen, opts = build(arch, res, n_tr, opt, device)
    B, C = micro, 4
    y = torch.randint(0, 1000, (B,), device=device)
    ts = []
    for it in range(iters):
        torch.cuda.synchronize(); t0 = time.time()
        x = torch.randn(B, C, res, res, device=device)
        t = torch.rand(B, device=device) * 10 + 0.1
        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                _ = frozen(x, t, y)                       # teacher score
            loss = 0
            for m in train:                               # student / critic / encoder
                loss = loss + m(x, t, y).float().pow(2).mean()
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if it >= 2:
            ts.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    res_ = dict(arch=arch, res=res, method=method, micro=micro, opt=opt,
                peak_gib=round(peak, 2), step_s=round(sum(ts) / len(ts), 4),
                samples_per_s=round(micro * len(ts) / sum(ts), 1))
    del train, frozen, opts
    torch.cuda.empty_cache()
    return res_


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="DiT-B/2"); p.add_argument("--res", type=int, default=32)
    p.add_argument("--method", default="dmd2"); p.add_argument("--micro", type=int, default=64)
    p.add_argument("--opt", default="adam_fp32"); p.add_argument("--sweep", action="store_true")
    p.add_argument("--out", default="results/probe.json")
    a = p.parse_args()
    out = []
    grid = (itertools.product(["DiT-B/2", "DiT-XL/2"], [32, 64],
                              ["dmd2", "invertible"], [8, 16, 32, 64, 128])
            if a.sweep else [(a.arch, a.res, a.method, a.micro)])
    for arch, res, meth, mic in grid:
        try:
            r = probe(arch, res, meth, mic, a.opt)
            print(json.dumps(r), flush=True); out.append(r)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps(dict(arch=arch, res=res, method=meth, micro=mic,
                                  peak_gib=None, oom=True)), flush=True)
            torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), indent=1)
