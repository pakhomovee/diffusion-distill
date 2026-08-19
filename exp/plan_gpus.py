import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.memcalc import Plan, dit_params, BACKBONES, wallclock_hours, USABLE_GIB
from ddgpu.dit import DIT_CONFIGS


def dit_meta(name, res):
    depth, hidden, _ = DIT_CONFIGS[name]
    patch = int(name.split("/")[1])
    tokens = (res // patch) ** 2
    params = dit_params(name, res)
    nonemb = depth * 12 * hidden ** 2          # attn(4H^2) + mlp(8H^2) per block
    return depth, hidden, tokens, params, nonemb


def row(label, name, res, method, n_tr, n_fr, opt, steps, gbatch, gpus, bwd, note=""):
    depth, hidden, tokens, params, nonemb = dit_meta(name, res)
    p = Plan(label, params, tokens, hidden, depth, n_trainable=n_tr, n_frozen=n_fr,
             opt=opt, n_bwd_passes=bwd, notes=note)
    tr, fr = p.weights_gib()
    mb = p.max_micro_batch()
    if mb == 0:
        print(f"{label:52s}  DOES NOT FIT ({tr+fr:.1f} GiB of state alone)")
        return
    per = math.ceil(gbatch / gpus)
    accum = max(1, math.ceil(per / mb))
    micro = math.ceil(per / accum)
    h, s = wallclock_hours(method, params, nonemb, tokens, depth, hidden,
                           gbatch, gpus, micro, accum, steps)
    print(f"{label:52s} {gpus:2d}x5090  state {tr+fr:5.1f}G  micro {micro:3d}x{accum:1d}"
          f"  {s:5.2f}s/step  {h:6.1f} h  ({h/24:.1f} d)")


print("=" * 118)
print("RUN PLAN  --  per-run GPU counts (RTX 5090 32GiB, DDP, grad ckpt, bf16 autocast)")
print("=" * 118)
print("\n-- L1/L2 dimension ladder: same arch, same data, 4096 vs 16384 dims (the controlled ablation) --")
row("DMD2 baseline   DiT-B/2 @256px (4096-d)", "DiT-B/2", 32, "dmd2", 2, 1, "adam_fp32", 50000, 256, 2, 2)
row("DMD2 baseline   DiT-B/2 @512px (16384-d)", "DiT-B/2", 64, "dmd2", 2, 1, "adam_fp32", 50000, 256, 4, 2)
row("Track A robust  DiT-B/2 @256px", "DiT-B/2", 32, "robust", 2, 1, "adam_fp32", 50000, 256, 2, 2)
row("Track A robust  DiT-B/2 @512px", "DiT-B/2", 64, "robust", 2, 1, "adam_fp32", 50000, 256, 4, 2)

print("\n-- L3 headline ImageNet number --")
row("DMD2 baseline   DiT-XL/2 @256px", "DiT-XL/2", 32, "dmd2", 2, 1, "adam_fp32", 50000, 256, 8, 2)
row("Track A robust  DiT-XL/2 @256px", "DiT-XL/2", 32, "robust", 2, 1, "adam_fp32", 50000, 256, 8, 2)
row("DMD2 baseline   DiT-XL/2 @512px", "DiT-XL/2", 64, "dmd2", 2, 1, "adam_bf16st", 50000, 256, 8, 2)

print("\n-- Track B (invertible): E REPLACES the fake-score critic, so still 2 trainable nets --")
row("Track B invert  DiT-B/2 @256px", "DiT-B/2", 32, "invertible", 2, 1, "adam_fp32", 50000, 256, 2, 3)
row("Track B invert  DiT-B/2 @512px", "DiT-B/2", 64, "invertible", 2, 1, "adam_fp32", 50000, 256, 4, 3)
row("Track B invert  DiT-XL/2 @256px", "DiT-XL/2", 32, "invertible", 2, 1, "adam_fp32", 50000, 256, 8, 3)

print("\n-- Track B preprocessing: teacher-anchor cache (one-off, reusable across runs) --")
for arch, res, npairs, nsteps in [("DiT-B/2", 32, 50000, 32), ("DiT-B/2", 64, 50000, 32),
                                  ("DiT-XL/2", 32, 50000, 32)]:
    depth, hidden, tokens, params, nonemb = dit_meta(arch, res)
    from ddgpu.memcalc import fwd_flops_per_sample, PEAK_BF16, MFU
    f = fwd_flops_per_sample(nonemb, tokens, depth, hidden)
    tot = npairs * nsteps * 2 * f                     # Heun = 2 evals per step
    for g in (1, 8):
        print(f"  anchors {arch} @{res}: {npairs//1000}k pairs x {nsteps} steps "
              f"-> {tot/1e15:6.2f} PFLOP, {tot/(PEAK_BF16*MFU)/3600/g:5.2f} h on {g} GPU")
