"""Emit a run config from (dataset env, mode, CLI overrides).

Configs are GENERATED, not hand-maintained, because FINDINGS.md 4.0 makes a
claim that only holds if they are: baseline and method must differ by exactly
the score-fusion switch, so that a measured difference cannot be an
implementation artefact. Twelve hand-edited JSON files cannot promise that; one
base dict plus a small per-mode delta can, and `--print-delta` shows the delta
so the claim is checkable rather than asserted.

  python3 scripts/mkconfig.py --mode robust --dataset imagenet256 \
      --arch DiT-XL/2 --latent-size 32 --data /data/in256 \
      --teacher ckpt/DiT-XL-2-256x256.pt --out runs/xl256_robust
"""
import argparse, json, sys

# Fields every run shares. Anything the artefacts determine (sigma_data, shape,
# n_classes, arch, latent_size, sigma_max) is left null and resolved at startup
# by ddgpu.train from meta.json / the teacher checkpoint.
BASE = dict(
    seed=0,
    track="A",
    precond="vp",            # released DiT teachers are VP eps-predictors
    n_timestep=1000,
    arch=None, latent_size=None, shape=None, n_classes=1000,
    learn_sigma=None, sigma_data=None, sigma_max=None,
    # t_min=20 is not cosmetic. Below it sigma < 0.08, and `empirical_score`
    # forms its distance matrix with cdist, whose ||a||^2+||b||^2-2a.b carries
    # ~1e-3 absolute error at d=16384; divided by 2*sigma^2 that becomes an O(1)
    # perturbation of the softmax logits and the nearest-neighbour weights go
    # arbitrary. t_max=979 is the mirror image at the top.
    sigma_dist="vp_uniform_t", t_min=20, t_max=979,
    P_mean=-1.2, P_std=1.2,          # only used when sigma_dist == "lognormal"
    micro_batch=32, steps=50000,
    lr_g=1e-5, lr_d=1e-5, clip=1.0,
    # The GAN head is random where the critic is teacher-initialised, so it
    # gets its own learning rate; None falls back to lr_d. And gan_d_weight
    # scales the discriminator's OWN objective, which gan_weight used to do
    # as a side effect -- see DMD2Trainer.step.
    lr_gan=1e-4, gan_d_weight=1.0,
    n_student_steps=1, cfg_scale=1.75, d_steps=1,
    ema_decay=0.999,
    log_every=50, diag_every=500, probe_every=2500, ckpt_every=5000,
    workers=4, amp=True, flip=False,
    data=None, teacher_ckpt=None, teacher_format="official",
    # Preferred teacher form: "<family>:<path>" through ddgpu.teachers, which
    # covers DiT / diffusers / EDM / SiT and clones the student from whatever it
    # loads. `teacher_ckpt` remains for the pre-registry DiT configs.
    teacher=None, repa_dir=None, edm_repo=None,
    validate_teacher=True, strict_teacher=True,
    init_from_teacher=True, out=None,
    gather_real=True,
)

# The complete difference between the baseline and the method. This is the diff
# that FINDINGS.md 4.0 promises: turning on the estimated lambda simultaneously
# turns OFF the hand-tuned GAN term, which is the thing it claims to replace.
MODES = {
    "dmd2":   dict(mode="teacher", gan_weight=0.001),
    "robust": dict(mode="robust", lam_estimator="dsm", lam_calib_every=4,
                   lam_ema=0.999, lam_min_count=256, lam_bins=32,
                   lam_gate_mult=None),
    # ablations
    "data":   dict(mode="data"),
    "fixed":  dict(mode="fixed", fixed_lam=0.5),
    "ratio":  dict(mode="robust", lam_estimator="ratio", lam_gate_mult=1.0,
                   lam_bins=32, lam_ema=0.999),
    "nogather": dict(mode="robust", lam_estimator="dsm", lam_calib_every=4,
                     lam_ema=0.999, lam_min_count=256, lam_bins=32,
                     lam_gate_mult=None, gather_real=False),
    # track B
    "invert": dict(track="B", mode="invert", w_mmd=1.0, w_ksd=0.1, w_moment=1.0,
                   w_rec=1.0, w_cyc=1.0, w_anchor=1.0, n_anchors=50000,
                   anchor_steps=32, lr_e=1e-5),
}


def build(a):
    c = dict(BASE)
    c.update(MODES[a.mode])
    for k, v in dict(micro_batch=a.micro_batch, steps=a.steps, data=a.data,
                     teacher_ckpt=a.teacher, teacher=a.teacher_spec,
                     sigma_dist=a.sigma_dist, repa_dir=a.repa_dir,
                     edm_repo=a.edm_repo,
                     out=a.out, n_classes=a.n_classes,
                     n_student_steps=a.n_student_steps, cfg_scale=a.cfg_scale,
                     workers=a.workers, seed=a.seed, arch=a.arch,
                     latent_size=a.latent_size).items():
        if v is not None:
            c[k] = v
    if c["teacher"]:
        # The teacher determines arch, shape and latent size; leaving them null
        # keeps a stale config value from overriding a fact about the artefact.
        c["arch"] = c["latent_size"] = c["shape"] = None
        c["teacher_ckpt"] = None
    elif c["latent_size"] and not c["shape"]:
        c["shape"] = [4, c["latent_size"], c["latent_size"]]
    if c["track"] == "B" and a.anchor_path:
        c["anchor_path"] = a.anchor_path
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            c[k] = json.loads(v)
        except json.JSONDecodeError:
            c[k] = v
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(MODES))
    p.add_argument("--dataset", default="")
    p.add_argument("--arch", default=None)
    p.add_argument("--latent-size", type=int, default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--teacher", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--micro-batch", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--n-student-steps", type=int, default=None)
    p.add_argument("--cfg-scale", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--anchor-path", default=None)
    p.add_argument("--teacher-spec", default=None,
                   help="'<family>:<path>' for ddgpu.teachers (preferred)")
    p.add_argument("--sigma-dist", default=None,
                   help="lognormal | vp_uniform_t | interp_uniform_t")
    p.add_argument("--repa-dir", default=None)
    p.add_argument("--edm-repo", default=None)
    p.add_argument("--set", nargs="*", default=[], help="key=json overrides")
    p.add_argument("--write", default=None, help="write here instead of stdout")
    p.add_argument("--print-delta", action="store_true",
                   help="show this mode's difference from the dmd2 baseline")
    a = p.parse_args()

    if a.print_delta:
        base, mine = dict(BASE, **MODES["dmd2"]), dict(BASE, **MODES[a.mode])
        keys = sorted(set(base) | set(mine))
        for k in keys:
            if base.get(k, "<absent>") != mine.get(k, "<absent>"):
                print(f"  {k}: {base.get(k, '<absent>')!r} -> {mine.get(k, '<absent>')!r}")
        return

    c = build(a)
    txt = json.dumps(c, indent=1)
    if a.write:
        open(a.write, "w").write(txt + "\n")
        print(a.write)
    else:
        sys.stdout.write(txt + "\n")


if __name__ == "__main__":
    main()
