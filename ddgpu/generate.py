"""Sample a trained student, decode, and score it. Emits an `eval.RunRecord`.

  torchrun --nproc_per_node=8 -m ddgpu.generate \
      --run-dir runs/dit_xl_256_robust --ckpt final --n-samples 50000 \
      --ref /data/in256/ref_256_50000.npz

Three things this deliberately does:

* **Reads `gpu_seconds` out of the checkpoint** and puts it in the record, so
  `eval.comparison_table` can enforce info.txt's matched-wall-clock rule. A
  checkpoint saved without it cannot be compared, which is intended.
* **Uses the EMA weights by default.** Every DMD/DMD2 number in the literature
  is an EMA number; scoring raw weights against them compares different things.
* **Featurises each batch as it is produced.** 50k decoded 256px images is 9.8
  GiB of uint8 -- buffering them to score at the end is the easiest way to OOM
  an eval that was supposed to be the cheap part.
* **Skips the VAE entirely in pixel space.** The cheap tier's student emits the
  image, so there is nothing to decode; see `is_pixel_space`.

The Inception features come from `pytorch_fid` (the canonical FID network) via
`ddgpu.prepare`, which is also what produced the reference `.npz`. Same module
on both sides, so nothing cancels incorrectly.
"""
import argparse, json, os
import numpy as np
import torch
import torch.distributed as dist

from .dit import make_dit
from .vp import VPSchedule
from .eval import RunRecord, fid_from_feats, precision_recall


def _setup():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r % torch.cuda.device_count())
    return r, w, torch.device("cuda", r % torch.cuda.device_count())


def build_student(c, device, state):
    """Rebuild the student exactly as training built it.

    Registry teachers (`c["teacher"]`) produced the student by cloning the
    teacher, so the sampler rebuilds it the same way -- load the teacher, clone,
    overwrite with the trained weights. Rebuilding from an arch config instead
    would silently diverge for any backbone that is not our own DiT.
    """
    from .train import wrap_precond, _teacher_schedule
    if c.get("teacher"):
        from .teachers import load_teacher
        kw = {k: c[k] for k in ("repa_dir", "edm_repo") if c.get(k)}
        G, _ = load_teacher(c["teacher"], device=device,
                            sigma_data=c["sigma_data"], **kw)
        G.load_state_dict(state)
        return G.eval(), _teacher_schedule(G)
    sch = VPSchedule(c.get("n_timestep", 1000)).to(device) \
        if c.get("precond", "edm") == "vp" else None
    net = make_dit(c["arch"], input_size=c["latent_size"], in_ch=c["shape"][0],
                   n_classes=c["n_classes"], learn_sigma=c.get("learn_sigma", False))
    G = wrap_precond(net, c, sch).to(device).eval()
    G.load_state_dict(state)
    return G, sch


def student_sigmas(c, sch, device):
    from .edm import edm_sigmas
    n = c["n_student_steps"]
    if n <= 1:
        return None
    if sch is not None:
        return sch.student_sigmas(n, sigma_max=c["sigma_max"]).to(device)
    return edm_sigmas(n, sigma_max=c["sigma_max"], device=device)


@torch.no_grad()
def sample_batch(G, b, c, sch, device, gen, cfg_scale=1.0):
    """One batch of student samples. Mirrors `DMD2Trainer.generate` exactly --
    a sampler that disagrees with the trainer measures a model nobody trained."""
    sig = student_sigmas(c, sch, device)
    y = torch.randint(0, c["n_classes"], (b,), device=device, generator=gen)
    x = torch.randn(b, *c["shape"], device=device, generator=gen) * c["sigma_max"]
    if sig is None:
        return G(x, torch.full((b,), c["sigma_max"], device=device), y), y
    for k in range(len(sig) - 1):
        x0 = G(x, sig[k].expand(b), y)
        x = (x0 + sig[k + 1] * torch.randn(x0.shape, device=device, generator=gen)
             if sig[k + 1] > 0 else x0)
    return x, y


def is_pixel_space(c):
    """Does this run's student emit images directly, with no VAE in the path?

    The cheap tier (CIFAR-10, ImageNet-64) trains in pixel space, so its student
    output IS the image and running it through the SD VAE decoder is not merely
    wasteful -- the decoder wants 4 latent channels and would be handed 3.

    `space` comes from the dataset's meta.json via `data.build_dataset` and is
    the authority. The channel count is a fallback for run directories written
    before `space` was resolved into the config: latents are 4-channel here,
    images are 3.
    """
    if "space" in c:
        return c["space"] == "pixel"
    return c["shape"][0] == 3


@torch.no_grad()
def decode(z, vae, scale, batch=32):
    """Model output -> uint8 (N,3,H,W), matching `prepare.cmd_refstats` exactly.

    The real side of the FID comparison converts with `((x + 1) * 127.5)`, so
    the fake side must too; a half-LSB difference here would be attributed to
    the student. `vae=None` is the pixel path -- same conversion, no decode.
    """
    if vae is None:
        return ((z.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    out = []
    for i in range(0, len(z), batch):
        x = vae.decode(z[i:i + batch].float() / scale).sample
        out.append(((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8))
    return torch.cat(out)


def _gather_cat(t, world):
    if world == 1:
        return t
    sizes = [torch.zeros(1, dtype=torch.long, device=t.device) for _ in range(world)]
    dist.all_gather(sizes, torch.tensor([len(t)], device=t.device))
    n = int(max(s.item() for s in sizes))
    pad = torch.zeros(n, t.shape[1], device=t.device, dtype=t.dtype)
    pad[:len(t)] = t
    buf = [torch.empty_like(pad) for _ in range(world)]
    dist.all_gather(buf, pad)
    return torch.cat([b[:int(s.item())] for b, s in zip(buf, sizes)])


def build_argparser():
    """Separate from `main` so tests can check the flag names without running.

    The names here are load-bearing -- see the comment below -- and the only
    cheap way to keep them that way is a test that parses them.
    """
    ap = argparse.ArgumentParser()
    # `--run-dir` and `--n-samples`, NOT `--run` and `--n`, because this module
    # is launched under torchrun. argparse abbreviation-matches every `--x` on
    # the command line against torchrun's OWN options before the script name's
    # REMAINDER can claim them, and both short spellings are ambiguous there:
    # `--run` prefixes --run-path/--run_path, and `--n` prefixes --nnodes,
    # --nproc-per-node, --node-rank, --no-python and their underscore aliases.
    # torchrun then exits with "ambiguous option" and the script never runs.
    # Whether it trips depends on the argparse version, so it fails on some
    # boxes and not others. The old spellings stay as aliases for direct
    # `python3 -m ddgpu.generate` use; do not "tidy" the long names away.
    ap.add_argument("--run-dir", "--run", dest="run",
                    help="run directory (uses config.resolved.json)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default="final", help="tag or explicit .pt path")
    ap.add_argument("--weights", default="ema", choices=["ema", "student"])
    ap.add_argument("--n-samples", "--n", dest="n", type=int, default=50000)
    ap.add_argument("--ref", required=True, help=".npz with 'feats' (N,2048)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pr-n", type=int, default=10000,
                    help="samples used for precision/recall (O(n^2) memory)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results/runs.json")
    ap.add_argument("--grid", type=int, default=64, help="also save an NxN sample grid")
    return ap


def main():
    a = build_argparser().parse_args()

    rank, world, dev = _setup()
    ck_path = a.ckpt if a.ckpt.endswith(".pt") else f"{a.run}/ckpt_{a.ckpt}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    c = json.load(open(a.config)) if a.config else \
        ck.get("config") or json.load(open(f"{a.run}/config.resolved.json"))

    state = ck["ema"] if (a.weights == "ema" and "ema" in ck) else ck["student"]
    if a.weights == "ema" and "ema" not in ck:
        print("[generate] no EMA in checkpoint; falling back to raw student weights")
    G, sch = build_student(c, dev, state)

    from .prepare import build_inception, inception_feats, LATENT_SCALE
    pixel = is_pixel_space(c)
    vae = None
    if not pixel:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(c.get("vae", "stabilityai/sd-vae-ft-mse")) \
            .to(dev).eval()
    elif rank == 0:
        print(f"[generate] pixel space ({c['shape'][0]}x{c['shape'][1]}x"
              f"{c['shape'][2]}); no VAE in the eval path")
    inc = build_inception(dev)
    scale = c.get("latent_scale", LATENT_SCALE)

    # Everything that can only fail AFTER the sampling run, exercised before it.
    # Sampling 50k images takes ~40 minutes; a bad reference file or a scipy
    # whose sqrtm signature has changed under us should cost seconds instead.
    f_real = None
    if rank == 0:
        f_real = torch.from_numpy(np.load(a.ref)["feats"]).float()
        if f_real.ndim != 2 or f_real.shape[1] != 2048:
            raise SystemExit(f"reference {a.ref} holds {tuple(f_real.shape)}; "
                             "expected (N, 2048) pool3 features from "
                             "`ddgpu.prepare refstats`")
        if len(f_real) < a.n:
            print(f"[generate] NOTE: reference has {len(f_real)} samples but "
                  f"--n-samples is {a.n}; FID is not comparable to numbers "
                  "computed against a 50k reference")
        fid_from_feats(f_real[:8, :16], f_real[8:16, :16])
        precision_recall(f_real[:8, :16], f_real[8:16, :16])
        print(f"[generate] preflight OK: reference {tuple(f_real.shape)}, "
              "FID and precision/recall callable", flush=True)

    per = (a.n + world - 1) // world
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)
    feats, grid_imgs, done = [], [], 0
    while done < per:
        b = min(a.batch, per - done)
        with torch.autocast(dev.type, torch.bfloat16, enabled=(dev.type == "cuda")):
            z, _ = sample_batch(G, b, c, sch, dev, gen)
        imgs = decode(z, vae, scale)
        feats.append(inception_feats(inc, imgs, dev, a.batch))
        if rank == 0 and len(grid_imgs) * a.batch < a.grid:
            grid_imgs.append(imgs.cpu())
        done += b
        if rank == 0 and done % (a.batch * 20) < a.batch:
            print(f"[generate] {done}/{per} per rank", flush=True)

    f_fake = _gather_cat(torch.cat(feats).to(dev), world).cpu()[:a.n]
    if rank != 0:
        return

    fid = fid_from_feats(f_real, f_fake)          # loaded in the preflight above
    k = min(a.pr_n, len(f_real), len(f_fake))
    prec, rec = precision_recall(f_real[:k], f_fake[:k])
    r = RunRecord(a.name or os.path.basename(a.run or os.path.dirname(ck_path)),
                  gpu_seconds=ck.get("gpu_seconds", 0.0), n_gpus=ck.get("n_gpus", 1),
                  step=ck.get("it", 0), fid=fid, precision=prec, recall=rec,
                  nfe=c["n_student_steps"],
                  extra=dict(weights=a.weights, n=len(f_fake), ckpt=ck_path))
    print(json.dumps(r.as_dict(), indent=1))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    recs = json.load(open(a.out)) if os.path.exists(a.out) else []
    recs = [x for x in recs if not (x["name"] == r.name and x["step"] == r.step
                                    and (x.get("extra") or {}).get("weights") == a.weights)]
    json.dump(recs + [r.as_dict()], open(a.out, "w"), indent=1)

    if grid_imgs and a.run:
        _save_grid(torch.cat(grid_imgs)[:a.grid], f"{a.run}/samples_{a.ckpt}.png")
        print(f"[generate] grid -> {a.run}/samples_{a.ckpt}.png")


def _save_grid(imgs, path, ncol=8):
    from PIL import Image
    n, _, h, w = imgs.shape
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    canvas = np.zeros((nrow * h, ncol * w, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, cc = divmod(i, ncol)
        canvas[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = im.permute(1, 2, 0).numpy()
    Image.fromarray(canvas).save(path)


if __name__ == "__main__":
    main()
