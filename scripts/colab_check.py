#!/usr/bin/env python3
"""Eyeball and score a trained student from a checkpoint alone -- built for Colab.

  python3 scripts/colab_check.py --ckpt hf:pakhomovee/distill:cifar10_dmd2/ckpt_final.pt
  python3 scripts/colab_check.py --ckpt runs/cifar10_dmd2/ckpt_final.pt --fid-n 10000

`ddgpu.train` writes `config` INTO the checkpoint, so a single `.pt` is
self-contained: no run directory, no `config.resolved.json`, no prepared
dataset, no reference `.npz`. That is the whole reason this can run on a box
that has never seen the training data.

`--fid-n 0` renders the grid and nothing else. That needs no reference data at
all -- no CIFAR-10, no Inception weights -- so it is the fast answer to "do the
samples look like images yet", which is usually the question.

The reference statistics are rebuilt here through `ddgpu.prepare` -- the same
`image_source` -> `((x+1)*127.5)` -> `inception_feats` path that produced them
on the training box -- so the FID printed here is comparable to
`ddgpu.generate`'s rather than merely similar to it. The images come from the
HF mirror (`prepare.HFParquetImages`, verified to hold the same 50 000 images
as the canonical tarball) because cs.toronto.edu throttles cloud notebooks to
~100 kB/s and the mirror runs at ~50 MB/s. Its row order differs, which leaves
FID over the full reference untouched but re-rolls which images
precision/recall sees -- so read P/R from here as indicative, and FID as
comparable. The result is cached, so a reconnected runtime never pays twice.
Sampling
likewise reuses `ddgpu.generate.sample_batch` and `decode` unchanged: a sampler
that disagrees with the eval path measures a model nobody scored.

`--compare-guard` renders the same seeds twice, with `VPPrecond`'s fp32 guard on
and then off. Off reproduces the bf16 catastrophic cancellation that made the
first CIFAR runs score FID ~325 (RUNPLAN.md section 6): `D = x - sigma*eps_hat`,
so a one-ulp bf16 error in `eps_hat` reaches `D` multiplied by sigma, and at
sigma_max=157.4 that is 0.26 of noise against a 0.5 image. If the two grids look
alike, this checkpoint's problem is NOT precision and the search moves on.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.generate import build_student, sample_batch, decode, is_pixel_space, _save_grid  # noqa: E402
from ddgpu.prepare import build_inception, inception_feats, image_source, LATENT_SCALE      # noqa: E402
from ddgpu.eval import fid_from_feats, precision_recall                                     # noqa: E402


def resolve_ckpt(spec):
    """Local path, or `hf:<repo_id>:<path/in/repo>` fetched from the Hub."""
    if not spec.startswith("hf:"):
        if not os.path.exists(spec):
            raise SystemExit(f"no such checkpoint: {spec}")
        return spec
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise SystemExit(f"expected hf:<repo_id>:<path/in/repo>, got {spec!r}")
    _, repo_id, path = parts
    from huggingface_hub import hf_hub_download
    for kind in ("dataset", "model"):
        try:
            p = hf_hub_download(repo_id=repo_id, filename=path, repo_type=kind)
            print(f"[colab] fetched {repo_id}:{path} ({kind}) -> {p}")
            return p
        except Exception as e:                                  # noqa: BLE001
            last = e
    raise SystemExit(f"could not fetch {path!r} from {repo_id!r} as dataset or "
                     f"model repo.\n  last error: {last}\n"
                     "  Check the file is actually uploaded:\n"
                     f"  python3 -c \"from huggingface_hub import list_repo_files as f; "
                     f"print(f('{repo_id}', repo_type='dataset'))\"")


def sample_images(G, c, sch, dev, n, batch, seed, use_bf16, vae, scale):
    """n decoded uint8 images, sampled exactly as `ddgpu.generate` samples."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    out, done = [], 0
    while done < n:
        b = min(batch, n - done)
        with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=use_bf16):
            z, _ = sample_batch(G, b, c, sch, dev, gen)
        out.append(decode(z, vae, scale).cpu())
        done += b
        if done % (batch * 20) < batch:
            print(f"[colab] sampled {done}/{n}", flush=True)
    return torch.cat(out)


def reference_feats(inc, dev, n, resolution, batch, source="cifar10-hf"):
    """Rebuild the FID reference, matching `prepare.cmd_refstats` exactly.

    Defaults to the HF mirror rather than torchvision: the same 50 000 images,
    verified byte-for-byte as a set, but ~50 MB/s instead of the ~100 kB/s
    cs.toronto.edu gives a cloud notebook. Row order differs; see
    `prepare.HFParquetImages` for what that does and does not affect.
    """
    ds = image_source(source, resolution)
    n = min(n, len(ds))
    # Evenly spaced, not a prefix: cmd_refstats does the same so that every
    # class is represented rather than the first few thousand.
    idx = np.linspace(0, len(ds) - 1, n).round().astype(np.int64).tolist()
    # num_workers=0 on purpose: torchvision's CIFAR-10 is a numpy array already
    # resident in RAM, so workers buy nothing and cost a real failure mode --
    # sandboxed and container runtimes routinely refuse to fork them.
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch,
                                     shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for bi, (x, _) in enumerate(dl):
            u8 = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            feats.append(inception_feats(inc, u8, dev, batch))
            if bi % 50 == 0:
                print(f"[colab] reference {bi * batch}/{n}", flush=True)
    return torch.cat(feats)


def load_or_build_reference(a, inc, dev, resolution):
    """Reference features, from the cheapest source that is actually available.

    In order: an existing refstats `.npz` (`--ref-npz`, local or `hf:`), this
    run's own cache, then a fresh pass over torchvision's CIFAR-10.

    The default source is already the fast one -- `prepare.HFParquetImages`,
    the HF mirror of the same images -- so the half-hour cs.toronto.edu fetch
    only happens if you ask for it with `--ref-source cifar10`. Two further
    ways to spend nothing at all:
      * `--ref-npz`, the same `ref_<res>_<n>.npz` `eval_all.sh` scores against,
        making the reference byte-identical to the training box's;
      * the cache written below, which survives anything but a new runtime.
    """
    if a.ref_npz:
        path = resolve_ckpt(a.ref_npz) if a.ref_npz.startswith("hf:") else a.ref_npz
        if not os.path.exists(path):
            raise SystemExit(f"no such reference: {path}")
        f = torch.from_numpy(np.load(path)["feats"]).float()
        if f.ndim != 2 or f.shape[1] != 2048:
            raise SystemExit(f"{path} holds {tuple(f.shape)}; expected (N, 2048) "
                             "pool3 features from `ddgpu.prepare refstats`")
        print(f"[colab] reference from {path}: {tuple(f.shape)}")
        return f

    cache = f"{a.out_dir}/ref_{resolution}_{a.ref_n}.npz"
    if os.path.exists(cache):
        f = torch.from_numpy(np.load(cache)["feats"]).float()
        print(f"[colab] reference from cache {cache}: {tuple(f.shape)}")
        return f

    f = reference_feats(inc, dev, a.ref_n, resolution, a.batch, a.ref_source)
    np.savez(cache, feats=f.numpy().astype(np.float32), resolution=resolution,
             n=len(f))
    print(f"[colab] reference cached -> {cache} (keep it; re-running is then free)")
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True,
                   help="local .pt, or hf:<repo_id>:<path/in/repo>")
    p.add_argument("--weights", default="ema", choices=["ema", "student"])
    p.add_argument("--out-dir", default="colab_out")
    p.add_argument("--grid", type=int, default=64, help="images in the PNG grid")
    p.add_argument("--fid-n", type=int, default=10000,
                   help="generated samples for FID; 0 skips scoring entirely")
    p.add_argument("--ref-n", type=int, default=50000, help="real images in the reference")
    p.add_argument("--ref-source", default="cifar10-hf",
                   choices=["cifar10-hf", "cifar10"],
                   help="cifar10-hf is the HF mirror (~50 MB/s, same images as "
                        "the tarball); cifar10 goes to cs.toronto.edu (~100 kB/s)")
    p.add_argument("--ref-npz", default=None,
                   help="reuse a refstats .npz instead of rebuilding from "
                        "torchvision; local path or hf:<repo_id>:<path/in/repo>")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"],
                   help="bf16 matches ddgpu.generate's sampling autocast")
    p.add_argument("--compare-guard", action="store_true",
                   help="also render a grid with the fp32 guard DISABLED")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[colab] {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'}"
          f"  torch {torch.__version__}")
    use_bf16 = a.precision == "bf16" and dev.type == "cuda"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        print("[colab] this GPU has no bf16 support; sampling in fp32 instead")
        use_bf16 = False

    ck = torch.load(resolve_ckpt(a.ckpt), map_location="cpu", weights_only=False)
    c = ck.get("config")
    if c is None:
        raise SystemExit("this checkpoint carries no embedded config; it predates "
                         "`train.save` writing one. Copy config.resolved.json "
                         "across and use ddgpu.generate --config instead.")
    print(f"[colab] mode={c.get('mode')}  step={ck.get('it')}  "
          f"nfe={c['n_student_steps']}  shape={tuple(c['shape'])}  "
          f"gpu_hours={ck.get('gpu_seconds', 0) / 3600:.2f}  "
          f"n_gpus={ck.get('n_gpus', 1)}")

    state = ck["ema"] if (a.weights == "ema" and "ema" in ck) else ck["student"]
    if a.weights == "ema" and "ema" not in ck:
        print("[colab] no EMA in checkpoint; using raw student weights")
    G, sch = build_student(c, dev, state)

    vae, scale = None, c.get("latent_scale", LATENT_SCALE)
    if is_pixel_space(c):
        print(f"[colab] pixel space; no VAE in the eval path")
    else:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(c.get("vae", "stabilityai/sd-vae-ft-mse")).to(dev).eval()

    os.makedirs(a.out_dir, exist_ok=True)
    imgs = sample_images(G, c, sch, dev, max(a.grid, a.fid_n), a.batch,
                         a.seed, use_bf16, vae, scale)
    grid_path = f"{a.out_dir}/samples.png"
    _save_grid(imgs[:a.grid], grid_path)
    print(f"[colab] grid -> {grid_path}")

    if a.compare_guard:
        margin = getattr(type(G), "FP32_MARGIN", None)
        if margin is None:
            print("[colab] --compare-guard needs a VPPrecond student; skipping")
        elif not use_bf16:
            print("[colab] --compare-guard is meaningless in fp32; skipping")
        else:
            type(G).FP32_MARGIN = 1e9          # the guard is what we want gone
            try:
                bad = sample_images(G, c, sch, dev, a.grid, a.batch,
                                    a.seed, True, vae, scale)
            finally:
                type(G).FP32_MARGIN = margin
            _save_grid(bad[:a.grid], f"{a.out_dir}/samples_guard_off.png")
            d = (imgs[:a.grid].float() - bad[:a.grid].float()).abs().mean() / 255
            print(f"[colab] guard-off grid -> {a.out_dir}/samples_guard_off.png")
            print(f"[colab] mean |guard-on - guard-off| = {d:.4f} of full scale "
                  "(large => the fp32 guard is doing real work here)")

    if a.fid_n <= 0:
        print("[colab] --fid-n 0; grid only, nothing scored")
        return

    inc = build_inception(dev)
    f_fake = inception_feats(inc, imgs[:a.fid_n], dev, a.batch)
    f_real = load_or_build_reference(a, inc, dev, c["shape"][-1])
    fid = fid_from_feats(f_real, f_fake)
    k = min(10000, len(f_real), len(f_fake))
    prec, rec = precision_recall(f_real[:k], f_fake[:k])

    rec_out = dict(ckpt=a.ckpt, weights=a.weights, step=ck.get("it"),
                   mode=c.get("mode"), nfe=c["n_student_steps"],
                   n_fake=int(len(f_fake)), n_real=int(len(f_real)),
                   # Provenance: P/R is only comparable against the same source.
                   ref_source=a.ref_npz or a.ref_source,
                   precision=a.precision, fid=float(fid),
                   prec=float(prec), recall=float(rec))
    json.dump(rec_out, open(f"{a.out_dir}/score.json", "w"), indent=1)
    print(json.dumps(rec_out, indent=1))
    if len(f_fake) < 50000:
        print(f"[colab] NOTE: FID over {len(f_fake)} samples is biased upward and is "
              "NOT comparable\n  to published 50k numbers. Use --fid-n 50000 for that.")
    print("[colab] a working one-step CIFAR student belongs in the low tens.\n"
          "  >100 means something is still structurally wrong, not merely undertrained.")


if __name__ == "__main__":
    main()
