"""Sample from a trained student and score it. Emits an eval.RunRecord.

  python -m ddgpu.generate --ckpt runs/dit_b_256_robust/ckpt_final.pt \
      --config configs/dit_b_256_robust.json --n 50000 --ref /data/in256_ref.npz

Deliberately reads `gpu_seconds` out of the checkpoint and puts it in the record,
so eval.comparison_table can enforce the matched-wall-clock rule. A checkpoint
saved without it cannot be compared, which is the intended behaviour.
"""
import argparse, json, os
import numpy as np
import torch

from .dit import make_dit
from .edm import EDMWrapper, edm_sigmas
from .eval import RunRecord, fid_from_feats, precision_recall


@torch.no_grad()
def sample_student(G, n, c, device, batch=64, cfg_scale=1.0, seed=0):
    """One-step (or few-step) generation, matching DMD2Trainer.generate."""
    g = torch.Generator(device=device).manual_seed(seed)
    sig = edm_sigmas(c["n_student_steps"], device=device) if c["n_student_steps"] > 1 else None
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        y = torch.randint(0, c["n_classes"], (b,), device=device, generator=g)
        x = torch.randn(b, *c["shape"], device=device, generator=g) * c["sigma_max"]
        if sig is None:
            x = G(x, torch.full((b,), c["sigma_max"], device=device), y)
        else:
            for k in range(len(sig) - 1):
                x0 = G(x, sig[k].expand(b), y)
                x = x0 + sig[k + 1] * torch.randn(x0.shape, device=device, generator=g) \
                    if sig[k + 1] > 0 else x0
        out.append(x.cpu())
    return torch.cat(out)[:n]


@torch.no_grad()
def decode_latents(z, vae, batch=32, device="cuda", scale=0.18215):
    imgs = []
    for i in range(0, len(z), batch):
        x = vae.decode(z[i:i + batch].to(device) / scale).sample
        imgs.append(((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu())
    return torch.cat(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--ref", required=True, help=".npz with 'feats' (N,2048) real features")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results/runs.json")
    a = ap.parse_args()

    c = json.load(open(a.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    G = EDMWrapper(make_dit(c["arch"], input_size=c["latent_size"], in_ch=c["shape"][0],
                            n_classes=c["n_classes"]), c["sigma_data"])
    G.load_state_dict(ck["student"]); G = G.to(dev).eval()

    z = sample_student(G, a.n, c, dev, a.batch)

    from diffusers import AutoencoderKL
    from torchvision.models import inception_v3
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dev).eval()
    imgs = decode_latents(z, vae, device=dev)

    inc = inception_v3(weights="IMAGENET1K_V1", aux_logits=True).to(dev).eval()
    inc.fc = torch.nn.Identity()
    from .eval import inception_features
    f_fake = inception_features(imgs, inc, device=dev)
    f_real = torch.from_numpy(np.load(a.ref)["feats"])

    fid = fid_from_feats(f_real, f_fake)
    prec, rec = precision_recall(f_real[:10000], f_fake[:10000])
    r = RunRecord(a.name or os.path.basename(os.path.dirname(a.ckpt)),
                  gpu_seconds=ck.get("gpu_seconds", 0.0), n_gpus=ck.get("n_gpus", 1),
                  step=ck.get("it", 0), fid=fid, precision=prec, recall=rec,
                  nfe=c["n_student_steps"])
    print(json.dumps(r.as_dict(), indent=1))
    recs = json.load(open(a.out)) if os.path.exists(a.out) else []
    recs.append(r.as_dict())
    json.dump(recs, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
