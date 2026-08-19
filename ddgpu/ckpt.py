"""Load facebook's released DiT checkpoints into `ddgpu.dit.DiT`.

`ddgpu/dit.py` is a from-scratch DiT so that the parameter counts and memory
estimates in RUNPLAN.md are *derived* rather than quoted. It is
tensor-shape-compatible with the official model but uses shorter attribute
names, so a state dict needs remapping:

    official                          ours
    x_embedder.proj.{weight,bias}     x_embed.{weight,bias}
    t_embedder.mlp.{0,2}.*            t_embed.mlp.{0,2}.*
    y_embedder.embedding_table.weight y_embed.emb.weight
    blocks.N.attn.{qkv,proj}.*        blocks.N.attn.{qkv,proj}.*      (same)
    blocks.N.mlp.{fc1,fc2}.*          blocks.N.mlp.{0,2}.*
    blocks.N.adaLN_modulation.1.*     blocks.N.ada.1.*
    final_layer.linear.*              final.lin.*
    final_layer.adaLN_modulation.1.*  final.ada.1.*
    pos_embed                         (buffer, recomputed -- VERIFIED, not loaded)

`pos_embed` is the one entry that is deliberately *not* loaded: ours is a
non-persistent buffer recomputed from `sincos_pos_embed`. A silent disagreement
there degrades samples without ever raising, so `load_official_dit` compares the
two and refuses the checkpoint if they differ. That check is the reason this is
a module and not three lines in `train.py`.
"""
import os
import re
import torch

from .dit import make_dit, DIT_CONFIGS

# Released weights. Mirrors are tried in order; on AutoDL, `network_turbo`
# plus HF_ENDPOINT usually makes the HF mirror the fast one.
OFFICIAL = {
    "DiT-XL-2-256x256": dict(
        arch="DiT-XL/2", input_size=32, urls=[
            "https://hf-mirror.com/facebook/DiT-XL-2-256/resolve/main/DiT-XL-2-256x256.pt",
            "https://huggingface.co/facebook/DiT-XL-2-256/resolve/main/DiT-XL-2-256x256.pt",
            "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
        ]),
    "DiT-XL-2-512x512": dict(
        arch="DiT-XL/2", input_size=64, urls=[
            "https://hf-mirror.com/facebook/DiT-XL-2-512/resolve/main/DiT-XL-2-512x512.pt",
            "https://huggingface.co/facebook/DiT-XL-2-512/resolve/main/DiT-XL-2-512x512.pt",
            "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
        ]),
}


def _remap_key(k):
    """Official key -> ours. Returns None for keys we drop on purpose."""
    if k == "pos_embed":
        return None                                  # verified separately
    k = k.replace("x_embedder.proj.", "x_embed.")
    k = k.replace("t_embedder.", "t_embed.")
    k = k.replace("y_embedder.embedding_table.", "y_embed.emb.")
    k = k.replace("final_layer.linear.", "final.lin.")
    k = k.replace("final_layer.adaLN_modulation.", "final.ada.")
    k = re.sub(r"^blocks\.(\d+)\.adaLN_modulation\.", r"blocks.\1.ada.", k)
    k = re.sub(r"^blocks\.(\d+)\.mlp\.fc1\.", r"blocks.\1.mlp.0.", k)
    k = re.sub(r"^blocks\.(\d+)\.mlp\.fc2\.", r"blocks.\1.mlp.2.", k)
    return k


def remap_state_dict(sd):
    """Remap an official DiT state dict, dropping the recomputed buffers."""
    out, dropped = {}, {}
    for k, v in sd.items():
        nk = _remap_key(k)
        if nk is None:
            dropped[k] = v
        else:
            out[nk] = v
    return out, dropped


def download_official(name, dest_dir):
    """Fetch a released checkpoint, trying each mirror. Returns the local path."""
    import urllib.request
    spec = OFFICIAL[name]
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{name}.pt")
    if os.path.exists(path) and os.path.getsize(path) > 1 << 20:
        return path
    last = None
    for url in spec["urls"]:
        try:
            print(f"[ckpt] downloading {url}", flush=True)
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
            return path
        except Exception as e:                        # noqa: BLE001 - try next mirror
            print(f"[ckpt]   failed: {type(e).__name__}: {e}", flush=True)
            last = e
    raise RuntimeError(f"could not download {name} from any mirror") from last


def load_official_dit(path, arch=None, input_size=None, n_classes=1000,
                      in_ch=4, grad_ckpt=False, pos_tol=1e-4, strict=True):
    """Build our DiT and load an official checkpoint into it.

    Returns `(model, info)`. `info["pos_max_err"]` is the max absolute
    disagreement between our recomputed sin-cos positional embedding and the one
    stored in the checkpoint -- a nonzero value there means the two grids are
    built differently and every sample would be subtly wrong.
    """
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("ema", sd))           # released files are raw dicts
    sd = {k[len("module."):] if k.startswith("module.") else k: v
          for k, v in sd.items()}

    # Infer the architecture from the weights rather than trusting the filename.
    hidden = sd["x_embedder.proj.weight"].shape[0]
    depth = 1 + max(int(m.group(1)) for k in sd
                    for m in [re.match(r"blocks\.(\d+)\.", k)] if m)
    patch = sd["x_embedder.proj.weight"].shape[-1]
    n_tok = sd["pos_embed"].shape[1]
    grid = int(round(n_tok ** 0.5))
    out_ch = sd["final_layer.linear.weight"].shape[0] // (patch * patch)
    learn_sigma = (out_ch == 2 * in_ch)
    if arch is None:
        arch = next((k for k, (d, h, _) in DIT_CONFIGS.items()
                     if d == depth and h == hidden and k.endswith(f"/{patch}")),
                    None)
        if arch is None:
            raise ValueError(f"unrecognised DiT: depth={depth} hidden={hidden} patch={patch}")
    exp_size = grid * patch
    if input_size is not None and input_size != exp_size:
        raise ValueError(f"checkpoint is for latent_size={exp_size}, config says {input_size}")

    model = make_dit(arch, input_size=exp_size, in_ch=in_ch, n_classes=n_classes,
                     learn_sigma=learn_sigma, grad_ckpt=grad_ckpt)

    # The positional grid is recomputed, never loaded -- so verify it.
    pos_err = (model.pos.squeeze(0) - sd["pos_embed"].squeeze(0).float()).abs().max().item()
    if pos_err > pos_tol:
        raise ValueError(
            f"positional embedding mismatch (max |err| = {pos_err:.3e} > {pos_tol}). "
            "ddgpu.dit.sincos_pos_embed disagrees with the checkpoint's grid; "
            "samples would be silently wrong.")

    mapped, _ = remap_state_dict(sd)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    # `pos` is non-persistent, so it is legitimately absent from the checkpoint.
    missing = [k for k in missing if k != "pos"]
    if strict and (missing or unexpected):
        raise ValueError(f"state dict mismatch: missing={missing} unexpected={unexpected}")

    info = dict(arch=arch, latent_size=exp_size, hidden=hidden, depth=depth,
                patch=patch, learn_sigma=learn_sigma, out_ch=out_ch,
                pos_max_err=pos_err, missing=missing, unexpected=unexpected,
                n_params=sum(p.numel() for p in model.parameters()))
    return model, info


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Download + verify an official DiT checkpoint")
    p.add_argument("--name", default="DiT-XL-2-256x256", choices=list(OFFICIAL))
    p.add_argument("--dir", default="ckpt")
    p.add_argument("--no-download", action="store_true")
    a = p.parse_args()
    path = (os.path.join(a.dir, f"{a.name}.pt") if a.no_download
            else download_official(a.name, a.dir))
    _, info = load_official_dit(path)
    print(json.dumps(info, indent=1))
