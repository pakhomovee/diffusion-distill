"""Latent datasets. Expects VAE latents precomputed once by `ddgpu.prepare`.

Precomputing is not optional at this scale: running the SD VAE encoder inside
the training loop would cost more than the distillation step itself and would
pin a third model in VRAM.

Two on-disk formats are supported, both float16 memmaps:

  "pixels"   (N, 3, H, W)  -- uint8, served in [-1,1]. The cheap tier
                             (CIFAR-10, ImageNet-64): no VAE in the loop at all.
  "moments"  (N, 8, H, W)  -- mean and logvar. The latent is *resampled* every
                             time the sample is read, which is what DiT trains
                             on and what keeps the dataset from being a frozen
                             draw. This is what `ddgpu.prepare latents` writes.
  "latents"  (N, 4, H, W)  -- a single fixed sample. Half the disk, no
                             resampling. Accepted for externally produced data.

`meta.json` written alongside carries `sigma_data`, `latent_scale` and the
latent shape; `load_meta` merges those into a run config so no config file has
to guess them.
"""
import json
import os
import numpy as np
import torch


def load_meta(root):
    """Read a dataset's meta.json, or return {} for synthetic/legacy data."""
    p = os.path.join(root, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", scale=None, flip=False):
        meta = load_meta(root)
        self.scale = scale if scale is not None else meta.get("latent_scale", 0.18215)
        self.flip = flip
        mom, lat = f"{root}/{split}_moments.npy", f"{root}/{split}_latents.npy"
        if os.path.exists(mom):
            self.x, self.fmt = np.load(mom, mmap_mode="r"), "moments"
        elif os.path.exists(lat):
            self.x, self.fmt = np.load(lat, mmap_mode="r"), "latents"
        else:
            raise FileNotFoundError(f"no {split}_moments.npy or {split}_latents.npy in {root}")
        self.y = np.load(f"{root}/{split}_labels.npy", mmap_mode="r")
        assert len(self.x) == len(self.y), f"{len(self.x)} latents vs {len(self.y)} labels"
        self.meta = meta

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        a = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32))
        if self.fmt == "moments":
            mean, logvar = a.chunk(2, dim=0)
            # Clamp as the SD VAE's own DiagonalGaussianDistribution does; a
            # stray logvar in fp16 would otherwise blow up one sample's scale.
            x = mean + (0.5 * logvar.clamp(-30.0, 20.0)).exp() * torch.randn_like(mean)
        else:
            x = a
        x = x * self.scale
        # Horizontal flip in LATENT space is only an approximation of encoding
        # the flipped image, so it is off by default. Distillation runs see far
        # less than one epoch of ImageNet, which is why this costs nothing.
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, int(self.y[i])


class PixelDataset(torch.utils.data.Dataset):
    """uint8 (N, 3, H, W) memmap, served in [-1, 1]. The cheap tier's data path.

    No VAE, no moments, no resampling -- pixels are the ground truth, so the
    only transform is the scale. `sigma_data` still comes from meta.json rather
    than a constant: CIFAR-10 in [-1,1] is not the same spread as ImageNet-64,
    and `dsm_weight` uses it at every noise level.
    """

    def __init__(self, root, split="train", flip=False):
        self.x = np.load(f"{root}/{split}_pixels.npy", mmap_mode="r")
        self.y = np.load(f"{root}/{split}_labels.npy", mmap_mode="r")
        self.flip, self.meta = flip, load_meta(root)
        assert len(self.x) == len(self.y), f"{len(self.x)} pixels vs {len(self.y)} labels"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32)) / 127.5 - 1.0
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, int(self.y[i])


class SyntheticLatents(torch.utils.data.Dataset):
    """Class-conditional low-rank Gaussian mixture in latent shape.

    Used for CPU smoke tests and for end-to-end sanity checks where the true
    distribution is known, so failures are attributable to the code and not to
    the data.
    """

    def __init__(self, n=4096, shape=(4, 8, 8), n_classes=10, rank=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        d = int(np.prod(shape))
        self.shape, self.n, self.n_classes = shape, n, n_classes
        self.mu = torch.randn(n_classes, d, generator=g) * 0.5
        self.U = torch.randn(n_classes, d, rank, generator=g) / np.sqrt(rank)
        self.y = torch.randint(0, n_classes, (n,), generator=g)
        w = torch.randn(n, rank, generator=g)
        self.x = (self.mu[self.y] + torch.einsum("nr,ndr->nd", w, self.U[self.y])
                  + 0.05 * torch.randn(n, d, generator=g)).reshape(n, *shape)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


class GaussianData(torch.utils.data.Dataset):
    """iid N(0, sd^2 I) samples, matching `teachers.GaussianTeacher`.

    The pair exists so `exp/10_lambda_real.py` can be validated against a target
    whose true score is known, before it is pointed at a real model whose answer
    nobody knows. Spec: `gaussian:c=4,hw=8,sd=0.5,n=8192`.
    """

    def __init__(self, spec="", c=4, hw=8, sd=0.5, n=8192, seed=0):
        kv = dict(c=c, hw=hw, sd=sd, n=n, seed=seed)
        for part in spec.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = float(v) if "." in v else int(v)
        g = torch.Generator().manual_seed(int(kv["seed"]))
        self.shape = (int(kv["c"]), int(kv["hw"]), int(kv["hw"]))
        self.sd = float(kv["sd"])
        self.x = torch.randn(int(kv["n"]), *self.shape, generator=g) * self.sd
        self.meta = dict(sigma_data=self.sd, shape=list(self.shape), n_classes=1,
                         space="synthetic", format="gaussian")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], 0


def build_dataset(c):
    """Dataset + the config fields the data itself determines.

    Returns `(dataset, resolved)` where `resolved` holds `sigma_data`, `shape`,
    `latent_size` and `n_classes` read from the data's meta.json. Letting the
    data set these removes a whole class of silent misconfiguration -- a config
    claiming `sigma_data=0.5` against unit-variance latents mis-weights the DSM
    loss at every noise level and looks like a bad hyperparameter, not a bug.
    """
    if c["data"] == "synthetic":
        return SyntheticLatents(c.get("n_synth", 8192), tuple(c["shape"]),
                                c["n_classes"]), {}
    if c["data"].startswith("gaussian"):
        ds = GaussianData(c["data"].split(":", 1)[-1] if ":" in c["data"] else "")
        return ds, dict(ds.meta)
    root = c["data"]
    # Fail on the actual problem. Without this the missing-directory case falls
    # through to LatentDataset (no meta.json -> format is unknown -> latent is
    # the default) and reports "no train_moments.npy", which describes a latent
    # dataset the caller may never have asked for. `train.sh` guards this
    # already; the bare `python3 -m ddgpu.teachers` / exp entry points do not.
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"dataset directory does not exist: {root}\n"
            "  * has it been prepared?  python3 -m ddgpu.prepare pixels "
            "--source cifar10 --dest <dir> --resolution 32\n"
            "  * does the path look like an unset variable expanded -- "
            "'/cifar10' from \"$DD_DATA_ROOT/cifar10\"?  export DD_DATA_ROOT first.")
    meta = load_meta(root)
    flip = c.get("flip", False)
    if meta.get("format") == "pixels":
        ds = PixelDataset(root, flip=flip)
    else:
        if not meta:
            # Legacy latent dirs predate meta.json and are still supported, so
            # this is a note on the way past, not a refusal.
            print(f"[data] no meta.json in {root}; reading it as a LATENT dataset")
        ds = LatentDataset(root, flip=flip)
    resolved = {}
    for k in ("sigma_data", "shape", "n_classes", "space"):
        if k in meta:
            resolved[k] = meta[k]
    if "latent_size" in meta:
        resolved["latent_size"] = meta["latent_size"]
    elif "resolution" in meta:
        # pixel data: `latent_size` is the grid the backbone patchifies, which
        # for a pixel model is the image itself.
        resolved["latent_size"] = meta["resolution"]
    return ds, resolved
