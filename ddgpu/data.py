"""Latent datasets. Expects VAE latents precomputed once by `ddgpu.prepare`.

Precomputing is not optional at this scale: running the SD VAE encoder inside
the training loop would cost more than the distillation step itself and would
pin a third model in VRAM.

Two on-disk formats are supported, both float16 memmaps:

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
    ds = LatentDataset(c["data"], flip=c.get("flip", False))
    m = ds.meta
    resolved = {}
    if m:
        resolved["sigma_data"] = m["sigma_data"]
        resolved["shape"] = m["shape"]
        resolved["latent_size"] = m["latent_size"]
        resolved["n_classes"] = m["n_classes"]
    return ds, resolved
