"""Latent dataset. Expects VAE latents precomputed once to a memmap.

Precomputing is not optional at this scale: running the SD VAE encoder inside the
training loop would cost more than the distillation step itself and would pin a
third model in VRAM. `precompute_latents` writes an (N, C, H, W) float16 memmap
plus an (N,) int32 label array.
"""
import os
import numpy as np
import torch


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", scale=0.18215, flip=True):
        self.x = np.load(f"{root}/{split}_latents.npy", mmap_mode="r")
        self.y = np.load(f"{root}/{split}_labels.npy", mmap_mode="r")
        self.scale, self.flip = scale, flip
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.x[i], dtype=np.float32)) * self.scale
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


@torch.no_grad()
def precompute_latents(image_dir, out_root, split, vae, batch=64, device="cuda",
                       resolution=256, num_workers=8):
    """One-off VAE encode of an image folder into a float16 memmap."""
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.Resize(resolution), transforms.CenterCrop(resolution),
        transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.ImageFolder(image_dir, tf)
    dl = torch.utils.data.DataLoader(ds, batch, num_workers=num_workers)
    os.makedirs(out_root, exist_ok=True)
    lat = None
    labs = np.zeros(len(ds), np.int32)
    i = 0
    for x, y in dl:
        z = vae.encode(x.to(device)).latent_dist.sample().mul_(1.0).cpu().numpy()
        if lat is None:
            lat = np.lib.format.open_memmap(
                f"{out_root}/{split}_latents.npy", "w+", np.float16,
                (len(ds),) + z.shape[1:])
        lat[i:i + len(z)] = z.astype(np.float16)
        labs[i:i + len(z)] = y.numpy()
        i += len(z)
    lat.flush()
    np.save(f"{out_root}/{split}_labels.npy", labs)
    return i
