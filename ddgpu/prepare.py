"""One-off dataset preparation: VAE latents and the FID reference statistics.

Three subcommands, all sharded across GPUs and all resumable:

    python -m ddgpu.prepare latents  --source /data/imagenet/train --dest /data/in256 \
                                     --resolution 256 --gpus 0,1,2,3
    python -m ddgpu.prepare pixels   --source cifar10 --dest /data/cifar10 \
                                     --resolution 32
    python -m ddgpu.prepare refstats --source /data/imagenet/train --dest /data/in256 \
                                     --resolution 256 --gpus 0,1,2,3 --n 50000

`latents` writes an (N, 8, H/8, W/8) float16 memmap of SD-VAE *moments*
(mean, logvar) rather than a single sample, which is what DiT trains on: the
latent is resampled every epoch, so the memmap is not a frozen draw. It also
measures `sigma_data` from the data and records it in `meta.json` -- the configs
must not guess it. EDM's `sigma_data=0.5` is calibrated for a *different* VAE;
SD-VAE latents scaled by 0.18215 are approximately unit variance, and getting
this wrong mis-weights the DSM loss at every noise level.

`refstats` writes the Inception pool3 features of the real images that FID and
precision/recall are measured against. It uses `pytorch_fid`'s InceptionV3 --
the canonical FID network, not torchvision's -- so the numbers are comparable
with the published literature. Fake features in `ddgpu.generate` come from the
same module, so any residual difference cancels in the comparison.

Both write into `dest` and skip work that is already there.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Image pipeline (ADM / DiT "center-crop-dhariwal")
# ---------------------------------------------------------------------------
def center_crop_arr(pil_image, image_size):
    """Dhariwal & Nichol's center crop, as used by ADM and DiT.

    Repeated BOX downsampling to just above the target, then a single BICUBIC
    resize, then a center crop. Reproduced exactly because the FID reference
    statistics depend on it: crop differently and every FID in the paper shifts.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.BICUBIC)
    arr = np.array(pil_image.convert("RGB"))
    cy = (arr.shape[0] - image_size) // 2
    cx = (arr.shape[1] - image_size) // 2
    return arr[cy:cy + image_size, cx:cx + image_size]


class ImageFolderFlat(torch.utils.data.Dataset):
    """ImageFolder with a deterministic, sharding-friendly index.

    Classes are the sorted subdirectory names; files are sorted within each
    class. The resulting order is stable across machines and across runs, which
    is what lets each GPU write its own contiguous slice of one shared memmap.
    """
    EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPEG")

    def __init__(self, root, resolution):
        self.res = resolution
        classes = sorted(d for d in os.listdir(root)
                         if os.path.isdir(os.path.join(root, d)))
        if not classes:                                   # flat dir, single class
            classes, root_is_flat = [""], True
        else:
            root_is_flat = False
        self.classes = classes
        self.samples = []
        for ci, c in enumerate(classes):
            d = root if root_is_flat else os.path.join(root, c)
            for f in sorted(os.listdir(d)):
                if f.endswith(self.EXT):
                    self.samples.append((os.path.join(d, f), ci))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        arr = center_crop_arr(Image.open(p), self.res)
        x = torch.from_numpy(arr).permute(2, 0, 1).float() / 127.5 - 1.0
        return x, y


class TorchvisionImages(torch.utils.data.Dataset):
    """A torchvision classification set, presented like `ImageFolderFlat`.

    Only used for the small standard benchmarks (CIFAR-10) where downloading
    through torchvision is far less friction than staging an image folder.
    Images come out in [-1, 1], the same convention as `ImageFolderFlat`.

    The root is *discovered* rather than fixed, because a box that has already
    fetched the tarball once should not fetch it again: see `find_root`.
    """
    # name: (torchvision class, n_classes, archive name, extracted folder)
    SETS = {"cifar10":  ("CIFAR10", 10, "cifar-10-python.tar.gz", "cifar-10-batches-py"),
            "cifar100": ("CIFAR100", 100, "cifar-100-python.tar.gz", "cifar-100-python")}
    DEFAULT_ROOT = "~/.cache/dd-data"

    @classmethod
    def find_root(cls, name):
        """First candidate root that already holds this dataset; else the default.

        torchvision downloads to `root` and skips the download when the archive
        (md5-checked) or the extracted folder is already there. So the only thing
        standing between "already have it" and "fetch 170 MB again" is pointing
        `root` at the right directory -- and the right directory differs per box.
        Checked in order, first hit wins:

            $DD_DATA_ROOT    the data volume the rest of the pipeline uses
            <repo>/data      the in-repo default when DD_DATA_ROOT is unset
            ~/.cache/dd-data where we download to when nobody has it yet

        `$DD_TV_ROOT` short-circuits the search entirely and is used whether or
        not the data is there yet, so that setting it also redirects the
        download. The rest are searched, not assumed.
        """
        _, _, archive, folder = cls.SETS[name]
        if os.environ.get("DD_TV_ROOT"):
            return os.path.expanduser(os.environ["DD_TV_ROOT"])
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [os.environ.get("DD_DATA_ROOT"),
                 os.path.join(repo, "data"), cls.DEFAULT_ROOT]
        for c in cands:
            if not c:
                continue
            c = os.path.expanduser(c)
            if os.path.exists(os.path.join(c, archive)) or \
               os.path.isdir(os.path.join(c, folder)):
                print(f"[prepare] {name}: using existing data in {c}")
                return c
        return os.path.expanduser(cls.DEFAULT_ROOT)

    def __init__(self, name, resolution, root=None, train=True):
        import torchvision
        cls, ncls = self.SETS[name][:2]
        root = root or self.find_root(name)
        self.ds = getattr(torchvision.datasets, cls)(
            os.path.expanduser(root), train=train, download=True)
        self.res, self.n_classes = resolution, ncls
        self.classes = [str(i) for i in range(ncls)]
        self.samples = [(None, int(y)) for _, y in self.ds]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]
        if img.size != (self.res, self.res):
            img = img.resize((self.res, self.res), resample=Image.BICUBIC)
        x = torch.from_numpy(np.array(img.convert("RGB"))).permute(2, 0, 1)
        return x.float() / 127.5 - 1.0, y


def image_source(source, resolution):
    """`--source cifar10` -> torchvision; anything else -> an image folder."""
    if source in TorchvisionImages.SETS:
        return TorchvisionImages(source, resolution)
    return ImageFolderFlat(source, resolution)


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------
def shard_bounds(n, world, rank):
    """Contiguous [lo, hi) slice for one shard. Contiguity is required: each
    worker writes straight into the shared memmap with no coordination."""
    per = (n + world - 1) // world
    return min(rank * per, n), min((rank + 1) * per, n)


def _spawn_shards(argv, gpus):
    """Re-invoke this module once per GPU, then wait. Simpler and more robust
    than torchrun here: the workers never need to talk to each other."""
    procs = []
    for r, g in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g),
                   DD_SHARD_RANK=str(r), DD_SHARD_WORLD=str(len(gpus)))
        procs.append(subprocess.Popen([sys.executable, "-m", "ddgpu.prepare"] + argv,
                                      env=env))
    bad = [p.wait() for p in procs]
    if any(bad):
        raise SystemExit(f"shard(s) failed with codes {bad}")


def _strip_gpus(argv):
    """Drop `--gpus X` / `--gpus=X` before handing argv to the shard workers;
    each child gets a single device through CUDA_VISIBLE_DEVICES instead."""
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == "--gpus":
            skip = True
        elif not x.startswith("--gpus="):
            out.append(x)
    return out


def _shard_id():
    return int(os.environ.get("DD_SHARD_RANK", 0)), int(os.environ.get("DD_SHARD_WORLD", 1))


# ---------------------------------------------------------------------------
# latents
# ---------------------------------------------------------------------------
LATENT_SCALE = 0.18215          # SD-VAE; the value DiT and LDM train with


def cmd_latents(a):
    rank, world = _shard_id()
    ds = image_source(a.source, a.resolution)
    n, lat_hw, dest = len(ds), a.resolution // 8, a.dest
    os.makedirs(dest, exist_ok=True)
    mom_path = f"{dest}/{a.split}_moments.npy"
    lab_path = f"{dest}/{a.split}_labels.npy"
    meta_path = f"{dest}/meta.json"

    if os.path.exists(meta_path) and not a.refresh:
        print(f"[latents] {meta_path} exists; nothing to do (use --refresh)")
        return

    # Rank 0 allocates the shared memmap; the others wait on a SENTINEL written
    # afterwards, not on the memmap itself -- the file appears before it is fully
    # sized, so waiting on it directly is a race that shows up as short writes.
    ready = f"{dest}/.alloc_done"
    if rank == 0:
        if not os.path.exists(mom_path):
            np.lib.format.open_memmap(mom_path, "w+", np.float16,
                                      (n, 8, lat_hw, lat_hw)).flush()
        np.save(lab_path, np.array([y for _, y in ds.samples], np.int32))
        open(ready, "w").write(str(n))
    if world > 1:
        _wait_for(ready)

    from diffusers import AutoencoderKL
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval()

    lo, hi = shard_bounds(n, world, rank)
    sub = torch.utils.data.Subset(ds, range(lo, hi))
    dl = torch.utils.data.DataLoader(sub, a.batch_size, num_workers=a.num_workers,
                                     shuffle=False, pin_memory=True)
    mom = np.lib.format.open_memmap(mom_path, "r+")
    i, sq, cnt = lo, 0.0, 0
    with torch.no_grad():
        for bi, (x, _) in enumerate(dl):
            d = vae.encode(x.to(dev)).latent_dist
            # Moments are stored in the VAE's own units; `LatentDataset`
            # applies `latent_scale` after sampling.
            m = torch.cat([d.mean, d.logvar], 1).float()
            mom[i:i + len(m)] = m.cpu().numpy().astype(np.float16)
            # sigma_data from a *sampled* latent, since that is what training sees
            z = (d.mean + d.std * torch.randn_like(d.mean)) * LATENT_SCALE
            sq += float(z.double().pow(2).sum()); cnt += z.numel()
            i += len(m)
            if bi % 50 == 0:
                print(f"[latents r{rank}] {i - lo}/{hi - lo}", flush=True)
    mom.flush()
    np.save(f"{dest}/.stat_{rank}.npy", np.array([sq, cnt], np.float64))

    if rank == 0:
        for r in range(world):
            _wait_for(f"{dest}/.stat_{r}.npy")
        tot = sum(np.load(f"{dest}/.stat_{r}.npy") for r in range(world))
        sigma_data = float(np.sqrt(tot[0] / tot[1]))
        json.dump(dict(n=n, latent_size=lat_hw, resolution=a.resolution,
                       shape=[4, lat_hw, lat_hw], vae=a.vae,
                       latent_scale=LATENT_SCALE, sigma_data=round(sigma_data, 4),
                       n_classes=len(ds.classes), format="moments"),
                  open(meta_path, "w"), indent=1)
        for r in range(world):
            os.remove(f"{dest}/.stat_{r}.npy")
        os.remove(ready)
        print(f"[latents] done: n={n} sigma_data={sigma_data:.4f} -> {meta_path}")


def _wait_for(path, timeout=1800):
    import time
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waiting for {path}")
        time.sleep(2)


# ---------------------------------------------------------------------------
# pixels  (the cheap tier: CIFAR-10, ImageNet-64 -- no VAE anywhere)
# ---------------------------------------------------------------------------
def cmd_pixels(a):
    """Write a uint8 (N, 3, H, W) memmap plus labels and a measured sigma_data.

    Pixel space is not a downgrade of the latent path, it is a different point
    on the same axis: CIFAR-10 at 32px is 3072 dims against 4096 for a 256px
    SD-VAE latent, and ImageNet-64 at 12288 dims is THREE TIMES the dimension of
    that latent. The VAE exists to make high-RESOLUTION cheap; it does not make
    the working dimension small. See FINDINGS.md 6.

    Stored as uint8 because that is lossless for images and a quarter the size
    of fp16; the [-1,1] conversion happens in `PixelDataset`.
    """
    rank, world = _shard_id()
    ds = image_source(a.source, a.resolution)
    n, dest = len(ds), a.dest
    os.makedirs(dest, exist_ok=True)
    px_path, lab_path = f"{dest}/{a.split}_pixels.npy", f"{dest}/{a.split}_labels.npy"
    meta_path = f"{dest}/meta.json"
    if os.path.exists(meta_path) and not a.refresh:
        print(f"[pixels] {meta_path} exists; nothing to do (use --refresh)")
        return

    ready = f"{dest}/.alloc_done"
    if rank == 0:
        if not os.path.exists(px_path):
            np.lib.format.open_memmap(px_path, "w+", np.uint8,
                                      (n, 3, a.resolution, a.resolution)).flush()
        np.save(lab_path, np.array([y for _, y in ds.samples], np.int32))
        open(ready, "w").write(str(n))
    if world > 1:
        _wait_for(ready)

    lo, hi = shard_bounds(n, world, rank)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(lo, hi)),
                                     a.batch_size, num_workers=a.num_workers,
                                     shuffle=False)
    px = np.lib.format.open_memmap(px_path, "r+")
    i, sq, cnt = lo, 0.0, 0
    for bi, (x, _) in enumerate(dl):
        px[i:i + len(x)] = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
        sq += float(x.double().pow(2).sum()); cnt += x.numel()
        i += len(x)
        if bi % 100 == 0:
            print(f"[pixels r{rank}] {i - lo}/{hi - lo}", flush=True)
    px.flush()
    np.save(f"{dest}/.stat_{rank}.npy", np.array([sq, cnt], np.float64))

    if rank == 0:
        for r in range(world):
            _wait_for(f"{dest}/.stat_{r}.npy")
        tot = sum(np.load(f"{dest}/.stat_{r}.npy") for r in range(world))
        sigma_data = float(np.sqrt(tot[0] / tot[1]))
        json.dump(dict(n=n, resolution=a.resolution, shape=[3, a.resolution, a.resolution],
                       dims=3 * a.resolution ** 2, sigma_data=round(sigma_data, 4),
                       n_classes=len(ds.classes), space="pixel", format="pixels",
                       latent_scale=1.0),
                  open(meta_path, "w"), indent=1)
        for r in range(world):
            os.remove(f"{dest}/.stat_{r}.npy")
        os.remove(ready)
        print(f"[pixels] done: n={n} dims={3 * a.resolution ** 2} "
              f"sigma_data={sigma_data:.4f} -> {meta_path}")


# ---------------------------------------------------------------------------
# refstats
# ---------------------------------------------------------------------------
def build_inception(device):
    """The canonical FID InceptionV3 (pytorch-fid), pool3 / 2048-d."""
    from pytorch_fid.inception import InceptionV3
    blk = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([blk]).to(device).eval()


@torch.no_grad()
def inception_feats(model, images_uint8, device, batch=64):
    """images_uint8: (N,3,H,W) uint8 in [0,255]. pytorch-fid expects [0,1]."""
    out = []
    for i in range(0, len(images_uint8), batch):
        x = images_uint8[i:i + batch].to(device).float() / 255.0
        out.append(model(x)[0].squeeze(-1).squeeze(-1).cpu())
    return torch.cat(out)


def cmd_refstats(a):
    rank, world = _shard_id()
    ds = image_source(a.source, a.resolution)
    n = min(a.n, len(ds)) if a.n else len(ds)
    os.makedirs(a.dest, exist_ok=True)
    out = f"{a.dest}/ref_{a.resolution}_{n}.npz"
    if os.path.exists(out) and not a.refresh:
        print(f"[refstats] {out} exists; nothing to do (use --refresh)")
        return

    # Evenly spaced subsample so every class is represented, not just the first
    # few thousand alphabetical ones -- a contiguous prefix of ImageNet is ~40
    # classes and would make FID meaningless.
    idx = np.linspace(0, len(ds) - 1, n).round().astype(np.int64)
    lo, hi = shard_bounds(n, world, rank)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_inception(dev)
    sub = torch.utils.data.Subset(ds, idx[lo:hi].tolist())
    dl = torch.utils.data.DataLoader(sub, a.batch_size, num_workers=a.num_workers,
                                     shuffle=False, pin_memory=True)
    feats = []
    with torch.no_grad():
        for bi, (x, _) in enumerate(dl):
            u8 = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            feats.append(inception_feats(model, u8, dev, a.batch_size))
            if bi % 50 == 0:
                print(f"[refstats r{rank}] {bi * a.batch_size}/{hi - lo}", flush=True)
    np.save(f"{a.dest}/.ref_{rank}.npy", torch.cat(feats).numpy())

    if rank == 0:
        for r in range(world):
            _wait_for(f"{a.dest}/.ref_{r}.npy")
        f = np.concatenate([np.load(f"{a.dest}/.ref_{r}.npy") for r in range(world)])
        np.savez(out, feats=f.astype(np.float32), resolution=a.resolution, n=len(f))
        for r in range(world):
            os.remove(f"{a.dest}/.ref_{r}.npy")
        print(f"[refstats] done: {f.shape} -> {out}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("latents", "pixels", "refstats"):
        q = sub.add_parser(name)
        q.add_argument("--source", required=True,
                       help="image root (class subdirs), or 'cifar10'/'cifar100'")
        q.add_argument("--dest", required=True)
        q.add_argument("--resolution", type=int, default=256)
        q.add_argument("--batch-size", type=int, default=64)
        q.add_argument("--num-workers", type=int, default=8)
        q.add_argument("--gpus", default="", help="e.g. 0,1,2,3 (parent process only)")
        q.add_argument("--refresh", action="store_true")
        if name == "latents":
            q.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
            q.add_argument("--split", default="train")
        elif name == "pixels":
            q.add_argument("--split", default="train")
        else:
            q.add_argument("--n", type=int, default=50000)
    a = p.parse_args()

    # Parent: fan out one process per GPU. Child: DD_SHARD_* is set, so run.
    if a.gpus and "DD_SHARD_RANK" not in os.environ:
        gpus = [g for g in a.gpus.split(",") if g != ""]
        if len(gpus) > 1:
            return _spawn_shards(_strip_gpus(sys.argv[1:]), gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    {"latents": cmd_latents, "pixels": cmd_pixels, "refstats": cmd_refstats}[a.cmd](a)


if __name__ == "__main__":
    main()
