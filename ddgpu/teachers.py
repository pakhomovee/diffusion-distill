"""One loader for every released teacher, plus the check that it was loaded right.

Four preconditioning families now coexist, and mixing them is the single largest
source of silent error in this project -- LOG.log ENTRY 012 (FINDING 19) is the
story of exactly one such mismatch producing a teacher that returned noise with
no exception anywhere. So all four go through one door, and that door validates.

    spec                                  family        space   wrapper
    ------------------------------------  ------------  ------  ------------------
    dit:ckpt/DiT-XL-2-256x256.pt          VP eps        latent  VPPrecond
    diffusers:google/ddpm-cifar10-32      VP eps        pixel   VPPrecond
    edm:ckpt/edm-imagenet-64x64-cond.pkl  EDM D(x,s)    pixel   EDMPklDenoiser
    sit:runs/in100/checkpoints/0300000.pt interpolant v latent  InterpolantPrecond
    synthetic:c=4,hw=32,sd=0.5,bias=0.1   analytic      n/a     GaussianTeacher

Every wrapper exposes the same `(forward, score, cfg_score, _coef)` surface, so
nothing downstream branches on which teacher it got.

`validate_teacher` is not optional plumbing. It measures the held-out denoising
loss and checks three properties that any correctly-wrapped denoiser has and a
mis-wrapped one does not:

  * `D(x, sigma) -> x`      as sigma -> 0   (nothing to denoise)
  * `D(x, sigma) -> E[x0]`  as sigma -> inf (the input carries no information)
  * the denoising loss at mid sigma is far below the loss of the trivial
    predictor `D = 0`, and below what a *deliberately wrong* noise convention
    gives -- which is how a x1000 timestep-scale error gets caught.

Run it on every teacher before it is used for anything, and record the numbers.
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from .vp import VPSchedule, VPPrecond
from .interpolant import LinearInterpolant, InterpolantPrecond


# ---------------------------------------------------------------------------
# Net-level adapters: normalise every backbone to net(x, c_noise, y, force_drop=)
# ---------------------------------------------------------------------------
class DiffusersUNetAdapter(nn.Module):
    """diffusers `UNet2DModel` -> our net signature.

    The released DDPM models (`google/ddpm-cifar10-32`, `-celebahq-256`, ...) are
    unconditional, so `y` and `force_drop` are accepted and ignored; CFG is a
    no-op on them and `cfg_scale` must be 1.0.
    """

    def __init__(self, unet, class_conditional=False):
        super().__init__()
        self.unet, self.class_conditional = unet, class_conditional

    def forward(self, x, t, y=None, force_drop=None, **kw):
        if self.class_conditional:
            return self.unet(x, t, class_labels=y).sample
        return self.unet(x, t).sample


class SiTNetAdapter(nn.Module):
    """REPA/SiT -> our net signature.

    Two mismatches to absorb, both of which would otherwise fail loudly at the
    wrong altitude or (worse) quietly:

    * `SiT.forward` returns `(prediction, projector_features)`. The tuple is
      unpacked here rather than in the preconditioner so a model that stops
      returning one is caught at the adapter.
    * `SiT.forward(x, t, y)` has NO `force_drop_ids` path -- it calls
      `self.y_embedder(y, self.training)` unconditionally. So classifier-free
      guidance cannot be expressed as a dropout flag and must be done by
      substituting the null class id, which is what `null_id` is for.
    """

    def __init__(self, sit, n_classes):
        super().__init__()
        self.sit, self.null_id = sit, n_classes

    def forward(self, x, t, y=None, force_drop=None, **kw):
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if force_drop is not None:
            y = torch.where(force_drop, self.null_id, y)
        out = self.sit(x, t, y)
        return out[0] if isinstance(out, (tuple, list)) else out


# ---------------------------------------------------------------------------
# EDM pickles are ALREADY preconditioned denoisers, so they need a passthrough
# ---------------------------------------------------------------------------
class EDMPklDenoiser(nn.Module):
    """NVlabs EDM `*Precond` module -> our denoiser surface.

    These checkpoints expose `D(x, sigma, class_labels)` directly: the Karras
    preconditioning is baked in. So unlike every other family there is nothing
    to convert -- wrapping it in `EDMWrapper` would apply the preconditioning
    twice, which is a mistake worth naming because it produces plausible-looking
    output rather than an error.
    """

    def __init__(self, precond, n_classes=0, sigma_data=0.5):
        super().__init__()
        self.net, self.n_classes, self.sigma_data = precond, n_classes, sigma_data
        self.sigma_min = float(getattr(precond, "sigma_min", 0.002))
        self.sigma_max = float(getattr(precond, "sigma_max", 80.0))

    def _labels(self, y, x, force_drop=None):
        if not self.n_classes:
            return None
        oh = torch.zeros(x.shape[0], self.n_classes, device=x.device, dtype=x.dtype)
        if y is not None:
            oh.scatter_(1, y.reshape(-1, 1).long(), 1.0)
        if force_drop is not None:                 # unconditional = all-zero label
            oh = oh * (~force_drop).reshape(-1, 1).to(oh.dtype)
        return oh

    def forward(self, x, sigma, y=None, force_drop=None, **kw):
        sigma = _as_batch(sigma, x)
        return self.net(x, sigma.reshape(-1, 1, 1, 1), self._labels(y, x, force_drop))

    def score(self, x, sigma, y=None, **kw):
        sigma = _as_batch(sigma, x)
        D = self.forward(x, sigma, y, **kw)
        return (D - x) / sigma.reshape(-1, 1, 1, 1) ** 2

    def cfg_score(self, x, sigma, y=None, scale=1.0, **kw):
        if scale == 1.0 or not self.n_classes:
            return self.score(x, sigma, y, **kw)
        sigma = _as_batch(sigma, x)
        keep = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        c = self.score(x, sigma, y, force_drop=keep, **kw)
        u = self.score(x, sigma, y, force_drop=~keep, **kw)
        return u + scale * (c - u)

    def _coef(self, sigma):
        return (torch.ones_like(sigma), sigma, torch.ones_like(sigma), sigma.log() / 4)


def _as_batch(sigma, x):
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=x.dtype)
    sigma = sigma.reshape(-1).to(x.device, x.dtype)
    return sigma.expand(x.shape[0]) if sigma.numel() == 1 else sigma


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_dit(path, device, n_classes=1000, in_ch=4, sigma_data=1.0, **kw):
    from .ckpt import load_official_dit
    net, info = load_official_dit(path, n_classes=n_classes, in_ch=in_ch)
    sch = VPSchedule().to(device)
    m = VPPrecond(net.to(device), sch, out_ch=in_ch, sigma_data=sigma_data).eval()
    ls = info["latent_size"]
    return m, dict(family="vp", space="latent", arch=info["arch"],
                   shape=[in_ch, ls, ls], dims=in_ch * ls * ls, n_classes=n_classes,
                   sigma_min=sch.sigma_min, sigma_max=sch.sigma_max,
                   n_params=info["n_params"], cfg_capable=True)


def _load_diffusers(repo, device, sigma_data=0.5, **kw):
    from diffusers import UNet2DModel
    unet = UNet2DModel.from_pretrained(repo).to(device).eval()
    # Read the schedule from the repo rather than assuming DDPM defaults: a
    # scaled-linear or cosine schedule with the same file layout would give a
    # completely different sigma(t) and no error.
    cfg = _scheduler_config(repo)
    beta_schedule = cfg.get("beta_schedule", "linear")
    if beta_schedule not in ("linear",):
        raise NotImplementedError(
            f"{repo} uses beta_schedule={beta_schedule!r}; ddgpu.vp.VPSchedule "
            "implements the linear schedule only. Add it there rather than "
            "approximating -- the sigma grid is what the network is conditioned on.")
    sch = VPSchedule(n_timestep=int(cfg.get("num_train_timesteps", 1000)),
                     beta_start=float(cfg.get("beta_start", 1e-4)),
                     beta_end=float(cfg.get("beta_end", 2e-2))).to(device)
    ch = unet.config.in_channels
    res = unet.config.sample_size
    cond = getattr(unet.config, "num_class_embeds", None)
    net = DiffusersUNetAdapter(unet, class_conditional=bool(cond))
    m = VPPrecond(net, sch, out_ch=ch, sigma_data=sigma_data).eval()
    return m, dict(family="vp", space="pixel", arch=repo,
                   shape=[ch, res, res], dims=ch * res * res,
                   n_classes=int(cond or 1), sigma_min=sch.sigma_min,
                   sigma_max=sch.sigma_max,
                   n_params=sum(p.numel() for p in unet.parameters()),
                   cfg_capable=bool(cond))


def _scheduler_config(repo):
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo, "scheduler/scheduler_config.json")
        return json.load(open(p))
    except Exception:                       # noqa: BLE001 - fall back to DDPM defaults
        return {}


def _load_edm(path, device, edm_repo=None, sigma_data=0.5, **kw):
    """Unpickle an NVlabs EDM checkpoint.

    The pickles reference `training.networks.*` and `torch_utils.persistence`,
    so the NVlabs/edm source tree has to be importable. Point `EDM_REPO` (or
    `--edm-repo`) at a clone; we do not vendor it because the pickle protocol
    ties the class paths to that exact layout.
    """
    repo = edm_repo or os.environ.get("EDM_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"unpickling {path} needs the NVlabs/edm source on sys.path "
            f"(missing: {e.name}). Clone it and set EDM_REPO=/path/to/edm.") from e
    precond = obj["ema"] if isinstance(obj, dict) and "ema" in obj else obj
    precond = precond.to(device).eval()
    ch = int(getattr(precond, "img_channels", 3))
    res = int(getattr(precond, "img_resolution", 64))
    nc = int(getattr(precond, "label_dim", 0))
    m = EDMPklDenoiser(precond, n_classes=nc,
                       sigma_data=float(getattr(precond, "sigma_data", sigma_data)))
    return m, dict(family="edm", space="pixel", arch=os.path.basename(path),
                   shape=[ch, res, res], dims=ch * res * res,
                   n_classes=max(nc, 1), sigma_min=m.sigma_min, sigma_max=m.sigma_max,
                   n_params=sum(p.numel() for p in precond.parameters()),
                   cfg_capable=bool(nc))


def _load_sit(path, device, repa_dir=None, weights="ema", sigma_data=1.0, **kw):
    """Load a REPA/SiT checkpoint through REPA's OWN model code.

    Deliberately not a state-dict remap. `ddgpu/ckpt.py` remaps DiT because we
    need our own DiT for the memory model, and it pays for that with a
    positional-grid verification. Here there is no such need, and importing
    `models.sit` removes an entire class of silent mismatch -- SiT has
    projectors, a decoder width and a final layer we would have to mirror
    exactly for no benefit.
    """
    repo = repa_dir or os.environ.get("REPA_DIR")
    if not repo:
        raise RuntimeError(
            "SiT teachers need REPA's source: set REPA_DIR=/path/to/repa-surgery/REPA")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from models.sit import SiT_models                      # noqa: E402

    ck = torch.load(path, map_location="cpu", weights_only=False)
    targs = ck["args"]
    state = ck[weights] if weights in ck else ck.get("model", ck)
    z_dims = _infer_z_dims(state)
    ls = targs.resolution // 8
    sit = SiT_models[targs.model](
        input_size=ls, num_classes=targs.num_classes,
        class_dropout_prob=getattr(targs, "class_dropout_prob", 0.1),
        z_dims=z_dims, encoder_depth=getattr(targs, "encoder_depth", 8))
    missing, unexpected = sit.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"SiT state dict missing keys: {missing[:8]}")
    sit = sit.to(device).eval()
    net = SiTNetAdapter(sit, n_classes=targs.num_classes)
    m = InterpolantPrecond(net, LinearInterpolant(), out_ch=targs.in_channels
                           if hasattr(targs, "in_channels") else 4,
                           sigma_data=sigma_data).eval()
    return m, dict(family="interpolant", space="latent", arch=targs.model,
                   shape=[4, ls, ls], dims=4 * ls * ls, n_classes=targs.num_classes,
                   sigma_min=m.path.sigma_min, sigma_max=m.path.sigma_max,
                   n_params=sum(p.numel() for p in sit.parameters()),
                   cfg_capable=True, step=ck.get("steps"), unexpected=len(unexpected))


def _infer_z_dims(state):
    """Projector output dims, as repa-surgery/training/sample.py does it."""
    import re
    out, pat = {}, re.compile(r"^projectors\.(\d+)\.(\d+)\.weight$")
    for k, v in state.items():
        m = pat.match(k)
        if m:
            i, layer = int(m.group(1)), int(m.group(2))
            if i not in out or layer >= 4:
                out[i] = v.shape[0]
    return [out[i] for i in sorted(out)]


class GaussianTeacher(nn.Module):
    """Analytic denoiser for a Gaussian target, with a controllable bias.

    Not a toy for its own sake -- it is the ONLY teacher for which the true score
    is known, so it is the only one against which `exp/10_lambda_real.py` can be
    validated rather than merely run. Data is N(0, sd^2 I), for which

        p(x_sigma) = N(0, (sd^2 + sigma^2) I),  s(x) = -x / (sd^2 + sigma^2)

    `bias` multiplies the returned score, standing in for a teacher that is
    systematically off; `bias=0` is an oracle. `true_score` exposes the exact
    answer so a test can check the DSM identity on real tensors.
    """

    def __init__(self, shape, sd=0.5, bias=0.0, n_classes=1):
        super().__init__()
        self.shape, self.sd, self.bias, self.n_classes = shape, sd, bias, n_classes
        self.sigma_data, self.sigma_min, self.sigma_max = sd, 1e-3, 1e3

    def true_score(self, x, sigma):
        sigma = _as_batch(sigma, x)
        return -x / (self.sd ** 2 + sigma.reshape(-1, 1, 1, 1) ** 2)

    def score(self, x, sigma, y=None, **kw):
        return (1.0 + self.bias) * self.true_score(x, sigma)

    def cfg_score(self, x, sigma, y=None, scale=1.0, **kw):
        return self.score(x, sigma, y, **kw)

    def forward(self, x, sigma, y=None, **kw):
        sigma = _as_batch(sigma, x)
        return x + sigma.reshape(-1, 1, 1, 1) ** 2 * self.score(x, sigma, y)

    def _coef(self, sigma):
        return (torch.ones_like(sigma), sigma, torch.ones_like(sigma), sigma.log() / 4)


class TinyDenoiser(nn.Module):
    """A few-thousand-parameter MLP denoiser, EDM-preconditioned.

    Exists so the cheap-tier code path -- registry load, `clone_trainable`,
    ConvGANHead (it has no `.trunk`), the DMD2 loop -- can be smoke-tested end
    to end on CPU without downloading anything. `GaussianTeacher` cannot serve
    that role: it has no parameters, so cloning it yields an optimiser over an
    empty list.

    It is a smoke fixture, not a teacher anyone should distil from.
    """

    def __init__(self, shape, hidden=64, sigma_data=0.5, n_classes=1):
        super().__init__()
        d = int(np.prod(shape))
        self.shape, self.sigma_data, self.n_classes = shape, sigma_data, n_classes
        self.net = nn.Sequential(nn.Linear(d + 1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.sigma_min, self.sigma_max = 0.002, 80.0

    def _coef(self, sigma):
        sd = self.sigma_data
        s2 = sigma ** 2 + sd ** 2
        return (sd ** 2 / s2, sigma * sd / s2.sqrt(), 1.0 / s2.sqrt(), sigma.log() / 4)

    def forward(self, x, sigma, y=None, force_drop=None, **kw):
        sigma = _as_batch(sigma, x)
        cs, co, ci, cn = self._coef(sigma)
        v = lambda t: t.reshape(-1, 1, 1, 1)
        h = torch.cat([(v(ci) * x).reshape(x.shape[0], -1), cn.reshape(-1, 1)], -1)
        F = self.net(h).reshape_as(x)
        return v(cs) * x + v(co) * F

    def score(self, x, sigma, y=None, **kw):
        sigma = _as_batch(sigma, x)
        return (self.forward(x, sigma, y, **kw) - x) / sigma.reshape(-1, 1, 1, 1) ** 2

    def cfg_score(self, x, sigma, y=None, scale=1.0, **kw):
        return self.score(x, sigma, y, **kw)


def _load_synthetic(spec, device, sigma_data=None, **kw):
    """`synthetic:c=4,hw=32,sd=0.5,bias=0.1` -> a GaussianTeacher."""
    kv = dict(c=4, hw=32, sd=0.5, bias=0.0, n_classes=1, net="gaussian")
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                kv[k] = float(v) if "." in v else int(v)
            except ValueError:
                kv[k] = v
    shape = [int(kv["c"]), int(kv["hw"]), int(kv["hw"])]
    if str(kv.get("net", "")) == "mlp" or "net=mlp" in spec:
        m = TinyDenoiser(shape, sigma_data=float(kv["sd"]),
                         n_classes=int(kv["n_classes"])).to(device).eval()
        return m, dict(family="synthetic", space="synthetic", arch="tiny-mlp",
                       shape=shape, dims=int(np.prod(shape)),
                       n_classes=int(kv["n_classes"]), sigma_min=m.sigma_min,
                       sigma_max=m.sigma_max,
                       n_params=sum(p.numel() for p in m.parameters()),
                       cfg_capable=False)
    m = GaussianTeacher(shape, sd=float(kv["sd"]), bias=float(kv["bias"]),
                        n_classes=int(kv["n_classes"])).to(device).eval()
    return m, dict(family="synthetic", space="synthetic", arch=f"gaussian(sd={kv['sd']})",
                   shape=shape, dims=int(np.prod(shape)), n_classes=int(kv["n_classes"]),
                   sigma_min=m.sigma_min, sigma_max=m.sigma_max, n_params=0,
                   cfg_capable=False)


LOADERS = {"dit": _load_dit, "diffusers": _load_diffusers, "edm": _load_edm,
           "sit": _load_sit, "synthetic": _load_synthetic}


def load_teacher(spec, device="cpu", **kw):
    """`load_teacher("diffusers:google/ddpm-cifar10-32")` -> (denoiser, meta)."""
    if ":" not in spec:
        raise ValueError(f"teacher spec must be '<family>:<path>', got {spec!r}. "
                         f"families: {sorted(LOADERS)}")
    fam, rest = spec.split(":", 1)
    if fam not in LOADERS:
        raise ValueError(f"unknown teacher family {fam!r}; expected {sorted(LOADERS)}")
    m, meta = LOADERS[fam](rest, device, **kw)
    for p in m.parameters():
        p.requires_grad_(False)
    meta["spec"] = spec
    return m, meta


# ---------------------------------------------------------------------------
# Validation -- run this on every teacher before trusting a single number
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate_teacher(denoiser, x0, y=None, sigmas=(0.01, 0.1, 0.5, 2.0, 20.0, 100.0),
                     device="cpu"):
    """Denoising loss across sigma, plus the two limits that catch mis-wrapping.

    Returns a dict of measurements and a `verdict`. Interpretation:

      `rel_mse[sigma]`  = E||D(x0+sigma*eps, sigma) - x0||^2 / E||x0||^2.
                          Should be ~0 at small sigma and rise to ~1 at large
                          sigma (where the best possible answer is E[x0], and
                          E||E[x0]-x0||^2 / E||x0||^2 -> 1 for centred data).
      `identity_err`    = ||D(x, sigma_tiny) - x|| / ||x||. A denoiser at
                          negligible noise must return its input.
      `mean_err`        = ||D(x, sigma_huge) - mean(x0)|| / ||x0||. At
                          overwhelming noise the only recoverable information is
                          the dataset mean.

    A teacher wrapped with the wrong noise convention (a x1000 timestep scale, a
    VP model read as EDM) typically shows `rel_mse` FLAT and near 1 everywhere,
    because the network is being evaluated at a noise level unrelated to the one
    actually applied. That is the signature to look for -- it is the failure that
    produces no exception.
    """
    x0 = x0.to(device)
    out, denom = {}, float(x0.pow(2).mean())
    for s in sigmas:
        sig = torch.full((x0.shape[0],), float(s), device=device)
        eps = torch.randn_like(x0)
        D = denoiser(x0 + sig.reshape(-1, 1, 1, 1) * eps, sig, y)
        out[f"rel_mse@{s:g}"] = float((D - x0).pow(2).mean()) / max(denom, 1e-12)

    tiny = torch.full((x0.shape[0],), 1e-3, device=device)
    xt = x0 + tiny.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
    identity_err = float((denoiser(xt, tiny, y) - xt).pow(2).mean()) / max(denom, 1e-12)

    huge_v = min(float(getattr(denoiser, "sigma_max", 100.0)), 100.0)
    huge = torch.full((x0.shape[0],), huge_v, device=device)
    xh = x0 + huge.reshape(-1, 1, 1, 1) * torch.randn_like(x0)
    mu = x0.mean(0, keepdim=True)
    mean_err = float((denoiser(xh, huge, y) - mu).pow(2).mean()) / max(denom, 1e-12)

    lo = out[f"rel_mse@{min(sigmas):g}"]
    hi = out[f"rel_mse@{max(sigmas):g}"]
    monotone = all(out[f"rel_mse@{a:g}"] <= out[f"rel_mse@{b:g}"] + 0.05
                   for a, b in zip(sigmas, sigmas[1:]))
    ok = (identity_err < 0.05) and (lo < 0.2) and (hi > 3 * lo) and monotone
    out.update(identity_err=identity_err, mean_err=mean_err, monotone=monotone,
               verdict="OK" if ok else "SUSPECT")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Load and validate a teacher")
    p.add_argument("--teacher", required=True, help="<family>:<path>")
    p.add_argument("--data", default=None,
                   help="prepared dataset dir, OR a source name ('cifar10-hf', "
                        "'cifar10') to pull the validation batch straight from "
                        "the images with nothing prepared")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--repa-dir", default=None)
    p.add_argument("--edm-repo", default=None)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {k: v for k, v in dict(repa_dir=a.repa_dir, edm_repo=a.edm_repo).items() if v}
    m, meta = load_teacher(a.teacher, device=dev, **kwargs)
    print(json.dumps(meta, indent=1, default=str))
    if a.data:
        # `validate_teacher` only wants a batch of real images in [-1,1]. If the
        # user names a source rather than a prepared directory, read the images
        # directly -- preparing a whole dataset to check a teacher is a large
        # detour, and on an ephemeral box it is the reason the check gets
        # skipped. Pixel teachers only: a latent teacher needs the VAE-encoded
        # dataset that `prepare latents` writes.
        from .prepare import TorchvisionImages, HFParquetImages, image_source
        if a.data in TorchvisionImages.SETS or a.data in HFParquetImages.SETS:
            if meta.get("space") != "pixel":
                raise SystemExit(
                    f"--data {a.data} reads raw images, but {a.teacher} works in "
                    f"{meta.get('space')} space; point --data at the prepared "
                    "latent dataset instead.")
            ds = image_source(a.data, meta["shape"][-1])
        else:
            from .data import build_dataset
            ds, _ = build_dataset(dict(data=a.data, shape=meta["shape"],
                                       n_classes=meta["n_classes"]))
        xb = torch.stack([ds[i][0] for i in range(a.batch)])
        yb = torch.tensor([ds[i][1] for i in range(a.batch)])
        print(json.dumps(validate_teacher(m, xb, yb.to(dev), device=dev), indent=1))
