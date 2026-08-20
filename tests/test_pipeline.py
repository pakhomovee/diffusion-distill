"""Invariants for the GPU pipeline: preconditioning, checkpoints, config wiring.

Same rule as tests/test_all.py -- each test guards a specific way a wrong number
could look plausible all the way into a plot. These cover the parts that only
exist because the released teacher is a VP model and ours is an EDM one, which
is exactly the seam where a silent mismatch would read as "distillation just
doesn't work very well".

  python3 tests/test_pipeline.py
"""
import json, math, os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.set_num_threads(2)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------------
def t_pos_embed_matches_official():
    """Our sin-cos grid must equal the official DiT one BIT FOR BIT.

    `ckpt.load_official_dit` verifies this at load time, but only against a
    checkpoint we might not have on this box. This is the same check against an
    independent transcription of facebookresearch/DiT's get_2d_sincos_pos_embed,
    so it runs everywhere. A mismatch is invisible in every loss curve and
    corrupts every sample.
    """
    from ddgpu.dit import sincos_pos_embed

    def get_1d(dim, pos):
        omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
        omega = 1.0 / 10000 ** omega
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    def get_2d(dim, g):
        gh = np.arange(g, dtype=np.float32)
        grid = np.stack(np.meshgrid(gh, gh), axis=0).reshape([2, 1, g, g])
        return np.concatenate([get_1d(dim // 2, grid[0]), get_1d(dim // 2, grid[1])], 1)

    for hidden, grid in ((1152, 16), (1152, 32), (768, 16)):
        err = (torch.from_numpy(get_2d(hidden, grid)).float()
               - sincos_pos_embed(hidden, grid)).abs().max().item()
        check(f"pos_embed == official (hidden={hidden}, grid={grid})", err == 0.0,
              f"max|err| {err:.1e}")


# --------------------------------------------------------------------------
def t_vp_schedule_roundtrip():
    """sigma <-> t must invert. `t_of_sigma` feeds the network's ONLY noise
    conditioning, so an error here is a model evaluated at the wrong noise
    level -- which looks like a bad teacher, not a bug."""
    from ddgpu.vp import VPSchedule
    s = VPSchedule()
    check("sigma_max == schedule top (not EDM's 80)", abs(s.sigma_max - 157.4) < 0.2,
          f"{s.sigma_max:.3f}")
    t = torch.tensor([0.0, 17.5, 249.0, 512.3, 999.0], dtype=torch.float64)
    sig = s.sigma_of_t(t)
    back = s.t_of_sigma(sig)
    check("t -> sigma -> t round trip", (back - t).abs().max().item() < 1e-3,
          f"max|dt| {(back - t).abs().max().item():.2e}")
    check("sigma(t) strictly increasing",
          bool((s.sigmas[1:] > s.sigmas[:-1]).all()))
    # out-of-range sigmas clamp instead of raising: samplers hit sigma=0.
    check("sigma below the schedule clamps to t=0",
          float(s.t_of_sigma(torch.tensor([1e-8]))) == 0.0)


def t_vp_precond_exact_on_gaussian():
    """With an ORACLE eps-predictor, VPPrecond.score must equal the analytic score.

    Data ~ N(0, sd^2 I) gives p(x_sigma) = N(0, (sd^2+sigma^2) I) and
    s(x) = -x / (sd^2 + sigma^2). The oracle net only sees the preconditioned
    input c_in*x and the timestep t, so this exercises the whole change of
    variables -- c_in, t_of_sigma, and the eps -> score conversion -- at once.
    Any one of them being wrong shows up here and nowhere else.
    """
    from ddgpu.vp import VPSchedule, VPPrecond
    sch, sd = VPSchedule(), 0.7

    class Oracle(torch.nn.Module):
        def forward(self, u, t, y, **kw):
            sig = sch.sigma_of_t(t.double()).to(u.dtype).reshape(-1, 1, 1, 1)
            x = u * (1 + sig ** 2).sqrt()                    # undo c_in
            return sig * x / (sd ** 2 + sig ** 2)            # = -sigma * s(x)

    m = VPPrecond(Oracle(), sch, out_ch=4)
    x = torch.randn(8, 4, 8, 8)
    for sig in (0.02, 0.3, 1.0, 7.0, 100.0):
        s = torch.full((8,), sig)
        got = m.score(x, s, torch.zeros(8, dtype=torch.long))
        want = -x / (sd ** 2 + sig ** 2)
        rel = ((got - want).norm() / want.norm()).item()
        check(f"VP score == analytic (sigma={sig})", rel < 1e-4, f"rel {rel:.2e}")
    # D and score must be the same object seen two ways
    s = torch.full((8,), 1.3)
    D = m(x, s, torch.zeros(8, dtype=torch.long))
    sc = m.score(x, s, torch.zeros(8, dtype=torch.long))
    rel = ((D - (x + s.reshape(-1, 1, 1, 1) ** 2 * sc)).abs().max()).item()
    check("D == x + sigma^2 * score", rel < 1e-4, f"max|err| {rel:.2e}")


def t_student_grid():
    """The few-step student grid must be the teacher's own timesteps.

    DMD2's ImageNet student steps at t = {999, 749, 499, 249}. An EDM rho=7 grid
    over the same sigma range visits noise levels the VP teacher was never
    trained on, and the student is initialised FROM that teacher."""
    from ddgpu.vp import VPSchedule
    s = VPSchedule()
    g = s.student_sigmas(4)
    want = s.sigma_of_t(torch.tensor([999.0, 749.0, 499.0, 249.0]))
    check("4-step student grid == DMD2 timesteps",
          (g[:4] - want).abs().max().item() < 1e-3)
    check("student grid ends at sigma=0", float(g[-1]) == 0.0)
    check("1-step student grid starts at sigma_max",
          abs(float(s.student_sigmas(1)[0]) - s.sigma_max) < 1e-2)


def t_sigma_sampler():
    from ddgpu.vp import VPSchedule, make_sigma_sampler
    s = VPSchedule()
    f = make_sigma_sampler(dict(sigma_dist="vp_uniform_t", t_min=20, t_max=979), s)
    v = f(4096, "cpu")
    check("vp_uniform_t stays inside the schedule window",
          bool((v >= float(s.sigmas[20]) - 1e-6).all() and
               (v <= float(s.sigmas[979]) + 1e-6).all()),
          f"[{v.min():.4f}, {v.max():.2f}]")
    g = make_sigma_sampler(dict(sigma_dist="lognormal", P_mean=-1.2, P_std=1.2))
    check("lognormal sampler still works", g(1024, "cpu").shape == (1024,))


# --------------------------------------------------------------------------
def t_checkpoint_remap():
    """An official-format state dict must load with zero missing/unexpected keys
    and reproduce the model bit-exactly. Built by inverting our own remap on a
    real model, so the test breaks if either side of the mapping drifts."""
    from ddgpu.dit import make_dit
    from ddgpu.ckpt import load_official_dit

    def to_official(k):
        k = re.sub(r"^blocks\.(\d+)\.mlp\.0\.", r"blocks.\1.mlp.fc1.", k)
        k = re.sub(r"^blocks\.(\d+)\.mlp\.2\.", r"blocks.\1.mlp.fc2.", k)
        k = re.sub(r"^blocks\.(\d+)\.ada\.", r"blocks.\1.adaLN_modulation.", k)
        k = k.replace("final.lin.", "final_layer.linear.")
        k = k.replace("final.ada.", "final_layer.adaLN_modulation.")
        k = k.replace("y_embed.emb.", "y_embedder.embedding_table.")
        k = k.replace("t_embed.", "t_embedder.")
        return k.replace("x_embed.", "x_embedder.proj.") if k.startswith("x_embed.") else k

    m = make_dit("DiT-S/2", input_size=16, in_ch=4, n_classes=100, learn_sigma=True)
    sd = {to_official(k): v.clone() for k, v in m.state_dict().items()}
    sd["pos_embed"] = m.pos.clone()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fake.pt")
        torch.save(sd, p)
        m2, info = load_official_dit(p, n_classes=100)
        check("remap: no missing keys", info["missing"] == [], str(info["missing"]))
        check("remap: no unexpected keys", info["unexpected"] == [], str(info["unexpected"]))
        check("remap: learn_sigma inferred", info["learn_sigma"] is True)
        check("remap: latent_size inferred", info["latent_size"] == 16)
        m.eval(); m2.eval()
        x, t, y = torch.randn(2, 4, 16, 16), torch.tensor([100., 900.]), torch.tensor([3, 7])
        with torch.no_grad():
            err = (m(x, t, y) - m2(x, t, y)).abs().max().item()
        check("remap: forward is bit-exact", err == 0.0, f"max|err| {err:.1e}")

        # A wrong positional grid must be REFUSED, not silently accepted.
        sd["pos_embed"] = sd["pos_embed"] + 1.0
        torch.save(sd, p)
        try:
            load_official_dit(p, n_classes=100)
            check("remap: bad pos_embed is rejected", False, "loaded anyway")
        except ValueError as e:
            check("remap: bad pos_embed is rejected", "positional" in str(e))


# --------------------------------------------------------------------------
def t_dataset_moments():
    """Moments format: resample per read, and sigma_data measured, not assumed."""
    from ddgpu.data import LatentDataset, build_dataset
    with tempfile.TemporaryDirectory() as td:
        n, hw = 64, 8
        mom = np.zeros((n, 8, hw, hw), np.float16)
        mom[:, :4] = np.random.randn(n, 4, hw, hw).astype(np.float16)
        mom[:, 4:] = np.float16(-2.0)                       # logvar
        np.save(f"{td}/train_moments.npy", mom)
        np.save(f"{td}/train_labels.npy", np.arange(n, dtype=np.int32) % 10)
        json.dump(dict(n=n, latent_size=hw, shape=[4, hw, hw], n_classes=10,
                       sigma_data=1.0, latent_scale=1.0, format="moments"),
                  open(f"{td}/meta.json", "w"))
        ds = LatentDataset(td)
        a, ya = ds[3]
        b, _ = ds[3]
        check("moments: shape is (4,H,W) after sampling", tuple(a.shape) == (4, hw, hw))
        check("moments: two reads of the same index differ (resampled)",
              (a - b).abs().max().item() > 0)
        check("moments: label preserved", ya == 3 % 10)
        _, resolved = build_dataset(dict(data=td, shape=[4, hw, hw], n_classes=10))
        check("build_dataset propagates sigma_data from meta.json",
              resolved["sigma_data"] == 1.0 and resolved["n_classes"] == 10)


# --------------------------------------------------------------------------
def t_config_delta():
    """FINDINGS.md 4.0 claims baseline and method share every line except the
    score-fusion switch. That claim is only true if it is enforced, so this
    pins the EXACT difference. If someone adds a knob to one mode and not the
    other, the "a measured difference cannot be an implementation artefact"
    argument quietly stops holding, and this test is what notices."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "scripts"))
    import mkconfig
    base = dict(mkconfig.BASE, **mkconfig.MODES["dmd2"])
    rob = dict(mkconfig.BASE, **mkconfig.MODES["robust"])
    diff = sorted(k for k in set(base) | set(rob)
                  if base.get(k, "<absent>") != rob.get(k, "<absent>"))
    expect = sorted(["mode", "gan_weight", "lam_estimator", "lam_calib_every",
                     "lam_ema", "lam_min_count", "lam_bins", "lam_gate_mult"])
    check("robust differs from dmd2 in exactly the documented keys",
          diff == expect, f"got {diff}")
    check("robust turns the hand-tuned GAN term OFF", "gan_weight" not in rob)
    check("dmd2 carries no lambda estimator",
          not any(k.startswith("lam_") for k in base))


def t_ema():
    from ddgpu.train import EMA
    m = torch.nn.Linear(4, 4)
    with torch.no_grad():
        m.weight.fill_(0.0)
    ema = EMA(m, decay=0.5)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)
    check("EMA moves halfway at decay=0.5",
          abs(float(ema.shadow.weight[0, 0]) - 0.5) < 1e-6,
          f"{float(ema.shadow.weight[0, 0]):.4f}")


def t_lambda_probe():
    """The drift probe must produce a bounded, finite statistic at arbitrary
    points -- that is the whole reason it exists rather than reusing the DSM
    calibrator, which needs held-out real data."""
    from ddgpu.robust import LambdaProbe
    from ddgpu.vp import VPSchedule, VPPrecond

    class Zero(torch.nn.Module):
        def forward(self, u, t, y, **kw):
            return torch.zeros_like(u)

    T = VPPrecond(Zero(), VPSchedule(), out_ch=4)
    pr = LambdaProbe(n_bins=8)
    real = torch.randn(64, 4, 4, 4)
    for _ in range(6):
        sig = torch.rand(32) * 2 + 0.1
        x = torch.randn(32, 4, 4, 4) * sig.reshape(-1, 1, 1, 1)
        pr.observe(T, x, torch.zeros(32, dtype=torch.long), sig, real)
    s, lam, vrel, cnt = pr.curve(min_count=1)
    seen = cnt > 0
    check("probe: lambda in [0,1] where sampled",
          bool(((lam[seen] >= 0) & (lam[seen] <= 1)).all()), str(lam[seen].tolist()[:4]))
    check("probe: unsampled buckets are NaN, not 0",
          bool(torch.isnan(lam[~seen]).all()) if (~seen).any() else True)
    check("probe: reset clears the accumulator",
          (pr.reset(), float(pr.cnt.sum()))[1] == 0.0)


def t_pixel_eval_path():
    """The cheap tier's eval must not route the student through the SD VAE.

    CIFAR-10 and ImageNet-64 train in PIXEL space: the student output is the
    image. `generate.main` was written for the latent tier and decoded
    unconditionally, which hands a 3-channel image to a decoder expecting 4
    latent channels -- so the entire cheap programme could be trained and then
    not scored. Nothing else in the suite executes `ddgpu.generate`.

    The second half is the subtler half: the fake side of a FID comparison must
    convert to uint8 exactly as `prepare.cmd_refstats` converts the real side.
    A different rounding or a missing clamp would land in the FID and be read as
    a property of the student.
    """
    from ddgpu.generate import is_pixel_space, decode, sample_batch
    from ddgpu.data import build_dataset

    check("pixel space from meta", is_pixel_space(dict(space="pixel", shape=[3, 32, 32])))
    check("latent space from meta", not is_pixel_space(dict(space="latent", shape=[4, 32, 32])))
    # Run dirs written before `space` was resolved into the config fall back to
    # the channel count; getting this backwards is silent in both directions.
    check("legacy config: 3ch -> pixel", is_pixel_space(dict(shape=[3, 32, 32])))
    check("legacy config: 4ch -> latent", not is_pixel_space(dict(shape=[4, 16, 16])))

    # A real pixel dataset dir, resolved the way training resolves it.
    n, hw = 8, 8
    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/train_pixels.npy",
                np.random.randint(0, 256, (n, 3, hw, hw), dtype=np.uint8))
        np.save(f"{td}/train_labels.npy", np.zeros(n, np.int64))
        json.dump(dict(n=n, resolution=hw, shape=[3, hw, hw], n_classes=1,
                       space="pixel", format="pixels", sigma_data=0.5,
                       latent_scale=1.0), open(f"{td}/meta.json", "w"))
        _, resolved = build_dataset(dict(data=td, shape=[3, hw, hw], n_classes=1))
    check("build_dataset resolves space=pixel", resolved.get("space") == "pixel")
    check("resolved pixel config takes the no-VAE path", is_pixel_space(resolved))

    # End-to-end: sample a pixel student and decode it with no VAE at all.
    from ddgpu.dit import make_dit
    from ddgpu.train import wrap_precond
    c = dict(shape=[3, hw, hw], latent_size=hw, n_classes=1, arch="DiT-S/2",
             sigma_data=0.5, sigma_max=80.0, n_student_steps=1, precond="edm",
             space="pixel", latent_scale=1.0)
    net = make_dit(c["arch"], input_size=hw, in_ch=3, n_classes=1, learn_sigma=False)
    G = wrap_precond(net, c, None).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        z, _ = sample_batch(G, 4, c, None, torch.device("cpu"), gen)
    check("pixel student emits image-shaped output", tuple(z.shape) == (4, 3, hw, hw),
          str(tuple(z.shape)))
    imgs = decode(z, None, c["latent_scale"])
    check("no-VAE decode gives uint8 (N,3,H,W)",
          imgs.dtype == torch.uint8 and tuple(imgs.shape) == (4, 3, hw, hw),
          f"{imgs.dtype} {tuple(imgs.shape)}")

    # Bit-for-bit against prepare.cmd_refstats' own conversion, out-of-range included.
    probe = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]).reshape(1, 1, 1, 7)
    ref = ((probe + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    check("uint8 conversion matches the FID reference side bit-for-bit",
          bool((decode(probe, None, 1.0) == ref).all()),
          f"{decode(probe, None, 1.0).flatten().tolist()} vs {ref.flatten().tolist()}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (t_pos_embed_matches_official, t_vp_schedule_roundtrip,
               t_vp_precond_exact_on_gaussian, t_student_grid, t_sigma_sampler,
               t_checkpoint_remap, t_dataset_moments, t_config_delta, t_ema,
               t_lambda_probe, t_pixel_eval_path):
        print(f"\n== {fn.__name__} ==")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
