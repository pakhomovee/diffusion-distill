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


def t_vp_precision_at_sigma_max():
    """A VP denoiser under bf16 autocast must still resolve x0 at sigma_max.

    `D = x - sigma*eps` is a catastrophic cancellation: both terms are O(sigma),
    the answer is O(sigma_data), so an error `d` in eps arrives in D as
    sigma*d. bf16's ulp is 2^-7, and at CIFAR's sigma_max=157.4 with
    sigma_data=0.5 that is rounding noise of std ~0.26 against a 0.5 signal --
    SNR 2. A one-step student generates at sigma_max on EVERY sample, so it
    cannot emit a clean image at all; the samples come out as recognisable
    structure buried in speckle, and FID lands around 325 instead of single
    digits. This cost a full 6-run CIFAR programme once.

    The oracle returns the EXACT eps for a known target, in the autocast dtype
    (which is what a real network's final conv does), so any error measured
    here is the parameterisation's, not the network's.
    """
    from ddgpu.vp import VPSchedule, VPPrecond
    sch, sd = VPSchedule(), 0.5

    def ac_dtype():
        try:
            return torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else None
        except TypeError:                                   # older torch
            return torch.get_autocast_cpu_dtype() if torch.is_autocast_enabled() else None

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target = None

        def forward(self, u, t, y, **kw):
            sig = sch.sigma_of_t(t.double()).to(torch.float32).reshape(-1, 1, 1, 1)
            out = (u.float() * (1 + sig ** 2).sqrt() - self.target) / sig
            d = ac_dtype()
            return out.to(d) if d is not None else out

    g = torch.Generator().manual_seed(0)
    target = torch.randn(32, 3, 32, 32, generator=g) * sd
    x = torch.randn(32, 3, 32, 32, generator=g) * sch.sigma_max
    s = torch.full((32,), sch.sigma_max)
    y = torch.zeros(32, dtype=torch.long)

    m = VPPrecond(Oracle(), sch, sigma_data=sd)
    m.net.target = target
    with torch.autocast("cpu", torch.bfloat16, enabled=True):
        D = m(x, s, y)
    err = (D.float() - target).std().item()
    check("VP denoiser resolves x0 at sigma_max under bf16 autocast",
          err < 0.02 * sd, f"noise std {err:.4f} vs image std {sd}")

    # The guard must be what is doing it -- otherwise this test passes for the
    # wrong reason on a build where autocast happens not to engage.
    m.FP32_MARGIN = 1e9                                     # disable it
    with torch.autocast("cpu", torch.bfloat16, enabled=True):
        D_bad = m(x, s, y)
    err_bad = (D_bad.float() - target).std().item()
    check("...and without the guard it is genuinely broken",
          err_bad > 0.2 * sd, f"noise std {err_bad:.4f} (SNR {sd / max(err_bad, 1e-9):.2f})")

    # Low sigma must NOT pay for fp32: there is no cancellation there.
    m.FP32_MARGIN = VPPrecond.FP32_MARGIN
    check("low sigma stays in the fast path", not m._fp32_needed(torch.full((4,), 1.0)))
    check("sigma_max takes the fp32 path", m._fp32_needed(s))

    # EDM preconditioning is safe by construction: c_out -> sigma_data, so the
    # network's contribution to D never gets amplified. Guarding it would only
    # cost speed. Pin the property so nobody "fixes" EDM the same way.
    from ddgpu.edm import EDMWrapper
    sig = torch.tensor([0.1, 1.0, 80.0, 157.4, 1000.0])
    e = EDMWrapper.__new__(EDMWrapper)
    e.sigma_data = sd
    _, c_out, _, _ = EDMWrapper._coef(e, sig)
    check("EDM c_out is bounded by sigma_data (no amplification)",
          bool((c_out.abs() <= sd + 1e-6).all()), f"max |c_out| {c_out.abs().max():.4f}")


def t_fid_math():
    """FID must be right, and must survive scipy moving under us.

    `fid_from_feats` called `sqrtm(..., disp=False)`; scipy 1.17 removed that
    argument, so the whole eval crashed -- AFTER sampling 50k images, because
    scoring is the last thing that happens. Testing the API alone would not be
    enough: the version shim also has to leave the number unchanged.

    The exact case: shifting every feature by a constant leaves the covariance
    identical, so `tr(C1 + C2 - 2*sqrtm(C1 C2))` vanishes and FID collapses to
    the squared mean distance. That pins the trace term and the shim at once.
    """
    from ddgpu.eval import fid_from_feats, _sqrtm

    g = torch.Generator().manual_seed(0)
    f1 = torch.randn(256, 16, generator=g)

    check("FID of a set against itself is 0", abs(fid_from_feats(f1, f1)) < 1e-6,
          f"{fid_from_feats(f1, f1):.3e}")

    delta = torch.linspace(0.1, 0.8, 16)
    f2 = f1 + delta                            # same covariance, shifted mean
    want = float((delta ** 2).sum())
    got = fid_from_feats(f1, f2)
    check("FID of a pure mean shift equals the squared mean distance",
          abs(got - want) < 1e-3 * max(want, 1.0), f"{got:.6f} vs {want:.6f}")

    # The matrix square root is the part scipy owns; check it squares back.
    c1 = np.cov(f1.numpy(), rowvar=False)
    c2 = np.cov((f1 * 1.7 + 0.3).numpy(), rowvar=False)
    s = _sqrtm(c1.dot(c2))
    if np.iscomplexobj(s):
        s = s.real
    check("_sqrtm(M) squared reproduces M",
          bool(np.allclose(s.dot(s), c1.dot(c2), atol=1e-6)),
          f"max err {np.abs(s.dot(s) - c1.dot(c2)).max():.2e}")

    # Scaling one set up must increase FID -- catches a sign or factor slip in
    # the trace term that the mean-shift case cannot see.
    near = fid_from_feats(f1, f1 * 1.05)
    far = fid_from_feats(f1, f1 * 1.50)
    check("FID grows with a covariance mismatch", 0 < near < far,
          f"{near:.4f} < {far:.4f}")


def t_torchrun_flag_safety():
    """No flag we hand a torchrun-launched module may abbreviate a torchrun one.

    argparse abbreviation-matches every `--x` on the command line against
    torchrun's own options before the training script's REMAINDER can claim
    them. Two ways that bites, both silent until runtime:

      * ambiguous (`--run` -> --run-path/--run_path) -> torchrun exits with
        "ambiguous option" and the script never starts;
      * unique (`--foo` matching exactly one torchrun option) -> torchrun eats
        the flag and the script silently runs with a default.

    Whether the ambiguous case trips depends on the argparse version, so it
    fails on the GPU box and passes on the laptop. This checks the invariant
    directly, against whatever torch is installed.
    """
    from torch.distributed.run import get_args_parser
    tr = {o for o in get_args_parser()._option_string_actions if o.startswith("--")}

    sh = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scripts", "eval_all.sh")).read()
    m = re.search(r"CMD=\((.*?)\)\n", sh, re.S)
    check("eval_all.sh: found the torchrun command", bool(m))
    if not m:
        return
    block = re.sub(r"#[^\n]*", "", m.group(1))          # strip comments
    check("eval_all.sh launches ddgpu.generate", "ddgpu.generate" in block)

    # Underscores included: torchrun's own flags are spelled --nproc_per_node
    # here, and truncating at the underscore would flag them as abbreviations.
    flags = sorted(set(re.findall(r"(?<![\w-])--[a-z][a-z0-9_-]*", block)))
    bad = {}
    for f in flags:
        if f in tr:                                     # torchrun's own, fine
            continue
        hits = [o for o in tr if o.startswith(f)]
        if hits:
            bad[f] = sorted(hits)
    check("no script flag abbreviates a torchrun option",
          not bad, "; ".join(f"{k} -> {v}" for k, v in bad.items()) or f"checked {flags}")

    # The module must accept the safe spellings, and still accept the old ones
    # for direct `python3 -m ddgpu.generate` use.
    from ddgpu.generate import build_argparser
    safe = build_argparser().parse_args(
        ["--run-dir", "R", "--n-samples", "7", "--ref", "F"])
    old = build_argparser().parse_args(["--run", "R", "--n", "7", "--ref", "F"])
    check("generate accepts --run-dir/--n-samples",
          (safe.run, safe.n) == ("R", 7), f"{safe.run!r} {safe.n}")
    check("old --run/--n still work when invoked directly",
          (old.run, old.n) == ("R", 7), f"{old.run!r} {old.n}")

    # The whole point: what eval_all.sh sends must survive torchrun's parser and
    # arrive at the script intact.
    parsed = get_args_parser().parse_args(
        ["--standalone", "--nproc_per_node=2", "-m", "ddgpu.generate",
         "--run-dir", "R", "--n-samples", "7", "--ref", "F"])
    check("torchrun passes the flags through untouched",
          parsed.training_script_args ==
          ["--run-dir", "R", "--n-samples", "7", "--ref", "F"],
          str(parsed.training_script_args))


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
def t_dataset_dir_errors():
    """A wrong --data path must say which wrong thing it is.

    This trap has bitten twice. Once as `/cifar10`, an unset $DD_DATA_ROOT
    expanding into a plausible absolute path. Once as an HF *download cache*
    passed to --data: the directory exists, so the isdir guard passes, and with
    no meta.json the loader assumed a legacy latent dir and reported a missing
    `train_moments.npy` -- sending the reader after a VAE problem they do not
    have. Both waste the same half hour, and neither raises where the mistake
    was made.

    Legacy latent directories genuinely predate meta.json and must keep
    loading, so the discriminator cannot be meta.json alone; it has to be
    whether any dataset file is present at all.
    """
    from ddgpu.data import build_dataset

    missing = "/definitely/not/here/cifar10"
    try:
        build_dataset(dict(data=missing, shape=[3, 32, 32], n_classes=1))
        check("nonexistent dir is refused", False)
    except FileNotFoundError as e:
        m = str(e)
        check("nonexistent dir names the unset-variable case",
              "does not exist" in m and "DD_DATA_ROOT" in m)

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "blobs"))
        os.makedirs(os.path.join(d, "snapshots"))
        try:
            build_dataset(dict(data=d, shape=[3, 32, 32], n_classes=1))
            check("a non-dataset directory is refused", False)
        except FileNotFoundError as e:
            m = str(e)
            check("existing non-dataset dir says so, not 'no train_moments.npy'",
                  "not a prepared dataset" in m)
            check("it lists what the directory actually holds", "it contains:" in m)
            check("it points at prepare and at the --data shortcut",
                  "ddgpu.prepare pixels" in m and "cifar10-hf" in m)

    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "train_latents.npy"),
                np.zeros((4, 4, 8, 8), np.float32))
        np.save(os.path.join(d, "train_labels.npy"), np.zeros((4,), np.int64))
        ds, _ = build_dataset(dict(data=d, shape=[4, 8, 8], n_classes=1))
        check("legacy latent dirs (no meta.json) still load", len(ds) == 4,
              f"{len(ds)} items")


# --------------------------------------------------------------------------
def t_cifar_mirror():
    """The HF CIFAR mirror must present EXACTLY as torchvision presents.

    `--source cifar10-hf` exists because cs.toronto.edu throttles cloud
    notebooks to ~100 kB/s. It is only safe because the two sources are
    interchangeable: same images, and the same `[-1,1]` presentation, so a
    reference built from either gives identical Inception features.

    The image SET being identical was verified once against the canonical
    tarball (md5 c58f30108f718f92721af3b95e74349a) and cannot be re-checked
    without downloading 170 MB. The row order is NOT the same, which leaves FID
    over the full 50 000 untouched (permutation-invariant) but does re-roll
    which images a sub-sampled reference or precision/recall sees -- see
    `HFParquetImages`.

    What IS cheap, and what actually rots, is the presentation arithmetic: if
    someone edits `TorchvisionImages.__getitem__` and not
    `HFParquetImages.__getitem__`, every FID computed on a mirror-built
    reference shifts, and nothing raises. That is what this pins.
    """
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except ImportError:
        print("  SKIP  pyarrow not installed (optional: --source cifar10-hf only)")
        return

    import io as _io, types
    from PIL import Image
    from ddgpu.prepare import HFParquetImages, TorchvisionImages, image_source

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    blobs = []
    for a in raw:
        buf = _io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")      # lossless, like the mirror
        blobs.append(buf.getvalue())
    tbl = pa.table({"img": pa.array([{"bytes": b, "path": None} for b in blobs],
                                    type=pa.struct([("bytes", pa.binary()),
                                                    ("path", pa.string())])),
                    "label": pa.array(list(range(8)), pa.int64())})

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "train.parquet")
        pq.write_table(tbl, path)
        stub = types.ModuleType("huggingface_hub")
        stub.hf_hub_download = lambda *a, **k: path
        saved = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = stub
        try:
            ds = image_source("cifar10-hf", 32)
            check("image_source dispatches cifar10-hf to the mirror",
                  isinstance(ds, HFParquetImages), type(ds).__name__)
            check("mirror name does not shadow the torchvision one",
                  "cifar10-hf" not in TorchvisionImages.SETS
                  and "cifar10" not in HFParquetImages.SETS)
            check("length and labels survive the round trip",
                  len(ds) == 8 and ds.labels == list(range(8)),
                  f"{len(ds)} {ds.labels}")

            # The invariant: identical to TorchvisionImages' own arithmetic.
            same = all(
                torch.equal(
                    ds[i][0],
                    torch.from_numpy(raw[i]).permute(2, 0, 1).float() / 127.5 - 1.0)
                for i in range(8))
            check("presentation matches TorchvisionImages bit-for-bit", same)
            check("PNG decode is lossless (no pixel drift through the mirror)",
                  np.array_equal(
                      np.array(Image.open(_io.BytesIO(blobs[3])).convert("RGB")),
                      raw[3]))
            check("labels come back as ints, matching the torchvision contract",
                  isinstance(ds[0][1], int) and ds.samples[2] == (None, 2),
                  f"{ds[0][1]!r} {ds.samples[2]}")

            big = HFParquetImages("cifar10-hf", 64)
            check("resolution resize path works", big[0][0].shape == (3, 64, 64),
                  str(tuple(big[0][0].shape)))
        finally:
            if saved is None:
                del sys.modules["huggingface_hub"]
            else:
                sys.modules["huggingface_hub"] = saved


# --------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (t_pos_embed_matches_official, t_vp_schedule_roundtrip,
               t_vp_precond_exact_on_gaussian, t_student_grid, t_sigma_sampler,
               t_checkpoint_remap, t_dataset_moments, t_config_delta, t_ema,
               t_lambda_probe, t_vp_precision_at_sigma_max, t_fid_math,
               t_torchrun_flag_safety, t_pixel_eval_path, t_cifar_mirror,
               t_dataset_dir_errors):
        print(f"\n== {fn.__name__} ==")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
