"""Invariants for the cheap tier: the interpolant family and the teacher zoo.

Same rule as the other test files -- each test guards a specific way a wrong
number could look plausible all the way into a plot. Four preconditioning
families now share one trainer, and a mismatch between any of them and its
checkpoint produces no exception (LOG.log ENTRY 012, FINDING 19). These are the
checks that would notice.

Deliberately tiny: this box has ~1 GB of usable RAM, so nothing here builds a
real backbone.

  python3 tests/test_teachers.py
"""
import json, math, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

torch.set_num_threads(1)
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------------
def t_interpolant_map():
    """sigma <-> t for the linear path must match REPA's OWN formula.

    REPA/loss.py's lognormal weighting computes `time_input = sigma / (1 + sigma)`
    for path_type='linear'. That is the map this whole family rests on, so it is
    checked against their expression rather than against my derivation of it."""
    from ddgpu.interpolant import LinearInterpolant
    p = LinearInterpolant()
    sig = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0])
    t_repa = sig / (1 + sig)                       # REPA/loss.py, verbatim
    err = (p.t_of_sigma(sig) - t_repa).abs().max().item()
    check("t(sigma) == REPA's sigma/(1+sigma)", err < 1e-6, f"max|err| {err:.2e}")
    back = p.sigma_of_t(p.t_of_sigma(sig))
    rel = ((back - sig).abs() / sig).max().item()
    check("sigma -> t -> sigma round trip", rel < 1e-4, f"max rel {rel:.2e}")


def t_interpolant_exact_on_gaussian():
    """With an ORACLE velocity field, InterpolantPrecond.score must equal the
    analytic score of a Gaussian target.

    Exercises c_in, t_of_sigma and the v -> score conversion together. Getting
    the sign of v wrong, or confusing t with 1-t, fails here and nowhere else --
    in training it would look like a teacher that simply does not help.
    """
    from ddgpu.interpolant import InterpolantPrecond, LinearInterpolant
    sd, path = 0.7, LinearInterpolant()

    class OracleV(torch.nn.Module):
        """E[eps - x0 | x_t] for x0 ~ N(0, sd^2 I).

        x_t = (1-t)x0 + t*eps is Gaussian with variance (1-t)^2 sd^2 + t^2, and
        both posteriors are linear in x_t, so the velocity is available in
        closed form."""
        def forward(self, u, t, y=None, **kw):
            t = t.reshape(-1, 1, 1, 1).to(u.dtype)
            var = (1 - t) ** 2 * sd ** 2 + t ** 2
            e_x0 = (1 - t) * sd ** 2 / var * u        # u IS x_t here (c_in applied)
            e_eps = t / var * u
            return e_eps - e_x0

    m = InterpolantPrecond(OracleV(), path, out_ch=4, tuple_out=False)
    x = torch.randn(16, 4, 4, 4)
    for s in (0.02, 0.3, 1.0, 7.0, 50.0):
        sig = torch.full((16,), s)
        got = m.score(x, sig, None)
        want = -x / (sd ** 2 + s ** 2)
        rel = ((got - want).norm() / want.norm()).item()
        check(f"interpolant score == analytic (sigma={s})", rel < 1e-4, f"rel {rel:.2e}")

    # The two limits that catch sign errors, asserted rather than reasoned about.
    tiny = torch.full((16,), 1e-3)
    check("D -> x as sigma -> 0",
          ((m(x, tiny, None) - x).norm() / x.norm()).item() < 1e-3)
    huge = torch.full((16,), 1e3)
    xh = torch.randn(16, 4, 4, 4) * 1e3
    check("D -> E[x0] = 0 as sigma -> inf",
          (m(xh, huge, None).norm() / xh.norm()).item() < 1e-2)


def t_student_grid_signature():
    """Both schedules must accept `student_sigmas(n, sigma_max=)`; the trainer
    and the sampler call whichever they were handed without branching."""
    from ddgpu.vp import VPSchedule
    from ddgpu.interpolant import LinearInterpolant
    for name, sch, smax in (("VPSchedule", VPSchedule(), 157.4),
                            ("LinearInterpolant", LinearInterpolant(), 49.0)):
        g = sch.student_sigmas(4, sigma_max=smax)
        check(f"{name}.student_sigmas(4, sigma_max=) -> 5 values", len(g) == 5)
        check(f"{name} grid starts at ~sigma_max",
              abs(float(g[0]) - smax) / smax < 0.02, f"{float(g[0]):.2f}")
        check(f"{name} grid ends at 0", float(g[-1]) == 0.0)
        check(f"{name} grid is decreasing", bool((g[:-1] > g[1:]).all()))


# --------------------------------------------------------------------------
def t_gaussian_teacher():
    from ddgpu.teachers import load_teacher
    m, meta = load_teacher("synthetic:c=4,hw=8,sd=0.5,bias=0.0")
    check("synthetic teacher dims", meta["dims"] == 256, str(meta["dims"]))
    x = torch.randn(8, 4, 8, 8)
    sig = torch.full((8,), 0.9)
    err = (m.score(x, sig) - m.true_score(x, sig)).abs().max().item()
    check("unbiased GaussianTeacher == true score", err < 1e-6, f"{err:.2e}")
    mb, _ = load_teacher("synthetic:c=4,hw=8,sd=0.5,bias=0.25")
    r = (mb.score(x, sig) / mb.true_score(x, sig)).mean().item()
    check("bias scales the score as declared", abs(r - 1.25) < 1e-5, f"{r:.4f}")


def t_validate_teacher_catches_miswrapping():
    """`validate_teacher` must pass a correct wrapper and FLAG a broken one.

    The broken case here is the realistic one: a noise-conditioning scale that
    is off by 1000, which is exactly the DiT-vs-SiT timestep convention hazard.
    A teacher evaluated at the wrong noise level produces no exception, so this
    check is the only thing standing between that bug and a week of runs.
    """
    from ddgpu.teachers import load_teacher, validate_teacher
    x0 = torch.randn(64, 4, 8, 8) * 0.5
    good, _ = load_teacher("synthetic:c=4,hw=8,sd=0.5,bias=0.0")
    v = validate_teacher(good, x0, sigmas=(0.01, 0.1, 0.5, 2.0, 20.0))
    check("correct wrapper -> OK", v["verdict"] == "OK", json.dumps(v)[:90])

    class Miswrapped(torch.nn.Module):
        """Same teacher, but told the wrong sigma -- the x1000 convention bug."""
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x, sigma, y=None, **kw):
            return self.inner(x, torch.as_tensor(sigma) * 1000.0, y, **kw)

    bad = validate_teacher(Miswrapped(good), x0, sigmas=(0.01, 0.1, 0.5, 2.0, 20.0))
    check("x1000 noise-scale bug -> SUSPECT", bad["verdict"] == "SUSPECT",
          f"identity_err={bad['identity_err']:.3g}")


def t_teacher_spec_errors():
    from ddgpu.teachers import load_teacher
    for spec, why in (("google/ddpm-cifar10-32", "no family prefix"),
                      ("nope:whatever", "unknown family")):
        try:
            load_teacher(spec)
            check(f"rejects {why}", False, "loaded anyway")
        except ValueError:
            check(f"rejects {why}", True)


# --------------------------------------------------------------------------
def t_dsm_identity():
    """THE theoretical claim, checked on real tensors.

    `E||lam*A + (1-lam)*B - g||^2 = MSE(lam) + const(sigma)` with const
    independent of lam. If that holds, the argmin of the held-out DENOISING loss
    (computable, no ground truth) equals the argmin of the true score MSE
    (not computable in general). Everything `exp/10_lambda_real.py` reports as
    falsifiable rests on it, so it is verified against a Gaussian target where
    the true score IS available.
    """
    from ddgpu.teachers import load_teacher
    from ddgpu.robust import dsm_lambda_terms, empirical_score
    torch.manual_seed(0)
    sd, n = 0.5, 512
    teacher, _ = load_teacher(f"synthetic:c=4,hw=4,sd={sd},bias=0.3")
    lam_grid = torch.linspace(0, 1, 41)
    for s in (0.4, 1.5):
        dsm_loss = torch.zeros(41, dtype=torch.float64)
        true_mse = torch.zeros(41, dtype=torch.float64)
        for _ in range(24):
            x0 = torch.randn(2 * n, 4, 4, 4) * sd
            y = torch.zeros(2 * n, dtype=torch.long)
            sig = torch.full((2 * n,), s)
            t = dsm_lambda_terms(teacher, x0, y, sig, lam_grid=lam_grid)
            dsm_loss += t["loss"].sum(-1).double()
            # the same A, B, evaluated against the TRUE score
            m = n
            eps = torch.randn_like(x0[:m])
            xt = x0[:m] + s * eps
            A = teacher.score(xt, sig[:m])
            B = empirical_score(xt, x0[m:], sig[:m])
            S = teacher.true_score(xt, sig[:m])
            f = lambda z: z.reshape(z.shape[0], -1)
            fA, fB, fS = f(A), f(B), f(S)
            true_mse += torch.stack([((l * fA + (1 - l) * fB - fS) ** 2).sum(-1).sum()
                                     for l in lam_grid]).double()
        a1 = float(lam_grid[int(dsm_loss.argmin())])
        a2 = float(lam_grid[int(true_mse.argmin())])
        check(f"argmin(held-out DSM loss) == argmin(true MSE) at sigma={s}",
              abs(a1 - a2) <= 0.051, f"dsm {a1:.3f} vs true {a2:.3f}")


def t_offline_matches_online():
    """The figure and the trained lambda must come from ONE implementation.

    `exp/10_lambda_real.py` and `LambdaEstimator.calibrate` both call
    `dsm_lambda_terms`. This pins that: a second, drifting copy would let us
    publish a curve that does not describe the run it is attached to.
    """
    from ddgpu.teachers import load_teacher
    from ddgpu.robust import dsm_lambda_terms, LambdaEstimator
    teacher, _ = load_teacher("synthetic:c=4,hw=4,sd=0.5,bias=0.2")
    x0 = torch.randn(256, 4, 4, 4) * 0.5
    y = torch.zeros(256, dtype=torch.long)
    sig = torch.full((256,), 1.0)
    torch.manual_seed(7)
    t = dsm_lambda_terms(teacher, x0, y, sig)
    lam_direct = float(t["num"].sum() / t["den"].sum())
    est = LambdaEstimator(n_bins=8, ema=0.0, min_count=0)
    torch.manual_seed(7)
    est.calibrate(teacher, x0, y, sig)
    lam_est = float(est.lam(torch.tensor([1.0]))[0])
    check("offline lambda == LambdaEstimator.calibrate",
          abs(lam_direct - lam_est) < 1e-4, f"{lam_direct:.6f} vs {lam_est:.6f}")


# --------------------------------------------------------------------------
def t_pixel_dataset():
    from ddgpu.data import PixelDataset, build_dataset
    with tempfile.TemporaryDirectory() as td:
        n, hw = 32, 8
        px = np.random.randint(0, 256, (n, 3, hw, hw), dtype=np.uint8)
        np.save(f"{td}/train_pixels.npy", px)
        np.save(f"{td}/train_labels.npy", np.arange(n, dtype=np.int32) % 10)
        json.dump(dict(n=n, resolution=hw, shape=[3, hw, hw], n_classes=10,
                       sigma_data=0.5, space="pixel", format="pixels",
                       latent_scale=1.0), open(f"{td}/meta.json", "w"))
        ds = PixelDataset(td)
        x, y = ds[5]
        check("pixels: shape", tuple(x.shape) == (3, hw, hw))
        check("pixels: served in [-1,1]", float(x.min()) >= -1.0 and float(x.max()) <= 1.0)
        exact = torch.from_numpy(px[5].astype(np.float32)) / 127.5 - 1.0
        check("pixels: lossless round trip", torch.equal(x, exact))
        _, res = build_dataset(dict(data=td, shape=[3, hw, hw], n_classes=10))
        check("build_dataset dispatches to pixels",
              res["space"] == "pixel" and res["sigma_data"] == 0.5, str(res))


def t_gaussian_data():
    from ddgpu.data import build_dataset
    ds, res = build_dataset(dict(data="gaussian:c=4,hw=4,sd=0.5,n=1024"))
    x = torch.stack([ds[i][0] for i in range(1024)])
    check("gaussian data has the declared sd", abs(float(x.std()) - 0.5) < 0.02,
          f"{float(x.std()):.4f}")
    check("gaussian meta carries sigma_data", res["sigma_data"] == 0.5)


def t_conv_gan_head():
    """The cheap tier's discriminator must accept the shapes it will be given."""
    from ddgpu.gan import ConvGANHead, d_loss, g_loss
    h = ConvGANHead(3)
    x = torch.randn(4, 3, 32, 32)
    s = torch.rand(4) * 2 + 0.1
    out = h(x, s)
    check("ConvGANHead -> (B,) logit", tuple(out.shape) == (4,), str(tuple(out.shape)))
    check("losses are finite",
          torch.isfinite(d_loss(out, out)) and torch.isfinite(g_loss(out)))
    # Softplus, matching DMD2 -- and BOUNDED, unlike the -logit.mean() it
    # replaced. An unbounded generator reward is a plausible cause of the
    # degenerate samples in DMD2_DIFF.md; pin that it saturates.
    z = torch.zeros(4)
    check("d_loss at zero logits is 2*log(2)",
          abs(float(d_loss(z, z)) - 2 * math.log(2)) < 1e-5, f"{float(d_loss(z,z)):.4f}")
    check("g_loss at zero logits is log(2)",
          abs(float(g_loss(z)) - math.log(2)) < 1e-5, f"{float(g_loss(z)):.4f}")
    check("g_loss saturates once the discriminator is fooled",
          float(g_loss(torch.full((4,), 20.0))) < 1e-6,
          f"{float(g_loss(torch.full((4,), 20.0))):.3e}")
    check("g_loss is bounded below by 0 (the old -logit.mean() was not)",
          float(g_loss(torch.full((4,), 1e3))) >= 0.0)


# --------------------------------------------------------------------------
def t_unet_trunk():
    """`DiffusersUNetAdapter.trunk` must return the UNet's OWN mid-block features.

    DMD2's discriminator is a head on the critic's features, not a separate
    network. `trunk()` makes that available on a diffusers backbone by replaying
    `UNet2DModel.forward` steps 0-4 and stopping at the bottleneck -- which
    couples us to diffusers' layout. A release that reorders those steps would
    not raise; it would feed the discriminator the wrong tensor and show up as
    "the GAN term does not help", which is unfalsifiable from the outside.

    So this captures the mid-block activation from a REAL `unet(...)` call with
    a hook and demands `trunk()` reproduce it exactly. LOG ENTRY 015 is why it
    matters: the standalone ConvGANHead that stood in for this collapsed the
    baseline it was meant to represent.
    """
    try:
        from diffusers import UNet2DModel
    except ImportError:
        print("  SKIP  diffusers not installed")
        return
    from ddgpu.teachers import DiffusersUNetAdapter
    from ddgpu.gan import GANHead, d_loss, g_loss

    torch.manual_seed(0)
    # norm_num_groups=4, not the default 32: this box has ~1 GB of usable RAM
    # and GroupNorm would otherwise force block_out_channels up to 32+.
    unet = UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
                       layers_per_block=1, block_out_channels=(8, 16),
                       norm_num_groups=4, attention_head_dim=8,
                       down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                       up_block_types=("AttnUpBlock2D", "UpBlock2D")).eval()
    ad = DiffusersUNetAdapter(unet, class_conditional=False)

    tok_dim, cond_dim = ad.trunk_dims
    check("trunk_dims reports mid channels and time-embedding width",
          tok_dim == unet.config.block_out_channels[-1]
          and cond_dim == unet.time_embedding.linear_2.out_features,
          f"{tok_dim}, {cond_dim}")

    x = torch.randn(2, 3, 8, 8)
    t = torch.tensor([13.0, 700.0])

    grabbed = {}
    h = unet.mid_block.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("mid", o.detach().clone()))
    h2 = unet.time_embedding.register_forward_hook(
        lambda m, i, o: grabbed.__setitem__("emb", o.detach().clone()))
    with torch.no_grad():
        unet(x, t)
    h.remove(); h2.remove()

    with torch.no_grad():
        tok, cond = ad.trunk(x, t)

    want = grabbed["mid"].flatten(2).transpose(1, 2)
    check("trunk tokens == the real forward's mid-block activation",
          tok.shape == want.shape and torch.allclose(tok, want, atol=1e-6),
          f"{tuple(tok.shape)} vs {tuple(want.shape)}, "
          f"max|d| {(tok - want).abs().max().item():.2e}")
    check("trunk conditioning == the real forward's time embedding",
          torch.allclose(cond, grabbed["emb"], atol=1e-6),
          f"max|d| {(cond - grabbed['emb']).abs().max().item():.2e}")
    check("token width matches trunk_dims", tok.shape[-1] == tok_dim)
    check("cond width matches trunk_dims", cond.shape[-1] == cond_dim)

    # Features must actually depend on the input -- a constant would pass the
    # shape checks and make the discriminator useless.
    with torch.no_grad():
        tok2, _ = ad.trunk(torch.randn(2, 3, 8, 8), t)
    check("trunk features vary with the input",
          (tok - tok2).abs().max().item() > 1e-4)

    # And the head must accept the two different widths.
    head = GANHead(tok_dim, cond_dim)
    with torch.no_grad():
        logit = head(tok, cond)
    check("GANHead on UNet features -> (B,) finite logit",
          logit.shape == (2,) and bool(torch.isfinite(logit).all()),
          str(tuple(logit.shape)))
    check("hinge/NS losses finite on it",
          bool(torch.isfinite(d_loss(logit, logit)) and torch.isfinite(g_loss(logit))))

    # The DiT path must be unchanged: equal widths, single-argument GANHead.
    from ddgpu.dit import make_dit
    dit = make_dit("DiT-T/2", input_size=8, in_ch=3, n_classes=2)
    td, cd = dit.trunk_dims
    check("DiT trunk_dims are equal widths (the old assumption)", td == cd,
          f"{td}, {cd}")
    dtok, dcond = dit.trunk(torch.randn(2, 3, 8, 8), torch.tensor([1.0, 2.0]),
                            torch.zeros(2, dtype=torch.long))
    check("GANHead(hidden) still works on DiT features",
          GANHead(td)(dtok, dcond).shape == (2,))


# --------------------------------------------------------------------------
def t_gan_param_groups():
    """The GAN head must get its own learning rate, and its own loss weight.

    Two separate defects, neither visible in a loss curve:

    * ONE learning rate for the critic and the head. The critic is 35.7M
      teacher-initialised parameters that need fine-tuning; the head is 0.33M
      RANDOM parameters that need to learn. At a shared 1e-5 the head stays at
      its initialisation -- measured on an A100, hinge d_loss is 1.0 at zero
      logits and after 300 steps had reached only 0.938.
    * `gan_weight` scaled the discriminator's OWN objective as well as its
      influence on the generator. Those are unrelated jobs, and a weight small
      enough to be safe for the generator attenuates the head's own gradient by
      the same factor -- 1000x at 1e-3 -- so the term cannot work at any setting
      that is also safe. gan_d_weight separates them.
    """
    try:
        from diffusers import UNet2DModel
    except ImportError:
        print("  SKIP  diffusers not installed")
        return
    from ddgpu.teachers import DiffusersUNetAdapter
    from ddgpu.vp import VPSchedule, VPPrecond
    from ddgpu.dmd2 import DMD2Trainer

    torch.manual_seed(0)

    def mk():
        u = UNet2DModel(sample_size=8, in_channels=3, out_channels=3,
                        layers_per_block=1, block_out_channels=(8, 16),
                        norm_num_groups=4, attention_head_dim=8,
                        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                        up_block_types=("AttnUpBlock2D", "UpBlock2D"))
        sch = VPSchedule(1000)
        return VPPrecond(DiffusersUNetAdapter(u), sch, out_ch=3, sigma_data=0.5), sch

    base = dict(sigma_data=0.5, shape=[3, 8, 8], n_classes=1, sigma_max=40.0,
                n_student_steps=1, sigma_dist="vp_uniform_t", t_min=20, t_max=979,
                lr_g=1e-5, lr_d=1e-5, clip=1.0, d_steps=1, mode="teacher",
                cfg_scale=1.0, track="A", ema_decay=0.999, gather_real=False,
                log_every=1)

    def trainer(**kw):
        G, sch = mk(); mu, _ = mk(); T, _ = mk()
        for q in T.parameters():
            q.requires_grad_(False)
        return DMD2Trainer(G, mu, T, dict(base, **kw), device="cpu", schedule=sch), mu

    tr, mu = trainer(gan_weight=1e-3, lr_gan=1e-4)
    gs = tr.opt_D.param_groups
    check("opt_D has a group per job", len(gs) == 2, f"{len(gs)} groups")
    head = {id(q) for q in tr.gan.parameters()}
    crit = {id(q) for q in mu.parameters()}
    by_lr = {g["lr"]: {id(q) for q in g["params"]} for g in gs}
    check("head group is at lr_gan", by_lr.get(1e-4) == head,
          f"lrs {sorted(by_lr)}")
    check("critic group is at lr_d", by_lr.get(1e-5) == crit)
    check("the two groups are disjoint", not (head & crit))
    check("every head parameter is optimised",
          head <= set().union(*by_lr.values()))

    tr2, _ = trainer(gan_weight=1e-3)                    # lr_gan omitted
    check("lr_gan defaults to lr_d", tr2.lr_gan == base["lr_d"], str(tr2.lr_gan))

    tr3, _ = trainer(gan_weight=0.0)
    check("no discriminator -> a single group",
          len(tr3.opt_D.param_groups) == 1 and tr3.gan is None)

    # gan_d_weight must drive the critic-side loss, independently of gan_weight.
    x = torch.randn(4, 3, 8, 8) * 0.5
    y = torch.zeros(4, dtype=torch.long)
    lo, _ = trainer(gan_weight=1e-3, gan_d_weight=0.0)
    hi, _ = trainer(gan_weight=1e-3, gan_d_weight=1.0)
    a = lo.step(x, y)["gan_pull_critic"]
    b = hi.step(x, y)["gan_pull_critic"]
    check("gan_d_weight=0 removes the GAN from the critic update", a == 0.0,
          f"{a:.3e}")
    check("gan_d_weight=1 restores it", b > 0.0, f"{b:.3e}")
    check("and it is NOT gan_weight that controls this",
          b > 100 * max(a, 1e-30), f"{a:.3e} vs {b:.3e}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    for fn in (t_interpolant_map, t_interpolant_exact_on_gaussian,
               t_student_grid_signature, t_gaussian_teacher,
               t_validate_teacher_catches_miswrapping, t_teacher_spec_errors,
               t_dsm_identity, t_offline_matches_online,
               t_pixel_dataset, t_gaussian_data, t_conv_gan_head,
               t_unet_trunk, t_gan_param_groups):
        print(f"\n== {fn.__name__} ==")
        fn()
    print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    sys.exit(1 if FAIL else 0)
