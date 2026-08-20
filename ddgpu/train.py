"""Unified entrypoint for all tracks.

  torchrun --nproc_per_node=8 -m ddgpu.train --config configs/dit_xl_256_robust.json
  python -m ddgpu.train --config configs/smokeA.json          # single process

Config resolution order, weakest first:
  1. the JSON file,
  2. the dataset's `meta.json`   (sigma_data, shape, latent_size, n_classes),
  3. the teacher checkpoint      (arch, latent_size, learn_sigma, sigma_max),
  4. `--override key=json` on the command line.

Steps 2 and 3 exist because those values are *facts about the artefacts*, not
choices: a config that hardcodes `sigma_data: 0.5` against unit-variance latents
or `sigma_max: 80` against a schedule that tops out at 157 is silently wrong in
a way that looks like a bad hyperparameter. The resolved config is written to
`<out>/config.resolved.json` so every run records what it actually ran with.
"""
import argparse, copy, json, os, time
import torch
import torch.distributed as dist

from .dit import make_dit
from .edm import EDMWrapper
from .vp import VPSchedule, VPPrecond
from .data import build_dataset
from .dmd2 import DMD2Trainer
from .invertible import InvertibleTrainer, AnchorCache


def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r % torch.cuda.device_count())
    return r, w, torch.device("cuda", r % torch.cuda.device_count())


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def wrap_precond(net, c, schedule):
    """Attach the preconditioning the config asks for.

    'vp'  -- the teacher is a released DDPM-style eps-predictor on discrete
             timesteps. Student and critic use the SAME wrapper, which is what
             makes `init_from_teacher` a genuine warm start rather than a
             re-interpretation of the weights under a different input scaling.
    'edm' -- Karras preconditioning, for teachers we train ourselves.
    """
    if c.get("precond", "edm") == "vp":
        return VPPrecond(net, schedule, out_ch=c["shape"][0],
                         sigma_data=c["sigma_data"])
    return EDMWrapper(net, c["sigma_data"])


def build_model(c, device, schedule, precond=True, grad_ckpt=True):
    net = make_dit(c["arch"], input_size=c["latent_size"], in_ch=c["shape"][0],
                   n_classes=c["n_classes"], grad_ckpt=grad_ckpt,
                   learn_sigma=c.get("learn_sigma", False))
    net = net.to(device)
    return wrap_precond(net, c, schedule) if precond else net


def load_teacher(c, device, schedule, log):
    """Frozen teacher, plus whatever the checkpoint tells us about the config.

    Preferred path is `c["teacher"] = "<family>:<path>"`, which goes through
    `ddgpu.teachers` and covers all four preconditioning families. The legacy
    `teacher_ckpt` + `teacher_format` path is kept for the DiT configs written
    before the registry existed.
    """
    if c.get("teacher"):
        from .teachers import load_teacher as _load
        kw = {k: c[k] for k in ("repa_dir", "edm_repo", "weights") if c.get(k)}
        m, meta = _load(c["teacher"], device=device, sigma_data=c["sigma_data"], **kw)
        log("teacher: " + json.dumps(meta, default=str))
        res = dict(shape=meta["shape"], n_classes=meta["n_classes"],
                   teacher_family=meta["family"], space=meta["space"],
                   teacher_sigma_max=meta["sigma_max"],
                   teacher_sigma_min=meta["sigma_min"],
                   latent_size=meta["shape"][-1])
        # A teacher that cannot do classifier-free guidance must not be asked
        # to: silently ignoring cfg_scale would make an unconditional model look
        # like a guided one in the logs.
        if not meta.get("cfg_capable", True) and c.get("cfg_scale", 1.0) != 1.0:
            raise ValueError(
                f"{c['teacher']} is unconditional but cfg_scale={c['cfg_scale']}. "
                "Set cfg_scale=1.0.")
        return m.eval(), res
    resolved = {}
    path = c.get("teacher_ckpt")
    if path and c.get("teacher_format", "official") == "official":
        from .ckpt import load_official_dit
        net, info = load_official_dit(path, n_classes=c["n_classes"],
                                      in_ch=c["shape"][0], grad_ckpt=False)
        log(f"teacher: {json.dumps({k: v for k, v in info.items() if k != 'missing'})}")
        resolved.update(arch=info["arch"], latent_size=info["latent_size"],
                        learn_sigma=info["learn_sigma"])
        c = {**c, **resolved}
        return wrap_precond(net.to(device), c, schedule).eval(), resolved
    net = build_model(c, device, schedule, precond=True, grad_ckpt=False)
    if path:
        net.net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval(), resolved


def wrap_ddp(m, rank, world, static_graph=False):
    """static_graph=True is REQUIRED whenever a module is invoked more than once
    per backward (Track B calls G and E on several legs). Without it DDP's
    reducer raises "Expected to mark a variable ready only once"."""
    if world == 1:
        return m
    # broadcast_buffers=False is not an optimisation. Our only buffer is the
    # deterministic sin-cos positional grid, identical on every rank, but DDP's
    # default buffer broadcast is a COLLECTIVE inside forward -- which makes any
    # rank-0-only diagnostic call (track B's `diagnose`) hang the other ranks.
    return torch.nn.parallel.DistributedDataParallel(
        m, device_ids=[rank % torch.cuda.device_count()],
        find_unused_parameters=False, gradient_as_bucket_view=True,
        broadcast_buffers=False, static_graph=static_graph)


def _teacher_schedule(teacher):
    """The noise schedule a registry teacher was trained on, if it has one.

    VPPrecond carries a `VPSchedule`, InterpolantPrecond a `LinearInterpolant`;
    both expose `sigma_of_t` / `student_sigmas` / `sigma_max`. EDM checkpoints
    carry none -- their sigma is already the free variable -- so they fall back
    to the EDM lognormal and an EDM rho=7 sampler grid.
    """
    return getattr(teacher, "sch", None) or getattr(teacher, "path", None)


def clone_trainable(teacher):
    """A trainable copy of the frozen teacher, same class and same weights."""
    m = copy.deepcopy(teacher).train()
    for p in m.parameters():
        p.requires_grad_(True)
    return m


class EMA:
    """Exponential moving average of the student, kept on the training device.

    Reported FID comes from these weights. Distillation students are trained
    with a very small LR against a noisy distribution-matching gradient, so the
    raw weights fluctuate on the same scale as the effect being measured; every
    DMD/DMD2 number in the literature is an EMA number and comparing raw weights
    against them would be comparing different things.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(_raw_module(model)).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, p in zip(self.shadow.state_dict().values(),
                        _raw_module(model).state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from ('auto' = latest in out/)")
    a = ap.parse_args()
    c = json.load(open(a.config))

    rank, world, dev = setup_dist()
    torch.manual_seed(c["seed"] + rank)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    # ---- data (resolves sigma_data / shape / n_classes) ----
    ds, from_data = build_dataset(c)
    if from_data:
        log(f"config from dataset meta.json: {from_data}")
        c.update(from_data)

    schedule = VPSchedule(c.get("n_timestep", 1000)).to(dev) \
        if c.get("precond", "edm") == "vp" else None

    # ---- teacher (resolves arch / latent_size / learn_sigma) ----
    teacher, from_ckpt = load_teacher(c, dev, schedule, log)
    c.update(from_ckpt)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Optional but strongly recommended: prove the teacher is wrapped correctly
    # before spending a run on it. LOG.log ENTRY 012 FINDING 19 is what happens
    # when a preconditioning mismatch goes unchecked -- no exception, just a
    # teacher that returns noise.
    if c.get("validate_teacher", True) and rank == 0:
        from .teachers import validate_teacher
        xb = torch.stack([ds[i][0] for i in range(min(64, len(ds)))]).to(dev)
        yb = torch.tensor([ds[i][1] for i in range(min(64, len(ds)))]).to(dev)
        v = validate_teacher(teacher, xb, yb, device=dev)
        log("TEACHER-CHECK " + json.dumps(v))
        if v["verdict"] != "OK" and c.get("strict_teacher", True):
            raise SystemExit(
                "teacher validation SUSPECT -- the denoising loss does not behave "
                "like a correctly-wrapped denoiser (see ddgpu/teachers.py:"
                "validate_teacher for how to read these). Set strict_teacher=false "
                "to override deliberately.")

    # The student generates from sigma_max. For a VP teacher that is the top of
    # its own schedule (157.4 for linear-1000), NOT EDM's 80: starting a
    # one-step student half way up a schedule it was initialised from is a
    # silent quality loss with no error message.
    # A registry teacher carries its own noise schedule; use it for both the
    # sampler grid and the training noise distribution rather than a config
    # constant. `sigma_max` is where a one-step student starts, so getting it
    # from anywhere other than the teacher is guesswork.
    sch = _teacher_schedule(teacher) or schedule
    if c.get("sigma_max") in (None, "schedule"):
        c["sigma_max"] = c.get("teacher_sigma_max") or (
            sch.sigma_max if sch else 80.0)
        # An interpolant teacher's nominal sigma_max is 1e4 (t -> 1), which no
        # sampler should start from; cap it at the trimmed training window.
        if c.get("teacher_family") == "interpolant":
            hi = float(c.get("t_hi", 0.98))
            c["sigma_max"] = hi / (1.0 - hi)
        log(f"sigma_max resolved from the teacher: {c['sigma_max']:.3f}")
    schedule = sch

    for kv in a.override:                       # CLI wins over everything
        k, v = kv.split("=", 1)
        c[k] = json.loads(v)
    os.makedirs(c["out"], exist_ok=True)
    if rank == 0:
        json.dump(c, open(f"{c['out']}/config.resolved.json", "w"), indent=1)
    log(json.dumps(c, indent=1))

    samp = torch.utils.data.distributed.DistributedSampler(ds) if world > 1 else None
    dl = torch.utils.data.DataLoader(ds, c["micro_batch"], shuffle=(samp is None),
                                     sampler=samp, num_workers=c.get("workers", 4),
                                     drop_last=True, pin_memory=(dev.type == "cuda"),
                                     persistent_workers=c.get("workers", 4) > 0)

    # ---- models ----
    # With a registry teacher the student and critic are CLONES of it. That is
    # not a convenience: it makes `init_from_teacher` exact by construction for
    # every backbone family, with no architecture config to get wrong and no
    # state-dict remap to drift. The DiT path keeps rebuilding from config
    # because RUNPLAN's memory model needs our own DiT.
    clone = bool(c.get("teacher"))
    student = clone_trainable(teacher) if clone else build_model(c, dev, schedule)
    if not clone and c.get("teacher_ckpt") and c.get("init_from_teacher", True):
        student.net.load_state_dict(teacher.net.state_dict())

    if c["track"] == "A":
        critic = clone_trainable(teacher) if clone else build_model(c, dev, schedule)
        if not clone and c.get("teacher_ckpt"):
            critic.net.load_state_dict(teacher.net.state_dict())
        tr = DMD2Trainer(wrap_ddp(student, rank, world), wrap_ddp(critic, rank, world),
                         teacher, c, device=dev, schedule=schedule)
    else:
        enc = build_model(c, dev, schedule, precond=False)   # raw DiT, see invertible.py
        anchors = None
        if c.get("anchor_path") and os.path.exists(c["anchor_path"]):
            anchors = AnchorCache(c["anchor_path"], device=dev)
        elif c.get("w_anchor", 0) > 0:
            log("building teacher anchor cache ...")
            anchors = AnchorCache(device=dev).build(
                teacher, c["n_anchors"], tuple(c["shape"]), c["n_classes"],
                c["sigma_max"], c.get("anchor_steps", 32), device=dev)
            if rank == 0 and c.get("anchor_path"):
                anchors.save(c["anchor_path"])
        tr = InvertibleTrainer(wrap_ddp(student, rank, world, static_graph=True),
                               wrap_ddp(enc, rank, world, static_graph=True),
                               teacher, c, anchors=anchors, device=dev)

    ema = EMA(student, c.get("ema_decay", 0.999)) if c.get("ema_decay", 0.999) else None

    # ---- resume ----
    hist, it, prior_gpu_s = [], 0, 0.0
    ck_path = _resume_path(a.resume, c["out"])
    if ck_path:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        _raw_module(student).load_state_dict(ck["student"])
        if ema and "ema" in ck:
            ema.shadow.load_state_dict(ck["ema"])
        if c["track"] == "A" and "critic" in ck:
            _raw_module(critic).load_state_dict(ck["critic"])
        it, prior_gpu_s = ck["it"], ck.get("gpu_seconds", 0.0)
        hp = f"{c['out']}/hist.json"
        hist = json.load(open(hp)) if os.path.exists(hp) else []
        log(f"resumed {ck_path} at it={it} ({prior_gpu_s / 3600:.2f} GPU-h)")

    # ---- loop ----
    # GPU-seconds, not steps: eval.comparison_table refuses to compare runs that
    # differ in wall-clock, and methods here have different per-step cost.
    t0 = time.time()
    it0 = it                      # steps already done BEFORE this session
    gpu_seconds = lambda: prior_gpu_s + (time.time() - t0) * world

    def save(tag):
        if rank != 0:
            return
        blob = dict(student=_sd(student), it=it, gpu_seconds=gpu_seconds(),
                    n_gpus=world, config=c)
        if ema:
            blob["ema"] = ema.state_dict()
        if c["track"] == "A":
            blob["critic"] = _sd(critic)
        torch.save(blob, f"{c['out']}/ckpt_{tag}.pt")
        if c["track"] == "A":
            s, l, n = tr.est.curve()
            json.dump(dict(it=it, sigma=s.tolist(), lam=l.tolist(), count=n.tolist()),
                      open(f"{c['out']}/lambda_curve_{tag}.json", "w"))
        json.dump(hist, open(f"{c['out']}/hist.json", "w"))

    while it < c["steps"]:
        if samp:
            samp.set_epoch(it)
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, torch.bfloat16, enabled=c.get("amp", True)):
                logs = tr.step(x, y)
            it += 1
            if ema:
                ema.update(student)
            if it % c["log_every"] == 0:
                # Rate over THIS session's steps, not `it`: after a resume `it`
                # already counts the previous session's work while t0 does not,
                # which would report a step time several times too fast and an
                # ETA to match.
                sps = (time.time() - t0) / max(it - it0, 1)
                logs["it"], logs["s_per_it"] = it, sps
                logs["pct"] = 100.0 * it / c["steps"]
                logs["eta_h"] = sps * (c["steps"] - it) / 3600
                logs["gpu_hours"] = gpu_seconds() / 3600
                hist.append(logs)
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in logs.items()))
            # FINDINGS 3.1: the lambda-drift measurement. Rare, but run on EVERY
            # rank -- it all-gathers the real batch, so a rank-0-only call would
            # deadlock the other ranks at the next collective.
            if (c["track"] == "A" and c.get("probe_every", 0)
                    and it % c["probe_every"] == 0):
                with torch.autocast(dev.type, torch.bfloat16, enabled=c.get("amp", True)):
                    pr = tr.probe_lambda(x, y)
                if rank == 0:
                    json.dump(dict(it=it, **pr), open(f"{c['out']}/probe_{it}.json", "w"))
                    log("PROBE " + " ".join(
                        f"{k}:lam@med={_median(v['lam_ratio']):.3f}" for k, v in pr.items()))
            if it % c["diag_every"] == 0 and c["track"] == "B" and rank == 0:
                d = tr.diagnose(x, y)
                log("DIAG " + " ".join(f"{k}={v:.4g}" for k, v in d.items()))
                hist.append(dict(it=it, **{f"diag_{k}": v for k, v in d.items()}))
            if it % c["ckpt_every"] == 0:
                save(str(it))
            if it >= c["steps"]:
                break
    save("final")
    log("done", time.time() - t0)


def _median(xs):
    v = sorted(x for x in xs if x == x)          # drop NaNs from empty buckets
    return v[len(v) // 2] if v else float("nan")


def _resume_path(spec, out):
    if not spec:
        return None
    if spec != "auto":
        return spec
    cks = [f for f in os.listdir(out) if f.startswith("ckpt_")] if os.path.isdir(out) else []
    if not cks:
        return None
    key = lambda f: (f == "ckpt_final.pt", int(f[5:-3]) if f[5:-3].isdigit() else -1)
    return os.path.join(out, max(cks, key=key))


def _raw_module(m):
    return m.module if hasattr(m, "module") else m


def _sd(m):
    return _raw_module(m).state_dict()


if __name__ == "__main__":
    main()
