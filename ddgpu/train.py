"""Unified entrypoint for all tracks.

  torchrun --nproc_per_node=2 -m ddgpu.train --config configs/dit_b_256_robust.json
  python -m ddgpu.train --config configs/smoke.json          # single process
"""
import argparse, json, os, time
import torch
import torch.distributed as dist

from .dit import make_dit
from .edm import EDMWrapper
from .data import LatentDataset, SyntheticLatents
from .dmd2 import DMD2Trainer
from .invertible import InvertibleTrainer, AnchorCache


def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group("nccl")
    r, w = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(r % torch.cuda.device_count())
    return r, w, torch.device("cuda", r % torch.cuda.device_count())


def build_model(c, device, precond=True, grad_ckpt=True):
    net = make_dit(c["arch"], input_size=c["latent_size"], in_ch=c["shape"][0],
                   n_classes=c["n_classes"], grad_ckpt=grad_ckpt)
    net = net.to(device)
    return EDMWrapper(net, c["sigma_data"]) if precond else net


def wrap_ddp(m, rank, world, static_graph=False):
    """static_graph=True is REQUIRED whenever a module is invoked more than once
    per backward (Track B calls G and E on several legs). Without it DDP's
    reducer raises "Expected to mark a variable ready only once"."""
    if world == 1:
        return m
    return torch.nn.parallel.DistributedDataParallel(
        m, device_ids=[rank % torch.cuda.device_count()],
        find_unused_parameters=False, gradient_as_bucket_view=True,
        static_graph=static_graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    c = json.load(open(a.config))
    for kv in a.override:
        k, v = kv.split("=", 1)
        c[k] = json.loads(v)

    rank, world, dev = setup_dist()
    torch.manual_seed(c["seed"] + rank)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    log(json.dumps(c, indent=1))

    # ---- data ----
    if c["data"] == "synthetic":
        ds = SyntheticLatents(c.get("n_synth", 8192), tuple(c["shape"]), c["n_classes"])
    else:
        ds = LatentDataset(c["data"])
    samp = torch.utils.data.distributed.DistributedSampler(ds) if world > 1 else None
    dl = torch.utils.data.DataLoader(ds, c["micro_batch"], shuffle=(samp is None),
                                     sampler=samp, num_workers=c.get("workers", 4),
                                     drop_last=True, pin_memory=(dev.type == "cuda"))

    # ---- models ----
    teacher = build_model(c, dev, precond=True, grad_ckpt=False).eval()
    if c.get("teacher_ckpt"):
        teacher.net.load_state_dict(torch.load(c["teacher_ckpt"], map_location="cpu"))
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = build_model(c, dev, precond=True)
    if c.get("teacher_ckpt") and c.get("init_from_teacher", True):
        student.net.load_state_dict(teacher.net.state_dict())

    if c["track"] == "A":
        critic = build_model(c, dev, precond=True)
        if c.get("teacher_ckpt"):
            critic.net.load_state_dict(teacher.net.state_dict())
        tr = DMD2Trainer(wrap_ddp(student, rank, world), wrap_ddp(critic, rank, world),
                         teacher, c, device=dev)
    else:
        enc = build_model(c, dev, precond=False)          # raw DiT, see invertible.py
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

    # ---- loop ----
    os.makedirs(c["out"], exist_ok=True)
    # GPU-seconds, not steps: eval.comparison_table refuses to compare runs that
    # differ in wall-clock, and methods here have different per-step cost.
    hist, t0, it = [], time.time(), 0
    gpu_seconds = lambda: (time.time() - t0) * world
    while it < c["steps"]:
        if samp:
            samp.set_epoch(it)
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, torch.bfloat16, enabled=c.get("amp", True)):
                logs = tr.step(x, y)
            it += 1
            if it % c["log_every"] == 0:
                logs["it"], logs["s_per_it"] = it, (time.time() - t0) / it
                logs["gpu_hours"] = gpu_seconds() / 3600
                hist.append(logs)
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in logs.items()))
            if it % c["diag_every"] == 0 and c["track"] == "B" and rank == 0:
                d = tr.diagnose(x, y)
                log("DIAG " + " ".join(f"{k}={v:.4g}" for k, v in d.items()))
                hist.append(dict(it=it, **{f"diag_{k}": v for k, v in d.items()}))
            if it % c["ckpt_every"] == 0 and rank == 0:
                torch.save(dict(student=_sd(student), it=it,
                                gpu_seconds=gpu_seconds(), n_gpus=world),
                           f"{c['out']}/ckpt_{it}.pt")
                if c["track"] == "A":
                    s, l, n = tr.est.curve()
                    json.dump(dict(sigma=s.tolist(), lam=l.tolist(), count=n.tolist()),
                              open(f"{c['out']}/lambda_curve_{it}.json", "w"))
                json.dump(hist, open(f"{c['out']}/hist.json", "w"))
            if it >= c["steps"]:
                break
    if rank == 0:
        torch.save(dict(student=_sd(student), it=it, gpu_seconds=gpu_seconds(),
                        n_gpus=world), f"{c['out']}/ckpt_final.pt")
        json.dump(hist, open(f"{c['out']}/hist.json", "w"))
    log("done", time.time() - t0)


def _sd(m):
    return (m.module if hasattr(m, "module") else m).state_dict()


if __name__ == "__main__":
    main()
