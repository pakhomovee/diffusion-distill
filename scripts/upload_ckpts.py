#!/usr/bin/env python3
"""Strip a run's checkpoints to what scoring needs and push them to the Hub.

  python3 scripts/upload_ckpts.py runs/cifar10_dmd2_s40_nogan \
      --repo pakhomovee/distill --token hf_xxx

Two things this does that a plain `api.upload_file` loop does not.

**It clears HF_ENDPOINT first.** `import ddgpu` calls `hfenv.setup_hf_env()`,
which points HF_ENDPOINT at hf-mirror.com whenever /etc/network_turbo exists --
i.e. on every AutoDL box. That mirror is READ-ONLY, so an upload aimed at it
does not go where you think. This module therefore imports no ddgpu, and pops
the variable before `huggingface_hub` is imported, because the endpoint is read
into a module constant at import time and setting it later does nothing.

**It strips the checkpoints.** `train.save` writes student + ema + critic, which
for a 35.7M UNet is ~429 MB each and ~3.9 GB for a nine-checkpoint run. Scoring
reads `ema` (or `student`) and the embedded `config`, nothing else -- so
dropping the rest is a 3x saving with no effect on any number, and the critic
is reconstructible from a resume anyway. `--keep-all` opts out if you intend to
resume training from these.
"""
import argparse
import os

# BEFORE huggingface_hub is imported anywhere. See the docstring.
_MIRROR_CLEARED = os.environ.pop("HF_ENDPOINT", None)

import gc                                                # noqa: E402
import glob                                              # noqa: E402
import json                                              # noqa: E402
import re                                                # noqa: E402
import resource                                          # noqa: E402
import tempfile                                          # noqa: E402

import torch                                             # noqa: E402


def _rss():
    """Peak resident memory so far -- this script has been OOM-killed before."""
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return f"  [peak RSS {kb / 1e6:.2f} GB]"


def ckpt_sort_key(path):
    """ckpt_1000.pt < ckpt_20000.pt < ckpt_final.pt, numerically not lexically."""
    tag = os.path.basename(path)[5:-3]
    return (tag == "final", int(tag) if tag.isdigit() else -1)


def slim(blob, keep_all=False):
    """Everything scoring reads, and nothing else."""
    if keep_all:
        return blob
    out = {k: blob[k] for k in ("it", "gpu_seconds", "n_gpus", "config")
           if k in blob}
    if "ema" in blob:
        out["ema"] = blob["ema"]
    else:                       # no EMA configured; scoring falls back to these
        out["student"] = blob["student"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--repo", required=True, help="e.g. pakhomovee/distill")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="or export HF_TOKEN")
    p.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    p.add_argument("--prefix", default=None,
                   help="path inside the repo (default: the run directory's name)")
    p.add_argument("--keep-all", action="store_true",
                   help="upload student+critic too, so the run can be resumed")
    p.add_argument("--force", action="store_true",
                   help="re-upload files already present in the repo")
    p.add_argument("--no-xet", action="store_true",
                   help="disable the Xet upload backend; it chunks in memory and "
                        "is the other half of the OOM on a small container")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    if not a.token and not a.dry_run:
        raise SystemExit("no token: pass --token or export HF_TOKEN "
                         "(HF settings -> Access Tokens, needs write scope)")
    if _MIRROR_CLEARED:
        print(f"[upload] cleared HF_ENDPOINT={_MIRROR_CLEARED} (read-only mirror)")

    run = a.run_dir.rstrip("/")
    prefix = a.prefix or os.path.basename(run)
    cks = sorted(glob.glob(f"{run}/ckpt_*.pt"), key=ckpt_sort_key)
    if not cks:
        raise SystemExit(f"no ckpt_*.pt in {run}")

    extras = [f for f in ("config.resolved.json", "train.log", "hist.json")
              if os.path.exists(f"{run}/{f}")]
    print(f"[upload] {len(cks)} checkpoints + {len(extras)} small files "
          f"-> {a.repo}:{prefix}/")

    if a.no_xet:
        # Must precede the huggingface_hub import: it reads this into a module
        # constant, exactly like HF_ENDPOINT.
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    api, already = None, set()
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=a.token)
        api.create_repo(a.repo, repo_type=a.repo_type, exist_ok=True)
        # Resume: an interrupted run has already pushed some of these, and each
        # is ~143 MB.
        already = {f for f in api.list_repo_files(a.repo, repo_type=a.repo_type)
                   if f.startswith(prefix + "/")}
        if already:
            print(f"[upload] {len(already)} file(s) already in {a.repo}:{prefix}/")

    with tempfile.TemporaryDirectory() as td:
        for c in cks:
            name = os.path.basename(c)
            if f"{prefix}/{name}" in already and not a.force:
                print(f"  {name:20} already in the repo, skipping (--force to redo)")
                continue

            # mmap so the 429 MB never lands in RSS: only the tensors `slim`
            # keeps are faulted in, when torch.save reads them.
            try:
                blob = torch.load(c, map_location="cpu", weights_only=False,
                                  mmap=True)
            except (TypeError, RuntimeError):        # torch < 2.1, or legacy format
                blob = torch.load(c, map_location="cpu", weights_only=False)
            small = slim(blob, a.keep_all)
            local = os.path.join(td, name)
            torch.save(small, local)
            before, after = os.path.getsize(c), os.path.getsize(local)
            step, keys = blob.get("it"), sorted(small)
            # Drop BOTH before uploading. Holding a 429 MB blob and a 143 MB copy
            # while the Xet client chunks the file is what got this OOM-killed on
            # a container the first time; the upload reads from disk and needs
            # neither.
            del small, blob
            gc.collect()
            print(f"  {name:20} {before/1e6:7.1f} MB -> {after/1e6:7.1f} MB "
                  f"(step {step}, keys {keys}){_rss()}")
            if api:
                api.upload_file(path_or_fileobj=local,
                                path_in_repo=f"{prefix}/{name}",
                                repo_id=a.repo, repo_type=a.repo_type)
            os.remove(local)

        for f in extras:
            if f"{prefix}/{f}" in already and not a.force:
                print(f"  {f} already in the repo, skipping")
                continue
            print(f"  {f}")
            if api:
                api.upload_file(path_or_fileobj=f"{run}/{f}",
                                path_in_repo=f"{prefix}/{f}",
                                repo_id=a.repo, repo_type=a.repo_type)

    if a.dry_run:
        print("[upload] --dry-run: nothing sent")
    else:
        print(f"[upload] done: https://huggingface.co/datasets/{a.repo}/tree/"
              f"main/{prefix}")


if __name__ == "__main__":
    main()
