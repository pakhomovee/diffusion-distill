"""Fetch a HuggingFace repo (checkpoints / datasets), with a disk guard.

  python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
      --dest /root/autodl-tmp/data/in100_raw

The guard is the point. The ImageNet-100 repo backing the SiT teacher is over
130 GB, and `snapshot_download` fills the disk and then dies part-way, leaving a
half-populated cache that looks like a complete one. So this refuses to start
unless the destination filesystem has `--require-gb` free (default 200), and it
prints what it is about to do first.

  --dry-run   print the plan and the free space, download nothing
  --allow-low-disk  override the guard deliberately

On AutoDL, source /etc/network_turbo first (scripts/lib/common.sh does it) and
put --dest on the data volume (/root/autodl-tmp), never on the system disk.
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddgpu.hfenv import setup_hf_env    # noqa: E402 -- must precede huggingface_hub


def free_gb(path):
    p = path
    while p and not os.path.isdir(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p or "/").free / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", required=True, help="e.g. pakhomovee/imagenet")
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset"])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--allow-patterns", nargs="*", default=None,
                    help="only fetch matching files, e.g. '*.pt' 'checkpoints/*'")
    ap.add_argument("--require-gb", type=float, default=200.0)
    ap.add_argument("--allow-low-disk", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    setup_hf_env()
    have = free_gb(a.dest)
    print(f"repo      : {a.repo} ({a.repo_type})")
    print(f"dest      : {a.dest}")
    print(f"patterns  : {a.allow_patterns or 'ALL FILES'}")
    print(f"free space: {have:.1f} GB (need {a.require_gb:.0f} GB)")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')}   "
          f"xet_disabled={os.environ.get('HF_HUB_DISABLE_XET')}")

    if a.dry_run:
        print("--dry-run: nothing downloaded")
        return
    if have < a.require_gb and not a.allow_low_disk:
        sys.exit(
            f"refusing to download: {have:.1f} GB free < {a.require_gb:.0f} GB required.\n"
            "A partial snapshot_download looks like a complete one, which is worse\n"
            "than not starting. Free space, pick a bigger volume, narrow the fetch\n"
            "with --allow-patterns, or pass --allow-low-disk deliberately.")

    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id=a.repo, repo_type=a.repo_type,
                          local_dir=a.dest, allow_patterns=a.allow_patterns)
    print(f"done -> {p}")


if __name__ == "__main__":
    main()
