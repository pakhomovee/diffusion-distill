"""Progress and ETA for every run under runs/, from the logs already on disk.

  python3 scripts/progress.py              # every run
  python3 scripts/progress.py -w 60        # refresh every 60s
  python3 scripts/progress.py --only cifar

Reads `<run>/train.log` and `<run>/config.json`, so it works on a job that is
already running -- including one started before `eta_h` was added to the log
line -- and needs nothing from the training process itself.

A run whose last log line has not moved is reported as STALLED rather than left
to look slow: a crashed `torchrun` leaves the log exactly as it was.
"""
import argparse
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def last_log_line(path):
    """Last line carrying `it=`, read from the tail rather than the whole file."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    with open(path, "rb") as f:
        f.seek(max(0, size - 65536))
        tail = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(tail):
        if "it=" in line:
            return line
    return None


def parse(line):
    return {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.eE+-]+)\b", line)
            if re.fullmatch(r"-?[\d.eE+-]+", v)}


def fmt_h(h):
    if h != h or h < 0:                      # NaN guard
        return "?"
    m = int(round(h * 60))
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def rows(only=None):
    out = []
    runs = os.path.join(REPO, "runs")
    for name in sorted(os.listdir(runs) if os.path.isdir(runs) else []):
        d = os.path.join(runs, name)
        log = os.path.join(d, "train.log")
        if not os.path.isfile(log) or (only and only not in name):
            continue
        cfg = os.path.join(d, "config.json")
        steps = 0
        if os.path.isfile(cfg):
            try:
                steps = int(json.load(open(cfg)).get("steps", 0))
            except (ValueError, OSError):
                steps = 0
        v = parse(last_log_line(log) or "")
        it, sps = v.get("it", 0), v.get("s_per_it", 0)
        # A finished run's log stops moving too, so age is only "stalled" when
        # there is work left to do.
        age = time.time() - os.path.getmtime(log)
        done = steps and it >= steps
        eta = sps * (steps - it) / 3600 if steps and sps and not done else 0.0
        out.append(dict(name=name, it=int(it), steps=steps, sps=sps, eta=eta,
                        gpu_h=v.get("gpu_hours", 0.0), age=age, done=bool(done)))
    return out


def show(only=None):
    rs = rows(only)
    if not rs:
        print("no runs with a train.log under runs/")
        return
    w = max(len(r["name"]) for r in rs)
    total = 0.0
    for r in rs:
        pct = 100.0 * r["it"] / r["steps"] if r["steps"] else 0.0
        if r["done"]:
            state = "done"
        elif r["age"] > 600:
            state = f"STALLED {fmt_h(r['age'] / 3600)}"
        else:
            state = f"eta {fmt_h(r['eta'])}"
            total = max(total, r["eta"])       # runs in parallel -> the slowest
        bar = int(pct / 5)
        print(f"{r['name']:<{w}}  {r['it']:>6d}/{r['steps'] or '?':<6}  "
              f"[{'#' * bar}{'.' * (20 - bar)}] {pct:5.1f}%  "
              f"{r['sps']:.3f}s/it  {r['gpu_h']:6.2f} GPU-h  {state}")
    if total:
        print(f"\nslowest unfinished run finishes in ~{fmt_h(total)} "
              f"(assumes the current rate holds)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--only", default=None, help="substring filter on the run name")
    p.add_argument("-w", "--watch", type=int, default=0, help="refresh every N seconds")
    a = p.parse_args()
    while True:
        if a.watch:
            sys.stdout.write("\033[2J\033[H")
        show(a.only)
        if not a.watch:
            return
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
