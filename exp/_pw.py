import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("e02", os.path.join(os.path.dirname(os.path.abspath(__file__)), "02_gauss_power.py"))
e02 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e02)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

reps = 100
rows = []
print("=== EXP03: power vs BATCH SIZE (d fixed per block) ===", flush=True)
for d in [4096, 16384]:
    print(f"\n-- d = {d} --", flush=True)
    hdr = f"{'perturb':>10} {'str':>6} {'B':>6} | " + " ".join(f"{s:>7}" for s in e02.STATS)
    print(hdr); print("-"*len(hdr), flush=True)
    for kind, s in [("scale", 0.02), ("lowrank", 0.10), ("mixture", 0.5), ("corr", 0.10)]:
        for B in [64, 128, 256, 512]:
            p = {}
            for st in e02.STATS:
                pw, n, a = e02.power(st, kind, B, d, s, reps)
                p[st] = pw
                rows.append(dict(exp="batch", perturb=kind, strength=s, d=d, B=B,
                                 stat=st, power=pw, null_mean=n, alt_mean=a, reps=reps))
            print(f"{kind:>10} {s:6.2f} {B:6d} | " + " ".join(f"{p[st]:7.2f}" for st in e02.STATS), flush=True)
        print(flush=True)
json.dump(rows, open(f"{ROOT}/results/exp03_gauss_power_batch.json","w"), indent=1)
print("done", flush=True)
