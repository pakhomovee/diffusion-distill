# Runbook — what to rent, and what to run, in order

Written after the first real GPU pre-flight: `scripts/smoke.sh --gpu` on an
**RTX 4090 (24 GiB)** AutoDL box. Raw output: `results/probe_4090_smoke.txt`.

---

## 1. Are we ready to run?

**Yes, for the cheap tier (FINDINGS §6). No, for the ImageNet-256/512 tier — not
on a 24 GiB card.**

What the 4090 pre-flight established:

| check | result |
|---|---|
| 29 invariant tests | ALL PASS |
| pipeline tests (VP precond, ckpt remap, config delta) | ALL PASS |
| teacher-zoo tests (interpolant, DSM identity, mis-wrap detection) | ALL PASS |
| Track A / Track B smoke, 8 steps | ran, losses finite, `DIAG eff_rank_frac` 0.88–0.92 |
| cheap-tier smoke (registry teacher → clone → train, both arms) | ran, `TEACHER-CHECK verdict OK` |
| Phase A self-test on a known-score target | `gain_vs_teacher = 0.15` at σ=8, λ̂ 0.42 vs grid argmin 0.40 |
| VRAM/throughput probe | ran; **see the OOM row below** |

Nothing in the codebase is blocking. What is *not* yet done is data prep and
teacher fetch — those are steps 1–3 below, and they are the only work between a
fresh box and Phase A.

Two non-blocking wrinkles seen in the output, worth knowing before they surprise
you mid-run:

* `invertible.py:179` emits a `requires_grad=True → scalar` UserWarning each
  step. Cosmetic (`float(v)` on an attached tensor in the log dict), Track B only.
* `exp/10_lambda_real.py:165` emits a numpy 2.0 `__array__ copy` DeprecationWarning.
  Cosmetic.

---

## 2. Which VM: 4090 or 5090?

**A 4090 is enough for everything the cheap tier runs. Rent 4090s, and rent
*four* of them rather than one bigger card.**

The binding constraint on this plan is **not VRAM** — it is FINDINGS §1.4's hard
floor of 256 on the effective real batch, which the launcher enforces as
`world_size × micro_batch ≥ 256` and refuses to start below. That floor is
bought with GPU *count*, not GPU *memory*.

### What the probe actually measured (4090, 24 GiB)

```
DiT-B/2  res 32  dmd2        micro 128  ->  7.07 GiB   0.83 s/step
DiT-B/2  res 64  dmd2        micro 128  -> 18.49 GiB   3.51 s/step
DiT-B/2  res 64  invertible  micro 128  ->  OOM
DiT-XL/2 (any res, any method, micro 8..128) -> OOM, every single cell
```

The DiT-XL/2 wipeout is not a surprise and not a bug: RUNPLAN's own arithmetic
puts XL's optimiser state at 21.4 GiB (675 M params × 16 B/param × 2 trainable
copies), which leaves under 3 GiB for activations on a 24 GiB card. On a 5090's
32 GiB it fits with ~8.6 GiB of headroom — that is exactly the gap the 5090 was
specified for. **So the 5090 requirement in RUNPLAN.md is real, and it belongs
entirely to the expensive tier we are not running yet.**

### Cheap-tier memory, by arm

Teachers are 19× smaller than DiT-XL/2 and student+critic are `deepcopy` clones
of the teacher (`train.clone_trainable`), so state scales with the teacher:

| arm | teacher | params | fp32-Adam state (student+critic+EMA+frozen) | verdict on 24 GiB |
|---|---|---:|---:|---|
| C · CIFAR-10 32px | `diffusers:google/ddpm-cifar10-32` | ≈36 M | ≈1.4 GiB | comfortable, ~20 GiB free for activations at micro 512 |
| D · ImageNet-64 | `edm:...-cond-adm.pkl` | ≈296 M | ≈11.2 GiB | **the tight one** — see below |
| E · IN-100 latent | `sit:...0300000.pt` | ≈130 M | ≈5 GiB | comfortable |
| A · λ ladder (incl. DiT-XL/2 @512, 16 384 dims) | any | — | forward only, bf16 | comfortable |

**Phase D is the only cell where 24 GiB is genuinely uncertain.** ADM at 64 px
carries no gradient checkpointing on the cloned path, so ~11 GiB of state plus
activations at micro 64 lands somewhere around 18–21 GiB. That is a 90-second
experiment, not a procurement decision — step 6a below runs 20 steps and tells
you. If it OOMs the fixes, in order of preference:

1. 8×4090 at `--micro-batch 32` (8 × 32 = 256, floor still met),
2. 2×4090 is *not* an option at micro 128 for the same memory reason,
3. 32 GiB cards (5090 / A6000-class) at micro 64 × 4.

### The honest 5090 argument

Not memory, for this tier — **throughput**. A 5090 is roughly 1.3–1.5× a 4090 on
bf16 dense with 1.8 TB/s vs 1.0 TB/s of bandwidth, so the ~150–250 GPU-h cheap
tier lands nearer 110–190 GPU-h. Against that: sm_120 needs CUDA 12.8+ / torch
≥ 2.7, one more thing that can be wrong on a fresh image, and the 4090 box in
`results/probe_4090_smoke.txt` is already proven working end to end.

**Recommendation: 4×4090 for Phases A/C/D/E. Move to ≥32 GiB cards only when the
cheap tier's gates pass and RUNPLAN's DiT-XL programme starts** — and that one
wants 8 of them, not a better single card.

---

## 3. Commands, in order

Assumes an AutoDL box with the repo at `~/autodl-tmp/diffusion-distill`.
Every step is idempotent; prep steps skip work that already exists.

### 0. Box setup (once)

```bash
cd ~/autodl-tmp/diffusion-distill
export DD_DATA_ROOT=/root/autodl-tmp/data
export DD_CKPT_ROOT=/root/autodl-tmp/ckpt
mkdir -p "$DD_DATA_ROOT" "$DD_CKPT_ROOT"

# sm_120 (5090) needs (12.8, torch>=2.7); a 4090 (8,9) is fine on anything modern
python3 -c "import torch,torchvision;print(torch.__version__, torchvision.__version__, torch.cuda.get_device_capability())"

scripts/smoke.sh --gpu          # done once on the 4090 -- redo on any new box
```

`torchvision` is required by the CIFAR-10 path and is **not** in
`requirements.txt` (torch/torchvision are deliberately unpinned). If the import
above fails, install the build matching the image's torch before anything else.

### 1. CIFAR-10 data (~0.15 GB, minutes)

```bash
python3 -m ddgpu.prepare pixels \
    --source cifar10 --dest "$DD_DATA_ROOT/cifar10" --resolution 32
python3 -m ddgpu.prepare refstats \
    --source cifar10 --dest "$DD_DATA_ROOT/cifar10" --resolution 32 --n 50000 --gpus 0
```

### 2. Verify the CIFAR teacher before trusting any number it produces

```bash
python3 -m ddgpu.teachers --teacher diffusers:google/ddpm-cifar10-32 \
    --data "$DD_DATA_ROOT/cifar10"
```

Want: `"verdict": "OK"`, `identity_err` ~1e-11, `rel_mse@0.01` ≈ 0 rising toward
1 at `rel_mse@100`. **Flat and near 1 everywhere = wrong noise convention**, and
nothing downstream is trustworthy. This is the first time `validate_teacher`
meets a real released checkpoint (LOG ENTRY 013's closing line).

### 3. Phase A — the λ ladder, low rung. Inference only, ~1 GPU-h

```bash
python3 exp/10_lambda_real.py \
    --teacher diffusers:google/ddpm-cifar10-32 \
    --data "$DD_DATA_ROOT/cifar10" \
    --batch 512 --batches 40 --sigma-max 157.4 --tag cifar10_3072
```

`--sigma-max 157.4`, not the script's default 80: `ddpm-cifar10-32` is a **VP**
model on a linear-β schedule and its σ tops out at 157.4 (RUNPLAN, LOG ENTRY 012).
The teacher meta printed at startup carries the true `sigma_max` — if it differs,
that value wins. Use 80 only for the EDM leg and ≈49 for the SiT leg.

**Read `gain_vs_teacher` first. If it is not below 1.0, stop and think — that is
the falsifiable result, and a negative one is worth having for one GPU-hour.**

### 4. Phase C — CIFAR-10 distillation pairs, 3 seeds per arm

One GPU each; `--micro-batch 512` is what clears both batch floors on a single
card (512 ≥ 256 hard floor; the λ-calibration half-split is 256 = soft floor, so
no warning). ~3–4 h per run on a 4090; the six runs pack onto four cards.

```bash
for s in 0 1 2; do
  scripts/train.sh -d cifar10 --mode dmd2   --gpus 0 --micro-batch 512 --seed $s
  scripts/train.sh -d cifar10 --mode robust --gpus 0 --micro-batch 512 --seed $s
done
```

Three seeds is not optional (FINDINGS §6.4): one-step CIFAR distillation is
near-saturated and the effect may be smaller than seed variance.

```bash
scripts/eval_all.sh -d cifar10 --gpus 0,1,2,3 --n 50000
python3 exp/09_plot_run.py runs/cifar10_robust runs/cifar10_dmd2
```

Read the **seed-grouped** table with its standard-error verdict, not the flat one.

### 5. Phase A — the rest of the ladder (needs the data from step 6/7)

```bash
python3 exp/10_lambda_real.py --teacher edm:$DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl \
    --edm-repo $EDM_REPO --data "$DD_DATA_ROOT/in64" --sigma-max 80 --tag in64_12288

python3 exp/10_lambda_real.py --teacher dit:$DD_CKPT_ROOT/DiT-XL-2-512x512.pt \
    --data "$DD_DATA_ROOT/in512" --sigma-max 157.4 --tag dit512_16384

python3 exp/10_lambda_real.py --teacher sit:$SIT_CKPT --repa-dir $REPA_DIR \
    --data "$DD_DATA_ROOT/in100_256" --sigma-max 49 --tag sit_latent_4096
```

The 16 384-dim DiT leg is forward-only and fits a 24 GiB card; it needs a 50 k
latent subset, not the 84 GB full prep.

### 6. Phase D — ImageNet-64 (required, not optional)

```bash
git clone https://github.com/NVlabs/edm /root/autodl-tmp/edm
export EDM_REPO=/root/autodl-tmp/edm
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
     -O $DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl

export IMAGENET_SRC=/root/autodl-tmp/imagenet/train      # 1000 class subdirs
python3 -m ddgpu.teachers --teacher edm:$DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl \
    --edm-repo $EDM_REPO --data "$DD_DATA_ROOT/in64"
```

**6a. Settle the 24 GiB question before committing 9 hours:**

```bash
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3 --prepare --steps 20
nvidia-smi --query-gpu=memory.used --format=csv   # in another shell
```

If that OOMs, rerun at `--gpus 0..7 --micro-batch 32` (still 256 effective).

**6b. The real pair (~9 h wall each on 4×4090):**

```bash
scripts/train.sh -d imagenet64 --mode dmd2   --gpus 0,1,2,3
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3
scripts/eval_all.sh -d imagenet64 --gpus 0,1,2,3 --n 50000
```

> **Open dependency:** Phase D wants full ImageNet train (~150 GB) on the box.
> If only the ImageNet-100 set is available, `IMAGENET_SRC` can point at it and
> the run still works — the EDM teacher was trained on the superset — but the FID
> reference is then a 100-class subset and the absolute number is not comparable
> to anything published. Decide that explicitly rather than by default.

### 7. Phase E — the latent leg

```bash
python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
    --dest $DD_DATA_ROOT/in100_raw --dry-run     # >130 GB; refuses under 200 GB free
python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
    --dest $DD_DATA_ROOT/in100_raw

export REPA_DIR=/root/autodl-tmp/repa-surgery/REPA
export SIT_CKPT=$DD_CKPT_ROOT/imagenet100_sit-b_2_baseline/checkpoints/0300000.pt
export IN100_SRC=$DD_DATA_ROOT/in100_raw/images

python3 -m ddgpu.teachers --teacher sit:$SIT_CKPT --repa-dir $REPA_DIR \
    --data "$DD_DATA_ROOT/in100_256"
scripts/train.sh -d in100_latent --mode dmd2   --gpus 0,1,2,3 --prepare
scripts/train.sh -d in100_latent --mode robust --gpus 0,1,2,3
scripts/eval_all.sh -d in100_latent --gpus 0,1,2,3 --n 25000
```

### 8. Only then: RUNPLAN.md's DiT-XL programme

Gated on Phase A's held-out gain and Phase C/D's seeded FIDs. Needs ≥32 GiB
cards — the 4090 probe OOMed on DiT-XL/2 at **every** micro-batch down to 8.

---

## 4. Watch-list during runs

* **Track A**: the launcher prints `effective real batch = N ... ok`. If it
  refuses, do not reach for `DD_ALLOW_SMALL_REAL_BATCH=1` — that turns the run
  into an ablation (FINDINGS §1.4).
* **Track B**: `DIAG eff_rank_frac` in the first 500 steps must stay near 1.0.
  If it falls the encoder is collapsing and the run is dead; the loss curves
  will not tell you.
* **λ**: `PROBE real:lam@med` vs `student:lam@med` in the log is the §3.1 drift
  measurement, and it rides along free on every robust run.
* **FID at matched wall-clock, not matched steps.** `ddgpu.eval` refuses the
  table beyond 15% GPU-second spread; that refusal is the guard working.
