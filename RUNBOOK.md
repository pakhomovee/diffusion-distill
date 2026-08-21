# Runbook — the cheap programme, step by step, with GPU counts

Every step below is annotated **`GPUs: N`** — the number of RTX 4090s that step
needs. Resize the VM between steps; nothing carries state in GPU memory across
them. The peak requirement in the whole programme is **4**, and only Phases D
and E need it.

Scope: **the cheap tier only** (FINDINGS §6 / LOG ENTRY 013). The DiT-XL
ImageNet-256/512 programme is deferred — see RUNPLAN §7 — and does not run on
24 GiB cards at any micro-batch.

Written after the first real GPU pre-flight: `scripts/smoke.sh --gpu` on an
**RTX 4090 (24 GiB)**. Raw output: `results/probe_4090_smoke.txt`.
GPU-count reasoning and memory arithmetic: **[`RUNPLAN.md`](RUNPLAN.md)**.

---

## At a glance

| step | what | GPUs | est. time |
|---|---|---:|---|
| 0 | box setup + `smoke.sh --gpu` | **1** | ~15 min |
| 1 | CIFAR-10 data | **1** (prep is CPU; refstats is GPU) | ~10 min |
| 2 | verify the CIFAR teacher | **1** | ~2 min |
| 3 | **Phase A** — λ ladder, CIFAR leg · *the first gate* | **1** | ~30 min |
| 4 | **Phase C** — CIFAR pairs, 3 seeds each (6 runs) | **1 per run** — 4 in parallel | ~8 h in 2 waves |
| 5 | Phase C eval + figures | **1–4** | ~30 min |
| 6 | ImageNet-64 teacher + data | **1** (prep is CPU; refstats is GPU) | ~1–2 h + download |
| 7 | **Phase D** — ImageNet-64 pair · *required, not optional* | **4** (or 8 at micro 32) | ~16 h |
| 8 | **Phase E** — IN-100 latent pair | **1** prep → **4** train | ~6 h + a 130 GB download |
| 9 | Phase A — the remaining ladder legs | **1** | ~1 h |

Total ≈ **1.5 days of wall-clock, ~130 GPU-h** on 4×4090 (RUNPLAN §3 for the
estimate's assumptions and its 455 GPU-h conservative ceiling).

---

## 1. Are we ready to run?

**Yes, for the cheap tier. No, for the deferred DiT-XL tier — not on 24 GiB.**

What the 4090 pre-flight established:

| check | result |
|---|---|
| 29 invariant tests | ALL PASS |
| pipeline tests (VP precond, ckpt remap, config delta) | ALL PASS |
| teacher-zoo tests (interpolant, DSM identity, mis-wrap detection) | ALL PASS |
| Track A / Track B smoke, 8 steps | ran, losses finite, `DIAG eff_rank_frac` 0.88–0.92 |
| cheap-tier smoke (registry teacher → clone → train, both arms) | ran, `TEACHER-CHECK verdict OK` |
| Phase A self-test on a known-score target | `gain_vs_teacher = 0.15` at σ=8, λ̂ 0.42 vs grid argmin 0.40 |
| VRAM/throughput probe | ran; DiT-XL/2 OOMs at every micro-batch (expected — RUNPLAN §2) |

Nothing in the codebase blocks a launch. The only work between a fresh box and
Phase A is data prep and teacher fetch — steps 1–3.

Two cosmetic warnings you will see and can ignore: a `requires_grad → float`
UserWarning from `invertible.py:179` (Track B only) and a numpy-2.0
`__array__ copy` DeprecationWarning from `exp/10_lambda_real.py:165`.

### Why 4090s, and why four of them

The binding constraint is **not VRAM** — it is FINDINGS §1.4's floor of 256 on
the effective real batch, enforced as `world_size × micro_batch ≥ 256`. That is
bought with GPU *count*, not GPU *memory*. Every cheap-tier teacher is ≤296 M
parameters, so optimiser state is 1.3–11.0 GiB and a 24 GiB card has room.

A 5090 buys ~1.3–1.5× throughput and nothing else at this tier; it becomes
*necessary* only for the deferred DiT-XL programme, whose 25.1 GiB of state does
not fit on 24 GiB at any micro-batch. Full arithmetic in RUNPLAN §2.

---

## 2. Commands, in order

Assumes an AutoDL box with the repo at `~/autodl-tmp/diffusion-distill`.
Every step is idempotent; prep steps skip work that already exists.

### 0. Box setup — **GPUs: 1**

```bash
cd ~/autodl-tmp/diffusion-distill

# Persist these -- do not just export them in one shell. Every later step
# interpolates "$DD_DATA_ROOT/..." into a path, and an unset variable expands to
# a working-looking absolute path: "$DD_DATA_ROOT/cifar10" becomes "/cifar10".
cat >> ~/.bashrc <<'EOF'
export DD_DATA_ROOT=/root/autodl-tmp/data
export DD_CKPT_ROOT=/root/autodl-tmp/ckpt
EOF
source ~/.bashrc
mkdir -p "$DD_DATA_ROOT" "$DD_CKPT_ROOT"
echo "data=$DD_DATA_ROOT ckpt=$DD_CKPT_ROOT"     # both must be non-empty

python3 -c "import torch,torchvision;print(torch.__version__, torchvision.__version__, torch.cuda.get_device_capability())"

scripts/smoke.sh --gpu          # done once on the 4090 -- redo on any new box
```

If a step ever fails with a path like `/cifar10` or `/in64`, that is this
variable being unset in the shell you ran it from — `build_dataset` now says so
rather than reporting a missing `train_moments.npy`. Check for a stray `/cifar10`
at the filesystem root from before the exports were set, and delete it.

`torchvision` is required by the CIFAR-10 path and is **not** in
`requirements.txt` (torch/torchvision are deliberately unpinned). If that import
fails, install the build matching the image's torch before anything else.

A 4090 reports capability `(8, 9)` and is fine on any modern CUDA build. A 5090
is `(12, 0)` and needs CUDA 12.8+ / torch ≥ 2.7.

**If a HuggingFace download crawls at single-digit kB/s** while printing a
`reconstructing file` progress bar, that is the **Xet** backend. `hf_xet` is a
Rust client with its own networking that ignores the `http_proxy` /
`https_proxy` variables `/etc/network_turbo` exports — so sourcing network_turbo
changes nothing, and the transfer does not fail, it just never finishes:

```
diffusion_pytorch_model.safetensors: downloading bytes: 134MB, 4.52kB/s
diffusion_pytorch_model.safetensors: reconstructing file: 56% | 80.5MB / 143MB
```

Importing `ddgpu` now sets `HF_HUB_DISABLE_XET=1` (and, on AutoDL only,
`HF_ENDPOINT=https://hf-mirror.com`) before `huggingface_hub` is imported, which
is the only moment it reads them. Anything you export yourself still wins. To
force it by hand, or on a box running an older checkout:

```bash
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
```

Kill the crawling download first — a resumed one will pick the fast path.

### 1. CIFAR-10 data — **GPUs: 1** (`pixels` is CPU-only; `refstats` needs the GPU)

```bash
python3 -m ddgpu.prepare pixels \
    --source cifar10 --dest "$DD_DATA_ROOT/cifar10" --resolution 32
python3 -m ddgpu.prepare refstats \
    --source cifar10 --dest "$DD_DATA_ROOT/cifar10" --resolution 32 --n 50000 --gpus 0
```

~0.15 GB on disk.

**Already have `cifar-10-python.tar.gz`?** Nothing to do — `prepare` looks for it
before downloading, checking `$DD_DATA_ROOT`, then `<repo>/data`, then
`~/.cache/dd-data`, and uses the first that holds either the archive or an
extracted `cifar-10-batches-py`. It prints which one it picked:

```
[prepare] cifar10: using existing data in /root/autodl-tmp/data
```

If the tarball lives somewhere else entirely, point at it explicitly — this also
redirects the download when the file is *not* there:

```bash
export DD_TV_ROOT=/path/to/the/folder/holding/the/tarball
```

torchvision md5-checks the archive, so a corrupt or truncated copy is re-fetched
rather than silently used.

### 2. Verify the CIFAR teacher — **GPUs: 1**

```bash
python3 -m ddgpu.teachers --teacher diffusers:google/ddpm-cifar10-32 \
    --data "$DD_DATA_ROOT/cifar10"
```

Want: `"verdict": "OK"`, `identity_err` ~1e-11, `rel_mse@0.01` ≈ 0 rising toward
1 at `rel_mse@100`. **Flat and near 1 everywhere = wrong noise convention**, and
nothing downstream is trustworthy. This is the first time `validate_teacher`
meets a real released checkpoint (LOG ENTRY 013's closing line).

### 3. Phase A — the λ ladder, CIFAR leg — **GPUs: 1** · inference only, ~30 min

```bash
python3 exp/10_lambda_real.py \
    --teacher diffusers:google/ddpm-cifar10-32 \
    --data "$DD_DATA_ROOT/cifar10" \
    --batch 512 --batches 40 --sigma-max 157.4 --tag cifar10_3072
```

`--sigma-max 157.4`, not the script's default 80: `ddpm-cifar10-32` is a **VP**
model on a linear-β schedule and its σ tops out at 157.4 (RUNPLAN §6, LOG
ENTRY 012). The teacher meta printed at startup carries the true `sigma_max` —
if it differs, that value wins.

**This is the programme's first gate. Read `gain_vs_teacher` before anything
else. If it is not below 1.0 at 3,072 dims, stop and think** — the method does
not work at that dimension, and that is a real negative result for one GPU-hour
instead of a week of distillation.

### 4. Phase C — CIFAR-10 distillation pairs, 3 seeds per arm — **GPUs: 1 per run** (4 in parallel)

`--micro-batch 512` on a single card clears both floors at once: 512 ≥ 256 hard,
and the λ-calibration half-split is 256 = the soft floor, so no warning. ~3.8 h
per run; six runs are two waves on a 4-card box.

```bash
export DD_SKIP_INSTALL=1        # deps are already in from step 0

# wave 1 -- four runs, one per card, all in the background
scripts/train.sh -d cifar10 --mode dmd2   --gpus 0 --micro-batch 512 --seed 0 &
scripts/train.sh -d cifar10 --mode robust --gpus 1 --micro-batch 512 --seed 0 &
scripts/train.sh -d cifar10 --mode dmd2   --gpus 2 --micro-batch 512 --seed 1 &
scripts/train.sh -d cifar10 --mode robust --gpus 3 --micro-batch 512 --seed 1 &
wait

# wave 2 -- the third seed
scripts/train.sh -d cifar10 --mode dmd2   --gpus 0 --micro-batch 512 --seed 2 &
scripts/train.sh -d cifar10 --mode robust --gpus 1 --micro-batch 512 --seed 2 &
wait
```

`DD_SKIP_INSTALL=1` matters for the parallel form specifically: every
`train.sh` otherwise runs `pip install -r requirements.txt`, and four of those
racing in the same site-packages is a good way to corrupt an install. Step 0
already did it once.

`torchrun --standalone` rendezvouses on a random free port, so parallel launches
do not collide and `MASTER_PORT` needs no setting. **One run per card** — two
runs sharing a card halves the usable micro-batch and breaks the batch floor.

Three seeds is not optional (FINDINGS §6.4): one-step CIFAR distillation is
near-saturated and the effect may be smaller than seed variance.

**Sanity-check the first run before launching the rest.** Kill one run at ~2000
steps, score it cheaply, and look at the grid:

```bash
torchrun --standalone --nproc_per_node=1 -m ddgpu.generate \
    --run-dir runs/cifar10_dmd2 --ckpt final --n-samples 2048 \
    --ref "$DD_DATA_ROOT/cifar10/ref_32_50000.npz" \
    --out results/probe.json --name probe
```

Open `runs/cifar10_dmd2/samples_final.png`. Structure buried in colour speckle
means the σ_max precision problem is back (RUNPLAN §6, first bullet) — that
signature cost a full six-run programme once. An FID in the hundreds with
`recall` exactly 0.000 is the same thing seen numerically. Twenty minutes here
saves five hours per arm.

**No GPU box to hand?** `scripts/colab_check.py` scores a single checkpoint
anywhere — Colab, a laptop, a fresh VM — because `train.save` writes the run's
`config` INTO the `.pt`. Nothing else from the training box is needed: it
rebuilds the FID reference from torchvision's CIFAR-10 through `ddgpu.prepare`,
the same path that produced `ref_32_50000.npz`, so the number is comparable
rather than merely similar.

```bash
!git clone -b worktree-runbook https://github.com/pakhomovee/diffusion-distill.git
%cd diffusion-distill
!pip install -q diffusers pytorch-fid

!python3 scripts/colab_check.py \
    --ckpt hf:pakhomovee/distill:cifar10_dmd2/ckpt_final.pt \
    --fid-n 10000 --compare-guard
```

`--compare-guard` renders the same seeds twice, with `VPPrecond`'s fp32 guard on
and off, and prints the mean difference. Off is the old bf16 failure; if the two
grids are indistinguishable, precision is *not* what is wrong with the run and
the search moves elsewhere.

**`--fid-n 0` needs no reference data at all** — no CIFAR-10, no Inception
weights — so it answers "do the samples look like images yet" in the time it
takes to pull the teacher. Reach for it first.

**Do not download CIFAR-10 from cs.toronto.edu.** That host throttles cloud
notebooks to ~100 kB/s, so torchvision's 170 MB fetch costs half an hour and a
reconnected runtime starts from zero. The Inception weights are *not* the
problem — those come from GitHub at full speed. `--source cifar10-hf`
(`prepare.HFParquetImages`) reads the same images from `uoft-cs/cifar10` on
HF's CDN instead: measured **2.4 s vs >10 min** for the train split from the
same box.

That the two hold the same images was checked rather than assumed — all 50 000
(image, label) pairs match the canonical tarball (md5
`c58f30108f718f92721af3b95e74349a`) byte-for-byte, **as a set**. The row order
differs, so:

* **FID over the full 50 000 is unaffected** — mean and covariance don't care
  about row order. This is the number in the comparison table.
* **Subsets are different subsets.** `--n <50000`, and precision/recall (which
  takes the first k rows), see different images than they would from
  torchvision — equally valid, different by sampling noise. Use one source per
  comparison; don't score one arm against a mirror reference and the other
  against a torchvision one.

`colab_check.py` defaults to the mirror; pass `--ref-source cifar10` to go the
slow way deliberately. It needs `pyarrow`, which Colab already has.

The mirror is available to the whole pipeline, not just Colab — useful on any
fresh VM that has no tarball staged:

```bash
python3 -m ddgpu.prepare refstats --source cifar10-hf \
    --dest "$DD_DATA_ROOT/cifar10" --resolution 32 --n 50000 --gpus 0
```

Two ways to spend nothing at all:

```bash
# Grid only — touches no reference data, no CIFAR, no Inception.
!python3 scripts/colab_check.py --ckpt hf:... --fid-n 0

# Or reuse the reference the training box already computed.
!python3 scripts/colab_check.py --ckpt hf:... \
    --ref-npz hf:pakhomovee/distill:ref_32_50000.npz
```

If you do have the tarball to hand, `prepare.find_root` still honours
`$DD_TV_ROOT` and torchvision will md5-check it and skip the download. Either
way the computed reference is cached to `--out-dir` as `ref_<res>_<n>.npz`, so a
second run in the same session is free; copy it to Drive and a *reconnected*
session is free too.

Serial equivalent on a 1-GPU box (~23 h):

```bash
for s in 0 1 2; do
  scripts/train.sh -d cifar10 --mode dmd2   --gpus 0 --micro-batch 512 --seed $s
  scripts/train.sh -d cifar10 --mode robust --gpus 0 --micro-batch 512 --seed $s
done
```

### 5. Phase C eval + figures — **GPUs: 1–4**

```bash
scripts/eval_all.sh -d cifar10 --gpus 0,1,2,3 --n 50000
python3 exp/09_plot_run.py runs/cifar10_robust runs/cifar10_dmd2
```

Read the **seed-grouped** table with its standard-error verdict, not the flat
one. A gap inside one standard error is "no measured difference", not a win.

`eval_all.sh` keeps going when a run fails to score and reports the names at the
end, so **the actual error is in the per-run log, not on your terminal**:

```bash
tail -30 runs/*/eval.log
```

If the table then says `no eval records`, that is the symptom of every run
having failed, not a separate problem — read the eval logs first.

### 6. ImageNet-64 teacher + data — **GPUs: 1** (`pixels` is CPU-only; `refstats` needs the GPU)

```bash
git clone https://github.com/NVlabs/edm /root/autodl-tmp/edm
export EDM_REPO=/root/autodl-tmp/edm
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
     -O $DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl

export IMAGENET_SRC=/root/autodl-tmp/imagenet/train      # 1000 class subdirs
python3 -m ddgpu.prepare pixels \
    --source "$IMAGENET_SRC" --dest "$DD_DATA_ROOT/in64" --resolution 64
python3 -m ddgpu.prepare refstats \
    --source "$IMAGENET_SRC" --dest "$DD_DATA_ROOT/in64" --resolution 64 --n 50000 --gpus 0

python3 -m ddgpu.teachers --teacher edm:$DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl \
    --edm-repo $EDM_REPO --data "$DD_DATA_ROOT/in64"
```

~15.7 GB of uint8 at 64px, and no VAE anywhere in the pipeline.

> **Open dependency:** Phase D wants full ImageNet train (~150 GB) on the box.
> If only the ImageNet-100 set is available, `IMAGENET_SRC` can point at it and
> the run still works — the EDM teacher was trained on the superset — but the
> FID reference is then a 100-class subset and the absolute number is not
> comparable to anything published. Decide that explicitly rather than by default.

### 7. Phase D — ImageNet-64 pair — **GPUs: 4** (micro 64 × 4 = 256)

**7a. Settle the 24 GiB question first — GPUs: 4, ~2 min.** State is 11.0 GiB
per card and the cloned UNet path has no gradient checkpointing, so activations
at micro 64 are the one unmeasured quantity in the plan (RUNPLAN §5):

```bash
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3 --steps 20
nvidia-smi --query-gpu=memory.used --format=csv   # from a second shell
```

If it OOMs, move to **8 GPUs at micro 32** (still 256 effective):

```bash
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3,4,5,6,7 --micro-batch 32 --steps 20
```

Do *not* drop to 2 GPUs at micro 128 — that raises per-card activations. Do
*not* set `DD_ALLOW_SMALL_REAL_BATCH=1`; that turns the run into the `nogather`
ablation (FINDINGS §1.4).

Note the `s_per_it` the trial prints: `wall-clock = steps × s_per_it / 3600`.
If it disagrees with RUNPLAN §3, the measurement wins.

**7b. The pair — GPUs: 4, ~8 h each:**

```bash
scripts/train.sh -d imagenet64 --mode dmd2   --gpus 0,1,2,3
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3
scripts/eval_all.sh -d imagenet64 --gpus 0,1,2,3 --n 50000
python3 exp/09_plot_run.py runs/imagenet64_robust runs/imagenet64_dmd2
```

Run them serially, not two-per-card. This rung is required, not optional:
without it the trained evidence stops at 3,072 dims while the measured evidence
reaches 16,384.

### 8. Phase E — the latent leg — **GPUs: 4** (micro 64 × 4 = 256)

Download and prep first — **GPUs: 1** for the prep (latents are VAE-encoded, so
`--prepare` does use the GPU; more cards just make it faster):

```bash
python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
    --dest $DD_DATA_ROOT/in100_raw --dry-run     # >130 GB; refuses under 200 GB free
python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
    --dest $DD_DATA_ROOT/in100_raw

export REPA_DIR=/root/autodl-tmp/repa-surgery/REPA
export SIT_CKPT=$DD_CKPT_ROOT/imagenet100_sit-b_2_baseline/checkpoints/0300000.pt
export IN100_SRC=$DD_DATA_ROOT/in100_raw/images

# latents use the SD-VAE, so this one is GPU work -- more cards just make it faster
python3 -m ddgpu.prepare latents \
    --source "$IN100_SRC" --dest "$DD_DATA_ROOT/in100_256" --resolution 256 --gpus 0
python3 -m ddgpu.prepare refstats \
    --source "$IN100_SRC" --dest "$DD_DATA_ROOT/in100_256" --resolution 256 --n 25000 --gpus 0
```

The dataset has to exist before the teacher can be validated against it — the
check needs a real batch. Then, in order:

```bash
python3 -m ddgpu.teachers --teacher sit:$SIT_CKPT --repa-dir $REPA_DIR \
    --data "$DD_DATA_ROOT/in100_256"
```

Then the pair — **GPUs: 4**, ~3 h each:

```bash
scripts/train.sh -d in100_latent --mode dmd2   --gpus 0,1,2,3
scripts/train.sh -d in100_latent --mode robust --gpus 0,1,2,3
scripts/eval_all.sh -d in100_latent --gpus 0,1,2,3 --n 25000
```

The same normalisation REPA trained the SiT with (SD-VAE, scale 0.18215) is what
`prepare latents` writes; a different one would put the teacher off-distribution
at every noise level, which `validate_teacher` would flag as `SUSPECT`.

This leg answers the one reviewer question the rest of the cheap tier cannot —
does any of this hold in latent space — and doubles as the teacher-capacity
probe, since the SiT is deliberately weaker (FID ≈16) than the released
teachers. λ should sit *lower* if λ is driven by teacher bias.

### 9. Phase A — the remaining ladder legs — **GPUs: 1** · inference only

These need the datasets from steps 6 and 8, so they run after them, not up front.

```bash
python3 exp/10_lambda_real.py --teacher edm:$DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl \
    --edm-repo $EDM_REPO --data "$DD_DATA_ROOT/in64" --sigma-max 80 --tag in64_12288

python3 exp/10_lambda_real.py --teacher sit:$SIT_CKPT --repa-dir $REPA_DIR \
    --data "$DD_DATA_ROOT/in100_256" --sigma-max 49 --tag sit_latent_4096
```

Optional 16,384-dim leg — the top of the measured ladder, forward-only, fits a
24 GiB card. It is the only place the deferred DiT-XL/2 teacher appears in this
programme, and it appears as inference, never as training.

`prepare latents` has no `--limit`, so the 50k subset is made by pointing
`--source` at a subset directory rather than the full train set — ~4 GB of
latents instead of the 84 GB full prep:

```bash
python3 -m ddgpu.ckpt --name DiT-XL-2-512x512 --dir $DD_CKPT_ROOT

# 50 images from each of the 1000 classes -> 50k, class balance preserved
mkdir -p $DD_DATA_ROOT/in512_src
for c in "$IMAGENET_SRC"/*/; do
  d=$DD_DATA_ROOT/in512_src/$(basename "$c"); mkdir -p "$d"
  ls "$c" | head -50 | while read -r f; do ln -sf "$c$f" "$d/$f"; done
done

python3 -m ddgpu.prepare latents --source $DD_DATA_ROOT/in512_src \
    --dest "$DD_DATA_ROOT/in512" --resolution 512 --gpus 0
python3 exp/10_lambda_real.py --teacher dit:$DD_CKPT_ROOT/DiT-XL-2-512x512.pt \
    --data "$DD_DATA_ROOT/in512" --sigma-max 157.4 --tag dit512_16384
```

---

## 3. Watch-list during runs

**Where is it up to?** There is no progress bar — training logs one line every
`log_every` steps, and with six runs that is the only signal. `scripts/progress.py`
turns those logs into progress and an ETA. It reads `<run>/train.log` and
`<run>/config.json` off disk, so it works on a job that is already running, and
on one started before `eta_h` existed:

```bash
python3 scripts/progress.py            # every run under runs/
python3 scripts/progress.py -w 60      # refresh once a minute
```

```
cifar10_dmd2    1100/20000   [#...................]   5.5%  0.825s/it   0.25 GPU-h  eta 4h20m

slowest unfinished run finishes in ~4h20m (assumes the current rate holds)
```

A run whose log has not moved for ten minutes reads `STALLED`, because a crashed
`torchrun` leaves its log looking exactly like a slow one. The training line
itself now also carries `pct=` and `eta_h=` directly.

* **Track A**: the launcher prints `effective real batch = N ... ok`. If it
  refuses, do not reach for `DD_ALLOW_SMALL_REAL_BATCH=1` — that turns the run
  into an ablation (FINDINGS §1.4).
* **λ drift**: `PROBE real:lam@med` vs `student:lam@med` in the training log is
  the §3.1 mechanism measurement, and it rides along free on every robust run.
  Phase A cannot produce it — the DSM identity does not extend to student samples.
* **FID at matched wall-clock, not matched steps.** `ddgpu.eval` refuses the
  table beyond 15% GPU-second spread; that refusal is the guard working.
* **Interrupted?** `scripts/train.sh ... --resume` restores the student, EMA,
  critic and the accumulated GPU-seconds — the last one matters, because a
  resumed run that forgot its history looks artificially cheap to the eval
  harness.
* **If you run Track B at all** (not part of this programme — RUNPLAN §3):
  `DIAG eff_rank_frac` must stay near 1.0 in the first 500 steps. If it falls,
  the encoder is collapsing and the run is dead; the loss curves will not tell you.
