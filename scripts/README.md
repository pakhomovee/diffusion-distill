# Running the experiments on a GPU box

> **Running this for the first time? Skip to [the cheap tier](#the-cheap-tier-start-here).**
> That is the ~230–455 GPU-hour plan (FINDINGS.md §6) and it is what to run now.
> Everything before it is the ~5,000 GPU-hour ImageNet-256/512 programme, kept
> intact for when the cheap tier justifies spending it.

One entry point — [`train.sh`](train.sh) — sets up the environment, fetches the
teacher, optionally builds the dataset, and launches `ddgpu.train` under
`torchrun`. The layout mirrors `repa-surgery/training/`, so a box configured for
that repo is already configured for this one.

```
scripts/
  smoke.sh        pre-flight: tests + both tracks on CPU (+ --gpu for the VRAM probe)
  progress.py     progress + ETA for every run, read off the logs already on disk
  precision_probe.py  noise each precision puts into D(x;sigma), per teacher/GPU
  colab_check.py  grid + FID for one checkpoint, on a box with no training data
  sample_stats.py what KIND of wrong a sample grid is: noise / blur / layout / collapse
  train.sh        single entry point for a training run
  eval_all.sh     score every run and print the matched-wall-clock table
  mkconfig.py     generates run configs (they are not hand-maintained -- see below)
  lib/common.sh   AutoDL network turbo, HF mirror, python env, batch-size guard
  envs/*.env      per-dataset config + prepare_dataset()
```

## Quick start (AutoDL, 8×5090)

```bash
git clone <this repo> && cd diffusion-distill

# 0. pre-flight. Do this first, on every fresh box.
scripts/smoke.sh --gpu
#    -> compare results/probe.json against RUNPLAN.md. The probe wins.

# 1. one-off: latents + FID reference + teacher checkpoint (~2-4 h for ImageNet)
IMAGENET_SRC=/root/autodl-tmp/imagenet/train \
  scripts/train.sh -d imagenet256 --mode dmd2 --gpus 0,1,2,3,4,5,6,7 --prepare --dry-run

# 2. Run 1: the DMD2 baseline reproduction. This is the gate for everything else.
scripts/train.sh -d imagenet256 --mode dmd2 --gpus 0,1,2,3,4,5,6,7

# 3. Run 1': the method, same everything else.
scripts/train.sh -d imagenet256 --mode robust --gpus 0,1,2,3,4,5,6,7

# 4. score both at matched wall-clock
scripts/eval_all.sh -d imagenet256 --gpus 0,1,2,3,4,5,6,7 --n 50000

# 5. figures
python3 exp/09_plot_run.py runs/imagenet256_robust runs/imagenet256_dmd2
```

`--dry-run` prints the exact `torchrun` command without launching. Anything
after `--` is passed verbatim to `ddgpu.train`.

## Modes

`--mode` selects the training variant and names the run (`imagenet256_robust`):

| mode | what it does |
|------|--------------|
| `dmd2` | The baseline: distribution matching against the frozen teacher, fake-score critic, and the hand-weighted real-data GAN term (`gan_weight=0.001`). |
| `robust` | Track A. Replaces that GAN term with the doubly-robust score fusion at an online-estimated λ(σ). **Turning the estimator on turns the hand-tuned term off** — that is the claim. |
| `invert` | Track B. Critic-free: an encoder E replaces the fake-score network, and the discrepancy is against N(0,I). Needs the teacher-anchor cache (built automatically, ~1 GPU-h, reused across runs). |
| `data` | Ablation: λ ≡ 0, empirical score only. |
| `fixed` | Ablation: hand-set λ — the hyperparameter this work removes. |
| `ratio` | Ablation: the legacy σ-gated ratio estimator that FINDINGS.md §1.3 measured failing at d=512. Kept so the failure is reproducible, not just asserted. |
| `nogather`| Ablation: `gather_real=false`, i.e. deliberately violating the effective-batch floor below. |

Configs are **generated**, not hand-maintained, because FINDINGS.md §4.0 makes a
claim that only holds if they are: baseline and method must differ by exactly
the score-fusion switch, so a measured difference cannot be an implementation
artefact. `train.sh` prints that delta at launch, and
`tests/test_pipeline.py:t_config_delta` pins it.

```bash
python3 scripts/mkconfig.py --mode robust --print-delta
```

To override anything: `--set key=json`, repeatable — e.g.
`--set lr_g=2e-5 --set d_steps=2`.

## The guard that will stop you launching

FINDINGS.md §1.4: with an **effective real batch below 256** the doubly-robust
fusion is *worse* than the plain teacher — mean MSE ratio 1.14, worst case 5.7×,
and every one of the five worst cells in the 234-cell sweep was an N=64 cell.
`robust.py:all_gather_batch` takes the effective N to `world_size × micro_batch`,
so the launcher checks it and **refuses to start** when it is too small:

```
[dd] effective real batch = 128 (micro 16 x 8 GPUs) < 256
[dd] FINDINGS.md 1.4: below 256 the robust fusion LOSES to the plain teacher.
[dd] refusing to launch. Raise --micro-batch or --gpus, or set
     DD_ALLOW_SMALL_REAL_BATCH=1 to run it as a deliberate ablation.
```

This bites at 512px, where the micro-batch is smaller. Raise `--micro-batch`,
add GPUs, or run it knowingly as the `nogather` ablation. It is a correctness
requirement, not an optimisation.

## Datasets

Each dataset is one file in [`envs/`](envs/) holding its paths, resolution,
teacher, defaults and a `prepare_dataset()`. `--prepare` runs it; it skips work
that already exists.

```bash
python3 -m ddgpu.prepare latents  --source $IMAGENET_SRC --dest data/in256 \
    --resolution 256 --gpus 0,1,2,3,4,5,6,7
python3 -m ddgpu.prepare refstats --source $IMAGENET_SRC --dest data/in256 \
    --resolution 256 --n 50000 --gpus 0,1,2,3,4,5,6,7
```

For CIFAR-10, `--source cifar10` goes through torchvision. It does **not**
re-download a tarball you already have: `prepare.TorchvisionImages.find_root`
checks `$DD_DATA_ROOT`, then `<repo>/data`, then `~/.cache/dd-data` for either
`cifar-10-python.tar.gz` or an extracted `cifar-10-batches-py`, uses the first
hit, and prints which. `$DD_TV_ROOT` short-circuits that search and also
redirects the download when the file is genuinely absent.

When the file is genuinely absent and the box is ephemeral — Colab, CI, a fresh
VM — prefer **`--source cifar10-hf`**. torchvision downloads from
cs.toronto.edu, which throttles such boxes to ~100 kB/s (half an hour for
170 MB, restarted on every reconnect); the mirror reads the same images from
`uoft-cs/cifar10` on HF's CDN in about two seconds. That they hold the same
images was verified, not assumed: all 50 000 (image, label) pairs match the
canonical tarball (md5 `c58f30108f718f92721af3b95e74349a`) byte-for-byte, as a
set.

The **row order differs**, which matters in exactly one place. FID over the full
50 000 is permutation-invariant and so is unaffected; but a sub-sampled
reference (`--n <50000`) or precision/recall (first k rows) draws a *different*
subset here than it would from torchvision — equally valid, equally
distributed, and different by sampling noise. Pick one source per comparison
and stay with it.

Needs `pyarrow`; `tests/test_pipeline.py::t_cifar_mirror` pins the presentation
arithmetic so the two sources cannot silently drift apart.

`latents` writes an (N, 8, H/8, W/8) fp16 memmap of SD-VAE **moments** — the
latent is resampled every read, as DiT trains — plus `meta.json` carrying the
**measured** `sigma_data`. Disk: ~21 GB for ImageNet-256, ~84 GB for 512.

`refstats` writes the Inception features FID and precision/recall are measured
against, using `pytorch_fid`'s InceptionV3 (the canonical FID network) — the
same module `ddgpu.generate` uses for the fake side.

Nothing in a config needs to guess `sigma_data`, `shape`, `n_classes`, `arch`,
`latent_size` or `sigma_max`: they are read from `meta.json` and from the
teacher checkpoint at startup, and the result is written to
`<run>/config.resolved.json`.

## The teacher

**DiT-XL/2 is the only publicly released latent DiT** (FINDINGS.md §4.0.1), so
it is the teacher for every run. `train.sh` fetches and verifies it:

```bash
python3 -m ddgpu.ckpt --name DiT-XL-2-256x256 --dir ckpt   # or DiT-XL-2-512x512
```

That checkpoint is a **VP** model: it predicts ε from a discrete timestep on a
linear-β schedule, not an EDM-preconditioned x₀. `ddgpu/vp.py` presents it in σ
coordinates so the trainers are unchanged, and student and critic use the *same*
preconditioning so `init_from_teacher` is a genuine warm start. Two consequences
that are easy to get wrong and produce no error message:

* `sigma_max` is **157.4**, the top of the schedule — not EDM's 80. A one-step
  student starting at 80 begins half way up a schedule it was initialised from.
* training noise levels come from `sigma_dist="vp_uniform_t"` (DMD2's choice),
  not EDM's lognormal, which would never visit the top three quarters of the σ
  range the teacher was trained on.

The loader verifies the recomputed sin-cos positional grid against the one in
the checkpoint and **refuses to load** on a mismatch, because that failure is
invisible in every loss curve.

## Evaluation

```bash
scripts/eval_all.sh -d imagenet256 --gpus 0,1,2,3 --n 50000 --weights ema
```

Scores every run under `runs/` with identical sampler settings and prints the
comparison table. Two rules from `info.txt` are enforced rather than documented:

* **matched wall-clock, not matched steps.** Track A and Track B have different
  per-step costs, so equal-step comparisons systematically favour the more
  expensive method. `ddgpu.eval` raises rather than print a table when the runs
  differ by more than `--tol` (default 15%) in GPU-seconds. That refusal is the
  guard working.
* **precision and recall alongside FID, always.** Mode-seeking objectives buy
  FID with diversity and that trade hides inside a single number.

FID comes from the **EMA** weights by default (`--weights student` for the raw
ones); every DMD/DMD2 number in the literature is an EMA number.

## Figures

```bash
python3 exp/09_plot_run.py runs/imagenet256_robust
```

* `lambda_curve.png` — λ(σ) at each checkpoint. The first measurement of the
  optimal teacher-versus-data weight on a real image model; a result independent
  of whether FID moves.
* `lambda_drift.png` — the §3.1 test: does λ at *student* samples drift toward λ
  at *real* samples as the student converges? That is the mechanism claim from
  §1.2, and it rides along on runs we are doing anyway.
* `loss_curves.png` — triage.

## Interrupted runs

```bash
scripts/train.sh -d imagenet256 --mode robust --gpus 0,1,2,3,4,5,6,7 --resume
```

`--resume` with no argument means `auto` (latest checkpoint in the run dir) and
restores the student, the EMA, the critic and the accumulated **GPU-seconds** —
which matters, because the eval harness compares on GPU-seconds and a resumed
run that forgot its history would look artificially cheap.

## AutoDL specifics

If `/etc/network_turbo` exists it is sourced automatically. HF traffic goes
through `HF_ENDPOINT=https://hf-mirror.com` with the Xet backend disabled
(`HF_HUB_DISABLE_XET=1`); both can be overridden by exporting them yourself.

Disabling Xet is the setting that matters. `hf_xet` does its own networking and
ignores the `http_proxy` / `https_proxy` variables network_turbo exports, so
behind that proxy it does not fail — it crawls at single-digit kB/s while
printing a `reconstructing file` progress bar that looks like progress.

That used to apply only to the bash entry points. The plain `python3 -m ...`
commands — `ddgpu.teachers`, `ddgpu.prepare`, `exp/10_lambda_real.py` — inherit
nothing from `common.sh`, so `ddgpu/hfenv.py` now applies the same policy on
import of `ddgpu`, which is before any `huggingface_hub` import and therefore
before the moment those variables are read. On a non-AutoDL box (no
`/etc/network_turbo`) it disables Xet but leaves `HF_ENDPOINT` alone, rather
than routing traffic through a mirror that would be the slow option there.

The pipeline uses the machine's **current Python** (the AutoDL base env) and
pip-installs `requirements.txt` — it does *not* create a conda env, because the
AutoDL conda mirror often fails behind the turbo proxy. Set `DD_CONDA_ENV=<name>`
to opt in, `DD_SKIP_INSTALL=1` or `--skip-setup` to skip installs.

`requirements.txt` deliberately does **not** pin torch: AutoDL images ship a CUDA
build matched to the driver, and replacing it is the fastest way to break the
box. On a 5090 (sm_120) you need CUDA 12.8+ / torch ≥ 2.7 — check with

```bash
python3 -c "import torch;print(torch.__version__, torch.cuda.get_device_capability())"
```

Data and checkpoints default to the repo root. On AutoDL put them on the data
volume instead:

```bash
export DD_DATA_ROOT=/root/autodl-tmp/data DD_CKPT_ROOT=/root/autodl-tmp/ckpt
```

---

# The cheap tier (start here)

`FINDINGS.md` §6 replaces the ~5,000 GPU-hour ImageNet-256/512 programme with a
~230–455 GPU-hour one that keeps the dimension ladder. Everything above still
works and is what you run *if* the cheap tier says the effect is real.

Three datasets, three preconditioning families, one launcher:

| env | space | dims | teacher | why |
|---|---|---:|---|---|
| `cifar10` | pixel | 3,072 | `diffusers:google/ddpm-cifar10-32` (≈36 M) | cheapest arm that can move FID |
| `imagenet64` | pixel | 12,288 | `edm:edm-imagenet-64x64-cond-adm.pkl` (≈296 M) | the rung that makes the ladder credible |
| `in100_latent` | **latent** | 4,096 | `sit:...0300000.pt` (≈130 M, self-trained) | latent space + a weaker-teacher capacity probe |

Note the dimensions before reading this as a retreat: **ImageNet-64 in pixel
space is 3× higher-dimensional than ImageNet-256 in latent space** (12,288 vs
4,096), and CIFAR-10 is 75% of it. The VAE makes high *resolution* cheap, not
high *dimension* small.

## Phase A — the λ ladder (inference only, ~10–15 GPU-h)

No student, no critic, no optimiser. This is the paper's Figure 1 and the
cheapest high-information experiment in the programme.

```bash
# self-test first: a target whose true score is known, so the script proves
# itself before it is pointed at a model whose answer nobody knows
python3 exp/10_lambda_real.py --teacher "synthetic:c=4,hw=8,sd=0.5,bias=0.15" \
    --data "gaussian:c=4,hw=8,sd=0.5,n=4096" --batch 128 --batches 3 \
    --n-sigma 5 --sigma-min 0.05 --sigma-max 8 --tag selftest

# then the ladder
python3 exp/10_lambda_real.py --teacher diffusers:google/ddpm-cifar10-32 \
    --data data/cifar10 --tag cifar10_3072
python3 exp/10_lambda_real.py --teacher edm:$DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl \
    --data data/in64 --tag in64_12288
python3 exp/10_lambda_real.py --teacher dit:$DD_CKPT_ROOT/DiT-XL-2-512x512.pt \
    --data data/in512 --tag dit512_16384        # 16,384 dims, never trained there
python3 exp/10_lambda_real.py --teacher sit:$SIT_CKPT --data data/in100_256 \
    --tag sit_latent_4096
```

Read the output in this order:

1. **`gain_vs_teacher`** — held-out denoising loss at λ̂ divided by the loss at
   λ=1 (plain DMD2). Below 1.0 means the fusion reduces *true* score MSE; the
   identity `E‖λA+(1−λ)B−g‖² = MSE(λ)+const` makes that exact with no ground
   truth. λ̂ is estimated on one split and tested on a fresh one, so it is not
   circular. **If this is not below 1.0, the method does not work at that
   dimension** — a real negative result for a few GPU-hours.
2. **λ_dsm(σ)** — FINDINGS §1.1's shape on a real image model.
3. **The off-manifold panel** — ratio statistic at blurred/shifted/collapsed real
   data. A *lower bound* on λ*: compare its legs, never the levels, and never
   against panel 2.

Phase A **cannot** produce the drift result (§1.2/§3.1): calibration evaluates at
noised *real* data, and the DSM identity does not extend to student samples.
That needs a training run, which logs it via `probe_every`.

## Phases C/D/E — the distillation pairs

```bash
# C: CIFAR-10, 3 seeds per arm. One GPU; micro 512 clears the batch floors.
scripts/train.sh -d cifar10 --mode dmd2   --gpus 0 --micro-batch 512 --seed 0
scripts/train.sh -d cifar10 --mode robust --gpus 0 --micro-batch 512 --seed 0
#  ... repeat with --seed 1 and --seed 2

# D: ImageNet-64 (required, not optional -- it is the ladder's high rung)
export EDM_REPO=/root/autodl-tmp/edm      # git clone https://github.com/NVlabs/edm
scripts/train.sh -d imagenet64 --mode dmd2   --gpus 0,1,2,3 --prepare
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3

# E: the latent leg, with your own SiT teacher
export REPA_DIR=/root/autodl-tmp/repa-surgery/REPA
export SIT_CKPT=$DD_CKPT_ROOT/imagenet100_sit-b_2_baseline/checkpoints/0300000.pt
scripts/train.sh -d in100_latent --mode dmd2   --gpus 0,1,2,3
scripts/train.sh -d in100_latent --mode robust --gpus 0,1,2,3
```

**Three seeds is not optional at CIFAR scale.** One-step CIFAR-10 distillation is
near-saturated, so the effect may be smaller than run-to-run variance and one run
per arm cannot tell the difference. `eval_all.sh` prints the seed-grouped table
with an explicit standard-error verdict:

```
best=cifar10_dmd2 (3.200) vs cifar10_robust (3.217); gap 0.017, se 0.105 -> 0.2 se
VERDICT: the gap is INSIDE one standard error. Report this as no measured
         difference, not as a win.
```

## Getting the teachers

```bash
# CIFAR-10: nothing to do, diffusers fetches and caches on first use.

# ImageNet-64 (EDM). The pickle needs NVlabs' source importable.
git clone https://github.com/NVlabs/edm /root/autodl-tmp/edm && export EDM_REPO=/root/autodl-tmp/edm
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl \
     -O $DD_CKPT_ROOT/edm-imagenet-64x64-cond-adm.pkl

# ImageNet-100 latent leg: the SiT checkpoint + images from HuggingFace.
# >130 GB -- fetch_hf.py REFUSES to start with less than 200 GB free, because a
# partial snapshot_download looks exactly like a complete one.
python3 scripts/fetch_hf.py --repo pakhomovee/imagenet --repo-type dataset \
    --dest $DD_DATA_ROOT/in100_raw --dry-run     # check space first
```

## Verify every teacher before trusting a number

Four preconditioning families now share one trainer, and a mismatch between any
of them and its checkpoint produces **no exception** — LOG.log ENTRY 012 is the
story of exactly that. So check, and record the numbers:

```bash
python3 -m ddgpu.teachers --teacher diffusers:google/ddpm-cifar10-32 --data data/cifar10
```

```
{"rel_mse@0.01": 0.0004, ..., "identity_err": 1.6e-11, "monotone": true, "verdict": "OK"}
```

`rel_mse` should be ~0 at small σ and rise toward 1 at large σ. A teacher wrapped
with the wrong noise convention shows it **flat and near 1 everywhere**, because
the network is being evaluated at a noise level unrelated to the one applied.
`ddgpu.train` runs this before every run and refuses to start on `SUSPECT`
(override with `--set strict_teacher=false`, deliberately).
