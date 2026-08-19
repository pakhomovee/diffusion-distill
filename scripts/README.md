# Running the experiments on a GPU box

One entry point — [`train.sh`](train.sh) — sets up the environment, fetches the
teacher, optionally builds the dataset, and launches `ddgpu.train` under
`torchrun`. The layout mirrors `repa-surgery/training/`, so a box configured for
that repo is already configured for this one.

```
scripts/
  smoke.sh        pre-flight: tests + both tracks on CPU (+ --gpu for the VRAM probe)
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
