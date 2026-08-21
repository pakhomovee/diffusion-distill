# Run plan — the cheap programme, on RTX 4090s

Target hardware: **NVIDIA RTX 4090, 24 GiB, 1–4 cards depending on phase.**
No NVLink (PCIe only), so DDP with full local replicas everywhere.

> **Scope: the cheap tier only.** FINDINGS.md §6 / LOG.log ENTRY 013 replace the
> ~5,000 GPU-hour ImageNet-256/512 programme with a ~130–200 GPU-hour one that
> keeps the dimension ladder (3,072 → 12,288 trained, → 16,384 measured) while
> cutting teacher parameters 19×. **That is the whole plan now.** The deferred
> DiT-XL programme is preserved in §7 and is not scheduled; it becomes live only
> if Phase A's held-out gain and Phase C/D's seeded FIDs both come out positive.

> The four teachers span **four preconditioning families** (VP ε, EDM denoiser,
> linear interpolant velocity, and the synthetic self-test). A mismatch between
> a family and its checkpoint raises **no exception** — LOG.log ENTRY 012 is the
> story of exactly that — so `validate_teacher` runs before every run and every
> λ measurement. See §6.

Ordered commands from a bare box to Phase E: **[`RUNBOOK.md`](RUNBOOK.md)**,
which carries the GPU count for every individual step.

---

## 0. What has actually been measured

`scripts/smoke.sh --gpu` on one RTX 4090 — full output in
`results/probe_4090_smoke.txt`. All three test suites pass, both tracks and both
cheap-tier arms train, and the Phase A self-test recovers λ* on a target whose
score is known (`gain_vs_teacher` 0.15 at σ=8, λ̂ 0.42 vs grid argmin 0.40).

The probe's memory rows, measured on the 4090:

```
DiT-B/2   res 32  dmd2        micro 128  ->  7.07 GiB   0.83 s/step
DiT-B/2   res 64  dmd2        micro 128  -> 18.49 GiB   3.51 s/step
DiT-B/2   res 64  invertible  micro 128  ->  OOM
DiT-XL/2  every res, every method, micro 8..128  ->  OOM
```

`ddgpu.probe` only builds DiT backbones, so **it cannot measure the cheap tier's
UNets**. Everything in §2 below is exact for optimiser state and *unmeasured*
for activations; §5 says how to close that gap in twenty steps.

---

## 1. GPU count is set by the batch floor, not by VRAM

FINDINGS §1.4, measured across 234 cells: with an effective real batch of
**N=64** the doubly-robust fusion is *worse* than the plain teacher — mean MSE
ratio 1.14, worst case **5.7×**, and every one of the five worst cells in the
sweep is an N=64 cell. At N=256 the mean is 0.91 and the worst 1.15.

`robust.py:all_gather_batch` gathers the real batch across ranks, so the
effective N is `world_size × micro_batch`, and `scripts/lib/common.sh` refuses
to launch below 256. That single inequality determines every GPU count in §3:

```
world_size × micro_batch ≥ 256          hard, launcher refuses below it
world_size × micro_batch ≥ 512          soft, warning only
```

The second line is a *different threshold on a different quantity*:
`LambdaEstimator.calibrate` splits the gathered batch — half supplies
calibration points, half supplies `B` — so the calibration leg's empirical score
sees `eff/2`. Below 256 there, calibration measures the optimal weight for a
worse `B` than training actually uses, which biases λ toward the teacher.
Conservative rather than catastrophic, hence a warning (FINDINGS §5).

**Consequence for renting**: more cards buys correctness, a bigger card does
not. A 24 GiB card at micro 256 and a 48 GiB card at micro 512 both satisfy the
floor on one GPU; four 24 GiB cards at micro 64 satisfy it at a quarter the
activation memory each. That is why the plan is 4×4090 and not 1×anything.

---

## 2. What fits in 24 GiB

Optimiser state is exact arithmetic, not a model. Per trainable copy fp32 Adam
is 16 B/param (params + grads + m + v); the EMA shadow is a fp32 `deepcopy` of
the student (4 B/param, always on — `ema_decay` defaults to 0.999); the frozen
teacher is loaded fp32 and run under bf16 autocast (`ddgpu/teachers.py` never
casts the weights), so 4 B/param:

```
state = 2 × 16 B/param   (student + critic)
      +     4 B/param    (EMA shadow)
      +     4 B/param    (frozen teacher)
      = 40 B/param
```

| arm | teacher | params | state | free on a 24 GiB card |
|---|---|---:|---:|---:|
| C · CIFAR-10 32px pixel | `diffusers:google/ddpm-cifar10-32` | 35.7 M | **1.33 GiB** | ~20.7 GiB |
| D · ImageNet-64 pixel | `edm:...-cond-adm.pkl` | 295.9 M | **11.02 GiB** | ~11.0 GiB |
| E · IN-100 256px latent | `sit:...0300000.pt` | 130.0 M | **4.84 GiB** | ~17.2 GiB |
| A · λ ladder, any leg | any | — | forward only, no optimiser | ~21 GiB |
| *(deferred)* DiT-XL/2 | `dit:DiT-XL-2-*.pt` | 674.8 M | **25.14 GiB** | **−3.1 GiB** |

The last row is why the probe OOMed on DiT-XL/2 at micro 8: it does not run out
of room for activations, it runs out of room for the *weights*. Nothing about
micro-batch can fix that.

The `dmd2` arm additionally trains a ~3 M-parameter `gan.ConvGANHead`
discriminator (~0.05 GiB of state). The `robust` arm has none — that is the
claim being tested — so the method arm is, if anything, the cheaper one.

**What is not in the table: activations.** The cheap tier's students and critics
are `deepcopy` clones of a UNet teacher (`train.clone_trainable`), and **the
clone path carries no gradient checkpointing** — `grad_ckpt` is a `ddgpu/dit.py`
option and these are not DiTs. So activation memory is whatever the third-party
UNet does at the chosen micro-batch, unmeasured, and Phase D is the one arm
where 11 GiB of state plus that unknown could plausibly exceed 24 GiB. §5.

---

## 3. Per-phase allocation

Wall-clock is an **estimate** — FLOP model at 165 TFLOP/s peak bf16, MFU 0.30
for the UNets and 0.35 for the SiT, `STEP_COST["dmd2"] = 11` forward-equivalents
per step (`ddgpu/memcalc.py`). Replace it with the measured `s_per_it` the
trainer prints; see §5.

| phase | what | GPUs | micro | effective N | state/GPU | est. wall-clock | est. GPU-h |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** | λ ladder, all legs, inference only | **1** | batch 512 | — | <2 GiB | ~2 h total | ~2–4 |
| **C** | CIFAR-10 `dmd2` vs `robust`, 3 seeds each (6 runs) | **1 per run**, 4 in parallel | 512 | 512 | 1.3 GiB | ~3.8 h/run → ~8 h in 2 waves | ~23 |
| **D** | ImageNet-64 pair (2 runs) | **4** | 64 | 256 | 11.0 GiB | ~8 h/run → ~16 h | ~65 |
| **E** | IN-100 latent pair (2 runs) | **4** | 64 | 256 | 4.8 GiB | ~3 h/run → ~6 h | ~25 |
| — | prep, teacher fetch, evals, figures | 1–4 | — | — | — | ~4 h | ~10 |
| | | | | | | **~1.5 days** | **~130** |

FINDINGS §6.3's envelope for the same programme is 230–455 GPU-h. The gap is
MFU optimism on my side versus deliberate conservatism on theirs; treat 130 as
the floor and 455 as the ceiling, and let the first measured `s_per_it` decide
which end you are at. Either way it is 10–35× under the deferred §7 plan.

**Phase C parallelism.** Six single-GPU runs on a 4-card box is two waves of
4 + 2, each run launched with its own `--gpus <id>`. `torchrun --standalone`
rendezvouses on `localhost:0` — a random free port — so concurrent launches do
not collide, and `MASTER_PORT` does not need setting. RUNPLAN's old warning
against co-locating runs still stands and is different: it is about *two runs
per card*, which halves the usable micro-batch and breaks the batch floor.

**Phase D fallback.** If micro 64 OOMs on 24 GiB, go to **8×4090 at micro 32**
(8 × 32 = 256, floor still met, half the activations per card). Do *not* go to
2 GPUs at micro 128 — that raises per-card activations rather than lowering
them. Do *not* set `DD_ALLOW_SMALL_REAL_BATCH=1`; that converts the run into the
`nogather` ablation.

**Track B is not in this programme.** The cheap tier is Track A only
(FINDINGS §6.3). If you want a cheap Track B signal, `-d cifar10 --mode invert`
on **1 GPU** is the fastest way to reach its kill criterion: watch
`DIAG eff_rank_frac` in the first 500 steps, and if it falls away from 1.0 the
encoder is collapsing onto a subspace and the run is dead (FINDINGS §3.3,
LOG ENTRY 004). The loss curves will not tell you this.

---

## 4. Sequencing and gates

**Do not launch a phase until the previous gate passes.**

| # | phase | gate to clear before the next |
|---|---|---|
| 0 | `scripts/smoke.sh --gpu`, teacher validation | every `verdict` is `OK`; `rel_mse` ≈0 at small σ rising toward 1 at large σ |
| 1 | **A** — λ ladder | `gain_vs_teacher < 1.0` at CIFAR's 3,072 dims. **If not, stop.** That is a real negative result for one GPU-hour |
| 2 | **C** — CIFAR pair, 3 seeds | read the *seed-grouped* table. A gap inside one standard error is "no measured difference", not a win |
| 3 | **D** — ImageNet-64 pair | does the effect survive 4× the dimension? This is the rung that makes the ladder credible, and it is required, not optional |
| 4 | **E** — IN-100 latent pair | does it hold in latent space, with a deliberately weaker teacher? λ should sit *lower* if λ is driven by teacher bias |
| 5 | decide on §7 | only if A and C/D are both positive |

Phase A's ladder legs beyond CIFAR (ImageNet-64, DiT@512, SiT) need their
datasets prepared, so in practice they run alongside phases D and E rather than
all up front. The CIFAR leg is the gate and it needs nothing but CIFAR-10.

**Phase A cannot produce the mechanism result.** `LambdaEstimator.calibrate`
evaluates at noised *real* data, and the DSM identity does not extend to student
samples — `g = −ε/σ` is unbiased for the score of whatever distribution `x0` came
from. §1.2/§3.1's drift claim needs a student, i.e. Phase C, where `probe_every`
logs `PROBE real:lam@med` against `student:lam@med` for free.

---

## 5. The one number this plan cannot derive, and how to get it

Activation memory and step time for third-party UNets. `ddgpu.probe` builds DiTs
only, and `memcalc`'s activation model is transformer-shaped — FINDINGS §6.3
says so explicitly about exactly these estimates.

Twenty steps settles both, per arm, before committing hours:

```bash
scripts/train.sh -d imagenet64 --mode robust --gpus 0,1,2,3 --steps 20
nvidia-smi --query-gpu=memory.used --format=csv    # from a second shell
```

The trainer prints `s_per_it` on every log line and accumulates `gpu_hours`, so
after twenty steps:

```
wall-clock hours = steps × s_per_it / 3600
```

If that disagrees with §3, **the measurement wins** and §3 should be corrected in
place. This is the same rule the old plan applied to `ddgpu.probe`, and it is the
reason the smoke script exists.

---

## 6. Things that will bite

- **bf16 cannot express a VP one-step student's output at σ_max.** This one is
  measured, not predicted: it destroyed the first CIFAR programme. The VP
  ε-parameterisation is `D = x − σ·ε̂` (`c_skip = 1`, `c_out = −σ`), so both
  terms are O(σ) and the answer is only O(σ_data) — an error `d` in `ε̂` lands
  in `D` multiplied by σ. bf16's ulp is 2⁻⁷, so at σ_max = 157.4 with
  σ_data = 0.5 the rounding noise has std **0.26 against a 0.5 signal, an SNR
  of 1.9**. A one-step student generates at σ_max on *every* sample, so it
  cannot emit a clean image no matter how well it is trained; the samples come
  out as real structure buried in speckle and FID lands near 325 instead of
  single digits. Both arms fail identically, which makes it look like a
  method-independent training failure rather than an arithmetic one.
  `VPPrecond.forward` now runs the network in fp32 whenever
  `σ·ulp(bf16) > 0.05·σ_data` (σ ≳ 3.2 at σ_data = 0.5); `score` needs no
  guard because it *divides* by σ. **EDM and interpolant preconditioning are
  immune** — their `c_out` tends to σ_data and −1 respectively, which is what
  that preconditioning is for, so Phases D and E are not exposed. Phase A is
  also unaffected: `exp/10_lambda_real.py` and `validate_teacher` never
  autocast.

- **Fixing the arithmetic does not fix the conditioning, and the conditioning is
  the bigger problem.** The bullet above is about *rounding*; this one is about
  what remains when rounding is gone. `D = x − σ·ε̂` amplifies **any** error in
  `ε̂` by σ, whatever its source — rounding, bias, or simply not having trained
  long enough. At σ_max = 157.4 that is a factor of 157; EDM's `c_out` at the
  same σ is σ_data = 0.5, so **the VP parameterisation demands ~315× more
  relative accuracy from the network for the same picture**:

  | target image SNR | noise in `D` | VP: error in `ε̂` | EDM: error in `F` |
  |---|---|---|---|
  | 1  | 0.500  | 3.2e-03 | 1.0e+00 |
  | 10 | 0.050  | 3.2e-04 | 1.0e-01 |
  | 30 | 0.017  | 1.1e-04 | 3.3e-02 |

  Measured on the first post-guard CIFAR grid (`scripts/sample_stats.py`): the
  student's `ε̂` is accurate to **6.6e-4**, giving image SNR **4.8** — grainy,
  structureless, and a systematic colour cast from a ~1.1e-3 *bias* in `ε̂`
  arriving as 15–30 levels. The network is not bad: behind EDM preconditioning
  that same 6.6e-4 would give SNR ~1500. It is the parameterisation that is
  expensive, and it is expensive in a way that **hurts both arms equally** —
  which is exactly why the first programme's two arms agreed to within 1 FID
  point of each other at FID ~325. A common-mode error this large swamps the
  treatment effect the experiment exists to measure.

  Three ways out, cheapest first. (a) **Lower σ_max**: error in the image is
  linear in the σ a one-step student starts from, so `--set sigma_max=40` cuts
  the grain 4× for nothing. It is a retrain, not a re-score — σ_max is where
  the generator is *trained*. (b) **Use an EDM CIFAR teacher** instead of
  `diffusers:google/ddpm-cifar10-32`; `edm:` is already a supported family
  (Phase D uses it) and removes the amplification by construction, at the cost
  of an NVlabs/edm clone and a `.pkl`. (c) **Train longer** — SNR 30 needs `ε̂`
  6× more accurate than it is now, which is the worst value of the three.

- **A teacher wrapped in the wrong preconditioning raises nothing.** Four
  families share one trainer (VP ε, EDM denoiser, interpolant velocity,
  synthetic). `validate_teacher` is what notices: `rel_mse` should be ~0 at
  small σ and rise toward 1 at large σ; **flat and near 1 everywhere means the
  network is being evaluated at a noise level unrelated to the one applied**.
  `ddgpu.train` refuses to start on `SUSPECT`. LOG ENTRY 012.
- **σ_max is per-family and it is not 80.** VP teachers (`diffusers` DDPM, DiT)
  top out at **157.4**; EDM at 80; the SiT interpolant near 49. A one-step
  student starting at the wrong σ_max begins half way up a schedule it was
  initialised from, and nothing errors. `exp/10_lambda_real.py` defaults to 80,
  so **pass `--sigma-max` explicitly** and cross-check it against the teacher
  meta printed at startup.
- **FID at matched wall-clock, not matched steps.** `ddgpu.eval` raises rather
  than print a table when runs differ by more than `--tol` (15%) in GPU-seconds.
  That refusal is the guard working.
- **Precision and recall alongside FID, always.** Mode-seeking objectives buy
  FID with diversity and that trade hides inside a single number.
- **Three seeds is not optional at CIFAR scale.** One-step CIFAR distillation is
  near-saturated (published FIDs ~2–4); the effect may be smaller than
  run-to-run variance, and one run per arm cannot tell the two apart.
  `ddgpu.eval --seeds` refuses to call a win inside one standard error, or with
  fewer than 3 seeds per arm.
- **The cheap tier's baseline FID is ours, not a reproduction.** Its teachers
  expose no token trunk, so the discriminator is a standalone `ConvGANHead`
  rather than DMD2's critic-feature head. Both arms use the identical head and
  the method arm uses none, so *our* comparison stands — the absolute number is
  not comparable to DMD2's published one.
- **Resuming.** `--resume` restores the student, EMA, critic and the accumulated
  **GPU-seconds**, which matters because the eval harness compares on
  GPU-seconds and a resumed run that forgot its history looks artificially cheap.

---

## 7. Deferred: the ImageNet-256/512 programme

Not scheduled. Kept because if the cheap tier's gates pass, this is what the
result justifies spending, and the pipeline for it is already built and tested.

**It needs ≥32 GiB cards, and more of them than the old table admitted.** The
original arithmetic here counted two trainable copies plus a bf16 frozen teacher
= 21.4 GiB for DiT-XL/2, and concluded a 32 GiB 5090 fits "with 8.6 GiB to
spare". The shipped code also keeps an fp32 EMA shadow and an fp32 teacher, so
the real figure is **25.14 GiB** (§2) and the spare on a 5090 is ~4.9 GiB of a
~30 GiB usable budget, at micro 32 with grad checkpointing on. That is not
impossible — DiT *does* get gradient checkpointing — but it is tight, and it
should be re-probed on the actual card before anyone books a node. On 24 GiB it
is arithmetically impossible, which the 4090 probe confirmed at every
micro-batch.

The rest of the old plan stands as written: 8×5090 for one node,
`{baseline, robust} × {256, 512}` as the controlled dimension ladder at fixed
parameters, DiT-XL/2 @256 as the headline pair, Track B invertible @256 as the
cheapest configuration that can fail fast, ~2 weeks of node time, and an SDXL
LoRA tier behind it (`ddgpu/lora.py`'s `AdapterSet` is unit-tested for adapter
isolation but has never met a real SDXL UNet). Regenerate its table with:

```bash
python exp/plan_gpus.py       # RTX 5090 assumptions, DiT backbones
```

Two structural facts from it that outlive the deferral:

- **Track B is critic-free by construction** — the encoder E *replaces* the
  fake-score network rather than adding to it, so Track B and the DMD2 baseline
  have the same VRAM footprint and can be compared without a memory confound.
  Add a critic back alongside E and you land on three trainable copies, where
  DiT-XL/2 stops fitting on any consumer card.
- **No NVLink is an architecture decision, not a footnote.** FSDP all-gathers
  parameters every layer; over PCIe (~50 GB/s effective versus 900 GB/s NVLink)
  that dominates the step. Use DDP with full local replicas wherever they fit,
  reach for ZeRO-2 before ZeRO-3, and full FSDP only for SDXL.

---

## 8. Track A has no tuned λ hyperparameter

The λ weight is the minimiser of a **held-out denoising loss**, computed online:

```
λ(σ) = E⟨A − B, g − B⟩ / E‖A − B‖²      g = −ε/σ on held-out real data
```

This is exact — `E‖λA+(1−λ)B − g‖² = MSE(λ) + const(σ)` — so it needs no ground
truth, no gate, and nothing calibrated per dimension. It replaced a σ-threshold
that was measured to fail at d=512 (3.2× worse than baseline, 21× in its worst
bucket) because the crossover it stood in for moves with dimension and with
teacher quality. Full derivation and numbers in LOG.log ENTRY 011.

Cost: one teacher forward on half the real batch every `lam_calib_every` (=4)
steps, plus one `cdist`. Under 3% of a step.

**Do not overlap the calibration split with the empirical-score split.** Half the
real batch supplies calibration points, the other half supplies `B`. If they
overlap, `B` has seen the sample it is being scored against and memorises it,
biasing the calibration in exactly the direction the method is about.

---

## 9. Sanity checks to run before each long job

```bash
scripts/smoke.sh --gpu      # invariant tests + both tracks + the VRAM/throughput probe
```

Tests, both tracks, both cheap-tier arms and the Phase A self-test, in order,
failing fast. On a fresh box also validate every teacher you are about to trust,
because that is the check with no exception behind it:

```bash
python3 -m ddgpu.teachers --teacher diffusers:google/ddpm-cifar10-32 --data data/cifar10
```

```
{"rel_mse@0.01": 0.0004, ..., "identity_err": 1.6e-11, "monotone": true, "verdict": "OK"}
```
