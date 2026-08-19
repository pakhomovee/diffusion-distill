# Findings and research direction

Consolidated state as of 2026-08-19. Companion documents: `LOG.log` (the running
narrative, 13 entries, chronological) and `RUNPLAN.md` (GPU budget and memory math).
This file is the decision document: what we learned, what to pursue, and what each
GPU run buys.

---

## 1. What was measured, and what it changed

Everything below was run on CPU against targets with a **closed-form score**, which
is the one class of experiment a GPU cannot do better. Total cost: a few hours of
two cores.

### 1.1 The standard explanation for real-data terms is wrong

DMD2, ADD and LADD all add a real-data adversarial term with a hand-tuned weight,
and the usual story is that it corrects the teacher's bias at **low** noise.

Measured against ground truth (`exp01`), the variance-optimal weight λ*(σ) on the
teacher is:

| σ/σ_data | 0.01 | 0.11 | 0.43 | 1.10 | 1.77 | 2.85 | 7.37 | 19.1 |
|---|---|---|---|---|---|---|---|---|
| **λ\*** | 1.00 | 1.00 | 0.99 | 0.86 | 0.62 | 0.48 | 0.42 | 0.22 |

Teacher-heavy at *low* noise, data-heavy at *high* noise — the opposite shape.
The mechanism is not subtle: the minibatch empirical score is a softmax over real
samples, which collapses onto the nearest neighbour as σ→0. It **memorises**, with
variance ~σ⁻⁴, while the teacher's bias falls faster. There is no low-noise regime
in which a finite real batch beats a trained teacher.

**Consequence:** whatever the adversarial term is doing in DMD2/ADD/LADD,
"correcting teacher bias at low noise on the data manifold" is not the mechanism.

### 1.2 …but the picture inverts off the data manifold, which reframes it

In real distillation the score is queried at **student** samples, not noised real
ones. Simulating that (`exp04`, d=128, N=256):

| eval points | λ\* at low σ | var_B / ‖s‖² | bias_A / ‖s‖² | oracle MSE gain |
|---|---|---|---|---|
| on-distribution | 1.00 | 90.8 | 0.032 | 1.00 (no gain) |
| blurred student | 0.76 | 0.6 | 0.376 | 0.63 |
| shifted student | 0.77 | 0.6 | 0.372 | 0.63 |
| **mode-collapsed student** | **0.58** | 0.6 | 0.852 | **0.42** |

Both terms move: the batch estimator's variance drops **150×** and the teacher's
bias rises **27×**. So the real-data term *is* valuable at low noise at student
samples — and most valuable exactly when the student has collapsed.

**The corrected mechanism:** the real-data term corrects the teacher **off the
manifold, where the teacher was never trained** — not on it. That is a different
claim from the folklore, and it makes a prediction the folklore does not:
*the term's value should shrink as the student converges.* See §3.1.

### 1.3 A σ-threshold rule collapses at high dimension; the fix removes it

The online rule `λ̂ = V_B / E‖A−B‖²` drops an unmeasurable term and under-shoots
badly at low σ. Gating λ=1 below a σ threshold fixes it — at d ≤ 128. At d=512:

| d | oracle λ\* mean | oracle best | σ of best | **σ-gate mean** | **σ-gate worst** |
|---|---|---|---|---|---|
| 8 | 0.905 | 0.622 | 10.5 σ_d | 0.921 | 1.00 |
| 32 | 0.849 | 0.548 | 3.4 σ_d | 0.857 | 1.00 |
| 128 | 0.907 | 0.623 | 2.0 σ_d | 0.943 | 1.15 |
| **512** | 0.869 | 0.493 | **0.19 σ_d** | **2.989** | **21.0** |

The *oracle* is fine at d=512 — the idea does not break with dimension. The
estimator-plus-gate breaks: 3× worse than just using the teacher. The teacher's
bias is 13× larger at d=512, which moves the variance/bias crossover below the
gate, so the gate forces λ=1 exactly where λ should be small.

A rule needing per-dimension re-tuning is unusable at 4096 or 16384 dims.

**The fix (`exp08`), which is the strongest result here.** On held-out real data an
unbiased estimate of the true score is free: `x_t = x₀ + σε` gives `g = −ε/σ` with
`E[g | x_t] = s(x_t)` exactly. A is deterministic given x_t and B depends only on
the training batch, so both are conditionally independent of the noise in g:

```
E‖ λA + (1−λ)B − g ‖²  =  MSE(λ)  +  const(σ)
```

Minimising a **held-out denoising loss** minimises the true score MSE *exactly*,
and the minimiser is closed form:

```
λ(σ) = E⟨A − B, g − B⟩ / E‖A − B‖²
```

No ground truth, no gate, nothing calibrated per dimension. At d=512 it reproduces
λ* to three decimals (0.985/0.809/0.515/0.062 against 0.985/0.808/0.516/0.062),
taking the method from **3.19× worse than baseline to 0.920** — within 0.007 of the
oracle. Track A now has **zero tuned hyperparameters** in its λ path.

Implementation detail that is not optional: the real batch must be **split**, half
for calibration points and half for the empirical score. If they overlap, B has
seen the sample it is being scored against.

*Negative result worth not retrying — and it is narrow.* Before arriving at the
formula above I tried to estimate the missing bias term `b_B` directly, by
comparing the empirical score at N samples against N/2 (Richardson extrapolation).
The reasoning: `b_B` vanishes as N→∞, so how the estimate *moves* with N should
reveal its size. It does not. At low σ the empirical score is dominated by the
nearest real sample, and nearest-neighbour distance among N points in d dimensions
shrinks like N^(−1/d). At d=512 that exponent is −0.002, so doubling N changes the
nearest-neighbour distance by 2^0.002 ≈ 1.001 — nothing. `B_{N/2} ≈ B_N`, and the
probe carries no information about the bias.

**This does not say the method fails.** It says one probe fails. `λ_dsm` never
estimates `b_B` at all; it sidesteps the term entirely. The note exists so nobody
retries Richardson.

### 1.4 Hard precondition: effective real batch ≥ 256

| N | mean MSE ratio | p90 | worst |
|---|---|---|---|
| 64 | 1.141 | 1.204 | **5.70** |
| 256 | 0.907 | 1.000 | 1.15 |
| 1024 | 0.910 | 1.000 | 1.00 |

At N=64 the method *loses*. Every one of the five worst cells across 234 is an
N=64 cell. Per-device micro-batch in the run plan is 32–128 — inside the failure
regime. `robust.py:all_gather_batch` takes N to `world_size × micro`; 8×32 = 256 is
the first safe row. **This is a correctness requirement, not an optimisation.**

### 1.5 Track B's Gaussianity term is provably insufficient alone

Detection power at 5% FPR, 100+100 replicates, B=256 (`exp02`):

| perturbation | stat | d=256 | d=1024 | d=4096 | d=16384 |
|---|---|---|---|---|---|
| mean 0.05 | all three | 1.00 | 1.00 | 1.00 | 1.00 |
| scale 2% | moment | 0.83 | 1.00 | 1.00 | 1.00 |
| | mmd | 0.29 | 0.77 | 1.00 | 1.00 |
| | ksd | 0.06 | 0.04 | 0.08 | **0.00** |
| lowrank 10% *(collusion signature)* | moment | 1.00 | 1.00 | 1.00 | 1.00 |
| | mmd | 0.17 | 0.09 | 0.10 | 0.00 |
| | ksd | 0.27 | 0.13 | 0.18 | 0.00 |
| **mixture** *(mean AND covariance matched)* | moment | 0.11 | 0.06 | 0.06 | 0.01 |
| | mmd | 0.06 | 0.05 | 0.07 | 0.00 |
| | ksd | 0.06 | 0.05 | 0.06 | 0.00 |
| corr 0.10 | ksd | 0.10 | 0.13 | 0.31 | **0.50** |

Four things follow:

1. **The cheap O(Bd) moment penalty beats both kernel discrepancies** on everything
   except correlation, at ~100× lower cost. Deflating for a Stein-based method.
2. **KSD is blind to isotropic scale error and gets blinder with d** (0.06 → 0.00).
3. **KSD's one real win is correlation, improving with both d and B** — 0.98 at
   d=16384, B=512 versus 0.21 at B=64. That is why the code all-gathers z.
4. **Nothing detects matched-moment non-Gaussianity**, at any dimension, and the
   gap does not close with batch size up to B=512. So the Gaussianity term
   *cannot* rule out a colluding encoder. **The teacher-ODE anchor is
   load-bearing**, not insurance; `w_anchor=0` is an ablation showing failure.

### 1.6 Two bugs that a single-GPU smoke test cannot catch

- **The encoder started at exactly the collapsed solution.** DiT zero-initialises
  its final layer — correct for a denoiser, whose EDM skip carries the signal — but
  an encoder has no skip, so a zero-init encoder outputs identically zero. Only the
  effective-rank diagnostic showed it; the loss curves looked fine. Adding a skip
  took effective rank from 0.06 to 0.88.
- **DDP:** `.score()` on a DDP-wrapped module is an AttributeError the moment world
  size > 1, and multiple grad-tracked forwards per backward trip the reducer. Both
  surface only on the first multi-GPU launch.

---

## 2. Where each track stands

| | Track A (doubly-robust) | Track B (invertible) |
|---|---|---|
| **Core mechanism** | validated to oracle accuracy on closed-form targets | structurally sound, untested at scale |
| **Hyperparameters** | zero in the λ path | ~5 loss weights, not yet swept |
| **Known failure mode** | none outstanding | encoder collusion; anchor is the only defence |
| **Memory vs DMD2** | identical | identical (E replaces the critic) |
| **Publishes if flat?** | yes — the contribution is explanatory | only via the inversion/editing fallback |
| **Biggest open risk** | 6–9% mean score-MSE may not move FID | Gaussianity term has a proven blind spot |

`exp06` (end-to-end toy distillation, in flight) is the first test of whether the
estimator improvement moves *samples*. Partial results at 3000 steps: energy
distance tied (4.17 baseline vs 4.20 robust), log-likelihood clearly better for
robust (35.0 vs 18.0 against a real-data 47.6), but **both students are heavily
mode-collapsed** (recall 0.08–0.10), so neither is near convergence and the
comparison is not yet meaningful. Re-running with a longer schedule and a fixed
coverage metric (the first version saturated at its floor — now corrected).

**Until exp06 or a real run lands, Track A is a well-measured estimator result, not
a demonstrated distillation improvement.** That distinction should survive into
any paper draft.

---

## 3. Most promising paths, in priority order

### 3.1 Highest value, nearly free: λ against training progress

§1.2 produced a mechanism claim — the real-data term corrects the teacher *off the
manifold* — and that claim makes a falsifiable prediction the folklore does not:
**λ should drift toward 1 as the student converges onto the data manifold.**

This costs almost nothing to test. The λ(σ) curve is already logged every
checkpoint. The only addition is a second calibration pass at *student* samples
alongside the held-out one, and a plot of the two curves over training.

Why this is the best next thing:
- It is the difference between "we tuned a weight automatically" (mildly
  interesting) and "we identified what the real-data term actually does, predicted
  its behaviour, and confirmed it" (a paper's thesis).
- If λ drifts toward 1, it directly motivates an **annealed real-data weight**,
  which nobody does, and which falls out of the theory rather than a sweep.
- If λ *doesn't* drift, the off-manifold explanation is wrong and we need a third
  hypothesis — still a result, and better to know before the XL runs.
- It rides along on runs we are doing anyway. Marginal cost ≈ 0.

### 3.2 Track A through the dimension ladder, then XL

The estimator is validated up to d=512 synthetic. Real latents are 4096 (256px)
and 16384 (512px) — 8× and 32× beyond the ladder. That gap is the single largest
unverified assumption in the whole programme, and §1.3 is a worked example of
exactly this kind of assumption failing silently. Runs 1→2 close it cheaply.

### 3.3 Track B, but gated hard on the collusion diagnostic

Worth doing, at DiT-B/2 only, and with an explicit kill criterion. The
`eff_rank_frac` diagnostic reports within the first few hundred steps; if it falls
below ~0.8 and keeps falling, the encoder is collapsing and the run should be
killed rather than left to finish. The fallback (native inversion and editing for
one-step models) is a real open problem with an established evaluation protocol,
so a partial success still yields something.

### 3.4 Deprioritise

- **Pure KSD/Stein distillation (info.txt #1) in data space.** §1.5 already shows
  that against a *known* target the kernel machinery barely beats moment matching.
  Against an unknown data distribution with a network-supplied score it will be
  worse. The Fisher-kernel geometry idea (#4) is worth keeping as an ablation
  inside Track B, not as a track.
- **DRO over noise level (info.txt #5).** Now partly redundant: λ(σ) already tells
  us where the fusion helps, and §1.3 shows that band moves with dimension. A
  DRO schedule would be re-deriving information we now measure directly. Keep as a
  two-page component if a paper needs one.

---

## 4. What each GPU run buys

### 4.0 How to read the run table (two things I stated ambiguously)

**"Either track" means TWO runs, not one.** Rows 2 and 3 are each a *pair* —
baseline and robust — because the comparison is the entire point. The GPU count is
per run. Run 2 is 4×5090 × 53 h × 2 = 424 GPU-hours; Run 3 is 8×5090 × 32 h × 2 =
512 GPU-hours. The phase schedule already assumed pairs; the table row did not say so.

**No code changes between Track A runs.** Track A is a config *mode*, not a fork —
baseline and method share every line except the score-fusion call, so a measured
difference cannot be an implementation artefact. The complete diffs:

| | changes from `dit_b_256_dmd2` |
|---|---|
| `dit_b_256_robust` | `mode: teacher→robust`; drop `gan_weight`; add `lam_estimator=dsm`, `lam_calib_every=4`, `lam_ema=0.999`, `lam_min_count=256` |
| `dit_b_512_dmd2` | `latent_size 32→64`, `shape [4,32,32]→[4,64,64]` |
| `dit_b_512_robust` | both of the above |
| `dit_xl_256_dmd2` | `arch DiT-B/2→DiT-XL/2`, `micro_batch 64→32`, XL teacher ckpt |
| `dit_b_256_invert` | `track: A→B` — this one *is* a different trainer |

So "turning on the adaptive λ" is literally `mode: robust` plus the `lam_*` block,
and it simultaneously turns *off* the hand-tuned GAN term (`gan_weight`), which is
the thing it claims to replace.

### 4.0.1 Teacher checkpoints — RESOLVED (option 1), and it had a hidden cost

Every run needs a pretrained latent-diffusion teacher. **Only DiT-XL/2 is publicly
released** (256px and 512px). DiT-B/2 is not, so Runs 1, 2 and 4 as specified
require training a DiT-B/2 teacher first — that is ImageNet pretraining, hundreds
of GPU-hours, and it is not in the budget above.

Three ways out, in order of preference:

1. **Run the dimension ladder on DiT-XL/2 @256 vs @512**, both public, same
   architecture and data — this is a *cleaner* ladder than DiT-B/2 would have
   been. Cost: 32 h + 131 h on 8×5090 per track. The 512px leg is 5.5 days, which
   is the real price of this option.
2. **Substitute a released small latent DiT** and accept a non-standard baseline.
3. **Train a DiT-B/2 teacher once** and reuse it across Runs 1/2/4 — worth it only
   if we expect many small-scale ablations.

**Decision: option 1.** Implemented in `scripts/envs/imagenet{256,512}.env` and
`ddgpu/ckpt.py`, which downloads and verifies the released weights. Runs 1/2/4
move to DiT-XL/2, and the DiT-B configs are kept only for the case where we ever
train our own teacher (`precond: "edm"`, `teacher_format: "raw"`).

**The hidden cost, which was not visible when this was written.** The released
checkpoints are **VP** models — eps-prediction on a discrete linear-beta
schedule — while `ddgpu/dit.py` + `ddgpu/edm.py` implement an **EDM** denoiser.
Loading one into the other does not raise; it produces a teacher that returns
noise, and `init_from_teacher: true` then propagates the same misreading into the
student. `ddgpu/vp.py` closes this with a change of variables that presents the VP
network through the identical `(forward, score, cfg_score, _coef)` surface, so
student, critic and teacher all share one preconditioning and the warm start is
exact. Two knock-on corrections:

* **`sigma_max` is 157.4, not 80.** The linear-1000 schedule tops out there. A
  one-step student generating from 80 starts half way up the schedule it was
  initialised from. It is now resolved from the schedule, never from a config.
* **The training noise distribution changes.** EDM's lognormal puts ~all mass in
  σ ∈ [0.03, 3]; the teacher's range is [0.01, 157]. Default is now
  `sigma_dist="vp_uniform_t"` over t ∈ [20, 979], which is what DMD2 does.

See LOG.log ENTRY 012, FINDINGS 19–21.

---

Ordering matters, and each run has a decision gate. Do not launch the next one
until the previous one's gate passes.

### Run 1 — DMD2 baseline · DiT-B/2 @256px · **2×5090 · 23 h**

**Question it answers:** is anything downstream trustworthy?

This is not a "baseline" in the box-ticking sense. Three things depend on it:

1. **Is the DMD2 implementation correct?** Reproducing a published distillation
   FID is the only check we have. Every comparison in the paper is a difference
   against this number; if it is wrong, so is everything else.
2. **Is the evaluation harness correct?** FID, precision/recall, and the
   matched-wall-clock guard all run for the first time here.
3. **It produces the paper's Figure 1.** λ_dsm runs on real image data for the
   first time and emits a measured λ(σ) curve — the first measurement of the
   optimal teacher-versus-data weight on a real image model. That figure is a
   result in its own right, independent of whether the method improves FID, and it
   is what makes §1.1's claim about the field checkable rather than synthetic.

**Gate:** if reproduced FID is far from published DiT-B distillation numbers, stop
and fix. Do not spend the other 108 GPU-hours on an unvalidated harness.

**Also produces:** the first real-data measurement of whether λ drifts over
training (§3.1) — no extra compute.

### Run 2 — DiT-B/2 @512px · **4×5090 · 53 h** (both tracks)

**Question it answers:** does the method transfer across dimension, at real scale?

This is the **controlled dimension ablation**: same architecture, same data, same
recipe, 4096 → 16384 latent dimensions. Paired with Run 1 it is the only clean 4×
dimension change available at low cost — a model swap (e.g. SD1.5 → SDXL) confounds
dimension with parameters, data, and conditioning, which is precisely why info.txt
recommends decoupling the two axes.

It is the highest-information run in the plan, because §1.3 is a worked example of
a dimension assumption failing silently and expensively. The synthetic ladder tops
out at d=512; this run is 32× beyond it. Specifically it tests:

- Does λ_dsm still track the optimum at 16384 dims, or does a new failure appear?
- Does the useful σ band keep sliding down with dimension, as the ladder predicts
  (10.5 → 3.4 → 2.0 → 0.19 σ_data across d = 8 → 512)? If so, any hard-coded noise
  schedule in the field is mis-specified at high resolution — a transferable claim.
- Does the effective-N requirement (§1.4) bite harder at higher dimension?

**Gate:** if λ_dsm degrades here, Track A does **not** go to SDXL, and the paper's
scope contracts to ImageNet with an honest dimension-limitation section.

### Run 3 — DiT-XL/2 @256px · **8×5090 · 32 h** (both tracks)

**Question it answers:** does it survive a 5× parameter increase, and is the number
competitive?

Two distinct purposes:

1. **The reviewer-facing table.** 675 M params on ImageNet-256 is the standard
   benchmark. Distillation reviewing has drifted such that small-scale-only results
   read as insufficient regardless of how clean the method is.
2. **A capacity probe on the mechanism.** λ is driven by the teacher's bias
   profile. A 5× larger, better-trained teacher should have lower bias, which
   should push λ *toward 1 everywhere* and shrink the real-data term's role. If
   that happens, it is direct evidence that the real-data term is compensating for
   teacher error and will matter less as teachers improve — a strong, quotable
   claim. If λ does *not* move with capacity, the off-manifold explanation (§1.2)
   is favoured over the teacher-error one, because student-manifold mismatch does
   not go away with a bigger teacher.

Either outcome is informative, which is why this run is worth 32 GPU-hours even
though the FID number alone might be unremarkable.

**Note:** Run 3 varies parameters at *fixed* dimension, while Run 2 varies
dimension at *fixed* parameters. Together they separate the two axes — that
separation is the reason to run both rather than jumping straight to DiT-XL @512px
(which would confound them, and costs 5.5 days).

### Run 4 — Track B invertible · DiT-B/2 @256px · **2×5090 · 39 h**

**Question it answers:** does the bijection survive contact with real data?

Track B is the highest-variance bet, so this is deliberately the cheapest
configuration that can test it, and it is designed to **fail fast**. Three
separable questions, and the run has value even if two fail:

1. **Does the encoder collude?** §1.5 proved the Gaussianity term cannot detect
   matched-moment non-Gaussianity, so collusion is a live risk that the loss
   curves will not reveal. The `eff_rank_frac` diagnostic answers this within the
   **first few hundred steps** — roughly an hour of the 39. This is the single
   cheapest high-information moment in the entire plan.
2. **Does a critic-free objective produce a usable generator?** If yes, that is the
   headline: no adversary, no fake-score network, no two-timescale inner loop.
3. **Does the inverse map work?** One-step diffusion models currently have no
   native inversion. Even if (2) fails, a working E gives editing and inversion —
   a real open problem with an established evaluation protocol, and a different
   paper rather than nothing.

**Prerequisite:** the teacher-anchor cache (~1 GPU-hour for DiT-B/2 @256px, one-off
and reusable across all Track B runs). §1.5 makes this non-negotiable — without
the anchor, the objective cannot distinguish a colluding encoder from a correct
one, so `w_anchor=0` is only an ablation.

**Gate:** kill at ~hour 1 if `eff_rank_frac` is falling.

### Suggested schedule on one 8×5090 node

| Phase | Runs | GPUs | Elapsed |
|---|---|---|---|
| 0 | `ddgpu.probe --sweep`, anchor cache | 1–8 | ~2 h |
| 1 | Run 1 (baseline @256px) — **gate** | 2 | ~1 d |
| 2 | Run 2 ×2 tracks + Run 4, in parallel | 4+4 | ~2.5 d |
| 3 | Run 3 ×2 (baseline, robust), serial | 8 | ~3 d |
| 4 | contingency / reruns | — | ~4 d |

≈ 11 days of node time; budget two weeks. Runs 1 and 4 are 2-GPU jobs and can be
co-scheduled with 4-GPU jobs on the same node — but see the co-location warning in
`RUNPLAN.md`: the micro-batch numbers assume exclusive access to each card.

---

## 5. Open risks not yet closed

- **λ_dsm is calibrated on-distribution.** There is no unbiased score estimate at a
  student sample, so the calibration cannot capture the off-manifold gain §1.2
  identified (oracle 0.65 versus shipped 0.93 for a collapsed student). The
  on-distribution curve is a safe default that never loses, but ~28% of available
  improvement sits unclaimed exactly where the student is worst.
- **The synthetic teacher may be unrealistically good at low σ.** Real image
  teachers are worse there. The σ⁻⁴ variance blow-up is universal and worsens with
  d, so the ordering should hold — but that is an expectation, not a result.
- **Nothing here tests KSD against a data distribution whose score comes from a
  network**, which is the setting where high-dimensional Stein operators are
  actually suspected to fail. §1.5 speaks only to the Gaussian-target case.
- **Track B's loss weights** (`w_mmd`, `w_ksd`, `w_moment`, `w_rec`, `w_cyc`,
  `w_anchor`) were set before the §1.5 evidence and should be revisited against it
  — the moment term is doing more work than its weight suggests, KSD less.
- **The activation-memory model is analytic.** `ddgpu.probe --sweep` measures the
  real numbers; if it disagrees with `RUNPLAN.md`, the probe wins.
- **Nothing in the pipeline has touched a GPU.** The VP change of variables is
  verified against an *oracle* eps-predictor on a Gaussian target (agreement to
  7e-8), not against the real DiT-XL/2 weights. `ddgpu/ckpt.py`'s positional-grid
  check is the first thing that will exercise the real file; run
  `python3 -m ddgpu.ckpt --name DiT-XL-2-256x256` on the VM before anything else.
- **λ_dsm has never been calibrated against a *network* teacher.** Every number
  in §1 comes from a synthetic teacher with a controllable bias profile. §5's
  "the synthetic teacher may be unrealistically good at low σ" is now testable
  cheaply: Run 1 emits the real λ(σ) curve within its first checkpoint interval.
- **The λ calibration leg sees N/2, not N.** `LambdaEstimator.calibrate` splits
  the gathered real batch — half supplies calibration points, half supplies `B` —
  and that split is not optional (overlap makes `B` memorise the sample it is
  scored against). But it means the calibration estimates the optimal weight for
  an empirical score built from **half** the samples the training-time fusion
  actually uses, which biases λ *toward the teacher*. Conservative rather than
  dangerous, and it shrinks as N grows. To clear §1.4's floor on **both** legs the
  effective real batch wants to be 512, i.e. micro_batch 64 on 8 GPUs. The
  launcher warns when it is not. Not measured; worth one cell of exp04 if the
  first real λ curve looks flatter than the synthetic one.
- **Phase A cannot produce the mechanism result.** `LambdaEstimator.calibrate`
  evaluates at noised *real* data, and the DSM identity does not extend to
  student samples, so §1.2/§3.1 still need a training run. §6.2.
- **The cheap tier's baseline FID is not a DMD2 reproduction.** Its teachers
  expose no token trunk, so the discriminator is a standalone conv head rather
  than DMD2's critic-feature head. Our arms are comparable to each other, not to
  the published number. §6.5.
- **The cheap tier's Phase C/D/E GPU-hour estimates are not derived.**
  `memcalc` is transformer-shaped and those backbones are UNets. Run
  `ddgpu.probe` before committing to a rental.
- **The drift probe is a lower bound, not λ\*.** `LambdaProbe` uses the ratio
  statistic, which drops the ⟨b_B, u⟩ term and is loosest at small σ. Read the
  *difference* between the real and student legs, never the levels. The CPU smoke
  run already shows the predicted split (1.000 vs 0.53–0.61), but against a
  synthetic target and an untrained student — that is a wiring check, not
  evidence.

---

## 6. The cheap tier — the plan we are actually running

§4's plan costs ~3,100 GPU-hours core and ~5,000 with ablations and contingency
(26 days on an 8×5090 node). This section replaces it with a ~200–375 GPU-hour
programme that keeps the axis the method is at risk on. §4 stays as written: it
is the plan to spend *if* the cheap tier says the effect is real.

Full derivation and the traps found along the way: **LOG.log ENTRY 013**.

### 6.1 What shrinks is parameters, not dimension

| tier | setup | working dims | teacher params |
|---|---|---:|---:|
| expensive | DiT-XL/2 @256px **latent** | 4,096 | 674.82 M |
| expensive | DiT-XL/2 @512px **latent** | 16,384 | 674.82 M |
| **cheap** | CIFAR-10 @32px **pixel** | 3,072 | ≈36 M |
| **cheap** | ImageNet-64 @64px **pixel** | 12,288 | ≈296 M |
| **cheap** | ImageNet-100 @256px **latent** | 4,096 | ≈130 M (self-trained SiT) |

The ladder ratio is preserved exactly — 4,096→16,384 becomes 3,072→12,288, both
4×, 25% lower in absolute terms. Parameters drop 19× at the low rung.

**The point that is easy to get backwards: ImageNet-64 in pixel space is 3×
higher-dimensional than ImageNet-256 in latent space** (12,288 vs 4,096). The
VAE exists to make high *resolution* cheap; it does not make the working
dimension small. Reading "256px → 64px" as a 16× retreat confuses resolution
with dimension, and dimension is what §1.3 says this method breaks on.

Relatedly, DiT-XL/2 is 674.82 M parameters at *both* 256px and 512px — identical
count, 4× the tokens. Capacity tracks distribution complexity and noise-level
coverage, not pixel count; that is why CIFAR-10 saturates at ≈36 M while
ImageNet-64, on a smaller grid, needs ≈296 M.

### 6.2 Phase A — the λ ladder is inference, and it is falsifiable

λ_dsm needs a teacher, real data, and forward passes. No student, no critic, no
optimiser. So the ladder from 3,072 to **16,384** dims — including the DiT-XL/2
@512 leg, on a 50k-image subset rather than the 84 GB full prep — is ~10 GPU-h.

`exp/10_lambda_real.py` emits three things, and they are not equally strong:

1. **Falsifiable — the held-out gain.** λ is estimated on split 1; the denoising
   loss `E‖λA+(1−λ)B−g‖²` is evaluated on a *fresh* split 2. Since that loss is
   `MSE(λ) + const(σ)`, a reduction in it *is* a reduction in true score MSE,
   with no ground truth. If `gain_vs_teacher` is not below 1.0, the method does
   not work at that dimension — a real negative result for a few GPU-hours.
   The identity itself is verified on real tensors against a Gaussian target in
   `tests/test_teachers.py:t_dsm_identity`.
2. **Descriptive — λ_dsm(σ).** §1.1's shape on a real image model.
3. **Comparative only — the off-manifold panel.** Ratio statistic at perturbed
   real data (blur/shift/collapse, mirroring §1.2's rows). A *lower bound* on
   λ*: compare its two legs, never the levels.

**Phase A cannot produce the mechanism result.** `LambdaEstimator.calibrate`
evaluates at noised *real* data (`robust.py`), and the DSM identity does not
extend to student samples — `g = −ε/σ` is unbiased for the score of whatever
distribution `x0` came from. §1.2 and §3.1 need a student, i.e. Phase C.

### 6.3 The plan

| phase | what | est. GPU-h |
|---|---|---:|
| A | λ ladder + held-out gain across 5–6 released teachers, inference only | ~10–15 |
| C | CIFAR-10 `dmd2` vs `robust`, **3 seeds each** | ~120–240 |
| D | ImageNet-64 pair — **required, not optional** | ~60–120 |
| E | ImageNet-100 latent leg (self-trained SiT teacher) | ~40–80 |
| | | **~230–455** |

**Phase D is required.** Without it the trained evidence stops at 3,072 dims
while the measured evidence reaches 16,384 — a 5× gap.

**Phase E is the answer to "does this hold in latent space?"** — the one
reviewer question the rest of the cheap tier cannot address. It costs no teacher
compute because the SiT-B/2 already exists (repa-surgery, ImageNet-100, 300k
steps, FID ≈16 at 250k). It doubles as §4 Run 3's teacher-capacity probe: that
teacher is deliberately weaker than the released ones, so λ should sit *lower*
if λ is driven by teacher bias.

The C/D/E estimates are **not derived** — `memcalc` is transformer-shaped and
these are UNets. Firm them up with `ddgpu.probe` before committing.

### 6.4 Three seeds, because the effect may be smaller than the noise

§2 already names "6–9% mean score-MSE may not move FID" as the biggest risk, and
going smaller does not reduce it: one-step CIFAR-10 distillation is
near-saturated (published FIDs ~2–4). One run per arm cannot distinguish "no
effect" from "effect smaller than noise".

`ddgpu.eval --seeds` groups `_sN` runs, reports mean ± sd, and prints an explicit
standard-error verdict — refusing to call a win inside one se, or with fewer
than 3 seeds per arm.

### 6.5 What the cheap tier gives up

- **No trained latent result at ImageNet scale**, and no competitive
  ImageNet-256 FID table. Phase E covers latent space at 4,096 dims and 100
  classes; that is not the same claim.
- **Trained evidence tops out at 12,288 dims; measured evidence reaches 16,384.**
- **The cheap tier's baseline FID is not comparable to DMD2's published number.**
  Its teachers expose no token trunk, so the discriminator is a standalone conv
  head (`gan.ConvGANHead`) rather than DMD2's critic-feature head. Both arms use
  the identical head and the method arm uses none, so *our* comparison stands —
  but the absolute number is ours, not a reproduction.

This is a de-risking plan, not a final-paper plan. If Phase A's held-out gain and
Phase C/D's seeded FIDs both come out positive, §4's budget is worth spending and
the ENTRY 012 pipeline is still there to spend it with.
