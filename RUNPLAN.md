# Run plan — GPU allocation for the distillation experiments

Target hardware: **NVIDIA RTX 5090, 32 GiB GDDR7, no NVLink (PCIe 5.0 x16 only).**

> **This document budgets the ~5,000 GPU-hour ImageNet-256/512 programme, which
> is NOT what we are running first.** `FINDINGS.md` §6 replaces it with a
> ~230–455 GPU-hour cheap tier (CIFAR-10 at 3,072 dims, ImageNet-64 at 12,288,
> a self-trained SiT latent leg at 4,096) that preserves the 4× dimension ladder
> while cutting teacher parameters 19×. Come back here when the cheap tier says
> the effect is real. The numbers below stay valid for that decision.

> **The teacher is DiT-XL/2 for every run** (FINDINGS.md §4.0.1, decided in
> favour of option 1). That makes the ladder XL@256 vs XL@512 rather than
> DiT-B/2, so the DiT-B rows below are only reachable if we ever pretrain our own
> teacher. The XL@512 leg is the 5.5-day row, and it is the real price of the
> decision. Launchers and configs for the XL path are in `scripts/`.
>
> The released checkpoints are **VP** models, not EDM ones; `ddgpu/vp.py` handles
> the change of variables and `sigma_max` is **157.4**, not 80. See LOG.log
> ENTRY 012 before changing anything about noise schedules.

All numbers below are **derived**, not quoted: parameter counts come from
instantiating `ddgpu/dit.py` on the meta device, memory from `ddgpu/memcalc.py`,
wall-clock from a FLOP model at 38% MFU. Regenerate with `python exp/plan_gpus.py`.

> **Before launching anything, run `python -m ddgpu.probe --sweep` on one card.**
> The activation-memory term is analytic and is the one number here I cannot
> validate without a GPU. The probe measures real `max_memory_allocated` and real
> step time; if it disagrees with the table, the probe wins and the GPU counts
> below should be recomputed.

## Two constraints that shape every choice

**1. 32 GiB, not 80 GiB.** DMD-style training holds three copies of the backbone
(student, fake-score critic, frozen teacher). With fp32 Adam that is 16 bytes per
parameter per trainable copy. The breakpoints:

| trainable copies | max params/model at 32 GiB | fits? |
|---|---|---|
| 2 (DMD2: student + critic) | ~850 M | DiT-XL/2 (675 M) fits with 8.6 GiB to spare |
| 2 (Track B: student + encoder) | ~850 M | same — **E replaces the critic, it does not add to it** |
| 3 (any variant keeping both) | ~560 M | DiT-XL/2 **does not fit** — 31.4 GiB of state alone |
| 2, LoRA r=64 + shared base | ~10 B | SDXL (2.6 B) fits with 23.6 GiB free |

Track B is critic-free by construction: the fake-score network is replaced by a
closed-form discrepancy against N(0,I), and the encoder E occupies the slot the
critic used to. So Track B and the DMD2 baseline have **the same VRAM footprint**
and can be compared without a memory confound. If you ever add a critic *back*
alongside E, you land in row 3 and DiT-XL stops fitting — reach for 8-bit Adam
(10 B/param → 20.1 GiB) or ZeRO-2 (optimizer-state sharding only, which avoids
the per-layer all-gather traffic that makes full FSDP bad on PCIe).

**2. No NVLink.** This is an architecture decision, not a footnote. FSDP
all-gathers parameters every layer; over PCIe 5.0 (~50 GB/s effective, versus
900 GB/s NVLink) that dominates the step. **Use DDP with full local replicas
wherever the replicas fit** — which, per the table above, is everywhere in the
ImageNet ladder. Reach for ZeRO-2 before ZeRO-3, and full FSDP only for SDXL.

## Per-run allocation

```
======================================================================================================================
RUN PLAN  --  per-run GPU counts (RTX 5090 32GiB, DDP, grad ckpt, bf16 autocast)
======================================================================================================================

-- L1/L2 dimension ladder: same arch, same data, 4096 vs 16384 dims (the controlled ablation) --
DMD2 baseline   DiT-B/2 @256px (4096-d)               2x5090  state   4.1G  micro 128x1   1.64s/step    22.8 h  (0.9 d)
DMD2 baseline   DiT-B/2 @512px (16384-d)              4x5090  state   4.1G  micro  64x1   3.78s/step    52.5 h  (2.2 d)
Track A robust  DiT-B/2 @256px                        2x5090  state   4.1G  micro 128x1   1.64s/step    22.8 h  (0.9 d)
Track A robust  DiT-B/2 @512px                        4x5090  state   4.1G  micro  64x1   3.78s/step    52.5 h  (2.2 d)

-- L3 headline ImageNet number --
DMD2 baseline   DiT-XL/2 @256px                       8x5090  state  21.4G  micro  32x1   2.28s/step    31.6 h  (1.3 d)
Track A robust  DiT-XL/2 @256px                       8x5090  state  21.4G  micro  32x1   2.28s/step    31.6 h  (1.3 d)
DMD2 baseline   DiT-XL/2 @512px                       8x5090  state  16.3G  micro  32x1   9.44s/step   131.1 h  (5.5 d)

-- Track B (invertible): E REPLACES the fake-score critic, so still 2 trainable nets --
Track B invert  DiT-B/2 @256px                        2x5090  state   4.1G  micro 128x1   2.82s/step    39.1 h  (1.6 d)
Track B invert  DiT-B/2 @512px                        4x5090  state   4.1G  micro  64x1   6.51s/step    90.4 h  (3.8 d)
Track B invert  DiT-XL/2 @256px                       8x5090  state  21.4G  micro  32x1   3.80s/step    52.7 h  (2.2 d)

-- Track B preprocessing: teacher-anchor cache (one-off, reusable across runs) --
  anchors DiT-B/2 @32: 50k pairs x 32 steps -> 146.89 PFLOP,  1.02 h on 1 GPU
  anchors DiT-B/2 @32: 50k pairs x 32 steps -> 146.89 PFLOP,  0.13 h on 8 GPU
  anchors DiT-B/2 @64: 50k pairs x 32 steps -> 680.32 PFLOP,  4.74 h on 1 GPU
  anchors DiT-B/2 @64: 50k pairs x 32 steps -> 680.32 PFLOP,  0.59 h on 8 GPU
  anchors DiT-XL/2 @32: 50k pairs x 32 steps -> 757.63 PFLOP,  5.27 h on 1 GPU
  anchors DiT-XL/2 @32: 50k pairs x 32 steps -> 757.63 PFLOP,  0.66 h on 8 GPU
```

## Text-to-image tier (only if the ImageNet ladder says the method survives)

| config | GPUs | state | micro×accum | note |
|---|---|---|---|---|
| SDXL-UNet 1024px, full finetune | — | 81.3 GiB | — | **impossible on 32 GiB, at any count** |
| SDXL-UNet 1024px, LoRA r=64 attn-only, shared base | 8×5090 | 6.4 GiB | 8×1 | 14/GPU ceiling |
| SDXL-UNet 1024px, LoRA r=64 attn+FF, shared base | 8×5090 | 9.8 GiB | 8×1 | 12/GPU ceiling — use this number, it is the pessimistic end |
| PixArt-Σ 0.6B 1024px | 4×5090 | 19.3 GiB | 8×2 | **SDXL's latent dim at ¼ the params — the right dimension probe** |

The shared-frozen-base trick is what makes SDXL tractable here: teacher, student
and critic are the *same* 2.6 B bf16 weights with three swappable LoRA adapters,
so we pay 4.8 GiB once instead of 14.3 GiB three times. Implemented in
`ddgpu/lora.py` (`AdapterSet`), unit-tested for adapter isolation — perturbing
the student adapter provably leaves the critic and the teacher untouched — but
**not yet run against a real SDXL UNet**; the target-name patterns are generic
and should be spot-checked against SDXL's block naming before the first run.

## What each run buys

Per-run scientific justification, decision gates and kill criteria live in
**`FINDINGS.md` §4**. Short version: Run 1 validates the harness and produces the
paper's first figure; Run 2 is the controlled dimension ablation (4096 → 16384
dims at fixed parameters) and is the highest-information run in the plan; Run 3
varies parameters at fixed dimension, so together with Run 2 it separates the two
axes; Run 4 tests Track B in the cheapest configuration that can fail fast.

**Do not launch a run until the previous gate passes.**

## Recommended allocation

**One 8×5090 node.** That covers every run in the table, with the two 2-GPU
ladder runs packing 4-at-a-time onto the node.

Sequenced, assuming the node is exclusive:

| phase | runs | GPUs used | wall-clock |
|---|---|---|---|
| 0. probe + harness validation | `ddgpu.probe --sweep`, toy CPU sweeps | 1 | ~1 h |
| 1. DMD2 baseline reproduction @256 | 1 run | 2 | ~1 d |
| 2. dimension ladder (4 runs: {baseline,robust} × {256,512}) | 4 runs | 2+2+4+4 = 12 → 2 waves | ~4 d |
| 3. headline DiT-XL/2 @256 (baseline + robust) | 2 runs | 8 each, serial | ~3 d |
| 4. Track B invertible @256 (+ ablations) | 3 runs | 2 each, parallel | ~2 d |
| 5. contingency / reruns | — | — | ~4 d |

**≈ 2 weeks on one 8×5090 node** for the full ImageNet story.
Add ~1 week and the same node for the SDXL LoRA tier if phase 2–3 justify it.

## Things that will bite

- **Two runs per card is a trap.** The micro-batch numbers assume exclusive
  access. Co-locating two runs on one 5090 halves the micro-batch and the
  activation model stops holding.
- **`torch.compile`** on a 5090 needs CUDA 12.8+ / PyTorch ≥ 2.7 for sm_120.
  Worth ~20–30% but compile it *after* the probe, not before, or the memory
  numbers shift under you.
- **Grad checkpointing is assumed on.** Turning it off raises the micro-batch
  ceiling by roughly 12× less memory headroom — i.e. it collapses.
- **FID must be at matched wall-clock**, not matched steps (info.txt's warning).
  Track A and Track B have different per-step costs (11 vs 19 fwd-equivalents),
  so equal-step comparisons systematically favour Track B. Report precision and
  recall separately as well; mode-seeking objectives buy FID with diversity.

## Track A has no tuned λ hyperparameter (as of the DSM calibrator)

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

## Hard precondition for Track A: effective real batch ≥ 256

Measured across 234 cells (exp04): with a real batch of **N=64** the doubly-robust
fusion is *worse* than the plain teacher — mean ratio 1.14, worst case **5.7×**.
Every one of the five worst cells in the whole sweep is an N=64 cell. At N=256 the
mean is 0.91 and the worst 1.15.

The per-device micro-batch in the table above is 32–128, so a naive implementation
sits **inside the failure regime**. `ddgpu/robust.py:all_gather_batch` gathers the
real batch across ranks, which takes N to `world_size × micro` — 8×32 = 256, the
first safe row. This is a correctness requirement, not an optimisation:

- **Never run Track A on fewer GPUs than `256 / micro_batch`**, or raise the micro
  batch to compensate.
- `gather_real` defaults to `true`; setting it false is an ablation, not a config.

## Sanity checks to run before each long job

```bash
scripts/smoke.sh --gpu      # invariant tests + both tracks + the VRAM/throughput probe
```

That is all three of the old commands plus `tests/test_pipeline.py`, in order,
failing fast. On a fresh box also run the teacher fetch once, because it is the
first thing that verifies the released weights against our DiT:

```bash
python3 -m ddgpu.ckpt --name DiT-XL-2-256x256 --dir ckpt
```

For Track B specifically, watch `DIAG eff_rank_frac` in the first 500 steps. It
should sit near 1.0. If it falls, the encoder is collapsing onto a subspace and
the run is dead — the loss curves will *not* tell you this. See LOG.log ENTRY 004
for the version of this failure that was already caught and fixed.
