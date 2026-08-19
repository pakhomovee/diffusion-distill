"""Derived VRAM budget + GPU-count planner for distillation runs on RTX 5090.

Everything here is computed from an architecture config, not quoted. Parameter
counts for DiT come from ddgpu/dit.py itself (instantiated on the meta device);
non-DiT backbones are a small table of published counts.

Card assumption: RTX 5090, 32 GiB GDDR7, no NVLink (PCIe 5.0 x16 only).
The no-NVLink point drives an architectural choice, not just a speed note:
FSDP parameter all-gathers every layer over PCIe are ~10x slower than over
NVLink, so we prefer DDP with FULL LOCAL REPLICAS wherever the replicas fit,
and fall back to FSDP/ZeRO-3 only when they do not.

Usable VRAM is taken as 30.0 GiB: the CUDA context, NCCL buffers, cuDNN
workspaces and allocator fragmentation reliably eat ~1.5-2 GiB on a 32 GiB card.
"""
import math

GIB = 1024 ** 3
CARD_GIB = 32.0
USABLE_GIB = 30.0          # after CUDA ctx + NCCL + fragmentation headroom

# bytes per parameter, by role and optimizer policy
OPT_BYTES = {
    # params + grads + Adam m + Adam v
    "adam_fp32":   4 + 4 + 4 + 4,     # 16  reference / most stable
    "adam_bf16st": 4 + 4 + 2 + 2,     # 12  bf16 optimizer states
    "adam8bit":    4 + 4 + 1 + 1,     # 10  bitsandbytes 8-bit Adam
}
FROZEN_BYTES = 2                      # bf16 inference-only replica


def dit_params(name, input_size):
    import torch
    from ddgpu.dit import make_dit
    with torch.device("meta"):
        m = make_dit(name, input_size=input_size)
    return sum(p.numel() for p in m.parameters())


BACKBONES = {
    # name: (params, tokens_at_train_res, hidden)  -- non-DiT, published counts
    "SDXL-UNet":    (2_567_000_000, 4096, 1280),   # 128x128 latent, 1024px
    "PixArt-Sigma": (610_000_000,   4096, 1152),   # 1024px, /2 patch on 64x64 latent
    "SD1.5-UNet":   (860_000_000,   4096, 1280),   # 64x64 latent, 512px
}


def activation_gib(batch, tokens, hidden, depth, grad_ckpt=True, dtype_bytes=2,
                   n_bwd_passes=1):
    """Activation memory for a transformer-ish backbone.

    With gradient checkpointing we store one (B, N, H) tensor per block boundary
    and recompute inside a block; the recompute peak is dominated by the MLP's
    4H intermediate plus qkv's 3H, i.e. ~12*H per token in bf16.
    Without checkpointing, every block keeps its full ~12H of intermediates.
    """
    per_tok = batch * tokens * hidden * dtype_bytes
    if grad_ckpt:
        stored = per_tok * depth          # block-boundary residual stream
        peak = per_tok * 12               # inside the one block being recomputed
    else:
        stored = per_tok * 12 * depth
        peak = per_tok * 12
    return n_bwd_passes * (stored + peak) / GIB


class Plan:
    def __init__(self, name, params, tokens, hidden, depth,
                 n_trainable=2, n_frozen=1, opt="adam_fp32", grad_ckpt=True,
                 lora_rank=None, lora_frac=None, share_frozen_base=False,
                 n_bwd_passes=2, notes=""):
        self.__dict__.update(locals()); del self.self

    def weights_gib(self):
        """Returns (trainable_state, frozen_weights) in GiB."""
        p = self.params
        if self.lora_rank:
            # LoRA: base weights frozen, only adapters carry optimizer state.
            trainable_p = int(p * self.lora_frac)
            # one shared frozen bf16 base serves teacher/student/critic if asked
            n_bases = 1 if self.share_frozen_base else (self.n_trainable + self.n_frozen)
            frozen = n_bases * p * FROZEN_BYTES
            train = self.n_trainable * trainable_p * OPT_BYTES[self.opt]
        else:
            trainable_p = p
            frozen = self.n_frozen * p * FROZEN_BYTES
            train = self.n_trainable * p * OPT_BYTES[self.opt]
        return train / GIB, frozen / GIB

    def max_micro_batch(self, cap=USABLE_GIB):
        tr, fr = self.weights_gib()
        room = cap - tr - fr
        if room <= 0.5:
            return 0
        lo, hi = 0, 1024
        while lo < hi:
            mid = (lo + hi + 1) // 2
            a = activation_gib(mid, self.tokens, self.hidden, self.depth,
                               self.grad_ckpt, 2, self.n_bwd_passes)
            if a <= room:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def report(self, global_batch, gpus_options=(1, 2, 4, 8)):
        tr, fr = self.weights_gib()
        mb = self.max_micro_batch()
        lines = [f"### {self.name}",
                 f"  params/model      : {self.params/1e6:.0f} M"
                 + (f"  (LoRA r={self.lora_rank}, {self.lora_frac*100:.1f}% trainable)"
                    if self.lora_rank else ""),
                 f"  trainable copies  : {self.n_trainable}   frozen copies: {self.n_frozen}"
                 + ("   [shared frozen base]" if self.share_frozen_base else ""),
                 f"  optimizer policy  : {self.opt} ({OPT_BYTES[self.opt]} B/param)",
                 f"  weights+opt state : {tr:.2f} GiB trainable + {fr:.2f} GiB frozen"
                 f" = {tr+fr:.2f} GiB",
                 f"  free for activs   : {USABLE_GIB-tr-fr:.2f} GiB of {USABLE_GIB:.1f} GiB usable",
                 f"  max micro-batch   : {mb} / GPU"
                 + ("  <-- DOES NOT FIT" if mb == 0 else "")]
        if mb:
            for g in gpus_options:
                per = math.ceil(global_batch / g)
                accum = math.ceil(per / mb)
                lines.append(f"    {g:2d} GPU: need {per:4d}/GPU -> micro {math.ceil(per/accum):3d}"
                             f" x accum {accum:2d}"
                             + ("   OK" if accum <= 8 else "   (accum high, slow)"))
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Throughput model
# ---------------------------------------------------------------------------
# RTX 5090 dense BF16 with FP32 accumulate: ~105 TFLOP/s peak (non-sparse).
# Achieved MFU for a DiT-style model with SDPA/flash and grad checkpointing is
# 35-45% in practice on consumer cards (lower than H100 because HBM->GDDR7 and
# no NVLink for the DDP all-reduce). We use 0.38.
PEAK_BF16 = 105e12
MFU = 0.38


def fwd_flops_per_sample(params_nonembed, tokens, depth=None, hidden=None):
    """2*N*T for the linear layers, plus 4*depth*T^2*hidden for attention."""
    f = 2.0 * params_nonembed * tokens
    if depth and hidden:
        f += 4.0 * depth * tokens ** 2 * hidden
    return f


# forward-equivalents per optimisation step, by method.
# fwd=1, fwd+bwd=3 (bwd is 2x fwd), +1x fwd for grad-checkpoint recompute.
STEP_COST = {
    # DMD2: student fwd+bwd+ckpt(4) | teacher fwd x2 for CFG(2) | fake fwd(1)
    #       | fake-score DSM fwd+bwd+ckpt(4)
    "dmd2": 11,
    # Track A adds the empirical/data score branch: no network, just a cdist and
    # a softmax over the real batch -> negligible FLOPs, but +1 teacher fwd is
    # NOT needed. Cost is identical to dmd2.
    "robust": 11,
    # Track B: + encoder fwd+bwd+ckpt(4) + G fwd on E(x) for the reconstruction
    #          leg (+4). Gaussianity term is closed form, ~free.
    "invertible": 19,
    # plain DSM pretraining, for reference
    "dsm": 4,
}


def step_seconds(method, params_nonembed, tokens, batch_per_gpu, n_gpu,
                 depth=None, hidden=None, allreduce_gib=0.0):
    f = fwd_flops_per_sample(params_nonembed, tokens, depth, hidden)
    flops = STEP_COST[method] * f * batch_per_gpu
    compute = flops / (PEAK_BF16 * MFU)
    # DDP all-reduce over PCIe 5.0 x16 (~50 GB/s effective, ring over n gpus)
    comm = 0.0
    if n_gpu > 1 and allreduce_gib:
        comm = 2 * (n_gpu - 1) / n_gpu * allreduce_gib * GIB / 50e9
    return compute + comm


def wallclock_hours(method, params, params_nonembed, tokens, depth, hidden,
                    global_batch, n_gpu, micro, accum, steps):
    ar = 2 * params * 4 / GIB          # fp32 grads of the two trainable nets
    s = accum * step_seconds(method, params_nonembed, tokens, micro, n_gpu,
                             depth, hidden, allreduce_gib=ar)
    return s * steps / 3600.0, s
