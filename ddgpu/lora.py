"""Minimal LoRA + adapter swapping, for the SDXL tier.

The memory plan for SDXL (RUNPLAN.md) rests on one trick: teacher, student and
critic are the SAME frozen bf16 base weights with three swappable low-rank
adapters. That turns 3 x 14.3 GiB of state into 4.8 GiB of base plus 1.6 GiB of
adapter state, which is the difference between "impossible on a 32 GiB card" and
"fits with 23 GiB free".

Status: implemented and unit-tested against ddgpu/dit.py. NOT yet run against a
real SDXL UNet -- the module-name patterns below are generic (any nn.Linear /
nn.Conv2d matching `targets`) but SDXL's attention blocks should be spot-checked
before the first run.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=64, alpha=None, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha or rank) / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + self.drop(x) @ self.A.T @ self.B.T * self.scale
        return y


def inject_lora(model, rank=64, targets=("attn.qkv", "attn.proj", "mlp", "final.lin"),
                exclude=("t_embed", "y_embed", "ada"), alpha=None, dropout=0.0):
    """Replace matching nn.Linear modules with LoRALinear. Returns the adapters.

    Matching is on the QUALIFIED name, not the child name: an nn.Sequential MLP
    exposes its children as "0"/"2", so matching on the child name alone silently
    skips every MLP -- which is most of the parameters and exactly what you
    wanted to adapt. (Caught by the unit test: 5 adapters instead of 9.)

    `exclude` keeps the adapters off the conditioning path: the timestep and
    label embedders are tiny, and the adaLN modulation layers are zero-init by
    design, so adapting them mostly adds state for no capacity.
    """
    adapters = {}
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            qname = f"{name}.{cname}" if name else cname
            if (isinstance(child, nn.Linear) and any(t in qname for t in targets)
                    and not any(e in qname for e in exclude)):
                lora = LoRALinear(child, rank, alpha, dropout)
                setattr(mod, cname, lora)
                adapters[qname] = lora
    for n, p in model.named_parameters():
        p.requires_grad_(".A" in n or ".B" in n)
    return adapters


def lora_state(adapters):
    return {k: dict(A=v.A.detach().clone(), B=v.B.detach().clone())
            for k, v in adapters.items()}


@torch.no_grad()
def load_lora(adapters, state):
    for k, v in adapters.items():
        v.A.copy_(state[k]["A"]); v.B.copy_(state[k]["B"])


class AdapterSet:
    """Several named adapters over ONE shared base model.

    Usage:
        aset = AdapterSet(base, ["student", "critic"], rank=64)
        with aset.use("student"):
            out = base(x, ...)
        with aset.use(None):          # base only == the frozen teacher
            ref = base(x, ...)

    `use(None)` disables every adapter, which is how the same weights serve as
    the frozen teacher without holding a second copy.
    """

    def __init__(self, base, names, rank=64, **kw):
        self.base = base
        self.adapters = inject_lora(base, rank, **kw)
        self.states = {n: lora_state(self.adapters) for n in names}
        for n in names:      # independent random init per adapter
            for k in self.states[n]:
                a = torch.empty_like(self.states[n][k]["A"])
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.states[n][k]["A"] = a
                self.states[n][k]["B"] = torch.zeros_like(self.states[n][k]["B"])
        self.active = None

    def parameters(self, name):
        return [p for k in self.adapters for p in (self.adapters[k].A, self.adapters[k].B)] \
            if name == self.active else []

    def use(self, name):
        return _AdapterCtx(self, name)

    def _set(self, name):
        if self.active is not None:
            self.states[self.active] = lora_state(self.adapters)
        for v in self.adapters.values():
            v.enabled = name is not None
        if name is not None:
            load_lora(self.adapters, self.states[name])
        self.active = name


class _AdapterCtx:
    def __init__(self, aset, name):
        self.aset, self.name, self.prev = aset, name, None

    def __enter__(self):
        self.prev = self.aset.active
        self.aset._set(self.name)
        return self.aset

    def __exit__(self, *a):
        self.aset._set(self.prev)
