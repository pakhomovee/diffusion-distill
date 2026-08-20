"""GPU codebase for the distillation experiments.

The one import-time side effect here is `hfenv.setup_hf_env()`, and its position
is the point: `huggingface_hub` reads `HF_HUB_DISABLE_XET` and `HF_ENDPOINT`
into module constants **when it is imported**, so setting them later has no
effect at all. Every `diffusers` / `huggingface_hub` import in this package is
deferred inside a function, which means this line always runs first.

It only fills in defaults -- anything already exported wins -- so the bash
layer (`scripts/lib/common.sh:setup_hf_env`) and a hand-exported override both
still take precedence. See `ddgpu/hfenv.py` for why Xet is the setting that
matters.
"""
from .hfenv import setup_hf_env

setup_hf_env()
