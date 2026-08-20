"""HuggingFace download settings, for the entry points bash does not wrap.

`scripts/lib/common.sh:setup_hf_env` sets these for anything launched through
`train.sh` / `smoke.sh` / `eval_all.sh`. But several documented commands are
plain `python3 -m ...` invocations -- `ddgpu.teachers`, `ddgpu.prepare`,
`exp/10_lambda_real.py` -- and those inherit nothing. This module is the same
policy for that path, so a teacher fetched by hand behaves like one fetched by
the launcher.

The setting that matters is **HF_HUB_DISABLE_XET**. `hf_xet` is a Rust client
that does its own networking and does not read the `http_proxy` / `https_proxy`
variables AutoDL's `/etc/network_turbo` exports. Behind that proxy it does not
fail -- it *crawls*, at single-digit kB/s, while printing a "reconstructing
file" progress bar that looks like normal progress:

    diffusion_pytorch_model.safetensors: downloading bytes: 134MB, 4.52kB/s

so the failure mode is a download that appears to be working and would finish
some time next week. Disabling Xet falls back to plain HTTPS, which does honour
the proxy.

`HF_ENDPOINT` is only pointed at the mirror when `/etc/network_turbo` exists,
i.e. on an AutoDL box. The bash layer sets it unconditionally; doing that here
would silently route a European or US box through a China mirror, which is the
opposite of a speed-up. Anything already exported always wins.
"""
import os

MIRROR = "https://hf-mirror.com"
AUTODL_MARKER = "/etc/network_turbo"


def setup_hf_env(verbose=False):
    """Set HF download defaults, without overriding anything already exported."""
    setdefault = lambda k, v: os.environ.setdefault(k, v)
    setdefault("HF_HUB_DISABLE_XET", "1")
    setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.path.exists(AUTODL_MARKER):
        setdefault("HF_ENDPOINT", MIRROR)
    if verbose:
        print(f"[hf] HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '(default)')} "
              f"HF_HUB_DISABLE_XET={os.environ['HF_HUB_DISABLE_XET']}")
    return dict(HF_ENDPOINT=os.environ.get("HF_ENDPOINT"),
                HF_HUB_DISABLE_XET=os.environ["HF_HUB_DISABLE_XET"])
