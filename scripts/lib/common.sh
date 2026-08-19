# Shared helpers for the diffusion-distillation pipeline. Sourced, not executed.
#
# Mirrors repa-surgery/training/lib/common.sh: same AutoDL conventions, same
# opt-in conda policy, same HF mirror handling -- so a box set up for one repo
# is already set up for the other.

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LIB_DIR/../.." && pwd)"

log()  { printf '\033[1;34m[dd]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dd]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[dd]\033[0m %s\n' "$*" >&2; exit 1; }

# --- AutoDL network acceleration -------------------------------------------
# On AutoDL GPU machines, sourcing /etc/network_turbo proxies outbound traffic
# (HuggingFace / torch.hub / github) through a fast mirror. No-op elsewhere.
setup_autodl_network() {
  if [[ -f /etc/network_turbo ]]; then
    log "AutoDL detected: sourcing /etc/network_turbo"
    # shellcheck disable=SC1091
    source /etc/network_turbo
  fi
}

# --- Hugging Face download settings ----------------------------------------
# Route HF traffic through hf-mirror.com and disable the Xet chunked-transfer
# backend, whose CAS server 401s through the network_turbo proxy. Only sets env
# vars, so it runs even with --skip-setup. Honors anything already exported.
setup_hf_env() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  log "HF_ENDPOINT=$HF_ENDPOINT HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
}

# --- Python environment -----------------------------------------------------
# Uses the currently active interpreter by default (AutoDL images ship a working
# CUDA-enabled base env) and just pip-installs the deps.
#
# Conda is OPT-IN via DD_CONDA_ENV=<name>: we deliberately do not auto-create an
# env, because on AutoDL the configured conda mirror often fails behind the
# network_turbo proxy. Skip installs entirely with DD_SKIP_INSTALL=1.
setup_python_env() {
  local env_name="${DD_CONDA_ENV:-}"
  if [[ -n "$env_name" ]]; then
    command -v conda >/dev/null 2>&1 || die "DD_CONDA_ENV=$env_name set but conda not found"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^\s*${env_name}\s"; then
      log "Creating conda env '$env_name' (python=3.10)"
      conda create -n "$env_name" python=3.10 -y
    fi
    log "Activating conda env '$env_name'"
    conda activate "$env_name"
  else
    log "Using current python: $(command -v python3) ($(python3 --version 2>&1))"
  fi

  if [[ "${DD_SKIP_INSTALL:-0}" != "1" ]]; then
    log "Installing dependencies (set DD_SKIP_INSTALL=1 to skip)"
    python3 -m pip install -q -r "$REPO_ROOT/requirements.txt"
  else
    log "DD_SKIP_INSTALL=1: skipping dependency install"
  fi
}

# Count comma-separated GPU ids: "0,1,2,3" -> 4
count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

# Read one key out of a JSON config without needing jq.
json_get() {
  python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" \
    "$1" "$2"
}

# --- Track A correctness precondition ---------------------------------------
# FINDINGS.md 1.4: with an effective real batch below 256 the doubly-robust
# fusion is *worse* than the plain teacher (mean MSE ratio 1.14, worst case
# 5.7x), and every one of the five worst cells in the 234-cell sweep was an
# N=64 cell. `robust.py:all_gather_batch` takes N to world_size * micro_batch,
# so the launcher can check it -- and should, because the failure is a quietly
# degraded result, not a crash.
check_effective_real_batch() {
  local mode="$1" gather="$2" micro="$3" ngpu="$4" min="${DD_MIN_REAL_BATCH:-256}"
  [[ "$mode" == "robust" ]] || return 0
  local eff=$micro
  [[ "$gather" == "True" || "$gather" == "true" || "$gather" == "" ]] && eff=$((micro * ngpu))
  if (( eff < min )); then
    warn "effective real batch = $eff (micro $micro x $ngpu GPUs) < $min"
    warn "FINDINGS.md 1.4: below 256 the robust fusion LOSES to the plain teacher."
    if [[ "${DD_ALLOW_SMALL_REAL_BATCH:-0}" != "1" ]]; then
      die "refusing to launch. Raise --micro-batch or --gpus, or set DD_ALLOW_SMALL_REAL_BATCH=1 to run it as a deliberate ablation."
    fi
    warn "DD_ALLOW_SMALL_REAL_BATCH=1: proceeding anyway (this is an ablation, not a run)"
  else
    log "effective real batch = $eff (micro $micro x $ngpu GPUs) -- ok"
  fi
}
