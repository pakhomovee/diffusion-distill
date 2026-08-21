#!/usr/bin/env bash
#
# Score every run under runs/ with IDENTICAL sampler settings, then print the
# matched-wall-clock comparison table.
#
#   scripts/eval_all.sh --dataset imagenet256 --gpus 0,1,2,3 --n 50000
#
# Two rules from info.txt are enforced, not merely documented:
#   * FID is compared at matched WALL-CLOCK, not matched steps. Track A and
#     Track B have different per-step costs, so equal-step comparisons
#     systematically favour the more expensive method. `ddgpu.eval` refuses to
#     emit a table when the runs differ by more than --tol in GPU-seconds.
#   * Precision and recall are printed alongside FID, always. Mode-seeking
#     objectives buy FID with diversity and that trade hides inside one number.
#
# Options:
#   -d, --dataset NAME    env to load (for REF_NPZ / defaults)   [required]
#       --gpus IDS        comma-separated ids (default: env default)
#       --n N             samples per run (default 50000)
#       --ckpt TAG        checkpoint tag to score (default: final)
#       --weights W       ema | student (default ema)
#       --tol F           allowed relative GPU-second spread (default 0.15)
#       --only REGEX      only runs whose name matches
#       --exclude REGEX   skip runs whose name matches (default '^smoke')
#       --skip-setup      skip network/python setup
#       --dry-run         print the commands and exit
#       --shutdown        power the box off when the sweep finishes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# CKPT="final" is the DEFAULT, not a recommendation. LOG ENTRY 014 measured a
# CIFAR run collapsing monotonically from step 5000 onward -- recall 0.104 ->
# 0.003, FID 122 -> 175 -- so ckpt_final.pt was the WORST checkpoint in it and
# scoring it silently reported the bottom of the curve. Scan the trajectory
# (--ckpt 5000, 10000, ...) before believing any single number.
DATASET=""; GPUS=""; N=50000; CKPT="final"; WEIGHTS="ema"; TOL=0.15
ONLY=""; EXCLUDE='^smoke'; SKIP_SETUP=0; DRY_RUN=0; SHUTDOWN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset) DATASET="$2"; shift 2 ;;
    --gpus)       GPUS="$2"; shift 2 ;;
    --n)          N="$2"; shift 2 ;;
    --ckpt)       CKPT="$2"; shift 2 ;;
    --weights)    WEIGHTS="$2"; shift 2 ;;
    --tol)        TOL="$2"; shift 2 ;;
    --only)       ONLY="$2"; shift 2 ;;
    --exclude)    EXCLUDE="$2"; shift 2 ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    --shutdown)   SHUTDOWN=1; shift ;;
    -h|--help)    sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$DATASET" ]] || die "Missing required --dataset"
ENV_FILE="$SCRIPT_DIR/envs/${DATASET}.env"
[[ -f "$ENV_FILE" ]] || die "No env file: $ENV_FILE"

# Fail fast on --shutdown rather than after hours of eval.
if [[ "$SHUTDOWN" == "1" ]]; then
  command -v "${SHUTDOWN_CMD%% *}" >/dev/null 2>&1 || \
    command -v shutdown >/dev/null 2>&1 || die "--shutdown requested but no shutdown binary"
fi

setup_hf_env
if [[ "$SKIP_SETUP" == "0" ]]; then setup_autodl_network; setup_python_env; fi
# shellcheck disable=SC1090
source "$ENV_FILE"

GPUS="${GPUS:-${DEFAULT_GPUS:-0}}"
NPROC="$(count_gpus "$GPUS")"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

[[ -f "$REF_NPZ" ]] || die "no FID reference at $REF_NPZ (scripts/train.sh --prepare)"

RESULTS="$REPO_ROOT/results/${DATASET}_runs.json"
mkdir -p "$REPO_ROOT/results"

FAILED=()
for d in "$REPO_ROOT"/runs/*/; do
  name="$(basename "$d")"
  [[ -f "$d/ckpt_${CKPT}.pt" ]] || { log "skip $name (no ckpt_${CKPT}.pt)"; continue; }
  [[ -z "$ONLY"    || "$name" =~ $ONLY    ]] || continue
  [[ -z "$EXCLUDE" || ! "$name" =~ $EXCLUDE ]] || { log "skip $name (excluded)"; continue; }

  CMD=(torchrun --standalone --nproc_per_node="$NPROC"
       --master_port="${MASTER_PORT:-29532}"
       # --run-dir / --n-samples, not --run / --n: the short spellings are
       # ambiguous abbreviations of torchrun's own options and it refuses the
       # command before the script starts. See ddgpu/generate.py's build_argparser().
       -m ddgpu.generate --run-dir "$d" --ckpt "$CKPT" --weights "$WEIGHTS"
       --n-samples "$N" --ref "$REF_NPZ" --out "$RESULTS" --name "$name")
  log "${CMD[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then continue; fi
  if ! "${CMD[@]}" 2>&1 | tee -a "$d/eval.log"; then
    warn "eval FAILED for $name (continuing)"
    FAILED+=("$name")
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi

log "Comparison table (matched wall-clock, tol=$TOL):"
python3 -m ddgpu.eval --records "$RESULTS" --tol "$TOL" || \
  warn "table refused -- see the message above; that is the guard working, not a bug"

# Runs differing only by _sN are seeds of one arm. At CIFAR scale the effect may
# be smaller than seed variance (FINDINGS.md 6), so the seed-grouped view with
# its explicit standard-error verdict is the one to read, not the flat table.
log "Seed-grouped table:"
python3 -m ddgpu.eval --records "$RESULTS" --tol "$TOL" --seeds || true

if [[ "$SHUTDOWN" == "1" ]]; then
  log "shutting down in ${SHUTDOWN_DELAY:-60}s (Ctrl-C to cancel)"
  sleep "${SHUTDOWN_DELAY:-60}"
  ${SHUTDOWN_CMD:-shutdown -h now}
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  die "runs that failed to evaluate: ${FAILED[*]}"
fi
