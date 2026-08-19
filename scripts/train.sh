#!/usr/bin/env bash
#
# Single entry point for distillation training.
#
# It (1) sets up the environment (AutoDL network turbo + python deps),
#    (2) loads a per-dataset env from scripts/envs/<dataset>.env,
#    (3) optionally prepares the dataset and fetches the teacher checkpoint,
#    (4) generates the run config, and (5) launches ddgpu.train under torchrun.
#
# Usage:
#   scripts/train.sh --dataset imagenet256 --mode robust --gpus 0,1,2,3,4,5,6,7
#
# Options:
#   -d, --dataset NAME      env to load: scripts/envs/NAME.env   [required]
#       --mode MODE         dmd2 | robust | invert | data | fixed | ratio | nogather
#       --gpus IDS          comma-separated ids, e.g. 0,1,2,3 (default: env)
#       --micro-batch N     per-device batch
#       --steps N           training steps
#       --seed N            RNG seed; run dir gets _sN when non-zero. Use 3 seeds
#                           at CIFAR scale -- FINDINGS.md 6 warns the effect may
#                           be smaller than run-to-run FID variance there.
#       --exp-name NAME     run dir under runs/ (default: <dataset>_<mode>)
#       --config FILE       use this config verbatim, skipping generation
#       --resume [PATH]     resume ('auto' = latest checkpoint in the run dir)
#       --prepare           build latents + FID reference before training
#       --skip-setup        skip network/python setup
#       --dry-run           print the launch command and exit
#       --set k=v           extra config overrides (repeatable)
#   --                      everything after this is passed to ddgpu.train
#
# Env: DD_CONDA_ENV (opt-in conda), DD_SKIP_INSTALL=1, DD_DATA_ROOT,
#      DD_CKPT_ROOT, MASTER_PORT (default 29531),
#      DD_ALLOW_SMALL_REAL_BATCH=1 (override the Track A batch floor).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

DATASET=""; MODE="robust"; GPUS=""; MICRO=""; STEPS=""; EXP_NAME=""; SEED=""
CONFIG=""; RESUME=""; DO_PREPARE=0; SKIP_SETUP=0; DRY_RUN=0
SETS=(); EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)    DATASET="$2"; shift 2 ;;
    --mode)          MODE="$2"; shift 2 ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --micro-batch)   MICRO="$2"; shift 2 ;;
    --steps)         STEPS="$2"; shift 2 ;;
    --seed)          SEED="$2"; shift 2 ;;
    --exp-name)      EXP_NAME="$2"; shift 2 ;;
    --config)        CONFIG="$2"; shift 2 ;;
    --resume)        if [[ "${2:-}" == --* || -z "${2:-}" ]]; then RESUME="auto"; shift
                     else RESUME="$2"; shift 2; fi ;;
    --set)           SETS+=("$2"); shift 2 ;;
    --prepare)       DO_PREPARE=1; shift ;;
    --skip-setup)    SKIP_SETUP=1; shift ;;
    --dry-run)       DRY_RUN=1; shift ;;
    --)              shift; EXTRA=("$@"); break ;;
    -h|--help)       sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$DATASET" ]] || die "Missing required --dataset (e.g. --dataset imagenet256)"
ENV_FILE="$SCRIPT_DIR/envs/${DATASET}.env"
[[ -f "$ENV_FILE" ]] || die "No env file: $ENV_FILE"

setup_hf_env
if [[ "$SKIP_SETUP" == "0" ]]; then
  setup_autodl_network
  setup_python_env
else
  log "--skip-setup: not touching network/python env"
fi

log "Loading dataset env: $ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"

GPUS="${GPUS:-${DEFAULT_GPUS:-0}}"
NPROC="$(count_gpus "$GPUS")"
MICRO="${MICRO:-$DEFAULT_MICRO_BATCH}"
STEPS="${STEPS:-$DEFAULT_STEPS}"
SEED="${SEED:-0}"
EXP_NAME="${EXP_NAME:-${DATASET}_${MODE}}"
if [[ "$SEED" != "0" ]]; then EXP_NAME="${EXP_NAME}_s${SEED}"; fi
# Per-dataset floor for the Track A effective real batch (FINDINGS.md 1.4 and 5).
export DD_MIN_REAL_BATCH="${DD_MIN_REAL_BATCH:-${DEFAULT_MIN_REAL_BATCH:-256}}"
RUN_DIR="$REPO_ROOT/runs/$EXP_NAME"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# ---- teacher checkpoint ----------------------------------------------------
# FINDINGS.md 4.0.1: DiT-XL/2 is the only publicly released latent DiT, so it is
# the teacher for every run here. Fetch + verify it before anything expensive.
if [[ -n "${TEACHER_NAME:-}" && -n "${TEACHER_CKPT:-}" && ! -f "$TEACHER_CKPT" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    log "--dry-run: would fetch teacher $TEACHER_NAME -> $CKPT_DIR (~2.7 GB)"
  else
    log "Fetching teacher $TEACHER_NAME -> $CKPT_DIR"
    python3 -m ddgpu.ckpt --name "$TEACHER_NAME" --dir "$CKPT_DIR"
  fi
fi

# A registry spec pointing at a local file (edm:..., sit:...) must exist; a
# hub id (diffusers:...) is fetched on first use and must not be checked here.
if [[ -n "${TEACHER_SPEC:-}" && "$DRY_RUN" == "0" ]]; then
  case "$TEACHER_SPEC" in
    edm:*|sit:*)
      _tp="${TEACHER_SPEC#*:}"
      [[ -f "$_tp" ]] || die "teacher not found: $_tp (see scripts/envs/${DATASET}.env)" ;;
  esac
fi

# ---- dataset ---------------------------------------------------------------
if [[ "$DO_PREPARE" == "1" ]]; then
  if declare -F prepare_dataset >/dev/null; then
    log "Preparing dataset '$DATASET' (this is a one-off; it skips existing work)"
    GPUS="$GPUS" prepare_dataset
  else
    warn "env '$DATASET' defines no prepare_dataset(); skipping"
  fi
fi
if [[ "${DATA_DIR:-}" != "synthetic" && ! -f "${DATA_DIR:-}/meta.json" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    warn "no dataset at $DATA_DIR -- a real launch would stop here (use --prepare)"
  else
    die "no dataset at $DATA_DIR (run with --prepare first)"
  fi
fi

# ---- config ----------------------------------------------------------------
mkdir -p "$RUN_DIR"
if [[ -z "$CONFIG" ]]; then
  CONFIG="$RUN_DIR/config.json"
  MK=(python3 "$SCRIPT_DIR/mkconfig.py" --mode "$MODE" --dataset "$DATASET"
      --data "$DATA_DIR" --out "runs/$EXP_NAME" --micro-batch "$MICRO"
      --steps "$STEPS" --n-classes "$NUM_CLASSES" --latent-size "$LATENT_SIZE"
      --n-student-steps "${DEFAULT_N_STUDENT_STEPS:-1}" --seed "$SEED"
      --cfg-scale "${DEFAULT_CFG_SCALE:-1.75}" --write "$CONFIG")
  # A registry teacher spec wins over a bare checkpoint path: it carries the
  # preconditioning family, which a path alone does not.
  if [[ -n "${TEACHER_SPEC:-}" ]]; then
    MK+=(--teacher-spec "$TEACHER_SPEC")
  elif [[ -n "${TEACHER_CKPT:-}" ]]; then
    MK+=(--teacher "$TEACHER_CKPT")
  fi
  if [[ -n "${DEFAULT_SIGMA_DIST:-}" ]]; then MK+=(--sigma-dist "$DEFAULT_SIGMA_DIST"); fi
  if [[ -n "${REPA_DIR:-}" ]]; then MK+=(--repa-dir "$REPA_DIR"); fi
  if [[ -n "${EDM_REPO:-}" ]]; then MK+=(--edm-repo "$EDM_REPO"); fi
  if [[ "$MODE" == "invert" ]]; then MK+=(--anchor-path "$CKPT_DIR/anchors_${DATASET}.pt"); fi
  if [[ ${#SETS[@]} -gt 0 ]]; then MK+=(--set "${SETS[@]}"); fi
  "${MK[@]}" >/dev/null
  log "Generated config: $CONFIG"
  log "Delta from the dmd2 baseline (FINDINGS.md 4.0 -- this must be small):"
  python3 "$SCRIPT_DIR/mkconfig.py" --mode "$MODE" --print-delta
fi

# ---- Track A correctness precondition --------------------------------------
CFG_MODE="$(json_get "$CONFIG" mode)"
CFG_GATHER="$(json_get "$CONFIG" gather_real)"
check_effective_real_batch "$CFG_MODE" "$CFG_GATHER" "$MICRO" "$NPROC"

# ---- launch ----------------------------------------------------------------
CMD=(torchrun --standalone --nproc_per_node="$NPROC"
     --master_port="${MASTER_PORT:-29531}"
     -m ddgpu.train --config "$CONFIG")
if [[ -n "$RESUME" ]]; then CMD+=(--resume "$RESUME"); fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then CMD+=("${EXTRA[@]}"); fi

log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "${CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  log "--dry-run: not launching"
  exit 0
fi
mkdir -p "$RUN_DIR"
"${CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
