#!/usr/bin/env bash
#
# Pre-flight checks. Run this before every long job, and on a fresh box before
# anything else. Everything here is CPU-safe and finishes in a couple of minutes.
#
#   scripts/smoke.sh            # invariant tests + all tracks + Phase A self-test
#   scripts/smoke.sh --gpu      # additionally probe real VRAM + step time
#
# RUNPLAN.md's GPU counts rest on an ANALYTIC activation-memory model. `--gpu`
# runs `ddgpu.probe --sweep`, which measures the real numbers; if the two
# disagree, the probe wins and the run plan needs recomputing before launch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

DO_GPU=0
[[ "${1:-}" == "--gpu" ]] && DO_GPU=1

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

log "1/4  invariant tests"
python3 tests/test_all.py

log "2/5  pipeline tests (VP preconditioning, checkpoint remap, config wiring)"
python3 tests/test_pipeline.py

log "3/5  teacher-zoo tests (interpolant math, DSM identity, mis-wrap detection)"
python3 tests/test_teachers.py

log "4/5  Track A smoke (8 steps, synthetic latents, CPU)"
python3 -m ddgpu.train --config configs/smokeA.json

log "     Track B smoke (8 steps, synthetic latents, CPU)"
python3 -m ddgpu.train --config configs/smokeB.json

log "     cheap-tier smoke: registry teacher -> clone -> train, both arms"
python3 -m ddgpu.train --config configs/smokeC.json
python3 -m ddgpu.train --config configs/smokeC_dmd2.json

log "     Phase A self-test against a target whose true score is known"
python3 exp/10_lambda_real.py --teacher "synthetic:c=4,hw=8,sd=0.5,bias=0.15" \
  --data "gaussian:c=4,hw=8,sd=0.5,n=4096" --batch 128 --batches 3 --n-sigma 5 \
  --sigma-min 0.05 --sigma-max 8 --tag selftest --out results/lambda_real

if [[ "$DO_GPU" == "1" ]]; then
  log "5/5  VRAM + throughput probe (this is the number RUNPLAN.md cannot derive)"
  python3 -m ddgpu.probe --sweep --out results/probe.json
  log "compare results/probe.json against RUNPLAN.md; the probe wins"
else
  log "5/5  skipped (pass --gpu on the GPU box to measure VRAM + step time)"
fi

log "smoke OK"
