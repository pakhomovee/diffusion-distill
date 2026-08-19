#!/usr/bin/env bash
#
# Pre-flight checks. Run this before every long job, and on a fresh box before
# anything else. Everything here is CPU-safe and finishes in a couple of minutes.
#
#   scripts/smoke.sh            # invariant tests + both tracks, 8 steps each
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

log "2/4  pipeline tests (VP preconditioning, checkpoint remap, config wiring)"
python3 tests/test_pipeline.py

log "3/4  Track A smoke (8 steps, synthetic latents, CPU)"
python3 -m ddgpu.train --config configs/smokeA.json

log "     Track B smoke (8 steps, synthetic latents, CPU)"
python3 -m ddgpu.train --config configs/smokeB.json

if [[ "$DO_GPU" == "1" ]]; then
  log "4/4  VRAM + throughput probe (this is the number RUNPLAN.md cannot derive)"
  python3 -m ddgpu.probe --sweep --out results/probe.json
  log "compare results/probe.json against RUNPLAN.md; the probe wins"
else
  log "4/4  skipped (pass --gpu on the GPU box to measure VRAM + step time)"
fi

log "smoke OK"
