#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/anima500k-human.yaml"

cd "${ROOT}"
export HF_HOME=/workspace/.cache/huggingface

run_stage() {
  local stage="$1"
  echo "pipeline stage start: ${stage}" >&2
  .venv/bin/anima-data --config "${CONFIG}" "${stage}"
  echo "pipeline stage complete: ${stage}" >&2
}

run_stage artist-only-mixture-functional-cache
run_stage artist-only-mixture-reference-generate
run_stage artist-only-mixture-reference-token-cache
run_stage detail-style-artist-only-mixture-train
