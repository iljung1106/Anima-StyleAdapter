#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/anima500k-human.yaml"
ACTIVE_PID="${1:-}"

if [[ -n "${ACTIVE_PID}" ]]; then
  echo "waiting for active artist-LoRA process ${ACTIVE_PID}" >&2
  while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
    sleep 60
  done
fi

cd "${ROOT}"
git pull --ff-only
export HF_HOME=/workspace/.cache/huggingface

run_stage() {
  local stage="$1"
  echo "pipeline stage start: ${stage}" >&2
  .venv/bin/anima-data --config "${CONFIG}" "${stage}"
  echo "pipeline stage complete: ${stage}" >&2
}

# Re-entering the trainer is intentional: completed artists are skipped and an
# interrupted active artist resumes from its periodic optimizer checkpoint.
run_stage artist-lora-train
run_stage lora-reference-generate
run_stage lora-reference-token-cache
run_stage lora-functional-teacher-cache
run_stage lora-oracle-bootstrap-train
run_stage lora-oracle-visual-projector-train
run_stage lora-oracle-joint-manifold-train
