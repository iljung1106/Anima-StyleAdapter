#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/anima500k-human.yaml"
ACTIVE_PID="${1:?active diverse-LoRA trainer PID is required}"
DATA="${ROOT}/data/anima500k-human"
TEACHERS="${DATA}/artist_lora_teachers_rank16_512_diverse_b2_v6"

echo "waiting for 512-artist LoRA trainer ${ACTIVE_PID}" >&2
while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
  sleep 30
done

python -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.load(open(root / "summary.json", encoding="utf-8"))
weights = len(list((root / "weights").glob("*.safetensors")))
if summary.get("artists_completed") != 512 or weights != 512:
    raise SystemExit(f"incomplete 512-artist bank: summary={summary}, weights={weights}")
' "${TEACHERS}"

cd "${ROOT}"
git pull --ff-only
export HF_HOME=/workspace/.cache/huggingface

run_stage() {
  local stage="$1"
  echo "pipeline stage start: ${stage}" >&2
  .venv/bin/anima-data --config "${CONFIG}" "${stage}"
  echo "pipeline stage complete: ${stage}" >&2
}

run_stage lora-reference-expansion-generate
run_stage lora-reference-expansion-token-cache
run_stage lora-functional-teacher-cache-512
run_stage lora-mixture-reference-generate
run_stage lora-mixture-reference-token-cache
run_stage direct-reference-kv-smoke
run_stage direct-reference-kv-train
