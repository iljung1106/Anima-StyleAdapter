#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_PID="${1:?cache PID is required}"
CACHE_SUMMARY="${ROOT}/data/anima500k-human/kv_lora_functional_teacher_bank_rank16_64_v2_broad256x4/summary.json"

echo "waiting for K/V LoRA teacher cache PID ${CACHE_PID}" >&2
while kill -0 "${CACHE_PID}" 2>/dev/null; do
  sleep 30
done

python -c '
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("complete_mixtures") != 64 or summary.get("individual") != 64:
    raise SystemExit(f"incomplete K/V teacher cache: {summary}")
' "${CACHE_SUMMARY}"

cd "${ROOT}"
export HF_HOME=/workspace/.cache/huggingface
echo "teacher cache complete; starting broad-content K/V oracle" >&2
exec .venv/bin/anima-data \
  --config configs/anima500k-human.yaml \
  kv-lora-oracle-bootstrap-train
