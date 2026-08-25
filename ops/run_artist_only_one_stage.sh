#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="configs/anima500k-human.yaml"

cd "${ROOT}"
export HF_HOME=/workspace/.cache/huggingface

# The bounded functional and materialized-reference caches are immutable inputs
# produced by run_artist_only_mixture_continuation.sh. Do not rebuild them here.
.venv/bin/anima-data \
  --config "${CONFIG}" \
  detail-style-artist-only-one-stage-train
