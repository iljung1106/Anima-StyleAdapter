#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-/workspace/Anima-StyleAdapter}"
log_dir="${2:-/workspace/logs/anima-data}"
config_path="${3:-configs/anima500k-human.yaml}"
teacher_pid_file="$log_dir/v2d-diverse-teacher-cache.pid"
teacher_summary="$repo_dir/data/anima500k-human/lora_functional_teacher_bank_rank16_256_v2d_diverse_broad12x6_v1/summary.json"

mkdir -p "$log_dir"
cd "$repo_dir"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

log_event() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1"
}

wait_for_teacher_cache() {
  if [[ -f "$teacher_summary" ]]; then
    log_event "teacher cache is already complete"
    return
  fi
  if [[ ! -f "$teacher_pid_file" ]]; then
    log_event "teacher cache PID file is missing"
    return 1
  fi

  local pid
  pid="$(<"$teacher_pid_file")"
  while ps -p "$pid" -o args= 2>/dev/null | grep -q 'v2d-diverse-functional-teacher-cache'; do
    sleep 60
  done
  if [[ ! -f "$teacher_summary" ]]; then
    log_event "teacher cache stopped without summary.json"
    return 1
  fi
  log_event "teacher cache completed"
}

run_stage() {
  local command="$1"
  local stage_log="$2"
  log_event "starting $command"
  .venv/bin/anima-data --config "$config_path" "$command" >"$stage_log" 2>&1
  log_event "completed $command"
}

wait_for_teacher_cache
run_stage v2d-diverse-mixture-reference-generate "$log_dir/v2d-diverse-mixture-reference.log"
run_stage v2d-diverse-mixture-reference-token-cache "$log_dir/v2d-diverse-mixture-reference-token-cache.log"
run_stage v2d-diverse-smoke "$log_dir/v2d-diverse-smoke.log"
run_stage v2d-diverse-train "$log_dir/v2d-diverse-train.log"
log_event "v2d diverse pipeline completed"
