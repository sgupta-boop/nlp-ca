#!/usr/bin/env bash
# Runs long LLM steps one after another, detached from the editor session.
# Every LLM answer is cached in cache/llm_responses.json, so a re-run only computes what is missing.
#   usage: bash scripts/run_llm_jobs.sh "python -m src.llm_layer adjudicate" "python -m src.llm_layer names" ...
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=reports/llm_jobs.log
for cmd in "$@"; do
  name=$(echo "$cmd" | awk '{print $5"_"$6}' | tr '.' '_')
  echo "$(date '+%F %T') START $cmd" >> "$LOG"
  .venv/Scripts/$cmd > "reports/run_${name}.txt" 2>&1
  rc=$?
  echo "$(date '+%F %T') END $cmd exit=$rc" >> "$LOG"
done
echo "$(date '+%F %T') ALL DONE" >> "$LOG"
