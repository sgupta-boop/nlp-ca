#!/usr/bin/env bash
# Runs the long LLM jobs one after another, detached from the editor session.
# Every LLM answer is cached, so re-running a job only computes what is missing.
#   usage: bash scripts/run_llm_jobs.sh <job> [<job> ...]   (jobs: llm_pave, llm_label)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=reports/llm_jobs.log
# wait for any extraction run that is already going
running() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*-m src.extract*' }).Count"; }
while [ "$(running | tr -d '\r')" != "0" ]; do sleep 30; done
for job in "$@"; do
  echo "$(date '+%F %T') START $job" >> "$LOG"
  case $job in
    llm_pave)  .venv/Scripts/python.exe -u -m src.extract llm_pave  > reports/phase3_run_llm_pave.txt 2>&1 ;;
    llm_label) .venv/Scripts/python.exe -u -m src.extract llm_label 1000 > reports/phase3_run_llm_label.txt 2>&1 ;;
  esac
  echo "$(date '+%F %T') END $job exit=$?" >> "$LOG"
done
