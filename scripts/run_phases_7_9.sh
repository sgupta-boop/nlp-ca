#!/usr/bin/env bash
# Runs Phases 7-9 unattended, after the Phase 6 LLM chain has finished.
# Each step logs to reports/run_<step>.txt; progress goes to reports/pipeline_7_9.log.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=reports/pipeline_7_9.log
until grep -q "ALL DONE" reports/llm_jobs.log; do sleep 15; done
run() {   # run <name> <python args...>
  local name=$1; shift
  echo "$(date '+%F %T') START $name" >> "$LOG"
  .venv/Scripts/python.exe -u "$@" > "reports/run_${name}.txt" 2>&1
  echo "$(date '+%F %T') END $name exit=$?" >> "$LOG"
}
run p7_train      -m src.finetune train
run p7_evaluate   -m src.finetune evaluate
run p8_prepare    -m src.evaluate prepare
run p8_llm        -m src.evaluate llm
run p8_report     -m src.evaluate report
run p8_clustering -m src.evaluate clustering
run p8_cost       -m src.evaluate cost
run p9_topics_fit -m src.topics fit
run p9_topics_label -m src.topics label
echo "$(date '+%F %T') ALL DONE" >> "$LOG"
