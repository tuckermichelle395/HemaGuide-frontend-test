#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_MODE="${LLM_MODE:-ollama-local}"
LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
DECISION_MODEL="${DECISION_MODEL:-$LLM_MODEL}"
EMBEDDING_MODE="${EMBEDDING_MODE:-ollama}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embeddinggemma:300m}"
RUN_DIR="${RUN_DIR:-results/test_runs/$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

PASS=0
FAIL=0
SKIP=0

run_step() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}.log"
  echo "[START] $name"
  echo "[COMMAND] $*" > "$log"
  "$@" >> "$log" 2>&1
  local code=$?
  if [ "$code" -eq 0 ]; then
    PASS=$((PASS + 1)); echo "[PASS] $name"
  else
    FAIL=$((FAIL + 1)); echo "[FAIL] $name (exit=$code; see $log)"
  fi
  return 0
}

echo "HemaGuide test run: $RUN_DIR"
echo "LLM_MODE=$LLM_MODE LLM_MODEL=$LLM_MODEL DECISION_MODEL=$DECISION_MODEL EMBEDDING_MODEL=$EMBEDDING_MODEL"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[BLOCKED] Python executable not found: $PYTHON_BIN"
  exit 2
fi

run_step preflight env PYTHON_BIN="$PYTHON_BIN" LLM_MODEL="$LLM_MODEL" EMBEDDING_MODEL="$EMBEDDING_MODEL" bash scripts/preflight.sh
run_step build_kb "$PYTHON_BIN" build_kb.py --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" --embedding-mode "$EMBEDDING_MODE" --embedding-model "$EMBEDDING_MODEL" --rebuild
run_step extract_queries "$PYTHON_BIN" process_query_input.py --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" --force-extract
run_step guideline_agent "$PYTHON_BIN" agent.py --llm-mode "$LLM_MODE" --decision-model "$DECISION_MODEL" --output-dir "$RUN_DIR/guideline"
run_step advanced_agent "$PYTHON_BIN" agent.py --llm-mode "$LLM_MODE" --decision-model "$DECISION_MODEL" --ignore-flowchart --output-dir "$RUN_DIR/advanced"
run_step advanced_no_cases "$PYTHON_BIN" agent.py --llm-mode "$LLM_MODE" --decision-model "$DECISION_MODEL" --ignore-flowchart --disable-case-retrieval --output-dir "$RUN_DIR/advanced_no_cases"
run_step advanced_no_literature "$PYTHON_BIN" agent.py --llm-mode "$LLM_MODE" --decision-model "$DECISION_MODEL" --ignore-flowchart --disable-pubmed-retrieval --disable-conference-retrieval --output-dir "$RUN_DIR/advanced_no_literature"

if find query_input -maxdepth 1 -type f -name 'SIM_MOL_*.docx' -print -quit | grep -q .; then
  run_step molecular_scenarios bash scripts/run_molecular_scenarios.sh
else
  SKIP=$((SKIP + 1)); echo "[SKIP] molecular_scenarios: no SIM_MOL query file"
fi

if [ "${RUN_LEGACY_SCENARIOS:-1}" = "1" ]; then
  run_step reason_scenarios env PYTHON_BIN="$PYTHON_BIN" LLM_MODE="$LLM_MODE" DECISION_MODEL="$DECISION_MODEL" bash scripts/run_reason_scenarios.sh
else
  SKIP=$((SKIP + 1)); echo "[SKIP] reason_scenarios: RUN_LEGACY_SCENARIOS=$RUN_LEGACY_SCENARIOS"
fi

if [ -x scripts/summarize_results.py ] || [ -f scripts/summarize_results.py ]; then
  run_step summarize "$PYTHON_BIN" scripts/summarize_results.py --input-dir "$RUN_DIR" --output-dir "$RUN_DIR"
fi

echo "SUMMARY pass=$PASS fail=$FAIL skip=$SKIP run_dir=$RUN_DIR"
exit 0
