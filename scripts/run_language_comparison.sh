#!/usr/bin/env bash
# Run the Chinese/English query-history-flowchart comparison cases.
set -u

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_MODE="${LLM_MODE:-ollama-local}"
LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
DECISION_MODEL="${DECISION_MODEL:-$LLM_MODEL}"
EMBEDDING_MODE="${EMBEDDING_MODE:-ollama}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embeddinggemma:300m}"
CASE_ROOT="${CASE_ROOT:-test_data/language_comparison}"
RUN_DIR="${RUN_DIR:-results/language_comparison/$(date +%Y%m%d_%H%M%S)}"
DISABLE_PUBMED="${DISABLE_PUBMED:-0}"
DISABLE_CONFERENCE="${DISABLE_CONFERENCE:-0}"

PASS=0
FAIL=0

run_case() {
  local name="$1" query_lang="$2" history_lang="$3" flowchart_lang="$4"
  local output_dir="$RUN_DIR/$name"
  local extracted_dir="$output_dir/extracted_data"
  local kb_storage_dir="$output_dir/kb_storage"
  local flowchart_dir="$CASE_ROOT/flowcharts/$flowchart_lang"
  echo "[START] $name query=$query_lang history=$history_lang flowchart=$flowchart_lang"
  mkdir -p "$output_dir"

  "$PYTHON_BIN" build_kb.py \
    --kb-dir "$CASE_ROOT/cases/$history_lang/kb" \
    --extracted-data-dir "$extracted_dir" \
    --kb-storage-dir "$kb_storage_dir" \
    --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" \
    --embedding-mode "$EMBEDDING_MODE" --embedding-model "$EMBEDDING_MODEL" --rebuild \
    > "$output_dir/build_kb.log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "[FAIL] $name build_kb (exit=$code)" | tee "$output_dir/status.txt"
    FAIL=$((FAIL + 1)); return 0
  fi

  "$PYTHON_BIN" process_query_input.py \
    --query-dir "$CASE_ROOT/cases/$query_lang/query" \
    --extracted-data-dir "$extracted_dir" \
    --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" --force-extract \
    > "$output_dir/process_query.log" 2>&1
  code=$?
  if [ "$code" -ne 0 ]; then
    echo "[FAIL] $name process_query_input (exit=$code)" | tee "$output_dir/status.txt"
    FAIL=$((FAIL + 1)); return 0
  fi

  local agent_args=(
    --query-dir "$CASE_ROOT/cases/$query_lang/query"
    --extracted-data-dir "$extracted_dir"
    --kb-storage-dir "$kb_storage_dir"
    --flowchart-dir "$flowchart_dir"
    --llm-mode "$LLM_MODE"
    --decision-model "$DECISION_MODEL"
    --output-dir "$output_dir"
  )
  if [ "$DISABLE_PUBMED" = "1" ]; then agent_args+=(--disable-pubmed-retrieval); fi
  if [ "$DISABLE_CONFERENCE" = "1" ]; then agent_args+=(--disable-conference-retrieval); fi
  "$PYTHON_BIN" agent.py "${agent_args[@]}" > "$output_dir/agent.log" 2>&1
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "[PASS] $name" | tee "$output_dir/status.txt"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $name agent (exit=$code)" | tee "$output_dir/status.txt"
    FAIL=$((FAIL + 1))
  fi
}

for lang in zh en; do
  for file in "$CASE_ROOT/flowcharts/$lang"/*.txt; do
    [ -f "$file" ] || { echo "Missing flowchart: $file" >&2; exit 2; }
  done
done

run_case zh_zh_zh zh zh zh
run_case en_en_en en en en
run_case zh_en_zh zh en zh
run_case en_zh_en en zh en
run_case zh_zh_en zh zh en
run_case en_en_zh en en zh

"$PYTHON_BIN" scripts/summarize_results.py --input-dir "$RUN_DIR" --output-dir "$RUN_DIR" > "$RUN_DIR/summarize.log" 2>&1 || true
"$PYTHON_BIN" scripts/compare_language_results.py --input-dir "$RUN_DIR" --output-dir "$RUN_DIR" > "$RUN_DIR/compare.log" 2>&1 || true
echo "SUMMARY pass=$PASS fail=$FAIL run_dir=$RUN_DIR"
exit 0
