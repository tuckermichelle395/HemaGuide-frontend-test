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

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hemaguide-language.XXXXXX")"
BACKUP_QUERY="$TMP_DIR/query_input"
BACKUP_KB="$TMP_DIR/kb_input_tumorboards"
BACKUP_FLOWCHART="$TMP_DIR/aml.txt"
mkdir -p "$BACKUP_QUERY" "$BACKUP_KB" "$RUN_DIR"

PASS=0
FAIL=0

restore_inputs() {
  find query_input -maxdepth 1 -type f -name '*.docx' -delete 2>/dev/null || true
  find kb_input/tumorboards -maxdepth 1 -type f -name '*.docx' -delete 2>/dev/null || true
  if [ -d "$BACKUP_QUERY" ]; then cp -p "$BACKUP_QUERY"/*.docx query_input/ 2>/dev/null || true; fi
  if [ -d "$BACKUP_KB" ]; then cp -p "$BACKUP_KB"/*.docx kb_input/tumorboards/ 2>/dev/null || true; fi
  if [ -f "$BACKUP_FLOWCHART" ]; then cp -p "$BACKUP_FLOWCHART" data/flowchart/aml.txt; fi
  rm -rf "$TMP_DIR"
}
trap restore_inputs EXIT INT TERM

if [ ! -d "$CASE_ROOT/cases/zh/query" ] || [ ! -d "$CASE_ROOT/cases/en/query" ]; then
  echo "Missing language test cases under $CASE_ROOT/cases" >&2
  exit 2
fi
if [ ! -f data/flowchart/aml.txt ] || [ ! -f data/flowchart/aml_en.txt ]; then
  echo "Missing Chinese or English AML flowchart under data/flowchart" >&2
  exit 2
fi

cp -p query_input/*.docx "$BACKUP_QUERY"/ 2>/dev/null || true
cp -p kb_input/tumorboards/*.docx "$BACKUP_KB"/ 2>/dev/null || true
cp -p data/flowchart/aml.txt "$BACKUP_FLOWCHART"

stage_language() {
  local query_lang="$1" history_lang="$2" flowchart_lang="$3"
  find query_input -maxdepth 1 -type f -name '*.docx' -delete 2>/dev/null || true
  find kb_input/tumorboards -maxdepth 1 -type f -name '*.docx' -delete 2>/dev/null || true
  cp -p "$CASE_ROOT/cases/$query_lang/query"/*.docx query_input/
  cp -p "$CASE_ROOT/cases/$history_lang/kb"/*.docx kb_input/tumorboards/
  cp -p "data/flowchart/aml${flowchart_lang:+_$flowchart_lang}.txt" data/flowchart/aml.txt
}

run_case() {
  local name="$1" query_lang="$2" history_lang="$3" flowchart_lang="$4"
  local output_dir="$RUN_DIR/$name"
  echo "[START] $name query=$query_lang history=$history_lang flowchart=${flowchart_lang:-zh}"
  stage_language "$query_lang" "$history_lang" "$flowchart_lang"
  mkdir -p "$output_dir"

  "$PYTHON_BIN" build_kb.py \
    --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" \
    --embedding-mode "$EMBEDDING_MODE" --embedding-model "$EMBEDDING_MODEL" --rebuild \
    > "$output_dir/build_kb.log" 2>&1
  local code=$?
  if [ "$code" -ne 0 ]; then
    echo "[FAIL] $name build_kb (exit=$code)" | tee "$output_dir/status.txt"
    FAIL=$((FAIL + 1)); return 0
  fi

  "$PYTHON_BIN" process_query_input.py \
    --llm-mode "$LLM_MODE" --extraction-model "$LLM_MODEL" --force-extract \
    > "$output_dir/process_query.log" 2>&1
  code=$?
  if [ "$code" -ne 0 ]; then
    echo "[FAIL] $name process_query_input (exit=$code)" | tee "$output_dir/status.txt"
    FAIL=$((FAIL + 1)); return 0
  fi

  local agent_args=(--llm-mode "$LLM_MODE" --decision-model "$DECISION_MODEL" --output-dir "$output_dir")
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

run_case zh_zh_zh zh zh ""
run_case en_en_en en en en
run_case zh_en_zh zh en ""
run_case en_zh_en en zh en
run_case zh_zh_en zh zh en
run_case en_en_zh en en ""

"$PYTHON_BIN" scripts/summarize_results.py --input-dir "$RUN_DIR" --output-dir "$RUN_DIR" > "$RUN_DIR/summarize.log" 2>&1 || true
echo "SUMMARY pass=$PASS fail=$FAIL run_dir=$RUN_DIR"
exit 0
