#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_MODE="${LLM_MODE:-ollama-local}"
DECISION_MODEL="${DECISION_MODEL:-qwen3:8b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

common_args=(
  --llm-mode "$LLM_MODE"
  --decision-model "$DECISION_MODEL"
  "$@"
)

echo "[1/4] 有流程图（正常路由）"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --output-dir results/reason_scenarios/with_flowchart

echo "[2/4] 无流程图，有历史病例"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --ignore-flowchart \
  --output-dir results/reason_scenarios/no_flowchart_with_cases

echo "[3/4] 无流程图，无历史病例"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --ignore-flowchart \
  --disable-case-retrieval \
  --output-dir results/reason_scenarios/no_flowchart_no_cases

echo "[4/4] 无流程图，无历史病例，无会议记录（保留 PubMed）"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --ignore-flowchart \
  --disable-case-retrieval \
  --disable-conference-retrieval \
  --output-dir results/reason_scenarios/no_flowchart_no_cases_no_conference

echo "完成：结果位于 results/reason_scenarios/"
