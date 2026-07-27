#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLM_MODE="${LLM_MODE:-ollama-local}"
DECISION_MODEL="${DECISION_MODEL:-qwen3:8b}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

common_args=(
  --llm-mode "$LLM_MODE"
  --decision-model "$DECISION_MODEL"
  --case-id SIM_MOL_KIT_query
  "$@"
)

echo "[1/2] MOLECULAR：开启基因匹配历史病例"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --output-dir results/molecular_scenarios/with_cases

echo "[2/2] MOLECULAR：关闭历史病例"
"$PYTHON_BIN" agent.py "${common_args[@]}" \
  --disable-case-retrieval \
  --output-dir results/molecular_scenarios/no_cases

echo "完成：结果位于 results/molecular_scenarios/"
