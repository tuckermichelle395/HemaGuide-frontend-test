#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"
LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embeddinggemma:300m}"

echo "Python: $($PYTHON_BIN --version 2>&1 || true)"
if ! command -v ollama >/dev/null 2>&1; then
  echo "WARN: ollama command not found"
else
  if ! ollama list >/tmp/hemaguide_ollama_list.$$ 2>&1; then
    echo "WARN: Ollama service is not reachable"
  else
    grep -q "^${LLM_MODEL}" /tmp/hemaguide_ollama_list.$$ || echo "WARN: model not found: $LLM_MODEL"
    grep -q "^${EMBEDDING_MODEL}" /tmp/hemaguide_ollama_list.$$ || echo "WARN: model not found: $EMBEDDING_MODEL"
  fi
  rm -f /tmp/hemaguide_ollama_list.$$
fi

for path in kb_input/tumorboards query_input; do
  if [ ! -d "$path" ]; then echo "WARN: missing directory: $path"; fi
done
echo "Preflight completed; warnings do not stop the test run."
exit 0
