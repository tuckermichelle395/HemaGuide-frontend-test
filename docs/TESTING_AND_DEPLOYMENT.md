# HemaGuide 测试与部署

本仓库使用本地修改版 HemaGuide-frontend-test。默认决策/抽取模型为 `qwen3:8b`，默认向量模型为 `embeddinggemma:300m`；模型均可通过环境变量覆盖，不写死旧模型配置。

## 一键运行

```bash
cd HemaGuide-frontend-test
PYTHON_BIN=.venv/bin/python bash scripts/run_all_tests.sh
```

脚本会依次执行预检、重建知识库、抽取 query、指南模式、禁用流程图、禁用病例、禁用文献、场景测试，并在单步失败时继续执行。结果和日志位于 `results/test_runs/<时间>/`，最后生成 `summary.md` 与 `summary.csv`。

可用环境变量：`LLM_MODE`、`LLM_MODEL`、`DECISION_MODEL`、`EMBEDDING_MODE`、`EMBEDDING_MODEL`、`RUN_DIR`、`RUN_LEGACY_SCENARIOS`。

中英文语言对照测试：

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_language_comparison.sh
```

语言结果比较也会由上述脚本自动生成；单独比较已有结果：

```bash
PYTHON_BIN=.venv/bin/python scripts/compare_language_results.py \
  --input-dir results/language_comparison/<时间目录>
```

后台运行：

```bash
nohup env PYTHON_BIN=.venv/bin/python LLM_MODE=ollama-local LLM_MODEL=qwen3:8b DECISION_MODEL=qwen3:8b \
  bash scripts/run_all_tests.sh > results/test_runs/latest.log 2>&1 &
```

## JumpServer

在 JumpServer 上克隆新 GitHub 路径后，创建虚拟环境并安装 `requirements.txt`，确认 Ollama 服务可访问并准备 `qwen3:8b` 与向量模型，然后运行：

```bash
PYTHON_BIN=.venv/bin/python bash scripts/preflight.sh
PYTHON_BIN=.venv/bin/python bash scripts/run_all_tests.sh
```

不要提交 `.env`、真实病例、抽取结果、知识库和 `results/`；凭据通过 JumpServer 环境变量或未入库的 `.env` 注入。远程上传前还需明确 GitHub 仓库地址、分支和 JumpServer 目录。
