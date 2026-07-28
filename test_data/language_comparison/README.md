# 中英文病例语言对照测试

这是 HemaGuide 的中英文新旧病例对照测试集，不是分子场景测试集。

## 数据组成

- `cases/zh/query/`：1 个中文新病例
- `cases/zh/kb/`：5 个中文历史病例
- `cases/en/query/`：1 个英文新病例
- `cases/en/kb/`：5 个英文历史病例

测试组合包括：中文查询/中文历史库、英文查询/英文历史库、中文查询/英文历史库、英文查询/中文历史库，以及流程图语言交叉组合。详细命中数、命中原文和最终 decision 见 [`docs/language_test_summary.md`](../../docs/language_test_summary.md)。

病例文件为测试用病例，不代表真实临床患者。

运行 6 组语言组合：

```bash
OPENAI_API_KEY=local-key \
OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
PYTHON_BIN=.venv/bin/python bash scripts/run_language_comparison.sh
```

脚本会使用 `qwen3:14b` 完成病例抽取，使用 OpenAI 兼容服务中的 `Qwen3.6-27B-UD-Q4_K_XL.gguf` 生成最终 decision，使用 `qwen3-embedding:8b` 建立向量库。它直接把病例、知识库、流程图和抽取缓存传给程序参数，不覆盖默认目录。PubMed/Crossref 默认开启；如需只观察病例和流程图影响，可设置 `DISABLE_PUBMED=1 DISABLE_CONFERENCE=1`。结果写入 `results/language_comparison/<时间>/`，并生成 `language_comparison.md` 和 `language_comparison.csv`。
