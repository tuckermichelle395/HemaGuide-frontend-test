# HemaGuide 项目记忆

> 这是项目维护记忆，不保存真实 API Key、病例结果或其他敏感凭据。

## 当前代码与分支

- 使用本地修改版 `HemaGuide-frontend-test`。
- 当前发布分支：`codex/micm-extraction`。
- GitHub：`https://github.com/tuckermichelle395/HemaGuide-frontend-test`。
- JumpServer 更新前执行 `git pull --ff-only origin codex/micm-extraction`。

## 模型约定

语言对照测试固定采用：

- 病例抽取：OpenAI 兼容服务，`Qwen3.6-27B-UD-Q4_K_XL.gguf`。
- 最终 decision：OpenAI 兼容服务，`Qwen3.6-27B-UD-Q4_K_XL.gguf`。
- 向量模型：Ollama，`qwen3-embedding:8b`。
- 不使用 `qwen3:14b`；不要把旧模型写死回通用核心默认值。

## OpenAI 兼容服务

JumpServer 当前服务地址：

```text
http://localhost:11433/v1
```

项目根目录 `.env` 中配置（只在本机保存，不提交真实密钥）：

```dotenv
OPENAI_API_KEY=local-key
OPENAI_BASE_URL=http://localhost:11433/v1
```

`localhost` 指运行 HemaGuide Python 进程的同一台机器。

## 中英文病例测试

测试数据位于 `test_data/language_comparison/`：中文 1 个 query、5 个历史病例、中文流程图；英文 1 个 query、5 个历史病例、英文流程图。脚本自动运行 6 组语言组合，每组使用独立缓存、知识库和结果目录。

一键运行：

```bash
PYTHON_BIN=.venv/bin/python \
EMBEDDING_MODEL=qwen3-embedding:8b \
bash scripts/run_language_comparison.sh
```

脚本生成 `language_comparison.md` 和 `language_comparison.csv`。单独比较已有结果：

```bash
PYTHON_BIN=.venv/bin/python scripts/compare_language_results.py \
  --input-dir results/language_comparison/<时间目录>
```

## JumpServer 快速更新

```bash
cd ~/project/HemaGuide-frontend-test-github
git fetch origin
git checkout codex/micm-extraction
git pull --ff-only origin codex/micm-extraction
cp .env.example .env  # 仅首次；然后检查 .env 中的地址
```

运行前检查：

```bash
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
ollama list
```

## 结果解释

- `GUIDELINE`：进入流程图模式，通常不会启动历史病例/PubMed/会议检索。
- `ADVANCED`：可检索历史病例、PubMed 和会议记录。
- `ADVANCED → PLAIN`：上下文为空、无效或综合失败，不能当作有证据支撑的匹配结论。
- PubMed“检索命中”和“最终进入 decision”是两个不同数字。
