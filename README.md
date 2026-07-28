# HemaGuide

Clinical decision support in hematological malignancies using a case-grounded AI agent.

> **Research Use Only** – This software is intended for research purposes only. Generated decisions require validation by qualified medical professionals and must not be used for clinical care without proper oversight.

## System Requirements

Developed and tested on Mac Studio M3 Ultra, 96GB RAM. We used Python 3.12.

### Prerequisites

Install Python 3.12+ from [python.org](https://www.python.org/downloads/) or via your package manager.

## Quick Start

The fastest path is the bundled launcher:

1. Install [Ollama](https://ollama.com/download)
2. Clone the repository:
```bash
git clone https://github.com/Friedrich-Lab/HemaGuide
```
3. Double-click **`HemaGuide.command`**

It installs Python tools, pulls the required models, creates the virtual environment, starts the server, and opens the web interface at `http://localhost:8000`. The interface ships pre-built — no Node.js required. Double-click **`HemaGuide-Stop.command`** to stop it.

Processing cases also needs a `.env` file and a built knowledge base — see [Step-by-step](#step-by-step).

## Step-by-step

### 1. Setup

```bash
python3 -m venv HemaGuide_venv             # Create virtual environment
source HemaGuide_venv/bin/activate         # Activate it
pip install -r requirements.txt             # Install dependencies
```

Create a `.env` file with:

```bash
# Required (at least one LLM provider)
OPENAI_API_KEY=sk-...        # OpenAI API
OPENAI_BASE_URL=...          # Optional OpenAI-compatible API base URL
OLLAMA_API_KEY=...           # Ollama Cloud (if using --llm-mode ollama-cloud)

# Required for PubMed
PUBMED_EMAIL=your@email.com  # NCBI Entrez API requires valid email

# Optional
NCBI_API_KEY=...             # Increases PubMed rate limit (3 → 10 req/sec)
ONCOKB_API_KEY=...           # OncoKB API for molecular classification
CROSSREF_EMAIL=...           # CrossRef API (falls back to PUBMED_EMAIL)
```

#### JumpServer Qwen3.6 本地服务配置

语言对照测试的抽取和 decision 都使用本机的 OpenAI 兼容 Qwen3.6 服务，地址固定为 `http://localhost:11433/v1`。在项目根目录执行一次即可持久保存到 `.env`，后续脚本会自动读取：

```bash
cd ~/project/HemaGuide-frontend-test-github
cp .env.example .env
```

如果已有 `.env`，只需确认其中包含：

```dotenv
OPENAI_API_KEY=local-key
OPENAI_BASE_URL=http://localhost:11433/v1
```

确认服务可访问：

```bash
curl -s "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

然后运行中英文病例对照测试：

```bash
PYTHON_BIN=.venv/bin/python \
EMBEDDING_MODEL=qwen3-embedding:8b \
bash scripts/run_language_comparison.sh
```

The following step is mandatory as we calculate the embeddings ALWAYS locally.

For default mode local Ollama embeddings and `--llm-mode ollama-local`:
Download Ollama from [ollama.com](https://ollama.com/download) and run it, then:

```bash
ollama pull embeddinggemma:300m             # Pull required models
```

### Optional: COSMIC Data (for hotspot classification)

For enhanced molecular variant classification, download COSMIC data:

1. Register at [COSMIC](https://cancer.sanger.ac.uk/cosmic/register)
2. Download from [COSMIC Downloads](https://cancer.sanger.ac.uk/cosmic/download):
   - File: **Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv** (~7GB)
3. Place in `data/cosmic/`:

```bash
mkdir -p data/cosmic
mv ~/Downloads/Cosmic_CompleteTargetedScreensMutant_v103_GRCh38.tsv data/cosmic/
```

Without this file, hotspot criteria will use the cancerhotspots.org API only.

### 2. CLI (run first to process data)

#### 一键测试与结果汇总

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_all_tests.sh
```

该脚本会在单个步骤报错后继续执行，并在运行目录下生成日志、JSON、`summary.md` 和 `summary.csv`。模型通过 `LLM_MODEL`、`DECISION_MODEL` 等环境变量配置，默认使用 `qwen3:8b`。详细的 JumpServer 部署步骤见 [`docs/TESTING_AND_DEPLOYMENT.md`](docs/TESTING_AND_DEPLOYMENT.md)。

```bash
python build_kb.py              # Build knowledge base
python process_query_input.py   # Extract queries
python agent.py                 # Run agent
```

### 3. Backend

```bash
cd backend && python main.py
```

### 4. Frontend

```bash
cd frontend && npm install && npm run dev
```

## Data Structure

```
kb_input/
└── tumorboards/
    └── *.docx          ← Clinical Case Memory cases

query_input/
└── *.docx              ← Query cases to process

results/
└── agent_decisions/
    └── *.json          ← Generated decisions (output)
```

### Bundled synthetic test data

This fork includes synthetic, non-patient test documents:

- `kb_input/tumorboards/SIM_*.docx` contains historical cases for building a test knowledge base.
- `query_input/SIM_*.docx` contains a sample query case.

Local flowcharts are not included. Add authorized flowchart files to
`data/flowchart/{entity_slug}.txt` before testing guideline mode.

## Disclaimer

This software is provided as proof-of-concept for **research purposes only** and is not intended for clinical use.

- We do not ship non-public data (flowcharts, COSMIC data) and offer a prebuilt UI for demonstration (frontend/dist/)
- All patient documents must be de-identified before processing online
- Generated clinical decisions require validation by qualified medical professionals
- Not approved for production clinical decision-making
- Users are responsible for ensuring compliance with local regulations and institutional policies
