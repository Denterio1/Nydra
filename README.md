# 🩺 Nydra — Autonomous Data Inspection Agent

> Drop your data. Get answers.

**Nydra** is an open-source autonomous agent that inspects, cleans, and prepares your data for Machine Learning — in seconds.

No code required. Just upload your file.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Inspection** | Missing values, duplicates, outliers, column stats |
| 🧹 **Auto Cleaning** | Fill missing values, remove duplicates |
| 📊 **HTML Reports** | Professional reports you can share |
| 🤖 **ML Readiness Score** | Know if your data is ready for ML |
| 🔗 **Relationship Detector** | Find correlations between columns |
| ⚡ **Data Preparation** | Encode + scale data for ML in one step |
| 📉 **Drift Detection** | Compare two datasets and find what changed |
| 🧠 **Data Memory** | Track inspection history across sessions |
| 💡 **AI Suggestions** | Get smart recommendations from any LLM |
| 🌐 **Web UI** | Beautiful browser interface — no Terminal needed |

**Supported file formats:** CSV · Excel (.xlsx) · JSON

---

## 🚀 Quick Start

### Option 1 — Full-Stack App (Recommended)

Nydra now features a modern Next.js frontend and a high-performance FastAPI backend.

```bash
# 1. Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# 2. Launch both Backend and Frontend
# Windows (PowerShell):
./run_nydra.ps1

# Windows (Command Prompt):
run_new_app.bat
```

**🔐 Automated HTTPS:** On the first run, Nydra will automatically generate self-signed SSL certificates (`cert.pem`, `key.pem`) for `localhost`.
> **Note:** Your browser will show a security warning for the self-signed certificate. Click **Advanced** -> **Proceed to localhost (unsafe)** to allow the frontend to communicate with the API.

---

### Option 2 — Streamlit UI (Legacy / Lightweight)

```bash
streamlit run app.py
```
Your browser will open at `http://localhost:8501`

---

### Option 3 — CLI (For developers)

```bash
# Full inspection report
python cli.py inspect examples/sample_sales.csv

# Interactive mode (guided, no flags needed)
python cli.py interactive

# See all commands
python cli.py --help
```

---

## 📦 Installation

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Denterio1/Nydra.git
cd Nydra
pip install -r requirements.txt
```

---

## 🖥️ Web UI Guide

1. Run `streamlit run app.py`
2. Upload your file in the sidebar
3. Explore 8 tabs:

| Tab | What you get |
|-----|-------------|
| 📊 Overview | Data preview + column type chart |
| 🔍 Quality | Missing values + outlier charts |
| 📈 Statistics | Interactive histograms per column |
| 🤖 ML Readiness | Score + detailed checks |
| 🔗 Relationships | Correlation heatmap |
| 🧹 Cleaning | Clean + download your data |
| 📉 Drift | Compare two datasets |
| 📂 Multi-File | Analyse and compare multiple files |

---

## ⌨️ CLI Commands

`Nydra` provides a comprehensive suite of commands for data analysis and ML preparation.

### ── Core Inspection
| Command | Description |
|---------|-------------|
| `inspect <file>` | Full inspection: issues + cleaning + stats + report |
| `clean <file>` | Clean data (remove dupes + fill missing) |
| `stats <file>` | Show column statistics only |
| `missing <file>` | Show missing value counts per column |
| `duplicates <file>` | Detect and show duplicate rows |
| `outliers <file>` | Detect outliers using IQR method |
| `report <file>` | Generate professional HTML report |
| `export <file>` | Inspect + export cleaned CSV to disk |

### ── Machine Learning & Prep
| Command | Description |
|---------|-------------|
| `ml <file>` | ML Readiness Score & detailed checks |
| `prepare <file>` | Prepare data for ML (encode + scale) |
| `target <file>` | Auto-detect best target column for ML |
| `split <file>` | Train/Test split advisor & code generator |
| `imbalance <file>` | Class imbalance detector & strategy advisor |
| `encoding <file>` | Smart Encoding Advisor for categorical data |
| `importance <file>` | Feature Importance using SHAP values |
| `automl <file>` | Run 5 models to find the best baseline |
| `pipeline <file>` | Export full Sklearn/Imblearn pipeline code |

### ── Advanced Analysis
| Command | Description |
|---------|-------------|
| `relations <file>` | Detect column relationships & correlations |
| `network <file>` | Build & visualize correlation network |
| `engineer <file>` | Auto Feature Engineering (Date/Text/Num) |
| `schema <file>` | Schema Validator (Infer/Validate/Export) |
| `dna <file>` | Cognitive Data DNA (Statistical identity) |
| `drift <base> <new>` | Detect data drift between two files |
| `suggest <file>` | AI-powered smart suggestions (needs .env) |
| `memory` | Track & compare inspection history |
| `interactive` | Guided interactive mode — no flags needed |

**Options:**
```bash
--strategy mean    # Fill missing with mean (default)
--strategy median  # Fill missing with median
--strategy mode    # Fill missing with most frequent value
--strategy drop    # Drop rows with missing values
--no-dedup         # Keep duplicate rows
```


---

## 💡 AI Suggestions Setup

Nydra supports any OpenAI-compatible API. Create a `.env` file:

```env
# Groq (free) — recommended
NYDRA_API_KEY=your_groq_key
NYDRA_BASE_URL=https://api.groq.com/openai/v1
NYDRA_MODEL=llama-3.3-70b-versatile

# Google Gemini (free)
# NYDRA_API_KEY=your_gemini_key
# NYDRA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# NYDRA_MODEL=gemini-1.5-flash

# OpenAI (paid)
# NYDRA_API_KEY=your_openai_key
# NYDRA_BASE_URL=https://api.openai.com/v1
# NYDRA_MODEL=gpt-4o-mini
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📁 Project Structure

```
Nydra/
├── app.py                  ← Web UI (Streamlit)
├── cli.py                  ← Command line interface
├── requirements.txt
├── .env                    ← API keys (not committed)
│
├── src/
│   ├── core/
│   │   └── agent.py        ← Main Nydra agent
│   ├── data/
│   │   ├── loader.py       ← CSV / Excel / JSON reader
│   │   ├── analyzer.py     ← Quality checks & statistics
│   │   ├── cleaner.py      ← Missing values & deduplication
│   │   ├── ml_readiness.py ← ML Readiness Score
│   │   ├── relationships.py← Column relationship detector
│   │   ├── preparator.py   ← ML data preparation pipeline
│   │   ├── drift.py        ← Data drift detection
│   │   ├── memory.py       ← Persistent inspection history
│   │   └── ai_suggestions.py ← AI-powered recommendations
│   └── report.py           ← HTML report generator
│
├── tests/
│   └── test_nydra.py  ← 38 pytest tests
│
└── examples/
    ├── sample_sales.csv
    ├── sample_employees.xlsx
    └── sample_products.json
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

38 tests — all passing.

---

## 🗺️ Roadmap

| Version | Features | Status |
|---------|----------|--------|
| **v0.1.0** | Core inspection, cleaning, ML prep, web UI | ✅ |
| **v0.2.0** | Auto Feature Engineering, Schema Validator | ✅ |
| **v0.3.0** | ML Baseline, Feature Importance, Pipeline Export | ✅ |
| **v0.4.0** | Cognitive DNA, Correlation Network, Target Detection | ✅ |
| **v0.5.0** | Database Connector, Parquet support, API Mode | 🚧 |
| **v1.0.0** | Docker, Cloud Deploy, Plugin System | 📅 |

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request
