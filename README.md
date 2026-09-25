# 🩺 Nydra

**Nydra** is an open-source, autonomous data intelligence platform. It inspects, cleans, analyzes, and prepares datasets for machine learning through a full-stack web application, a command-line interface, and a REST API — backed by an intelligent agent that plans, executes, and self-heals its own data pipelines.

Formerly known as **dataDoctor**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Capabilities](#key-capabilities)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [AI Chat & LLM Providers](#ai-chat--llm-providers)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Example Datasets](#example-datasets)
- [Roadmap](#roadmap)
- [Security Notes](#security-notes)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Nydra goes beyond static data profiling. Its core agent (`Nydra`, in `src/core/agent.py`) plans a pipeline based on the shape and quality of the data it's given, executes each step with error isolation and automatic fallback, caches results to avoid recomputation, and keeps a full audit log and undo/redo history for every session.

The platform is accessible through three interfaces:

| Interface | Description |
|---|---|
| **Web App** | Next.js/React frontend + FastAPI backend, with authentication, job queues, and an integrated AI chat assistant |
| **CLI** | `cli.py` — scriptable access to every analysis and cleaning capability |
| **REST API** | `api.py` — the same backend the web app runs on, usable independently |

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                      User Interfaces                      │
│   CLI (cli.py)   │   Web UI (Next.js)   │   REST API      │
└─────────┬──────────────────┬───────────────────┬──────────┘
          │                  │                   │
┌─────────▼──────────────────▼───────────────────▼──────────┐
│              Nydra Agent (src/core/agent.py)               │
│   Pipeline Planner │ Step Executor │ Cache Manager         │
│   Session Manager  │ Decision Log  │ Plugin System         │
└─────────────────────────────┬──────────────────────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        │                      │                        │
┌───────▼────────┐   ┌─────────▼─────────┐   ┌──────────▼────────┐
│   Data Layer    │   │     ML Layer      │   │    Infra Layer     │
│ loader, cleaner │   │ AutoML, feature   │   │ JWT + Google OAuth  │
│ analyzer, drift │   │ importance, split │   │ Job queue + WS chat │
│ advanced stats  │   │ advisor, imbalance│   │ Security vault      │
│ advanced outlier│   │ detector, pipeline│   │ SQLite (SQLAlchemy) │
│ advanced imputer│   │ export            │   │                     │
└─────────────────┘   └───────────────────┘   └─────────────────────┘
```

**Backend:** FastAPI, SQLAlchemy + SQLite (WAL mode), Pydantic v2, `httpx`, `aiofiles`
**Frontend:** Next.js / React (TypeScript)
**Compute core:** `nydra-core`, a Rust extension (PyO3 + Rayon) for parallelized statistical computation
**Auth:** JWT (access + refresh, with proactive silent refresh) and Google OAuth (popup + `postMessage` exchange flow)
**Real-time:** WebSocket-based job progress and AI chat, with full multi-turn conversation history

---

## Key Capabilities

### Core Data Intelligence
- Smart inspection: missing values, duplicates, outliers, column statistics
- Auto-cleaning: imputation, deduplication, encoding, scaling
- ML Readiness Score with a detailed, itemized breakdown
- Correlation and relationship detection, including a full correlation network
- Data drift detection between two datasets
- Persistent inspection memory across sessions
- Cognitive Data DNA — a statistical fingerprint of a dataset

### Advanced Analytics (v0.5.5)
- **10+ imputation methods** — KNN, MICE, MissForest, gradient-boosting, ensemble, and an auto-selecting `SmartImputer`
- **11 outlier detectors** — Isolation Forest, LOF, DBSCAN, One-Class SVM, Mahalanobis, Elliptic Envelope, and more, unified behind a `SmartOutlierDetector` and ensemble voting
- **15+ AutoML models** with Optuna-driven Bayesian hyperparameter search, nested cross-validation, and a CASH (Combined Algorithm Selection and Hyperparameter) optimizer
- **Advanced statistics** — normality testing, distribution fitting across 20 candidate distributions, hypothesis testing with automatic parametric/non-parametric selection, and transformation recommendations

### Vision & Document Intelligence (v0.6.0)
- Image dataset analysis: quality scoring, duplicate detection (perceptual hashing), automated cleaning and normalization
- Document ingestion: PDF, DOCX, TXT, HTML, and Markdown
- Text analysis: readability, sentiment, topic modeling, keyword extraction, named entity recognition
- Training data auditing: leakage detection, label quality, bias and fairness metrics, and an overall training-readiness score

### AI Assistant
- A built-in chat assistant with full multi-turn memory, connected to a real LLM of your choice (see [AI Chat & LLM Providers](#ai-chat--llm-providers))
- Context-aware: can reference the results of a job you've already run

---

## Getting Started

### Requirements
- Python 3.11+
- Node.js (for the frontend)

### Installation

```bash
git clone https://github.com/Denterio1/Nydra.git
cd Nydra
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Running the full application

Two processes run independently and must both stay up:

```bash
# Terminal 1 — backend
python api.py

# Terminal 2 — frontend
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:3000` and will communicate with the backend at `http://127.0.0.1:8000`.

> Nydra currently runs in development mode only. A `Dockerfile` and `docker-compose.yml` are included in the repository for containerized deployment, but production hosting is not yet configured.

### CLI-only usage

For scripted or headless analysis without the web app:

```bash
python cli.py inspect examples/sample_sales.csv
python cli.py interactive   # guided mode, no flags required
python cli.py --help        # full command list
```

---

## Configuration

Authentication, job handling, and file-size limits are configured via environment variables (see `.env.example` if present, or `api.py`'s settings section). Sensitive values — including any LLM API keys you connect — are encrypted at rest using `src/security_vault.py` (PBKDF2-derived key, Fernet/AES-128) and are never stored or returned in plaintext.

---

## AI Chat & LLM Providers

Nydra's chat assistant is **bring-your-own-key**: no provider, model, or API key is hardcoded anywhere in the codebase. Each user connects their own key through the Settings panel, and it's encrypted before storage.

Supported providers:

| Provider | Style |
|---|---|
| Groq | OpenAI-compatible |
| OpenRouter | OpenAI-compatible |
| Google Gemini | Native |
| OpenAI | OpenAI-compatible |
| Anthropic | Native |
| Mistral AI | OpenAI-compatible |
| DeepSeek | OpenAI-compatible |
| Cerebras | OpenAI-compatible |
| Together AI | OpenAI-compatible |
| Cohere | Native |
| Custom | Any OpenAI-compatible endpoint |

Model lists are **fetched live** from each provider's own API (`GET /api/v1/config/llm/models?provider=...`) rather than hardcoded, since provider lineups change frequently. A small fallback list is kept per provider for the rare case an endpoint is briefly unreachable.

Chat supports full multi-turn conversation history over both REST (`POST /api/v1/chat`) and WebSocket (`WS /api/v1/ws/chat`), with real token streaming for OpenAI-compatible providers.

---

## CLI Reference

### Core Inspection
| Command | Description |
|---|---|
| `inspect <file>` | Full inspection: issues, cleaning, stats, and report |
| `clean <file>` | Clean data (deduplicate + fill missing) |
| `stats <file>` | Column statistics |
| `missing <file>` | Missing-value counts per column |
| `duplicates <file>` | Detect duplicate rows |
| `outliers <file>` | Outlier detection |
| `report <file>` | Generate an HTML report |
| `export <file>` | Inspect and export cleaned CSV |

### Machine Learning & Preparation
| Command | Description |
|---|---|
| `ml <file>` | ML Readiness Score and detailed checks |
| `prepare <file>` | Encode and scale data for ML |
| `target <file>` | Auto-detect the best target column |
| `split <file>` | Train/test split advisor and code generator |
| `imbalance <file>` | Class-imbalance detection and strategy advice |
| `encoding <file>` | Categorical encoding advisor |
| `importance <file>` | Feature importance via SHAP |
| `automl <file>` | Run and rank multiple candidate models |
| `pipeline <file>` | Export a full scikit-learn/imbalanced-learn pipeline |

### Advanced Analysis
| Command | Description |
|---|---|
| `relations <file>` | Column relationships and correlations |
| `network <file>` | Correlation network visualization |
| `engineer <file>` | Automated feature engineering |
| `schema <file>` | Schema inference, validation, export |
| `dna <file>` | Cognitive Data DNA (statistical fingerprint) |
| `drift <base> <new>` | Drift detection between two files |
| `audit <train> <test> --target <col>` | Full training-data audit |
| `images <folder/>` | Image dataset quality and duplicate analysis |
| `text <file>` | Document/text analysis |
| `memory` | Inspection history |
| `interactive` | Guided, flag-free mode |

Run `python cli.py --help` for the complete, current list.

---

## Project Structure

```
Nydra/
├── api.py                      ← FastAPI backend (REST + WebSocket)
├── cli.py                      ← Command-line interface
├── requirements.txt
├── Dockerfile / docker-compose.yml
│
├── frontend/                   ← Next.js / React web app
│
├── src/
│   ├── core/
│   │   └── agent.py            ← Nydra agent: planner, executor, cache, session
│   ├── security_vault.py       ← Encrypted API key storage
│   ├── config_manager.py       ← Versioned configuration store
│   ├── auth.py                 ← JWT + Google OAuth
│   ├── api_schemas.py          ← Pydantic models
│   ├── nydra_core.py           ← Bindings to the Rust statistics engine
│   │
│   ├── data/
│   │   ├── loader.py, analyzer.py, cleaner.py, preparator.py
│   │   ├── ml_readiness.py, relationships.py, drift.py, memory.py
│   │   ├── feature_engineer.py, target_detector.py, correlation_network.py
│   │   ├── auto_ml.py, feature_importance.py, split_advisor.py
│   │   ├── imbalance_detector.py, pipeline_export.py
│   │   ├── cognitive_dna.py, dna_memory.py
│   │   ├── db_connector.py, db_query.py, db_converter.py
│   │   ├── advanced_imputer.py, advanced_outlier.py
│   │   ├── advanced_automl.py, advanced_stats.py, quality_score.py
│   │   └── data_chat.py        ← Unified multi-provider LLM client
│   │
│   └── vision_nlp/
│       ├── image_loader.py, image_quality.py, image_analyzer.py
│       ├── image_cleaner.py, image_report.py
│       ├── document_reader.py, text_analyzer.py, text_quality.py
│       └── data_leakage.py, label_quality.py, bias_detector.py
│           training_readiness.py, audit_orchestrator.py
│
└── tests/
    ├── conftest.py
    ├── test_nydra.py
    ├── test_nydra_core.py
    ├── test_audit_layer.py
    └── test_api.py
```

---

## Testing

```bash
pytest tests/ -v
```

The suite covers the core pipeline, the Rust/Python statistics engine, the vision/text audit layer, and the full FastAPI backend (auth, jobs, WebSocket, settings).

---

## Example Datasets

The `examples/` directory previously bundled third-party sample datasets for local testing. These are no longer committed to the repository — see `.gitignore` — to keep it lightweight and avoid redistributing data that isn't Nydra's to redistribute. To reproduce the same local testing setup, fetch them yourself from their original sources:

- **Cats vs. Dogs (image classification)** — [Kaggle: Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- **MNIST (handwritten digits)** — [Yann LeCun's MNIST database](http://yann.lecun.com/exdb/mnist/), or via `torchvision.datasets.MNIST` / `tensorflow.keras.datasets.mnist`

Place downloaded files under `examples/` (already excluded from version control) and point the relevant CLI command or notebook at that path.

---

## Roadmap

| Version | Highlights | Status |
|---|---|---|
| v0.1.0 – v0.4.0 | Core inspection, cleaning, ML prep, feature engineering, schema validation, cognitive DNA | ✅ |
| v0.5.0 | Database connectivity, large-file support | ✅ |
| v0.5.5 | Advanced imputation, outlier detection, AutoML, statistics; agent rebuild | ✅ |
| v0.6.0 | Image and document/text intelligence, training-data auditing | ✅ (module build complete; full API wiring in progress) |
| v1.0.0 | Production backend: auth, job queue, WebSocket infrastructure | ✅ |
| v1.1.0 | Google OAuth, real multi-provider AI chat, encrypted key vault | ✅ |
| v1.2.0 | Live model-fetching, 5 additional LLM providers, WebSocket routing fix | ✅ |
| Unreleased | WebSocket multi-turn chat memory | ✅ (merged, pending next tag) |
| Planned | Security hardening layer (input sanitization, rate limiting, audit logging); true token streaming for all providers; production deployment | 📅 |

---

## Security Notes

- API keys and other secrets are encrypted at rest and are never logged or returned in plaintext.
- If you fork or self-host Nydra, verify your own environment's secret handling before deploying — in particular, ensure any database credentials (e.g. Supabase Row Level Security policies, if used) are configured correctly for your deployment.
- Report security concerns privately rather than via a public issue.

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss the approach.

```bash
git checkout -b feature/your-feature
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

Then open a Pull Request.

---

## License

MIT — free to use, modify, and distribute.

---

**Author:** Kader ([Denterio1](https://github.com/Denterio1))
**Repository:** [github.com/Denterio1/Nydra](https://github.com/Denterio1/Nydra)