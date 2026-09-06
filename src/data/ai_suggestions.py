"""
ai_suggestions.py — Smart AI suggestions using any LLM provider.

Supports any OpenAI-compatible API:
    - Groq          (free)  base_url: https://api.groq.com/openai/v1
    - Google Gemini (free)  base_url: https://generativelanguage.googleapis.com/v1beta/openai
    - OpenRouter    (free)  base_url: https://openrouter.ai/api/v1
    - OpenAI        (paid)  base_url: https://api.openai.com/v1

Usage — set these environment variables before running:

    Groq (free):
        set NYDRA_API_KEY=your_groq_key
        set NYDRA_BASE_URL=https://api.groq.com/openai/v1
        set NYDRA_MODEL=llama3-8b-8192

    Gemini (free):
        set NYDRA_API_KEY=your_gemini_key
        set NYDRA_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
        set NYDRA_MODEL=gemini-1.5-flash

    OpenRouter (free tier):
        set NYDRA_API_KEY=your_openrouter_key
        set NYDRA_BASE_URL=https://openrouter.ai/api/v1
        set NYDRA_MODEL=mistralai/mistral-7b-instruct

    OpenAI (paid):
        set NYDRA_API_KEY=your_openai_key
        set NYDRA_BASE_URL=https://api.openai.com/v1
        set NYDRA_MODEL=gpt-4o-mini
"""

from __future__ import annotations
import json
import urllib.request
import urllib.error
from typing import Any


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL    = "llama3-8b-8192"


def _build_prompt(source, shape, missing, dupes, stats, outliers, relationships, ml_score):
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    outlier_cols = list(outliers.keys())
    rel_summary  = [
        f"{r['col_a']} <-> {r['col_b']} (strength={r['strength']}, {r['direction']})"
        for r in relationships[:5]
    ]
    col_types = {
        col: ("numeric" if s["type"] == "numeric" else "categorical")
        for col, s in stats.items()
    }
    checks_text = "\n".join(f"- {c['name']}: {c['status']} - {c['detail']}" for c in ml_score['checks'])

    return f"""You are a senior data scientist reviewing a dataset before ML training.
Here is the full analysis. Give 5-7 specific, actionable suggestions.

Dataset: {source}
Shape: {shape['rows']} rows x {shape['columns']} columns
ML Readiness Score: {ml_score['score']}/100 (grade {ml_score['grade']})

Column types: {json.dumps(col_types)}
Missing values: {json.dumps(missing_cols) if missing_cols else 'none'}
Duplicate rows: {dupes}
Outliers in: {outlier_cols if outlier_cols else 'none'}
Column relationships: {rel_summary if rel_summary else 'none found'}

ML checks:
{checks_text}

Instructions:
- Reference actual column names from the dataset.
- Focus on what improves ML model performance the most.
- Each suggestion starts with an action verb.
- Keep each suggestion to 1-2 sentences.
- Format as a numbered list (1. 2. 3. etc).
- Plain English only, no markdown."""


def get_ai_suggestions(
    api_key: str,
    source: str,
    shape: dict,
    missing: dict,
    dupes: int,
    stats: dict,
    outliers: dict,
    relationships: list,
    ml_score: dict,
    base_url: str = DEFAULT_BASE_URL,
    model: str    = DEFAULT_MODEL,
) -> str:
    prompt  = _build_prompt(source, shape, missing, dupes, stats, outliers, relationships, ml_score)
    url     = f"{base_url.rstrip('/')}/chat/completions"

    payload = json.dumps({
        "model":       model,
        "max_tokens":  600,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"API error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}")
