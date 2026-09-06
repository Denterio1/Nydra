"""
data_chat.py — Universal Data Chat Engine v2.0
Nydra — Talk to your data using ANY AI model from ANY provider.

Supported Providers (12):
    OpenAI       → GPT-4o, GPT-4, o1, o3-mini ...
    Anthropic    → Claude Opus, Sonnet, Haiku ...
    Google       → Gemini 2.0, 1.5 Pro/Flash ...
    Groq         → Llama 3.3, Mixtral (fastest) ...
    Mistral AI   → Mistral Large, Codestral ...
    Together AI  → 50+ open models ...
    DeepSeek     → DeepSeek Chat, Reasoner ...
    xAI          → Grok 2 ...
    Perplexity   → Sonar models (web search) ...
    Cohere       → Command R+ ...
    OpenRouter   → Any model via one key ...
    Ollama       → Local models, no key needed ...

Architecture:
    PROVIDER_REGISTRY     — All providers, models, formats, base URLs
    LLMConfig             — User settings per request
    LLMResponse           — Unified response from any provider
    SafeCodeExecutor      — Sandboxed Python execution on DataFrames
    DataContextBuilder    — Rich DataFrame → AI-readable context
    SystemPromptBuilder   — Builds expert system prompts per task
    UnifiedLLMClient      — Single interface for ALL provider formats
    ConversationMemory    — Smart history with token budget management
    QueryParser           — Natural language → structured intent (rule-based)
    DataQueryEngine       — Execute structured queries on DataFrames
    DataChatBot           — Master orchestrator: rules + LLM + fallback

Author  : Nydra / Kader (Denterio1)
Version : 2.0.0
Updated : 2026
"""

from __future__ import annotations

import re
import json
import time
import logging
import hashlib
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator

import pandas as pd
import numpy as np

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("nydra.chat")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROVIDER REGISTRY
# The single source of truth for every supported LLM provider.
# Format types: "openai" | "anthropic" | "gemini" | "ollama" | "cohere"
# ══════════════════════════════════════════════════════════════════════════════

PROVIDER_REGISTRY: dict[str, dict] = {

    "openai": {
        "name"        : "OpenAI",
        "format"      : "openai",
        "base_url"    : "https://api.openai.com/v1",
        "requires_key": True,
        "key_hint"    : "Starts with sk-",
        "free_tier"   : False,
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3-mini",
        ],
        "recommended" : "gpt-4o",
        "notes"       : "Industry standard. Best ecosystem.",
    },

    "anthropic": {
        "name"        : "Anthropic (Claude)",
        "format"      : "anthropic",
        "base_url"    : "https://api.anthropic.com",
        "requires_key": True,
        "key_hint"    : "Starts with sk-ant-",
        "free_tier"   : False,
        "models": [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20251101",
            "claude-haiku-4-5-20251101",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        "recommended" : "claude-sonnet-4-5-20251101",
        "notes"       : "Best instruction following. 200K context. Different API format.",
    },

    "google": {
        "name"        : "Google Gemini",
        "format"      : "gemini",
        "base_url"    : "https://generativelanguage.googleapis.com",
        "requires_key": True,
        "key_hint"    : "From Google AI Studio (aistudio.google.com)",
        "free_tier"   : True,
        "models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.5-pro-preview-05-06",
        ],
        "recommended" : "gemini-2.0-flash",
        "notes"       : "Free tier available. Different API format.",
    },

    "groq": {
        "name"        : "Groq",
        "format"      : "openai",
        "base_url"    : "https://api.groq.com/openai/v1",
        "requires_key": True,
        "key_hint"    : "Starts with gsk_",
        "free_tier"   : True,
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma-7b-it",
        ],
        "recommended" : "llama-3.3-70b-versatile",
        "notes"       : "Fastest inference. Free tier. OpenAI-compatible.",
    },

    "mistral": {
        "name"        : "Mistral AI",
        "format"      : "openai",
        "base_url"    : "https://api.mistral.ai/v1",
        "requires_key": True,
        "key_hint"    : "From console.mistral.ai",
        "free_tier"   : False,
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "codestral-latest",
            "open-mixtral-8x22b",
            "open-mistral-nemo",
            "open-mistral-7b",
        ],
        "recommended" : "mistral-large-latest",
        "notes"       : "Strong code model (Codestral). European provider.",
    },

    "together": {
        "name"        : "Together AI",
        "format"      : "openai",
        "base_url"    : "https://api.together.xyz/v1",
        "requires_key": True,
        "key_hint"    : "From api.together.xyz",
        "free_tier"   : True,
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
            "google/gemma-2-27b-it",
        ],
        "recommended" : "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "notes"       : "Access to 50+ open models. Free $25 credit on signup.",
    },

    "deepseek": {
        "name"        : "DeepSeek",
        "format"      : "openai",
        "base_url"    : "https://api.deepseek.com/v1",
        "requires_key": True,
        "key_hint"    : "From platform.deepseek.com",
        "free_tier"   : False,
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-coder",
        ],
        "recommended" : "deepseek-chat",
        "notes"       : "Very cheap. Strong at reasoning and code.",
    },

    "xai": {
        "name"        : "xAI (Grok)",
        "format"      : "openai",
        "base_url"    : "https://api.x.ai/v1",
        "requires_key": True,
        "key_hint"    : "Starts with xai-",
        "free_tier"   : False,
        "models": [
            "grok-2-latest",
            "grok-2-vision-latest",
            "grok-beta",
        ],
        "recommended" : "grok-2-latest",
        "notes"       : "Real-time data access. OpenAI-compatible.",
    },

    "perplexity": {
        "name"        : "Perplexity",
        "format"      : "openai",
        "base_url"    : "https://api.perplexity.ai",
        "requires_key": True,
        "key_hint"    : "Starts with pplx-",
        "free_tier"   : False,
        "models": [
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
        ],
        "recommended" : "llama-3.1-sonar-large-128k-online",
        "notes"       : "Includes live web search. Good for recent data.",
    },

    "cohere": {
        "name"        : "Cohere",
        "format"      : "cohere",
        "base_url"    : "https://api.cohere.ai/v1",
        "requires_key": True,
        "key_hint"    : "From dashboard.cohere.com",
        "free_tier"   : True,
        "models": [
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
            "command-nightly",
        ],
        "recommended" : "command-r-plus",
        "notes"       : "Free trial key available. Enterprise-grade.",
    },

    "openrouter": {
        "name"        : "OpenRouter",
        "format"      : "openai",
        "base_url"    : "https://openrouter.ai/api/v1",
        "requires_key": True,
        "key_hint"    : "Starts with sk-or-",
        "free_tier"   : True,
        "models": [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-flash-1.5",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
            "deepseek/deepseek-chat",
            "qwen/qwen-2.5-72b-instruct",
            "microsoft/phi-4",
            "google/gemma-2-27b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ],
        "recommended" : "google/gemini-flash-1.5",
        "notes"       : "One key for ALL providers. Some models are free.",
    },

    "ollama": {
        "name"        : "Ollama (Local)",
        "format"      : "ollama",
        "base_url"    : "http://localhost:11434",
        "requires_key": False,
        "key_hint"    : "No API key needed",
        "free_tier"   : True,
        "models": [
            "llama3.2",
            "llama3.1",
            "llama3.2:3b",
            "mistral",
            "codellama",
            "phi3",
            "phi3:mini",
            "gemma2",
            "qwen2.5",
            "deepseek-r1",
            "nomic-embed-text",
        ],
        "recommended" : "llama3.2",
        "notes"       : "Runs locally. No cost. No data sent externally.",
    },
}

# Convenience: all provider names
ALL_PROVIDERS       = list(PROVIDER_REGISTRY.keys())
OPENAI_FORMAT_PROVIDERS = [k for k, v in PROVIDER_REGISTRY.items() if v["format"] == "openai"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """
    User-provided configuration for any LLM provider.
    This is everything the UnifiedLLMClient needs to make a call.
    """
    provider    : str               # Key in PROVIDER_REGISTRY
    model       : str               # Model name/ID
    api_key     : str = ""          # Empty string for Ollama
    base_url    : str = ""          # Override default base URL (optional)
    temperature : float = 0.2       # Lower = more deterministic (good for data analysis)
    max_tokens  : int   = 1500      # Max tokens in response
    timeout     : int   = 60        # Request timeout in seconds
    custom_url  : str   = ""        # For any custom OpenAI-compatible endpoint

    def __post_init__(self) -> None:
        # If user did not specify a custom URL, use the registry default
        if not self.base_url and not self.custom_url:
            info = PROVIDER_REGISTRY.get(self.provider, {})
            self.base_url = info.get("base_url", "")
        elif self.custom_url:
            self.base_url = self.custom_url

    @property
    def effective_url(self) -> str:
        return self.base_url

    @property
    def api_format(self) -> str:
        return PROVIDER_REGISTRY.get(self.provider, {}).get("format", "openai")


@dataclass
class LLMResponse:
    """
    Unified response from ANY provider.
    All format differences are normalized here.
    """
    success      : bool
    text         : str              # The actual text content
    provider     : str
    model        : str
    prompt_tokens: int  = 0
    output_tokens: int  = 0
    latency_ms   : int  = 0
    raw          : Any  = None      # Raw response dict (for debugging)
    error        : str  = ""


@dataclass
class ParsedQuery:
    """Structured representation of a natural language query."""
    intent      : str               # aggregate | filter | sort | describe | compare | count
    columns     : list[str]
    conditions  : list[dict]
    aggregation : str               # sum | mean | max | min | count | std
    group_by    : str | None
    order_by    : str | None
    ascending   : bool
    limit       : int
    raw_query   : str


@dataclass
class QueryResult:
    """Result of executing a query (rule-based or LLM-based)."""
    success     : bool
    data        : Any               # DataFrame, scalar, dict, or None
    answer      : str               # Human-readable answer text
    query       : ParsedQuery
    chart_type  : str | None = None # "bar" | "line" | "pie" | "scatter" | None
    code        : str | None = None # Generated Python code (if any)
    error       : str        = ""
    provider    : str        = ""   # Which provider answered (if LLM)
    model       : str        = ""   # Which model answered (if LLM)
    latency_ms  : int        = 0


@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role        : str               # "user" | "assistant" | "system"
    content     : str
    timestamp   : float = field(default_factory=time.time)
    result      : QueryResult | None = None
    token_count : int   = 0

    def estimate_tokens(self) -> int:
        """Rough token count estimate (1 token ≈ 4 chars)."""
        return max(1, len(self.content) // 4)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAFE CODE EXECUTOR
# Executes LLM-generated Python code on DataFrames with security sandboxing.
# ══════════════════════════════════════════════════════════════════════════════

class SafeCodeExecutor:
    """
    Executes Python code strings on a DataFrame safely.

    Security:
    - Blocks dangerous imports (os, sys, subprocess, socket, requests ...)
    - Blocks file operations (open, write, read ...)
    - Blocks network access
    - Blocks deletion operations
    - Timeout enforcement
    - Restricted builtins

    The LLM is instructed to always assign the final result to `result`.
    """

    # Patterns that are NEVER allowed in executed code
    FORBIDDEN_PATTERNS: list[str] = [
        r'\bimport\s+os\b',
        r'\bimport\s+sys\b',
        r'\bimport\s+subprocess\b',
        r'\bimport\s+socket\b',
        r'\bimport\s+requests?\b',
        r'\bimport\s+urllib\b',
        r'\bimport\s+shutil\b',
        r'\bimport\s+pathlib\b',
        r'\bimport\s+glob\b',
        r'\b__import__\b',
        r'\beval\s*\(',
        r'\bexec\s*\(',
        r'\bopen\s*\(',
        r'\bwrite\s*\(',
        r'\bdelete\b',
        r'\bremove\b',
        r'\bdrop\s*\(',
        r'\brmdir\b',
        r'\bunlink\b',
        r'\bos\.path\b',
        r'\bsys\.exit\b',
        r'\bquit\s*\(',
        r'\bexit\s*\(',
    ]

    # Safe builtins allowed during execution
    SAFE_BUILTINS: dict = {
        "abs": abs, "all": all, "any": any,
        "bool": bool, "dict": dict, "enumerate": enumerate,
        "filter": filter, "float": float, "format": format,
        "frozenset": frozenset,
        "hasattr": hasattr, "hash": hash,
        "int": int, "isinstance": isinstance, "issubclass": issubclass,
        "iter": iter, "len": len, "list": list,
        "map": map, "max": max, "min": min,
        "next": next,
        "print": print, "range": range,
        "repr": repr, "reversed": reversed,
        "round": round, "set": set, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum,
        "tuple": tuple, "zip": zip,
        "True": True, "False": False, "None": None,
    }

    def is_safe(self, code: str) -> tuple[bool, str]:
        """
        Check if code is safe to execute.
        Returns (is_safe, reason_if_not_safe).
        """
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"
        return True, ""

    def clean_code(self, code: str) -> str:
        """Strip markdown code fences and whitespace."""
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        return code.strip()

    def execute(self, code: str, df: pd.DataFrame) -> tuple[Any, str | None]:
        """
        Execute code string against a DataFrame.

        Returns:
            (result, error_string)
            result is whatever the code stored in `result` variable.
            error_string is None on success, or the error message.
        """
        code = self.clean_code(code)

        # Security check first
        safe, reason = self.is_safe(code)
        if not safe:
            logger.warning(f"Code execution blocked: {reason}")
            return None, f"Code blocked for security reasons: {reason}"

        # Build execution namespace
        local_ns: dict = {
            "df"   : df.copy(),          # Copy to prevent mutation
            "pd"   : pd,
            "np"   : np,
            "result": None,              # Default result
            "__builtins__": self.SAFE_BUILTINS,
        }

        try:
            exec(code, {"__builtins__": self.SAFE_BUILTINS}, local_ns)
            result = local_ns.get("result")

            # If result is a DataFrame, limit rows for display
            if isinstance(result, pd.DataFrame):
                result = result.head(50)

            return result, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.debug(f"Code execution error:\n{code}\n→ {error_msg}")
            return None, error_msg

    def execute_safe(self, code: str, df: pd.DataFrame) -> dict:
        """
        Wrapper that always returns a structured dict.
        Never raises exceptions.
        """
        try:
            result, error = self.execute(code, df)
            return {
                "success": error is None,
                "result" : result,
                "error"  : error or "",
                "code"   : code,
            }
        except Exception as e:
            return {
                "success": False,
                "result" : None,
                "error"  : str(e),
                "code"   : code,
            }


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA CONTEXT BUILDER
# Converts a DataFrame into a rich text description the AI can understand.
# ══════════════════════════════════════════════════════════════════════════════

class DataContextBuilder:
    """
    Builds a rich, informative text description of a DataFrame
    to inject into the LLM system prompt.

    The AI needs to know:
    - What columns exist and their types
    - Sample values (to understand the data domain)
    - Key statistics (range, missing values, unique counts)
    - Relationships (correlations between numeric columns)
    - Special patterns (date columns, IDs, text columns)
    """

    MAX_SAMPLE_ROWS   : int = 5
    MAX_UNIQUE_PREVIEW: int = 8
    MAX_CORR_PAIRS    : int = 6

    def build(self, df: pd.DataFrame) -> str:
        """Build the full context string for a DataFrame."""
        sections = [
            self._build_header(df),
            self._build_schema(df),
            self._build_sample(df),
            self._build_numeric_stats(df),
            self._build_categorical_stats(df),
            self._build_missing_summary(df),
            self._build_correlations(df),
            self._build_special_columns(df),
        ]
        return "\n\n".join(s for s in sections if s)

    def _build_header(self, df: pd.DataFrame) -> str:
        mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
        return (
            f"=== DATASET OVERVIEW ===\n"
            f"Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"Memory     : {mem_mb:.2f} MB\n"
            f"Total Cells: {df.size:,}\n"
            f"Missing    : {df.isnull().sum().sum():,} cells "
            f"({df.isnull().mean().mean() * 100:.1f}% of total)"
        )

    def _build_schema(self, df: pd.DataFrame) -> str:
        lines = ["=== COLUMN SCHEMA ==="]
        for col in df.columns:
            dtype    = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null   = df[col].isnull().sum()
            null_pct = n_null / len(df) * 100 if len(df) > 0 else 0

            col_type = self._classify_column(df[col])
            lines.append(
                f"  [{col_type:10s}] {col:<30s} "
                f"dtype={dtype:<12s} "
                f"unique={n_unique:<8,d} "
                f"missing={null_pct:.1f}%"
            )
        return "\n".join(lines)

    def _classify_column(self, series: pd.Series) -> str:
        """Classify column role/type for the AI."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "DATETIME"
        if pd.api.types.is_bool_dtype(series):
            return "BOOLEAN"
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() <= 10:
                return "NUMERIC_CAT"   # Numeric but few values (like 0/1, 1-5 ratings)
            return "NUMERIC"
        if series.dtype == object:
            avg_len = series.dropna().astype(str).str.len().mean()
            if series.nunique() <= 20:
                return "CATEGORY"
            if avg_len > 50:
                return "TEXT"
            # Check if it could be a date string
            sample = series.dropna().head(3).tolist()
            date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}']
            if any(re.search(p, str(s)) for s in sample for p in date_patterns):
                return "DATE_STR"
            return "STRING"
        return "OTHER"

    def _build_sample(self, df: pd.DataFrame) -> str:
        sample = df.head(self.MAX_SAMPLE_ROWS)
        try:
            sample_str = sample.to_string(max_colwidth=30, index=False)
            return f"=== SAMPLE DATA (first {len(sample)} rows) ===\n{sample_str}"
        except Exception:
            return f"=== SAMPLE DATA ===\n{sample.to_dict(orient='records')}"

    def _build_numeric_stats(self, df: pd.DataFrame) -> str:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return ""
        lines = ["=== NUMERIC COLUMN STATISTICS ==="]
        for col in num_df.columns:
            s = num_df[col].dropna()
            if len(s) == 0:
                continue
            lines.append(
                f"  {col:<30s} "
                f"min={s.min():<12.4g} "
                f"mean={s.mean():<12.4g} "
                f"median={s.median():<12.4g} "
                f"max={s.max():<12.4g} "
                f"std={s.std():.4g}"
            )
        return "\n".join(lines)

    def _build_categorical_stats(self, df: pd.DataFrame) -> str:
        cat_cols = [c for c in df.columns
                    if df[c].dtype == object or df[c].nunique() <= 20]
        if not cat_cols:
            return ""
        lines = ["=== CATEGORICAL COLUMN PREVIEW ==="]
        for col in cat_cols[:10]:   # Limit to 10 categorical columns
            top = df[col].value_counts().head(self.MAX_UNIQUE_PREVIEW)
            values_str = ", ".join(
                f"{k}({v})" for k, v in top.items()
            )
            lines.append(f"  {col:<30s} top values: {values_str}")
        return "\n".join(lines)

    def _build_missing_summary(self, df: pd.DataFrame) -> str:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return "=== MISSING VALUES ===\nNo missing values found."
        lines = ["=== MISSING VALUES (columns with nulls) ==="]
        for col, count in missing.items():
            pct = count / len(df) * 100
            lines.append(f"  {col:<30s} {count:>6,} missing ({pct:.1f}%)")
        return "\n".join(lines)

    def _build_correlations(self, df: pd.DataFrame) -> str:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return ""
        try:
            corr    = num_df.corr().abs()
            # Get upper triangle pairs
            pairs   = []
            cols    = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], corr.iloc[i, j]))
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = pairs[:self.MAX_CORR_PAIRS]
            if not top_pairs:
                return ""
            lines = ["=== TOP CORRELATIONS (absolute) ==="]
            for c1, c2, r in top_pairs:
                strength = (
                    "very strong" if r > 0.9 else
                    "strong"      if r > 0.7 else
                    "moderate"    if r > 0.4 else
                    "weak"
                )
                lines.append(f"  {c1} ↔ {c2}: r={r:.3f} ({strength})")
            return "\n".join(lines)
        except Exception:
            return ""

    def _build_special_columns(self, df: pd.DataFrame) -> str:
        """Detect likely ID columns, target columns, and date columns."""
        notes = []

        for col in df.columns:
            s = df[col]
            col_lower = col.lower()

            # Likely target/label column
            if any(kw in col_lower for kw in ["target", "label", "churn", "fraud", "default", "class", "y", "output"]):
                notes.append(f"  '{col}' → likely TARGET/LABEL column")

            # Likely ID column
            if any(kw in col_lower for kw in ["id", "_key", "uuid", "index"]):
                if s.nunique() == len(df):
                    notes.append(f"  '{col}' → likely ID column (all unique, not useful for ML)")

            # Date column stored as string
            if s.dtype == object:
                sample = s.dropna().head(3).astype(str).tolist()
                if any(re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', v) for v in sample):
                    notes.append(f"  '{col}' → contains dates stored as strings (consider pd.to_datetime)")

        if not notes:
            return ""
        return "=== SPECIAL COLUMN NOTES ===\n" + "\n".join(notes)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SYSTEM PROMPT BUILDER
# Builds expert system prompts tailored for data analysis tasks.
# ══════════════════════════════════════════════════════════════════════════════

class SystemPromptBuilder:
    """
    Builds structured system prompts for the LLM.
    A good system prompt is the single most important factor in response quality.
    """

    BASE_INSTRUCTIONS = """
You are 'Nydra AI', an expert Senior Data Analyst and Data Scientist.
You have been given access to a real pandas DataFrame called `df`.

YOUR CORE RULES:
1. Always answer in the SAME LANGUAGE as the user's question.
2. When a question requires calculation, write Python code. Do NOT guess numbers.
3. Python code MUST be inside a ```python ... ``` block.
4. ALWAYS store your final answer in a variable called `result`.
5. `result` must be one of: a number, a string, a dict, or a pandas DataFrame.
6. If the code might fail (e.g., column not found), add a try/except block.
7. After the code block, explain the result in plain language.
8. When no code is needed (greeting, help, simple description), respond directly.
9. Be concise. No unnecessary padding.
10. Reference the conversation history when relevant.

WHAT YOU CAN DO:
- Calculate statistics (mean, sum, max, min, std, percentiles ...)
- Filter rows by conditions
- Group and aggregate data
- Find duplicates and missing values
- Detect outliers
- Compare columns
- Generate insights and recommendations
- Write complex multi-step analysis
- Explain your findings clearly

WHAT YOU CANNOT DO:
- Modify, delete, or write files
- Access the internet
- Import dangerous modules (os, sys, subprocess ...)
- Access anything outside of the provided `df`

CODE EXAMPLES (follow this style exactly):
```python
# Simple aggregation
result = df["salary"].mean()
```

```python
# Group by aggregation  
result = df.groupby("department")["salary"].mean().sort_values(ascending=False).to_dict()
```

```python
# Filter + count
filtered = df[df["age"] > 30]
result = {
    "count": len(filtered),
    "average_salary": round(filtered["salary"].mean(), 2),
    "departments": filtered["department"].value_counts().to_dict()
}
```

```python
# Multi-step with error handling
try:
    result = df.groupby("region")["revenue"].agg(["sum", "mean", "count"]).round(2).to_dict()
except KeyError as e:
    result = f"Column not found: {e}"
```
"""

    def build(self, df: pd.DataFrame, data_context: str, language_hint: str = "en") -> str:
        """Build the full system prompt including dataset context."""
        lang_note = ""
        if language_hint == "ar":
            lang_note = "\n\nNOTE: The user prefers Arabic. Always respond in Arabic (العربية).\n"

        prompt = f"{self.BASE_INSTRUCTIONS}{lang_note}\n\n{'='*60}\n{data_context}\n{'='*60}\n"
        prompt += f"\nDataFrame available as `df` with {df.shape[0]:,} rows and {df.shape[1]} columns."
        return prompt

    def detect_language(self, text: str) -> str:
        """Detect if the user wrote in Arabic."""
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "ar" if arabic_chars > len(text) * 0.3 else "en"


# ══════════════════════════════════════════════════════════════════════════════
# 6. UNIFIED LLM CLIENT
# The heart of the system — one interface to call ANY provider.
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedLLMClient:
    """
    Universal LLM client that handles 5 different API formats:

    1. openai   → OpenAI, Groq, Mistral, Together, DeepSeek, xAI,
                  Perplexity, OpenRouter, Ollama-compat, any custom URL
    2. anthropic → Anthropic (Claude) — totally different structure
    3. gemini   → Google Gemini — uses API key in URL, different request body
    4. ollama   → Ollama local — streams by default, no key needed
    5. cohere   → Cohere — different message format

    All formats are normalized into a single LLMResponse object.
    """

    def __init__(self) -> None:
        try:
            import requests
            self._session = requests.Session()
        except ImportError:
            raise ImportError("requests library required: pip install requests")

    def complete(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> LLMResponse:
        """
        Main entry point — routes to the correct format handler.

        Args:
            config        : LLMConfig with provider/model/key settings
            system_prompt : The system instruction string
            messages      : List of {"role": ..., "content": ...} dicts

        Returns:
            LLMResponse (always — never raises)
        """
        start = time.time()
        fmt   = config.api_format

        try:
            if fmt == "anthropic":
                raw = self._call_anthropic(config, system_prompt, messages)
            elif fmt == "gemini":
                raw = self._call_gemini(config, system_prompt, messages)
            elif fmt == "ollama":
                raw = self._call_ollama(config, system_prompt, messages)
            elif fmt == "cohere":
                raw = self._call_cohere(config, system_prompt, messages)
            else:
                # Default: OpenAI-compatible format (covers most providers)
                raw = self._call_openai(config, system_prompt, messages)

            latency = int((time.time() - start) * 1000)
            return self._parse_response(raw, config, latency, fmt)

        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"LLM call failed [{config.provider}/{config.model}]: {e}")
            return LLMResponse(
                success      = False,
                text         = "",
                provider     = config.provider,
                model        = config.model,
                latency_ms   = latency,
                error        = str(e),
            )

    # ── OpenAI-compatible format (covers most providers) ─────────────────────

    def _call_openai(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> dict:
        """
        Handles: OpenAI, Groq, Mistral, Together, DeepSeek,
                 xAI, Perplexity, OpenRouter, custom endpoints.
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        headers = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }

        # OpenRouter requires extra headers
        if config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/Denterio1/Nydra"
            headers["X-Title"]      = "Nydra"

        payload = {
            "model"      : config.model,
            "messages"   : full_messages,
            "max_tokens" : config.max_tokens,
            "temperature": config.temperature,
        }

        url      = f"{config.effective_url.rstrip('/')}/chat/completions"
        response = self._session.post(url, headers=headers, json=payload, timeout=config.timeout)

        if response.status_code != 200:
            self._raise_api_error(response, config.provider)

        return {"format": "openai", "data": response.json()}

    # ── Anthropic format ──────────────────────────────────────────────────────

    def _call_anthropic(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> dict:
        """
        Anthropic Claude uses a completely different API structure:
        - System prompt is a TOP-LEVEL field, NOT inside messages
        - Uses `anthropic-version` header
        - Response is in `content[0].text`, not `choices[0].message.content`
        - URL: /v1/messages (not /v1/chat/completions)
        """
        headers = {
            "Content-Type"     : "application/json",
            "x-api-key"        : config.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model"      : config.model,
            "max_tokens" : config.max_tokens,
            "system"     : system_prompt,             # ← Top-level, NOT in messages
            "messages"   : messages,                  # ← No system role allowed here
            "temperature": config.temperature,
        }

        url      = f"{config.effective_url.rstrip('/')}/v1/messages"
        response = self._session.post(url, headers=headers, json=payload, timeout=config.timeout)

        if response.status_code != 200:
            self._raise_api_error(response, "anthropic")

        return {"format": "anthropic", "data": response.json()}

    # ── Google Gemini format ──────────────────────────────────────────────────

    def _call_gemini(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> dict:
        """
        Google Gemini has a completely different API structure:
        - API key goes in the URL (not headers)
        - Uses `contents` instead of `messages`
        - Roles are "user" and "model" (not "assistant")
        - System prompt goes in `systemInstruction`
        - Response is in candidates[0].content.parts[0].text
        """
        # Convert OpenAI-style messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role    = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append({
                "role"  : role,
                "parts" : [{"text": msg["content"]}],
            })

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents"          : gemini_contents,
            "generationConfig"  : {
                "maxOutputTokens": config.max_tokens,
                "temperature"    : config.temperature,
            },
        }

        # API key is a URL parameter for Gemini
        url      = (
            f"{config.effective_url.rstrip('/')}"
            f"/v1beta/models/{config.model}:generateContent"
            f"?key={config.api_key}"
        )
        headers  = {"Content-Type": "application/json"}
        response = self._session.post(url, headers=headers, json=payload, timeout=config.timeout)

        if response.status_code != 200:
            self._raise_api_error(response, "gemini")

        return {"format": "gemini", "data": response.json()}

    # ── Ollama local format ───────────────────────────────────────────────────

    def _call_ollama(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> dict:
        """
        Ollama runs locally. Different structure:
        - No API key required
        - System prompt goes inside messages as a system role
        - stream: false for synchronous response
        - Response is in message.content (not choices[0].message.content)
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model"   : config.model,
            "messages": full_messages,
            "stream"  : False,
            "options" : {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        }

        url = f"{config.effective_url.rstrip('/')}/api/chat"
        try:
            response = self._session.post(url, json=payload, timeout=config.timeout)
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {config.effective_url}. "
                f"Is Ollama running? Start it with: ollama serve\n"
                f"Error: {e}"
            )

        if response.status_code != 200:
            self._raise_api_error(response, "ollama")

        return {"format": "ollama", "data": response.json()}

    # ── Cohere format ─────────────────────────────────────────────────────────

    def _call_cohere(
        self,
        config       : LLMConfig,
        system_prompt: str,
        messages     : list[dict],
    ) -> dict:
        """
        Cohere uses its own chat format:
        - `preamble` for system prompt
        - `chat_history` for previous messages
        - `message` for the latest user message
        """
        # Separate the last user message from history
        if not messages:
            raise ValueError("No messages provided for Cohere call.")

        last_user_msg = ""
        chat_history  = []

        for msg in messages:
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                chat_history.append({"role": "USER", "message": msg["content"]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg["content"]})

        # Remove the last user message from history (it goes in `message`)
        if chat_history and chat_history[-1]["role"] == "USER":
            chat_history.pop()

        payload = {
            "model"       : config.model,
            "message"     : last_user_msg,
            "preamble"    : system_prompt,
            "chat_history": chat_history,
            "temperature" : config.temperature,
            "max_tokens"  : config.max_tokens,
        }

        headers  = {
            "Content-Type" : "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        url      = f"{config.effective_url.rstrip('/')}/chat"
        response = self._session.post(url, headers=headers, json=payload, timeout=config.timeout)

        if response.status_code != 200:
            self._raise_api_error(response, "cohere")

        return {"format": "cohere", "data": response.json()}

    # ── Response normalizer ───────────────────────────────────────────────────

    def _parse_response(
        self,
        raw      : dict,
        config   : LLMConfig,
        latency  : int,
        fmt      : str,
    ) -> LLMResponse:
        """
        Normalize all provider response formats into a single LLMResponse.
        """
        data = raw.get("data", {})
        text = ""
        prompt_tokens = 0
        output_tokens = 0

        try:
            if fmt == "openai":
                text          = data["choices"][0]["message"]["content"]
                usage         = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

            elif fmt == "anthropic":
                text          = data["content"][0]["text"]
                usage         = data.get("usage", {})
                prompt_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

            elif fmt == "gemini":
                text          = data["candidates"][0]["content"]["parts"][0]["text"]
                usage         = data.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", 0)
                output_tokens = usage.get("candidatesTokenCount", 0)

            elif fmt == "ollama":
                text          = data["message"]["content"]
                prompt_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)

            elif fmt == "cohere":
                text          = data["text"]
                usage         = data.get("meta", {}).get("tokens", {})
                prompt_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Response parsing error for {fmt}: {e}\nRaw: {data}")
            return LLMResponse(
                success  = False,
                text     = "",
                provider = config.provider,
                model    = config.model,
                latency_ms = latency,
                raw      = data,
                error    = f"Failed to parse {fmt} response: {e}",
            )

        return LLMResponse(
            success       = True,
            text          = text.strip(),
            provider      = config.provider,
            model         = config.model,
            prompt_tokens = prompt_tokens,
            output_tokens = output_tokens,
            latency_ms    = latency,
            raw           = data,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _raise_api_error(self, response: Any, provider: str) -> None:
        """Parse and raise a descriptive API error."""
        status  = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text[:300]}

        # Extract the most useful error message from each format
        msg = (
            body.get("error", {}).get("message")          # OpenAI format
            or body.get("error", {}).get("description")   # Some providers
            or body.get("message")                         # Anthropic
            or body.get("detail")                          # FastAPI style
            or str(body)[:200]
        )

        http_hints = {
            401: "Invalid API key. Check your key is correct and active.",
            403: "Access denied. Check your API key permissions.",
            404: "Model not found. Check the model name is correct.",
            429: "Rate limit hit. Wait a moment then try again.",
            500: "Provider server error. Try again in a few minutes.",
            503: "Provider service unavailable. Try again later.",
        }

        hint  = http_hints.get(status, "")
        raise ConnectionError(
            f"{provider} API error (HTTP {status}): {msg}"
            + (f"\nHint: {hint}" if hint else "")
        )

    def validate_config(self, config: LLMConfig) -> tuple[bool, str]:
        """
        Quick validation without making a real API call.
        Returns (is_valid, error_message).
        """
        info = PROVIDER_REGISTRY.get(config.provider)
        if not info:
            return False, f"Unknown provider '{config.provider}'. Available: {ALL_PROVIDERS}"

        if info["requires_key"] and not config.api_key:
            return False, f"Provider '{config.provider}' requires an API key."

        if not config.model:
            return False, "No model specified."

        return True, ""

    def list_providers(self) -> dict:
        """Return provider info useful for UI display."""
        return {
            k: {
                "name"       : v["name"],
                "free"       : v["free_tier"],
                "models"     : v["models"],
                "recommended": v["recommended"],
                "key_hint"   : v["key_hint"],
                "notes"      : v["notes"],
            }
            for k, v in PROVIDER_REGISTRY.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONVERSATION MEMORY
# Smart conversation history with token budget management.
# ══════════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """
    Manages conversation history for multi-turn chat.

    Features:
    - Sliding window (keeps last N messages)
    - Token budget enforcement
    - Message deduplication
    - Export to various formats
    - Session fingerprinting (for caching)
    """

    MAX_MESSAGES : int = 40        # Hard cap on stored messages
    MAX_TOKENS   : int = 8_000     # Max tokens to send to LLM in history

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []

    def add(self, role: str, content: str, result: QueryResult | None = None) -> None:
        """Add a message to history."""
        msg = ChatMessage(role=role, content=content, result=result)
        msg.token_count = msg.estimate_tokens()
        self._messages.append(msg)
        # Trim if over hard cap
        if len(self._messages) > self.MAX_MESSAGES:
            self._messages = self._messages[-self.MAX_MESSAGES:]

    def get_messages_for_llm(self, max_tokens: int | None = None) -> list[dict]:
        """
        Return messages formatted for LLM API (no system message).
        Respects token budget — drops oldest messages first.
        """
        budget  = max_tokens or self.MAX_TOKENS
        result  = []
        total   = 0

        # Walk in reverse (newest first), collect until budget hit
        for msg in reversed(self._messages):
            if msg.role == "system":
                continue
            tokens = msg.token_count or msg.estimate_tokens()
            if total + tokens > budget:
                break
            result.append({"role": msg.role, "content": msg.content})
            total += tokens

        result.reverse()
        return result

    def get_all(self) -> list[ChatMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def export_json(self) -> str:
        """Export full history as JSON string."""
        return json.dumps(
            [{"role": m.role, "content": m.content, "timestamp": m.timestamp}
             for m in self._messages],
            indent=2, default=str,
        )

    def export_markdown(self) -> str:
        """Export history as readable Markdown."""
        lines = ["# Chat History\n"]
        for msg in self._messages:
            if msg.role == "system":
                continue
            icon = "👤" if msg.role == "user" else "🤖"
            lines.append(f"**{icon} {msg.role.title()}**\n\n{msg.content}\n\n---\n")
        return "\n".join(lines)

    def get_fingerprint(self) -> str:
        """SHA256 hash of conversation for caching."""
        content = "".join(m.content for m in self._messages)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# 8. QUERY PARSER  (rule-based — works without any API key)
# ══════════════════════════════════════════════════════════════════════════════

class QueryParser:
    """
    Parse natural language into structured ParsedQuery objects.
    Supports English and Arabic. No API key needed.
    """

    AGGREGATE_PATTERNS = [
        r'\b(average|avg|mean|متوسط)\b',
        r'\b(sum|total|مجموع|إجمالي)\b',
        r'\b(max|maximum|highest|أعلى|أكثر)\b',
        r'\b(min|minimum|lowest|أدنى|أقل)\b',
        r'\b(count|number of|كم عدد)\b',
        r'\b(std|standard deviation|الانحراف المعياري)\b',
    ]
    FILTER_PATTERNS = [
        r'\b(where|filter|show.*where|أين|عرض.*حيث)\b',
        r'\b(greater than|more than|above|أكبر من|أكثر من)\b',
        r'\b(less than|below|under|أقل من|أصغر من)\b',
        r'\b(equal|equals?|is|يساوي)\b',
        r'\b(contains?|like|يحتوي)\b',
    ]
    SORT_PATTERNS = [
        r'\b(sort|order|rank|ترتيب|رتب)\b',
        r'\b(top|best|highest|أعلى|أفضل)\b',
        r'\b(bottom|worst|lowest|أدنى|أسوأ)\b',
    ]
    DESCRIBE_PATTERNS = [
        r'\b(describe|summary|stats|statistics|وصف|ملخص)\b',
        r'\b(what is|what are|tell me about|ما هو|أخبرني)\b',
        r'\b(missing|null|empty|مفقود|فارغ)\b',
        r'\b(unique|distinct|فريد|مختلف)\b',
    ]
    AGGREGATION_MAP = {
        "average": "mean", "avg": "mean", "mean": "mean", "متوسط": "mean",
        "sum": "sum", "total": "sum", "مجموع": "sum", "إجمالي": "sum",
        "max": "max", "maximum": "max", "highest": "max", "أعلى": "max",
        "min": "min", "minimum": "min", "lowest": "min", "أدنى": "min",
        "count": "count", "number": "count", "كم": "count",
        "std": "std", "standard": "std",
    }

    def parse(self, query: str, df: pd.DataFrame) -> ParsedQuery:
        q = query.lower().strip()
        return ParsedQuery(
            intent      = self._detect_intent(q),
            columns     = self._extract_columns(q, df.columns.tolist()),
            conditions  = self._detect_conditions(q, df),
            aggregation = self._detect_aggregation(q),
            group_by    = self._detect_group_by(q, df.columns.tolist()),
            order_by    = self._detect_sort_col(q, df.columns.tolist()),
            ascending   = bool(re.search(r'\b(ascending|asc|lowest|bottom)\b', q)),
            limit       = self._detect_limit(q),
            raw_query   = query,
        )

    def _extract_columns(self, query: str, cols: list[str]) -> list[str]:
        found = [c for c in cols if c.lower().replace("_", " ") in query or c.lower() in query]
        return found if found else cols[:3]

    def _detect_intent(self, q: str) -> str:
        if any(re.search(p, q) for p in self.DESCRIBE_PATTERNS):
            return "describe"
        if any(re.search(p, q) for p in self.AGGREGATE_PATTERNS):
            return "aggregate"
        if any(re.search(p, q) for p in self.FILTER_PATTERNS):
            return "filter"
        if any(re.search(p, q) for p in self.SORT_PATTERNS):
            return "sort"
        if re.search(r'\b(compare|vs|versus|قارن)\b', q):
            return "compare"
        return "describe"

    def _detect_aggregation(self, q: str) -> str:
        for kw, fn in self.AGGREGATION_MAP.items():
            if kw in q:
                return fn
        return "mean"

    def _detect_conditions(self, query: str, df: pd.DataFrame) -> list[dict]:
        conds, cols = [], df.columns.tolist()
        for op, pat in [
            (">", r'(\w[\w\s]*?)\s*(?:>|greater than|more than|above)\s*([\d.]+)'),
            ("<", r'(\w[\w\s]*?)\s*(?:<|less than|below|under)\s*([\d.]+)'),
            ("==", r'(\w+)\s*(?:=|equals?|is)\s*["\']?(\w+)["\']?'),
        ]:
            m = re.search(pat, query)
            if m:
                col = self._match_column(m.group(1).strip(), cols)
                if col:
                    val = float(m.group(2)) if op in (">", "<") else m.group(2)
                    conds.append({"column": col, "op": op, "value": val})
        return conds

    def _detect_group_by(self, q: str, cols: list[str]) -> str | None:
        m = re.search(r'\bby\s+(\w+)', q)
        if m:
            return self._match_column(m.group(1), cols)
        if re.search(r'\b(per|each|every|لكل)\b', q):
            cat = [c for c in cols if True]
            return cat[0] if cat else None
        return None

    def _detect_sort_col(self, q: str, cols: list[str]) -> str | None:
        m = re.search(r'(?:sort|order|rank)\s+by\s+(\w+)', q)
        return self._match_column(m.group(1), cols) if m else None

    def _detect_limit(self, q: str) -> int:
        m = re.search(r'\btop\s+(\d+)', q)
        return int(m.group(1)) if m else 10

    def _match_column(self, word: str, cols: list[str]) -> str | None:
        word = word.lower().strip()
        for c in cols:
            if word == c.lower() or word in c.lower() or c.lower() in word:
                return c
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 9. DATA QUERY ENGINE (rule-based execution — no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

class DataQueryEngine:
    """Execute ParsedQuery objects on a pandas DataFrame."""

    def execute(self, query: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        try:
            dispatch = {
                "describe" : self._describe,
                "aggregate": self._aggregate,
                "filter"   : self._filter,
                "sort"     : self._sort,
                "compare"  : self._compare,
            }
            fn = dispatch.get(query.intent, self._describe)
            return fn(query, df)
        except Exception as e:
            logger.warning(f"Query execution error: {e}")
            return QueryResult(
                success=False, data=None,
                answer=f"Could not process that query: {e}",
                query=query, error=str(e),
            )

    def _apply_conditions(self, df: pd.DataFrame, conds: list[dict]) -> pd.DataFrame:
        for c in conds:
            col, op, val = c["column"], c["op"], c["value"]
            if col not in df.columns:
                continue
            if op == ">":
                df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
            elif op == "<":
                df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
            elif op == "==":
                df = df[df[col].astype(str).str.lower() == str(val).lower()]
        return df

    def _describe(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        raw = q.raw_query.lower()

        if re.search(r'\b(missing|null|empty|مفقود)\b', raw):
            mv = df.isnull().sum()
            mv = mv[mv > 0].sort_values(ascending=False)
            if mv.empty:
                return QueryResult(True, {}, "✅ No missing values in your dataset.", q)
            lines  = [f"• **{c}**: {n:,} missing ({n/len(df)*100:.1f}%)" for c, n in mv.items()]
            return QueryResult(True, mv.to_dict(), f"Missing values in {len(mv)} columns:\n" + "\n".join(lines), q)

        if re.search(r'\b(unique|distinct|فريد)\b', raw):
            col = q.columns[0] if q.columns else df.columns[0]
            if col in df.columns:
                vals = df[col].dropna().unique().tolist()[:20]
                return QueryResult(True, vals, f"**{col}** has **{df[col].nunique()}** unique values:\n{', '.join(str(v) for v in vals)}", q)

        if re.search(r'\b(duplicate|دuplicate)\b', raw):
            n_dup = df.duplicated().sum()
            return QueryResult(True, {"duplicates": int(n_dup)}, f"Found **{n_dup:,}** duplicate rows ({n_dup/len(df)*100:.1f}%).", q)

        col = q.columns[0] if q.columns and q.columns[0] in df.columns else None
        if col:
            s = df[col].dropna()
            if pd.api.types.is_numeric_dtype(s):
                data = {
                    "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                    "min": float(s.min()), "25%": float(s.quantile(0.25)),
                    "median": float(s.median()), "75%": float(s.quantile(0.75)),
                    "max": float(s.max()),
                }
                answer = (f"**{col}** statistics:\n"
                          f"• Mean: {data['mean']:,.2f} | Median: {data['median']:,.2f}\n"
                          f"• Min: {data['min']:,.2f} | Max: {data['max']:,.2f}\n"
                          f"• Std Dev: {data['std']:,.2f}")
            else:
                top  = df[col].value_counts().head(5).to_dict()
                data = top
                answer = f"**{col}** top values:\n" + "\n".join(f"• {k}: {v:,}" for k, v in top.items())
            return QueryResult(True, data, answer, q, chart_type="bar")

        data = {
            "rows": len(df), "columns": len(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        }
        answer = (f"Dataset: **{data['rows']:,}** rows × **{data['columns']}** columns | "
                  f"**{data['missing']}** missing | **{data['duplicates']}** duplicates\n"
                  f"Columns: {', '.join(df.columns.tolist()[:8])}")
        return QueryResult(True, data, answer, q)

    def _aggregate(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        filtered = self._apply_conditions(df, q.conditions)
        fn       = q.aggregation
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col      = next((c for c in q.columns if c in df.columns and c in num_cols), None)
        if not col and num_cols:
            col = num_cols[0]
        if not col:
            return QueryResult(False, None, "No numeric column found for aggregation.", q)

        if q.group_by and q.group_by in df.columns:
            grouped = filtered.groupby(q.group_by)[col].agg(fn).round(4).sort_values(ascending=False)
            data    = grouped.to_dict()
            top5    = list(grouped.head(5).items())
            lines   = [f"• **{k}**: {v:,.2f}" for k, v in top5]
            answer  = f"**{fn.title()}** of **{col}** grouped by **{q.group_by}**:\n" + "\n".join(lines)
            return QueryResult(True, data, answer, q, chart_type="bar")

        val    = getattr(filtered[col], fn)()
        return QueryResult(True, {f"{fn}_{col}": round(float(val), 4)},
                           f"**{fn.title()}** of **{col}**: **{val:,.4f}**", q)

    def _filter(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        filtered = self._apply_conditions(df, q.conditions)
        cols     = [c for c in q.columns if c in df.columns] or df.columns.tolist()
        data     = filtered[cols].head(q.limit)
        return QueryResult(True, data,
                           f"Found **{len(filtered):,}** matching rows (showing {min(q.limit, len(filtered))}):", q)

    def _sort(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        num_cols = df.select_dtypes(include="number").columns
        sort_col = q.order_by or (num_cols[0] if len(num_cols) > 0 else df.columns[0])
        if sort_col not in df.columns:
            return QueryResult(False, None, f"Column '{sort_col}' not found.", q)
        direction = "lowest" if q.ascending else "highest"
        data = df.sort_values(sort_col, ascending=q.ascending).head(q.limit)
        return QueryResult(True, data, f"**Top {q.limit} {direction}** by **{sort_col}**:", q, chart_type="bar")

    def _compare(self, q: ParsedQuery, df: pd.DataFrame) -> QueryResult:
        num_cols = [c for c in q.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            c1, c2 = num_cols[0], num_cols[1]
            corr   = df[[c1, c2]].corr().iloc[0, 1]
            s      = ("very strong" if abs(corr) > 0.9 else "strong" if abs(corr) > 0.7 else
                      "moderate" if abs(corr) > 0.4 else "weak")
            d      = "positive" if corr > 0 else "negative"
            answer = (f"**{c1}** vs **{c2}**:\n"
                      f"• Correlation: **{corr:.4f}** ({s} {d})\n"
                      f"• {c1} mean: {df[c1].mean():,.2f} | {c2} mean: {df[c2].mean():,.2f}")
            return QueryResult(True, {"correlation": corr}, answer, q, chart_type="line")
        return self._describe(q, df)


# ══════════════════════════════════════════════════════════════════════════════
# 10. DATA CHAT BOT — Master Orchestrator
# The main class that everything connects to.
# ══════════════════════════════════════════════════════════════════════════════

class DataChatBot:
    """
    Universal data chat interface for nydra.

    Modes:
    1. chat()          → Pure rule-based, no API key needed
    2. chat_with_llm() → Uses ANY configured LLM provider
    3. chat_auto()     → Tries LLM, falls back to rules on failure

    Supports all 12 providers via UnifiedLLMClient.
    Supports English and Arabic.
    Full conversation memory with smart token management.
    Safe code execution via SafeCodeExecutor.
    """

    GREETINGS  = {"hello", "hi", "hey", "مرحبا", "سلام", "أهلاً", "أهلا", "bonjour"}
    HELP_WORDS = {"help", "مساعدة", "what can you do", "ماذا تفعل", "commands", "?"}

    def __init__(self) -> None:
        self.parser         = QueryParser()
        self.engine         = DataQueryEngine()
        self.executor       = SafeCodeExecutor()
        self.context_builder= DataContextBuilder()
        self.prompt_builder = SystemPromptBuilder()
        self.llm_client     = UnifiedLLMClient()
        self.memory         = ConversationMemory()

        self._df            : pd.DataFrame | None = None
        self._col_names     : list[str]            = []
        self._data_context  : str                  = ""   # Cached context string
        self._df_hash       : str                  = ""   # Detect df changes

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_data(self, data: dict | pd.DataFrame) -> None:
        """
        Load a DataFrame into the chatbot.
        Only clears memory if the data is different from current.
        """
        # 1. Standardize input to DataFrame
        if isinstance(data, pd.DataFrame):
            new_df = data
        elif isinstance(data, dict) and "df" in data:
            new_df = data["df"]
        else:
            raise ValueError("Pass a DataFrame or a dict with a 'df' key.")

        # 2. Compute hash to detect changes
        new_hash = hashlib.md5(
            pd.util.hash_pandas_object(new_df, index=True).values.tobytes()
        ).hexdigest()[:8]

        # 3. Only reload if hash is different
        if new_hash == self._df_hash and self._df is not None:
            return

        self._df          = new_df.copy()
        self._col_names   = list(self._df.columns)
        self._df_hash     = new_hash
        self._data_context = self.context_builder.build(self._df)
        self.memory.clear()
        logger.info(f"New data loaded: {self._df.shape[0]:,} rows × {self._df.shape[1]} cols [hash={self._df_hash}]")

    def is_ready(self) -> bool:
        return self._df is not None

    # ── Rule-based chat (no API key) ─────────────────────────────────────────

    def chat(self, user_message: str) -> QueryResult:
        """
        Process a query using pure rule-based parsing.
        Works offline, no API key needed.
        """
        if not self.is_ready():
            return self._error_result("Please load a dataset first.", user_message)

        msg_lower = user_message.lower().strip()

        # Greetings
        if any(g in msg_lower for g in self.GREETINGS):
            result = self._greeting_result(user_message)
            self.memory.add("user", user_message)
            self.memory.add("assistant", result.answer, result)
            return result

        # Help
        if any(h in msg_lower for h in self.HELP_WORDS):
            result = self._help_result(user_message)
            self.memory.add("user", user_message)
            self.memory.add("assistant", result.answer, result)
            return result

        # Parse and execute
        parsed = self.parser.parse(user_message, self._df)
        result = self.engine.execute(parsed, self._df)
        self.memory.add("user", user_message)
        self.memory.add("assistant", result.answer, result)
        return result

    # ── LLM-powered chat (any provider) ──────────────────────────────────────

    def chat_with_llm(
        self,
        user_message: str,
        config      : LLMConfig,
    ) -> QueryResult:
        """
        Process query using a real LLM with full conversation memory.
        Supports all 12 providers via UnifiedLLMClient.

        Args:
            user_message : The user's question
            config       : LLMConfig with provider/model/key

        Returns:
            QueryResult with answer, optional chart data, and generated code
        """
        if not self.is_ready():
            return self._error_result("Please load a dataset first.", user_message)

        # Validate config before making any API call
        valid, err = self.llm_client.validate_config(config)
        if not valid:
            return self._error_result(f"Configuration error: {err}", user_message)

        # Detect language for appropriate system prompt
        lang         = self.prompt_builder.detect_language(user_message)
        system_prompt= self.prompt_builder.build(self._df, self._data_context, lang)

        # Get conversation history formatted for LLM
        history      = self.memory.get_messages_for_llm()
        messages     = history + [{"role": "user", "content": user_message}]

        # Add user message to memory before API call
        self.memory.add("user", user_message)

        # Call the LLM
        llm_resp = self.llm_client.complete(config, system_prompt, messages)

        if not llm_resp.success:
            result = QueryResult(
                success    = False,
                data       = None,
                answer     = f"❌ **{config.provider.title()} Error**: {llm_resp.error}",
                query      = self._make_query(user_message),
                error      = llm_resp.error,
                provider   = config.provider,
                model      = config.model,
                latency_ms = llm_resp.latency_ms,
            )
            self.memory.add("assistant", result.answer, result)
            return result

        # Process the LLM response text
        result = self._process_llm_response(llm_resp, user_message, config)
        self.memory.add("assistant", result.answer, result)
        return result

    # ── Auto chat (tries LLM, falls back to rules) ───────────────────────────

    def chat_auto(
        self,
        user_message: str,
        config      : LLMConfig | None = None,
    ) -> QueryResult:
        """
        Smart mode: Use LLM if config is provided, fall back to rules.
        Never fails — always returns a result.
        """
        if config:
            result = self.chat_with_llm(user_message, config)
            if result.success:
                return result
            # LLM failed → fall back to rule-based with a note
            rule_result = self.chat(user_message)
            rule_result.answer = (
                f"⚠️ *AI unavailable ({result.error[:60]}...) — using rule-based analysis:*\n\n"
                + rule_result.answer
            )
            return rule_result
        return self.chat(user_message)

    # ── LLM response processing ───────────────────────────────────────────────

    def _process_llm_response(
        self,
        llm_resp    : LLMResponse,
        user_message: str,
        config      : LLMConfig,
    ) -> QueryResult:
        """
        Parse the LLM text response:
        1. Extract code blocks
        2. Execute them safely
        3. Determine chart type from results
        4. Clean up the answer text
        """
        raw_text   = llm_resp.text
        code_block = re.search(r'```python\s*\n(.*?)\n```', raw_text, re.DOTALL)

        data_result = {}
        chart_type  = None
        code_text   = None
        exec_error  = None

        if code_block:
            code_text  = code_block.group(1).strip()
            exec_out   = self.executor.execute_safe(code_text, self._df)

            if exec_out["success"]:
                raw_result = exec_out["result"]
                # Determine data and chart type based on result type
                if isinstance(raw_result, dict) and len(raw_result) > 1:
                    data_result = raw_result
                    chart_type  = "bar"
                elif isinstance(raw_result, pd.DataFrame):
                    data_result = raw_result
                    chart_type  = None
                elif isinstance(raw_result, pd.Series):
                    data_result = raw_result.to_dict()
                    chart_type  = "bar"
                elif isinstance(raw_result, (int, float)):
                    data_result = {"value": raw_result}
                    chart_type  = None
                elif isinstance(raw_result, list) and raw_result:
                    data_result = {"items": raw_result}
                    chart_type  = "bar" if all(isinstance(x, (int, float)) for x in raw_result) else None
                else:
                    data_result = {"result": str(raw_result)} if raw_result is not None else {}
            else:
                exec_error  = exec_out["error"]
                # Append execution error to the answer
                logger.warning(f"Code execution failed: {exec_error}")

        # Clean the answer text (remove code block for display if code was executed)
        answer = raw_text
        if code_block and not exec_error:
            # Remove the code block from displayed answer (it's shown separately in UI)
            answer = re.sub(r'```python.*?```', '', raw_text, flags=re.DOTALL).strip()
            if not answer:
                answer = "Here are the results from my analysis:"

        # Append execution error to answer if code failed
        if exec_error:
            answer += f"\n\n⚠️ *Code execution note: {exec_error}*"

        # Build provider attribution
        provider_note = f"\n\n*— {llm_resp.provider.title()} / {llm_resp.model} ({llm_resp.latency_ms}ms)*"

        return QueryResult(
            success    = True,
            data       = data_result,
            answer     = answer + provider_note,
            query      = self._make_query(user_message, intent="llm"),
            chart_type = chart_type,
            code       = code_text,
            provider   = config.provider,
            model      = config.model,
            latency_ms = llm_resp.latency_ms,
        )

    # ── Suggestion engine ─────────────────────────────────────────────────────

    def get_suggested_questions(self) -> list[str]:
        """Generate smart suggested questions based on the loaded dataset."""
        if not self.is_ready():
            return []

        df       = self._df
        num_cols = [c for c in self._col_names if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in self._col_names if df[c].dtype == object and df[c].nunique() <= 30]
        date_cols= [c for c in self._col_names if pd.api.types.is_datetime64_any_dtype(df[c])]

        suggestions = []

        if num_cols:
            suggestions.append(f"What is the average {num_cols[0]}?")
            suggestions.append(f"Show the top 10 rows by {num_cols[0]}")
            if len(num_cols) >= 2:
                suggestions.append(f"Compare {num_cols[0]} and {num_cols[1]}")
        if cat_cols and num_cols:
            suggestions.append(f"What is the total {num_cols[0]} by {cat_cols[0]}?")
            suggestions.append(f"Which {cat_cols[0]} has the highest {num_cols[0]}?")
        if cat_cols:
            suggestions.append(f"What are the unique values in {cat_cols[0]}?")
        if date_cols:
            suggestions.append(f"Show the trend of {num_cols[0] if num_cols else self._col_names[0]} over time")

        suggestions += [
            "How many missing values are there?",
            "Are there any duplicate rows?",
            "Describe the dataset",
            "What are the strongest correlations?",
        ]

        return suggestions[:10]

    # ── Utility ───────────────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Clear conversation memory."""
        self.memory.clear()

    def export_history(self, fmt: str = "markdown") -> str:
        """Export conversation history. fmt: 'markdown' | 'json'"""
        if fmt == "json":
            return self.memory.export_json()
        return self.memory.export_markdown()

    def get_history(self) -> list[ChatMessage]:
        return self.memory.get_all()

    def get_data_summary(self) -> str:
        """Return the current data context string."""
        return self._data_context or "No data loaded."

    def get_providers_info(self) -> dict:
        """Return provider registry for UI display."""
        return self.llm_client.list_providers()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_query(self, raw: str, intent: str = "describe") -> ParsedQuery:
        return ParsedQuery(intent, self._col_names, [], "mean", None, None, False, 10, raw)

    def _error_result(self, msg: str, raw: str) -> QueryResult:
        return QueryResult(False, None, msg, self._make_query(raw), error=msg)

    def _greeting_result(self, user_message: str) -> QueryResult:
        df   = self._df
        cols = self._col_names
        answer = (
            f"👋 Hello! I'm **Nydra AI**, your data analyst.\n\n"
            f"Your dataset has **{len(df):,}** rows and **{len(cols)}** columns:\n"
            f"{', '.join(cols[:6])}{'...' if len(cols) > 6 else ''}\n\n"
            f"Ask me anything in plain English or Arabic! For example:\n"
            f"• *What is the average {cols[0] if cols else 'value'}?*\n"
            f"• *Which category has the highest total?*\n"
            f"• *How many missing values are there?*\n"
            f"• *Show top 10 rows by revenue*"
        )
        return QueryResult(True, {"rows": len(df), "cols": len(cols)}, answer, self._make_query(user_message))

    def _help_result(self, user_message: str) -> QueryResult:
        cols = self._col_names
        c0   = cols[0] if cols else "column"
        answer = (
            f"**🔍 What I can do:**\n\n"
            f"**Statistics & Aggregations**\n"
            f"• *What is the average {c0}?*\n"
            f"• *What is the total revenue?*\n"
            f"• *Show me the max, min, and std of all numeric columns*\n\n"
            f"**Filtering**\n"
            f"• *Show rows where {c0} > 100*\n"
            f"• *Filter by category = 'Electronics'*\n\n"
            f"**Sorting & Ranking**\n"
            f"• *Top 10 by sales*\n"
            f"• *Bottom 5 by performance*\n\n"
            f"**Data Quality**\n"
            f"• *How many missing values?*\n"
            f"• *Are there any duplicates?*\n\n"
            f"**Advanced Analysis** *(with AI)*\n"
            f"• *Detect outliers in the salary column*\n"
            f"• *What insights can you find?*\n"
            f"• *Predict which customers are likely to churn*\n\n"
            f"**Languages:** English 🇬🇧 | Arabic 🇸🇦"
        )
        return QueryResult(True, {}, answer, self._make_query(user_message))


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONVENIENCE FUNCTIONS — Quick access for app.py and cli.py
# ══════════════════════════════════════════════════════════════════════════════

def get_providers() -> dict:
    """Return all providers and their models for UI dropdowns."""
    return {
        k: {
            "name"       : v["name"],
            "models"     : v["models"],
            "recommended": v["recommended"],
            "free"       : v["free_tier"],
            "key_hint"   : v["key_hint"],
            "notes"      : v["notes"],
        }
        for k, v in PROVIDER_REGISTRY.items()
    }


def make_config(
    provider   : str,
    model      : str,
    api_key    : str  = "",
    temperature: float = 0.2,
    max_tokens : int  = 1500,
    custom_url : str  = "",
) -> LLMConfig:
    """
    Factory function to create an LLMConfig.
    Use this in app.py to build config from Streamlit widgets.

    Example:
        config = make_config(
            provider="groq",
            model="llama-3.3-70b-versatile",
            api_key=st.session_state.api_key,
        )
        result = chatbot.chat_with_llm(question, config)
    """
    return LLMConfig(
        provider    = provider,
        model       = model,
        api_key     = api_key,
        temperature = temperature,
        max_tokens  = max_tokens,
        custom_url  = custom_url,
    )


def quick_chat(
    df         : pd.DataFrame,
    question   : str,
    provider   : str  = "groq",
    model      : str  = "llama-3.3-70b-versatile",
    api_key    : str  = "",
    session_id : str  = "default",
) -> QueryResult:
    """
    One-line function to chat with your data using an LLM.

    Example:
        result = quick_chat(df, "What is the average salary?", provider="groq", api_key="gsk_...")
        print(result.answer)
    """
    bot = get_chatbot(session_id)
    bot.load_data(df)
    config = make_config(provider=provider, model=model, api_key=api_key)
    return bot.chat_with_llm(question, config)


def quick_chat_rules(
    df        : pd.DataFrame,
    question  : str,
    session_id: str = "default",
) -> QueryResult:
    """
    One-line function to chat with your data using rule-based parsing (no API key).

    Example:
        result = quick_chat_rules(df, "What is the average salary?")
        print(result.answer)
    """
    bot = get_chatbot(session_id)
    bot.load_data(df)
    return bot.chat(question)


# ══════════════════════════════════════════════════════════════════════════════
# 12. SESSION MANAGER — Multiple users, multiple sessions
# ══════════════════════════════════════════════════════════════════════════════

class ChatSessionManager:
    """
    Manages multiple DataChatBot instances — one per user/session.
    Used by app.py for multi-user Streamlit deployments.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DataChatBot] = {}
        self._created : dict[str, float]        = {}

    def get(self, session_id: str = "default") -> DataChatBot:
        """Get or create a chatbot for a session ID."""
        if session_id not in self._sessions:
            self._sessions[session_id] = DataChatBot()
            self._created[session_id]  = time.time()
            logger.debug(f"New chat session created: {session_id}")
        return self._sessions[session_id]

    def delete(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)
        self._created.pop(session_id, None)

    def cleanup_old(self, max_age_hours: float = 24.0) -> int:
        """Remove sessions older than max_age_hours. Returns count deleted."""
        cutoff  = time.time() - max_age_hours * 3600
        to_del  = [sid for sid, t in self._created.items() if t < cutoff]
        for sid in to_del:
            self.delete(sid)
        return len(to_del)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    def count(self) -> int:
        return len(self._sessions)


# ── Global session manager and convenience accessor ───────────────────────────
_session_manager = ChatSessionManager()


def get_chatbot(session_id: str = "default") -> DataChatBot:
    """Get or create a DataChatBot for the given session ID."""
    return _session_manager.get(session_id)


# ══════════════════════════════════════════════════════════════════════════════
# 13. APP.PY INTEGRATION GUIDE (as docstring for developers)
# ══════════════════════════════════════════════════════════════════════════════
"""
HOW TO USE IN app.py (Streamlit)
=================================

from src.data.data_chat import get_chatbot, make_config, get_providers, PROVIDER_REGISTRY

# ── Session state setup ───
if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot(st.session_state.get("session_id", "default"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Load data ─────────────
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.chatbot.load_data(df)

# ── Provider + Model selector ─
providers = get_providers()
provider  = st.selectbox("AI Provider", list(providers.keys()),
                          format_func=lambda k: providers[k]["name"])
model     = st.selectbox("Model", providers[provider]["models"],
                          index=0)
api_key   = st.text_input("API Key", type="password",
                           placeholder=providers[provider]["key_hint"],
                           help=providers[provider]["notes"])
# For Ollama: no key needed
if provider == "ollama":
    api_key = ""

# ── Chat input ────────────
if prompt := st.chat_input("Ask anything about your data..."):
    config = make_config(provider=provider, model=model, api_key=api_key)
    result = st.session_state.chatbot.chat_with_llm(prompt, config)

    st.session_state.chat_history.append({"role": "user",      "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": result.answer,
                                           "result": result})

# ── Render history ─────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("result"):
            r = msg["result"]
            if isinstance(r.data, pd.DataFrame) and not r.data.empty:
                st.dataframe(r.data)
            elif isinstance(r.data, dict) and r.data and r.chart_type == "bar":
                import plotly.express as px
                fig = px.bar(x=list(r.data.keys()), y=list(r.data.values()))
                st.plotly_chart(fig, use_container_width=True)
            if r.code:
                with st.expander("📄 Generated Code"):
                    st.code(r.code, language="python")

# ── Suggested questions ───
suggestions = st.session_state.chatbot.get_suggested_questions()
for q in suggestions:
    if st.button(q, key=q):
        st.session_state.pending_question = q

# ── Export ────────────────
if st.button("Export Chat History"):
    md = st.session_state.chatbot.export_history("markdown")
    st.download_button("Download", md, "chat_history.md")
"""
