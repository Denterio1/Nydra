"""
encoding_advisor.py — Smart Encoding Advisor for Nydra v0.2.0

Analyzes categorical columns and recommends the best encoding strategy
based on cardinality, distribution, target correlation, and ML context.

Encoding strategies covered:
    - One-Hot Encoding      : low cardinality, no ordinal relationship
    - Label Encoding        : medium cardinality or tree-based models
    - Ordinal Encoding      : when natural order is detected
    - Target Encoding       : high cardinality with numeric target
    - Binary Encoding       : medium-high cardinality
    - Hash Encoding         : very high cardinality
    - Frequency Encoding    : when frequency itself carries signal
    - Leave as-is           : already numeric or boolean
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import numpy as np


# ── Thresholds ────────────────────────────────────────────────────────────────

LOW_CARD_THRESHOLD    = 10    # <= 10 unique  → One-Hot
MEDIUM_CARD_THRESHOLD = 50    # <= 50 unique  → Label / Binary
HIGH_CARD_THRESHOLD   = 200   # <= 200 unique → Target / Frequency
# > 200 unique → Hash Encoding

# Common ordinal patterns (order matters)
ORDINAL_PATTERNS: list[list[str]] = [
    ["low", "medium", "high"],
    ["small", "medium", "large", "xlarge"],
    ["none", "low", "medium", "high", "critical"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["very bad", "bad", "neutral", "good", "very good"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["bronze", "silver", "gold", "platinum"],
    ["junior", "mid", "senior", "lead", "principal"],
    ["beginner", "intermediate", "advanced", "expert"],
    ["daily", "weekly", "monthly", "quarterly", "yearly"],
    ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"],
    ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
    ["q1", "q2", "q3", "q4"],
    ["xs", "s", "m", "l", "xl", "xxl"],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cardinality_label(n_unique: int, n_rows: int) -> str:
    ratio = n_unique / max(n_rows, 1)
    if n_unique <= LOW_CARD_THRESHOLD:
        return "low"
    if n_unique <= MEDIUM_CARD_THRESHOLD:
        return "medium"
    if n_unique <= HIGH_CARD_THRESHOLD:
        return "high"
    return "very_high"


def _detect_ordinal(values: list[str]) -> list[str] | None:
    """Return the matching ordinal scale if the column values fit one."""
    cleaned = [str(v).strip().lower() for v in values if pd.notna(v)]
    value_set = set(cleaned)
    for pattern in ORDINAL_PATTERNS:
        pattern_set = set(pattern)
        if value_set <= pattern_set and len(value_set) >= 2:
            # Return only the levels that appear, in order
            return [p for p in pattern if p in value_set]
    return None


def _is_boolean_like(series: pd.Series) -> bool:
    vals = set(series.dropna().astype(str).str.lower().unique())
    bool_pairs = [
        {"true", "false"}, {"yes", "no"}, {"1", "0"},
        {"y", "n"}, {"t", "f"}, {"on", "off"},
    ]
    return any(vals <= bp for bp in bool_pairs)


def _distribution_entropy(series: pd.Series) -> float:
    """Shannon entropy of the value distribution — higher = more uniform."""
    counts = series.value_counts(normalize=True)
    return float(-sum(p * math.log2(p) for p in counts if p > 0))


def _imbalance_ratio(series: pd.Series) -> float:
    """Ratio of most-common to least-common value count."""
    counts = series.value_counts()
    if len(counts) < 2:
        return 1.0
    return float(counts.iloc[0] / counts.iloc[-1])


def _target_correlation(series: pd.Series, target: pd.Series | None) -> float | None:
    """Mean target value variance across categories (eta-squared proxy)."""
    if target is None or not pd.api.types.is_numeric_dtype(target):
        return None
    combined = pd.DataFrame({"cat": series, "target": target}).dropna()
    if combined.empty:
        return None
    group_means = combined.groupby("cat")["target"].mean()
    overall_mean = combined["target"].mean()
    ss_between = sum(
        (combined["cat"] == cat).sum() * (mean - overall_mean) ** 2
        for cat, mean in group_means.items()
    )
    ss_total = ((combined["target"] - overall_mean) ** 2).sum()
    return round(float(ss_between / ss_total), 4) if ss_total > 0 else 0.0


def _high_cardinality_risk(n_unique: int, n_rows: int) -> str:
    ratio = n_unique / max(n_rows, 1)
    if ratio > 0.9:
        return "ID-like column — likely not useful for ML"
    if ratio > 0.5:
        return "Very sparse — consider dropping or hashing"
    return "Manageable with target or frequency encoding"


# ── Core advisor ──────────────────────────────────────────────────────────────

def _advise_column(
    col: str,
    series: pd.Series,
    n_rows: int,
    target: pd.Series | None,
    model_type: str,
) -> dict[str, Any]:
    """Produce a full encoding recommendation for one column."""

    series = series.copy()
    n_missing  = int(series.isna().sum())
    n_unique   = int(series.nunique())
    card_label = _cardinality_label(n_unique, n_rows)
    entropy    = round(_distribution_entropy(series.dropna()), 4)
    imbalance  = round(_imbalance_ratio(series.dropna()), 2)
    top_values = series.value_counts().head(5).to_dict()
    ordinal_scale  = _detect_ordinal(series.dropna().tolist())
    is_bool        = _is_boolean_like(series)
    target_corr    = _target_correlation(series, target)
    high_card_note = _high_cardinality_risk(n_unique, n_rows) if card_label in ("high", "very_high") else None

    # ── Decision logic ────────────────────────────────────────────────────────
    if is_bool:
        strategy    = "Binary Encoding"
        reason      = "Column has boolean-like values (yes/no, true/false)."
        code        = f"df['{col}'] = df['{col}'].map({{v: i for i, v in enumerate(df['{col}'].dropna().unique())}})"
        sklearn_tip = "Use OrdinalEncoder with 2 categories."
        risk        = "low"

    elif ordinal_scale is not None:
        strategy    = "Ordinal Encoding"
        order_str   = " < ".join(ordinal_scale)
        reason      = f"Natural order detected: {order_str}"
        mapping     = {v: i for i, v in enumerate(ordinal_scale)}
        code        = f"df['{col}'] = df['{col}'].str.lower().map({mapping})"
        sklearn_tip = "Use OrdinalEncoder with categories=[" + str(ordinal_scale) + "]."
        risk        = "low"

    elif n_unique <= LOW_CARD_THRESHOLD:
        if model_type in ("linear", "neural"):
            strategy = "One-Hot Encoding"
            code     = f"df = pd.get_dummies(df, columns=['{col}'], drop_first=True)"
            sklearn_tip = "Use OneHotEncoder(drop='first', sparse_output=False)."
        else:
            strategy = "One-Hot Encoding"
            code     = f"df = pd.get_dummies(df, columns=['{col}'])"
            sklearn_tip = "Use OneHotEncoder(sparse_output=False)."
        reason = f"Low cardinality ({n_unique} unique values) — safe for one-hot."
        risk   = "low"

    elif n_unique <= MEDIUM_CARD_THRESHOLD:
        if model_type in ("tree", "boosting"):
            strategy    = "Label Encoding"
            code        = f"df['{col}'] = df['{col}'].astype('category').cat.codes"
            sklearn_tip = "Use LabelEncoder or OrdinalEncoder."
            reason      = f"Medium cardinality ({n_unique}) — label encoding fine for tree models."
            risk        = "low"
        else:
            strategy    = "Binary Encoding"
            code        = f"# pip install category_encoders\nimport category_encoders as ce\nenc = ce.BinaryEncoder(cols=['{col}'])\ndf = enc.fit_transform(df)"
            sklearn_tip = "Use category_encoders.BinaryEncoder."
            reason      = f"Medium cardinality ({n_unique}) — binary encoding reduces dimensionality vs one-hot."
            risk        = "medium"

    elif n_unique <= HIGH_CARD_THRESHOLD:
        if target_corr is not None and target_corr > 0.05:
            strategy    = "Target Encoding"
            code        = f"# pip install category_encoders\nimport category_encoders as ce\nenc = ce.TargetEncoder(cols=['{col}'])\ndf['{col}'] = enc.fit_transform(df['{col}'], target)"
            sklearn_tip = "Use category_encoders.TargetEncoder — apply on train only, transform test."
            reason      = f"High cardinality ({n_unique}) with target correlation {target_corr} — target encoding captures signal."
            risk        = "medium"
        else:
            strategy    = "Frequency Encoding"
            code        = f"freq = df['{col}'].value_counts() / len(df)\ndf['{col}_freq'] = df['{col}'].map(freq)"
            sklearn_tip = "No sklearn equivalent — compute manually; always fit on train set."
            reason      = f"High cardinality ({n_unique}) — frequency encodes how often each value appears."
            risk        = "medium"

    else:
        strategy    = "Hash Encoding"
        code        = f"# pip install category_encoders\nimport category_encoders as ce\nenc = ce.HashingEncoder(cols=['{col}'], n_components=16)\ndf = enc.fit_transform(df)"
        sklearn_tip = "Use category_encoders.HashingEncoder with n_components tuned to dataset size."
        reason      = f"Very high cardinality ({n_unique}) — hashing keeps dimensionality fixed."
        risk        = "high"

    # Imbalance warning
    warnings = []
    if imbalance > 20:
        warnings.append(f"Severe imbalance: dominant class appears {imbalance:.0f}× more than rarest. Consider grouping rare values into 'Other'.")
    if n_missing > 0:
        pct = round(n_missing / n_rows * 100, 1)
        warnings.append(f"{n_missing} missing values ({pct}%) — fill before encoding.")
    if high_card_note:
        warnings.append(high_card_note)

    return {
        "column":         col,
        "n_unique":       n_unique,
        "cardinality":    card_label,
        "is_ordinal":     ordinal_scale is not None,
        "ordinal_scale":  ordinal_scale,
        "is_boolean":     is_bool,
        "entropy":        entropy,
        "imbalance":      imbalance,
        "target_corr":    target_corr,
        "top_values":     top_values,
        "strategy":       strategy,
        "reason":         reason,
        "code":           code,
        "sklearn_tip":    sklearn_tip,
        "risk":           risk,
        "warnings":       warnings,
    }


def encoding_advisor(
    data: dict[str, Any],
    target_col: str | None = None,
    model_type: str = "tree",
) -> dict[str, Any]:
    """
    Analyse all categorical columns and return encoding recommendations.

    Args:
        data:       Nydra data dict (must have 'df' key).
        target_col: optional name of the target/label column.
        model_type: one of 'tree', 'boosting', 'linear', 'neural'.
                    Affects strategy choice (e.g. linear models need one-hot).

    Returns:
        {
          'model_type':    str,
          'target_col':    str | None,
          'columns':       list[dict],   # one per categorical column
          'summary':       str,
          'pipeline_code': str,          # ready-to-paste sklearn pipeline snippet
        }
    """
    df        = data["df"]
    n_rows    = len(df)
    target    = df[target_col] if target_col and target_col in df.columns else None

    results: list[dict] = []

    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # skip numeric columns
        results.append(
            _advise_column(col, df[col], n_rows, target, model_type)
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    strategy_counts: dict[str, int] = {}
    for r in results:
        strategy_counts[r["strategy"]] = strategy_counts.get(r["strategy"], 0) + 1

    summary_parts = [f"{v}× {k}" for k, v in sorted(strategy_counts.items())]
    summary = (
        f"Found {len(results)} categorical column(s). "
        f"Recommended: {', '.join(summary_parts)}."
        if results else "No categorical columns found."
    )

    # ── Pipeline code snippet ─────────────────────────────────────────────────
    ohe_cols   = [r["column"] for r in results if r["strategy"] == "One-Hot Encoding"]
    label_cols = [r["column"] for r in results if r["strategy"] == "Label Encoding"]
    ord_cols   = [(r["column"], r["ordinal_scale"]) for r in results if r["strategy"] == "Ordinal Encoding"]
    bin_cols   = [r["column"] for r in results if r["strategy"] == "Binary Encoding"]
    freq_cols  = [r["column"] for r in results if r["strategy"] == "Frequency Encoding"]
    tgt_cols   = [r["column"] for r in results if r["strategy"] == "Target Encoding"]
    hash_cols  = [r["column"] for r in results if r["strategy"] == "Hash Encoding"]

    pipeline_lines = [
        "from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder",
        "from sklearn.compose import ColumnTransformer",
        "from sklearn.pipeline import Pipeline",
        "",
    ]

    if ohe_cols:
        pipeline_lines.append(f"ohe_cols   = {ohe_cols}")
    if label_cols:
        pipeline_lines.append(f"label_cols = {label_cols}")
    if ord_cols:
        for col, scale in ord_cols:
            pipeline_lines.append(f"# Ordinal: {col} → {scale}")

    if ohe_cols or label_cols:
        pipeline_lines += [
            "",
            "ct = ColumnTransformer(transformers=[",
        ]
        if ohe_cols:
            pipeline_lines.append(f"    ('ohe',   OneHotEncoder(drop='first', sparse_output=False), ohe_cols),")
        if label_cols:
            pipeline_lines.append(f"    ('label', OrdinalEncoder(), label_cols),")
        pipeline_lines.append("], remainder='passthrough')")

    if freq_cols:
        pipeline_lines += ["", "# Frequency encoding (apply before ColumnTransformer):"]
        for col in freq_cols:
            pipeline_lines.append(f"freq_{col} = df['{col}'].value_counts() / len(df)")
            pipeline_lines.append(f"df['{col}'] = df['{col}'].map(freq_{col})")

    pipeline_code = "\n".join(pipeline_lines)

    return {
        "model_type":    model_type,
        "target_col":    target_col,
        "columns":       results,
        "summary":       summary,
        "pipeline_code": pipeline_code,
    }
