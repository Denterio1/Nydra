"""
ml_readiness.py — Auto ML Readiness Score.

Analyses a dataset and returns a score (0-100) with detailed
feedback on what needs to be fixed before training an ML model.
"""

from __future__ import annotations
import pandas as pd
from typing import Any


# ── Weights (must sum to 100) ─────────────────────────────────────────────────
W_SIZE      = 20   # enough rows
W_MISSING   = 25   # missing values
W_OUTLIERS  = 15   # outliers
W_DUPLICATES= 10   # duplicate rows
W_USELESS   = 15   # useless columns (constant / all-unique IDs)
W_TYPES     = 15   # data types need encoding/scaling


def ml_readiness(data: dict[str, Any], outliers: dict) -> dict[str, Any]:
    """
    Compute an ML Readiness Score for the dataset.

    Args:
        data:     Structured data dict from loader.
        outliers: Output from analyzer.detect_outliers().

    Returns:
        {
            "score":    int (0-100),
            "grade":    str ("A" / "B" / "C" / "D" / "F"),
            "checks":   list of check dicts,
            "summary":  str,
        }
    """
    df     = data["df"] if "df" in data else __import__("pandas").DataFrame(data.get("rows", []))
    checks = []
    total  = 0

    # ── 1. Dataset size ───────────────────────────────────────────────────────
    n_rows = len(df)
    if n_rows >= 1000:
        pts, status, msg = W_SIZE, "pass", f"{n_rows:,} rows — good for most ML models."
    elif n_rows >= 200:
        pts, status, msg = int(W_SIZE * 0.7), "warn", f"{n_rows} rows — acceptable but more data is better."
    elif n_rows >= 50:
        pts, status, msg = int(W_SIZE * 0.4), "warn", f"{n_rows} rows — very small dataset, results may be unreliable."
    else:
        pts, status, msg = 0, "fail", f"{n_rows} rows — too few rows for reliable ML training."

    checks.append({"name": "Dataset size", "status": status, "points": pts, "max": W_SIZE, "detail": msg})
    total += pts

    # ── 2. Missing values ─────────────────────────────────────────────────────
    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct == 0:
        pts, status, msg = W_MISSING, "pass", "No missing values — perfect."
    elif missing_pct <= 5:
        pts, status, msg = int(W_MISSING * 0.85), "pass", f"{missing_pct:.1f}% missing — low, easy to handle."
    elif missing_pct <= 15:
        pts, status, msg = int(W_MISSING * 0.6), "warn", f"{missing_pct:.1f}% missing — moderate, needs imputation strategy."
    elif missing_pct <= 30:
        pts, status, msg = int(W_MISSING * 0.3), "warn", f"{missing_pct:.1f}% missing — high, consider dropping affected columns."
    else:
        pts, status, msg = 0, "fail", f"{missing_pct:.1f}% missing — critical, data collection may be flawed."

    checks.append({"name": "Missing values", "status": status, "points": pts, "max": W_MISSING, "detail": msg})
    total += pts

    # ── 3. Outliers ───────────────────────────────────────────────────────────
    # Handle both new OutlierReport object and old dictionary format
    if hasattr(outliers, "method_results"):
        n_outlier_cols = len(set(r.column for r in outliers.method_results if r.column and r.outlier_count > 0))
    else:
        n_outlier_cols = len(outliers)
        
    n_numeric      = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))

    if n_outlier_cols == 0:
        pts, status, msg = W_OUTLIERS, "pass", "No outliers detected in numeric columns."
    elif n_numeric > 0 and n_outlier_cols / n_numeric <= 0.25:
        pts, status, msg = int(W_OUTLIERS * 0.7), "warn", f"{n_outlier_cols} column(s) have outliers — consider capping or log-transform."
    elif n_numeric > 0 and n_outlier_cols / n_numeric <= 0.5:
        pts, status, msg = int(W_OUTLIERS * 0.4), "warn", f"{n_outlier_cols} column(s) have outliers — significant, review before training."
    else:
        pts, status, msg = int(W_OUTLIERS * 0.1), "fail", f"{n_outlier_cols} column(s) have outliers — severe, will distort model."

    checks.append({"name": "Outliers", "status": status, "points": pts, "max": W_OUTLIERS, "detail": msg})
    total += pts

    # ── 4. Duplicate rows ─────────────────────────────────────────────────────
    dupe_pct = df.duplicated().sum() / max(len(df), 1) * 100
    if dupe_pct == 0:
        pts, status, msg = W_DUPLICATES, "pass", "No duplicate rows."
    elif dupe_pct <= 5:
        pts, status, msg = int(W_DUPLICATES * 0.7), "warn", f"{dupe_pct:.1f}% duplicate rows — remove before training."
    else:
        pts, status, msg = 0, "fail", f"{dupe_pct:.1f}% duplicate rows — data leakage risk in train/test split."

    checks.append({"name": "Duplicate rows", "status": status, "points": pts, "max": W_DUPLICATES, "detail": msg})
    total += pts

    # ── 5. Useless columns ────────────────────────────────────────────────────
    useless = []
    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique <= 1:
            useless.append(f"'{col}' (constant)")
        elif n_unique == len(df) and not pd.api.types.is_numeric_dtype(df[col]):
            useless.append(f"'{col}' (all unique — likely an ID)")

    if not useless:
        pts, status, msg = W_USELESS, "pass", "No constant or ID-only columns detected."
    elif len(useless) == 1:
        pts, status, msg = int(W_USELESS * 0.5), "warn", f"1 useless column: {useless[0]} — drop it."
    else:
        pts, status, msg = 0, "fail", f"{len(useless)} useless columns: {', '.join(useless)} — drop them."

    checks.append({"name": "Useless columns", "status": status, "points": pts, "max": W_USELESS, "detail": msg})
    total += pts

    # ── 6. Data types ─────────────────────────────────────────────────────────
    text_cols    = [c for c in df.columns if df[c].dtype == object]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    needs_encode = len(text_cols)
    needs_scale  = len(numeric_cols)

    issues = []
    if needs_encode:
        issues.append(f"{needs_encode} text column(s) need encoding (Label/OneHot)")
    if needs_scale:
        issues.append(f"{needs_scale} numeric column(s) may need scaling (StandardScaler/MinMax)")

    if not issues:
        pts, status, msg = W_TYPES, "pass", "All columns are numeric — ready for most ML models."
    elif needs_encode == 0:
        pts, status, msg = int(W_TYPES * 0.8), "pass", " | ".join(issues)
    elif needs_encode <= 2:
        pts, status, msg = int(W_TYPES * 0.6), "warn", " | ".join(issues)
    else:
        pts, status, msg = int(W_TYPES * 0.3), "warn", " | ".join(issues)

    checks.append({"name": "Data types", "status": status, "points": pts, "max": W_TYPES, "detail": msg})
    total += pts

    # ── Grade ─────────────────────────────────────────────────────────────────
    grade = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else "D" if total >= 40 else "F"

    # ── Summary ───────────────────────────────────────────────────────────────
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]

    if not fails and not warns:
        summary = "Data is in excellent shape — ready for ML training."
    elif not fails:
        summary = f"Data is mostly ready. Fix {len(warns)} warning(s) for best results."
    else:
        summary = f"Data needs work. Fix {len(fails)} critical issue(s) before training."

    return {
        "score":   total,
        "grade":   grade,
        "checks":  checks,
        "summary": summary,
    }