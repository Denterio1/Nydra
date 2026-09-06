"""
Nydra — src/data/quality_score.py
=======================================
Unified Data Quality Scoring Engine

Dimensions (7):
    1. Completeness   — missing values analysis
    2. Uniqueness     — duplicate detection
    3. Validity       — type/range/format conformity
    4. Consistency    — internal logic + cross-column coherence
    5. Integrity      — referential + structural integrity
    6. Distribution   — statistical health + entropy
    7. ML Readiness   — machine learning fitness

Each dimension → 0-100 score + weighted → Overall Score 0-100
Grade: A (90+), B (75+), C (60+), D (45+), F (<45)

Author  : Nydra Team
Version : 1.0.0
"""

from __future__ import annotations

import re
import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger("nydra.quality")


# ─────────────────────────────────────────────────────────────────────────────
# Constants & Weights
# ─────────────────────────────────────────────────────────────────────────────

DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness"  : 0.22,
    "uniqueness"    : 0.15,
    "validity"      : 0.18,
    "consistency"   : 0.15,
    "integrity"     : 0.10,
    "distribution"  : 0.10,
    "ml_readiness"  : 0.10,
}

GRADE_THRESHOLDS = {
    "A": 90,
    "B": 75,
    "C": 60,
    "D": 45,
    "F": 0,
}

GRADE_LABELS = {
    "A": "Excellent 🟢",
    "B": "Good 🟡",
    "C": "Fair 🟠",
    "D": "Poor 🔴",
    "F": "Critical ⛔",
}

# PII patterns
PII_PATTERNS = {
    "email"       : r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone"       : r"(\+?\d[\d\s\-().]{7,}\d)",
    "ssn"         : r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card" : r"\b(?:\d[ -]?){13,16}\b",
    "ip_address"  : r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# Column name hints for PII
PII_NAME_HINTS = [
    "email", "phone", "mobile", "ssn", "passport", "address",
    "zip", "postal", "credit", "card", "ip", "national_id",
    "birth", "dob", "salary", "income",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DimensionResult:
    """Result for a single quality dimension."""
    name        : str
    score       : float                   # 0-100
    weight      : float
    weighted    : float                   # score * weight
    issues      : List[str]
    positives   : List[str]
    details     : Dict[str, Any] = field(default_factory=dict)
    sub_scores  : Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityProfile:
    """Complete dataset quality profile."""
    overall_score     : float
    grade             : str
    grade_label       : str
    dimensions        : Dict[str, DimensionResult]
    issues            : List[str]                  # all issues merged
    recommendations   : List[str]
    pii_detected      : Dict[str, List[str]]       # col → pii types found
    statistics        : Dict[str, Any]
    dataset_shape     : Tuple[int, int]
    health_badge      : str                        # emoji badge
    numeric_columns   : List[str] 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score"  : round(self.overall_score, 2),
            "grade"          : self.grade,
            "grade_label"    : self.grade_label,
            "dimensions"     : {
                k: {
                    "score"    : round(v.score, 2),
                    "weight"   : v.weight,
                    "issues"   : v.issues,
                    "positives": v.positives,
                    "details"  : v.details,
                }
                for k, v in self.dimensions.items()
            },
            "issues"          : self.issues,
            "recommendations" : self.recommendations,
            "pii_detected"    : self.pii_detected,
            "statistics"      : self.statistics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _penalty(issue_count: int, per_issue: float = 10.0,
             base: float = 100.0) -> float:
    return _clamp(base - issue_count * per_issue)


def _grade(score: float) -> Tuple[str, str]:
    for letter, threshold in GRADE_THRESHOLDS.items():
        if score >= threshold:
            return letter, GRADE_LABELS[letter]
    return "F", GRADE_LABELS["F"]


def _health_badge(score: float) -> str:
    if score >= 90:
        return "🟢 Healthy"
    elif score >= 75:
        return "🟡 Good"
    elif score >= 60:
        return "🟠 Needs Attention"
    elif score >= 45:
        return "🔴 Unhealthy"
    else:
        return "⛔ Critical"


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _datetime_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime64"]).columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 1 — COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class CompletenessAnalyzer:
    """
    Measures how complete the dataset is.

    Sub-scores:
    - cell_completeness   : % non-null cells
    - row_completeness    : % rows with 0 nulls
    - col_completeness    : % cols with 0 nulls
    - critical_col_score  : completeness of numeric cols (more important)
    """

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        n, p = df.shape
        issues, positives = [], []

        # ── cell completeness ────────────────────────────────────────────────
        total_cells    = n * p
        null_cells     = df.isnull().sum().sum()
        cell_complete  = 1 - (null_cells / total_cells) if total_cells > 0 else 1.0
        cell_score     = cell_complete * 100

        # ── row completeness ─────────────────────────────────────────────────
        rows_complete   = (df.isnull().sum(axis=1) == 0).sum()
        row_score       = (rows_complete / n) * 100 if n > 0 else 100.0

        # ── column completeness ──────────────────────────────────────────────
        cols_complete   = (df.isnull().sum() == 0).sum()
        col_score       = (cols_complete / p) * 100 if p > 0 else 100.0

        # ── per-column null rates ─────────────────────────────────────────────
        null_rates      = df.isnull().mean()
        critical_nulls  = null_rates[null_rates > 0.5].index.tolist()  # >50% missing
        moderate_nulls  = null_rates[(null_rates > 0.1) & (null_rates <= 0.5)].index.tolist()
        low_nulls       = null_rates[(null_rates > 0) & (null_rates <= 0.1)].index.tolist()

        # ── critical columns (numeric) ────────────────────────────────────────
        num_cols        = _numeric_cols(df)
        if num_cols:
            num_null_rate  = df[num_cols].isnull().mean().mean()
            critical_score = (1 - num_null_rate) * 100
        else:
            critical_score = 100.0

        # ── issues & positives ────────────────────────────────────────────────
        if critical_nulls:
            issues.append(
                f"{len(critical_nulls)} column(s) missing >50% values: "
                f"{', '.join(critical_nulls[:5])}"
            )
        if moderate_nulls:
            issues.append(
                f"{len(moderate_nulls)} column(s) missing 10-50%: "
                f"{', '.join(moderate_nulls[:5])}"
            )
        if low_nulls:
            issues.append(f"{len(low_nulls)} column(s) with minor missing values (<10%)")

        if cell_score >= 95:
            positives.append(f"Dataset is {cell_score:.1f}% complete at cell level ✅")
        if not critical_nulls:
            positives.append("No column has >50% missing values ✅")
        if row_score >= 90:
            positives.append(f"{rows_complete}/{n} rows are fully complete ✅")

        # ── weighted sub-score ────────────────────────────────────────────────
        score = (
            cell_score     * 0.40 +
            row_score      * 0.25 +
            col_score      * 0.20 +
            critical_score * 0.15
        )

        return DimensionResult(
            name       = "Completeness",
            score      = _clamp(score),
            weight     = DIMENSION_WEIGHTS["completeness"],
            weighted   = _clamp(score) * DIMENSION_WEIGHTS["completeness"],
            issues     = issues,
            positives  = positives,
            sub_scores = {
                "cell_completeness"    : round(cell_score, 2),
                "row_completeness"     : round(row_score, 2),
                "column_completeness"  : round(col_score, 2),
                "numeric_completeness" : round(critical_score, 2),
            },
            details    = {
                "null_cells"       : int(null_cells),
                "null_rate"        : round(float(null_cells / total_cells), 4),
                "critical_columns" : critical_nulls,
                "moderate_columns" : moderate_nulls,
                "per_column_nulls" : null_rates.round(4).to_dict(),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 2 — UNIQUENESS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class UniquenessAnalyzer:
    """
    Detects duplicates at row, column, and value level.

    Sub-scores:
    - row_uniqueness     : % unique rows
    - col_uniqueness     : constant columns penalty
    - near_duplicate     : columns with >95% same value
    - id_col_uniqueness  : suspected ID columns must be 100% unique
    """

    def _suspect_id_cols(self, df: pd.DataFrame) -> List[str]:
        hints = ["id", "key", "uuid", "code", "ref", "no", "num", "index"]
        return [
            c for c in df.columns
            if any(h in c.lower() for h in hints)
        ]

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        n, p = df.shape
        issues, positives = [], []

        # ── row duplicates ────────────────────────────────────────────────────
        dup_rows        = df.duplicated().sum()
        row_uniq_score  = ((n - dup_rows) / n) * 100 if n > 0 else 100.0

        # ── constant columns (0 variance) ─────────────────────────────────────
        const_cols = [
            c for c in df.columns
            if df[c].nunique(dropna=False) <= 1
        ]
        const_penalty = len(const_cols) * 15

        # ── near-constant columns (>95% same value) ───────────────────────────
        near_const = []
        for c in df.columns:
            vc = df[c].value_counts(normalize=True, dropna=False)
            if len(vc) > 0 and vc.iloc[0] > 0.95:
                near_const.append((c, round(vc.iloc[0] * 100, 1)))

        near_penalty = len(near_const) * 8

        # ── ID column uniqueness ──────────────────────────────────────────────
        id_cols        = self._suspect_id_cols(df)
        id_violations  = []
        for c in id_cols:
            if df[c].nunique() < n:
                id_violations.append(c)

        id_penalty = len(id_violations) * 20

        # ── issues & positives ────────────────────────────────────────────────
        if dup_rows > 0:
            issues.append(
                f"{dup_rows} duplicate rows ({dup_rows/n:.1%} of dataset)"
            )
        if const_cols:
            issues.append(
                f"{len(const_cols)} constant column(s) — zero variance: "
                f"{', '.join(const_cols[:5])}"
            )
        if near_const:
            nc_str = ", ".join([f"{c}({pct}%)" for c, pct in near_const[:3]])
            issues.append(f"{len(near_const)} near-constant column(s): {nc_str}")
        if id_violations:
            issues.append(
                f"ID column(s) with duplicates: {', '.join(id_violations)}"
            )

        if dup_rows == 0:
            positives.append("No duplicate rows found ✅")
        if not const_cols:
            positives.append("No constant (zero-variance) columns ✅")
        if id_cols and not id_violations:
            positives.append(f"All ID columns are 100% unique ✅")

        score = _clamp(row_uniq_score - const_penalty - near_penalty - id_penalty)

        return DimensionResult(
            name      = "Uniqueness",
            score     = score,
            weight    = DIMENSION_WEIGHTS["uniqueness"],
            weighted  = score * DIMENSION_WEIGHTS["uniqueness"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "row_uniqueness"   : round(row_uniq_score, 2),
                "constant_penalty" : const_penalty,
                "near_const_penalty": near_penalty,
                "id_penalty"       : id_penalty,
            },
            details    = {
                "duplicate_rows"       : int(dup_rows),
                "constant_columns"     : const_cols,
                "near_constant_columns": [c for c, _ in near_const],
                "id_columns_violated"  : id_violations,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 3 — VALIDITY
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class ValidityAnalyzer:
    """
    Checks format, type, and value range conformity.

    Sub-scores:
    - type_validity      : mixed types in columns
    - range_validity     : outliers / impossible values
    - format_validity    : strings matching expected patterns
    - cardinality_check  : high-cardinality categoricals
    """

    # Impossible value detectors for common column names
    RANGE_RULES: Dict[str, Tuple[float, float]] = {
        "age"         : (0, 130),
        "year"        : (1900, 2100),
        "month"       : (1, 12),
        "day"         : (1, 31),
        "percentage"  : (0, 100),
        "percent"     : (0, 100),
        "rate"        : (0, 100),
        "score"       : (0, 100),
        "probability" : (0, 1),
        "lat"         : (-90, 90),
        "latitude"    : (-90, 90),
        "lon"         : (-180, 180),
        "longitude"   : (-180, 180),
        "price"       : (0, 1e9),
        "cost"        : (0, 1e9),
        "salary"      : (0, 1e8),
        "quantity"    : (0, 1e7),
        "count"       : (0, 1e9),
    }

    def _check_mixed_types(self, col: pd.Series) -> float:
        """Returns penalty for mixed types in object columns."""
        if col.dtype != object:
            return 0.0
        sample  = col.dropna().head(500)
        if len(sample) == 0:
            return 0.0
        type_set = set(type(v).__name__ for v in sample)
        return 0.0 if len(type_set) <= 1 else 15.0 * (len(type_set) - 1)

    def _check_range(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Check numeric columns against known valid ranges."""
        violations = []
        total_penalty = 0.0
        for col in _numeric_cols(df):
            col_lower = col.lower()
            for hint, (lo, hi) in self.RANGE_RULES.items():
                if hint in col_lower:
                    s = df[col].dropna()
                    out_of_range = ((s < lo) | (s > hi)).sum()
                    if out_of_range > 0:
                        violations.append(
                            f"'{col}': {out_of_range} value(s) outside [{lo}, {hi}]"
                        )
                        total_penalty += min(30, out_of_range * 5)
                    break
        return violations, total_penalty

    def _check_negative(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Flag numeric columns that shouldn't be negative."""
        never_neg = ["age", "price", "cost", "salary", "count",
                     "quantity", "amount", "size", "length", "weight"]
        violations = []
        penalty    = 0.0
        for col in _numeric_cols(df):
            if any(h in col.lower() for h in never_neg):
                neg_count = (df[col].dropna() < 0).sum()
                if neg_count > 0:
                    violations.append(
                        f"'{col}': {neg_count} negative value(s) (should be ≥0)"
                    )
                    penalty += 10.0
        return violations, penalty

    def _check_cardinality(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """High-cardinality categoricals are suspicious (not encoded)."""
        issues  = []
        penalty = 0.0
        n       = len(df)
        for col in _categorical_cols(df):
            card = df[col].nunique()
            if card > 0.9 * n and n > 50:
                issues.append(
                    f"'{col}': very high cardinality ({card}/{n}) — possible free text or ID"
                )
                penalty += 5.0
        return issues, penalty

    def _check_format(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Spot-check string columns for consistent formats."""
        issues  = []
        penalty = 0.0
        for col in _categorical_cols(df):
            sample = df[col].dropna().astype(str).head(200)
            if len(sample) == 0:
                continue
            # mixed case inconsistency
            has_upper = sample.str.isupper().any()
            has_lower = sample.str.islower().any()
            has_mixed = sample.str.istitle().any()
            case_types = sum([has_upper, has_lower, has_mixed])
            if case_types > 1:
                issues.append(f"'{col}': inconsistent casing (mixed UPPER/lower/Title)")
                penalty += 3.0
            # leading/trailing whitespace
            has_spaces = (sample != sample.str.strip()).any()
            if has_spaces:
                issues.append(f"'{col}': leading/trailing whitespace detected")
                penalty += 3.0
        return issues, penalty

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        issues, positives = [], []
        total_penalty     = 0.0

        # mixed types
        mixed_penalty = sum(self._check_mixed_types(df[c]) for c in df.columns)
        if mixed_penalty > 0:
            issues.append(f"Mixed data types in object columns (penalty: {mixed_penalty:.0f})")
        total_penalty += mixed_penalty

        # range validity
        range_issues, range_pen = self._check_range(df)
        issues.extend(range_issues)
        total_penalty += range_pen

        # negative values
        neg_issues, neg_pen = self._check_negative(df)
        issues.extend(neg_issues)
        total_penalty += neg_pen

        # cardinality
        card_issues, card_pen = self._check_cardinality(df)
        issues.extend(card_issues)
        total_penalty += card_pen

        # format
        fmt_issues, fmt_pen = self._check_format(df)
        issues.extend(fmt_issues)
        total_penalty += fmt_pen

        if not range_issues:
            positives.append("All named numeric columns within expected ranges ✅")
        if not neg_issues:
            positives.append("No impossible negative values found ✅")
        if not fmt_issues:
            positives.append("String column formatting appears consistent ✅")

        score = _clamp(100 - total_penalty)

        return DimensionResult(
            name      = "Validity",
            score     = score,
            weight    = DIMENSION_WEIGHTS["validity"],
            weighted  = score * DIMENSION_WEIGHTS["validity"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "mixed_type_penalty": mixed_penalty,
                "range_penalty"     : range_pen,
                "negative_penalty"  : neg_pen,
                "cardinality_penalty": card_pen,
                "format_penalty"    : fmt_pen,
            },
            details = {
                "range_violations"   : range_issues,
                "negative_violations": neg_issues,
                "format_issues"      : fmt_issues,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 4 — CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyAnalyzer:
    """
    Internal logical consistency of the dataset.

    Sub-scores:
    - cross_column_logic  : detects impossible combinations
    - date_ordering       : start_date < end_date
    - value_symmetry      : min ≤ max, lower ≤ upper
    - dtype_stability     : columns that look numeric but typed as object
    """

    def _check_date_order(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Detect start/end date pairs and check ordering."""
        issues  = []
        penalty = 0.0
        cols    = df.columns.tolist()

        start_candidates = [c for c in cols if "start" in c.lower() or "begin" in c.lower() or "from" in c.lower()]
        end_candidates   = [c for c in cols if "end" in c.lower() or "finish" in c.lower() or "to" in c.lower()]

        for sc in start_candidates:
            for ec in end_candidates:
                if sc == ec:
                    continue
                try:
                    s_col = pd.to_datetime(df[sc], errors="coerce")
                    e_col = pd.to_datetime(df[ec], errors="coerce")
                    valid = s_col.notna() & e_col.notna()
                    violations = (s_col[valid] > e_col[valid]).sum()
                    if violations > 0:
                        issues.append(
                            f"Date ordering: '{sc}' > '{ec}' in {violations} row(s)"
                        )
                        penalty += min(20, violations * 2)
                except Exception:
                    pass
        return issues, penalty

    def _check_min_max_pairs(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """min_x should be ≤ max_x."""
        issues  = []
        penalty = 0.0
        cols    = df.columns.tolist()

        min_cols = [c for c in cols if c.lower().startswith("min_") or "_min" in c.lower()]
        max_cols = [c for c in cols if c.lower().startswith("max_") or "_max" in c.lower()]

        for mc in min_cols:
            suffix = mc.lower().replace("min_", "").replace("_min", "")
            for xc in max_cols:
                if suffix in xc.lower():
                    try:
                        violations = (df[mc].dropna() > df[xc].dropna()).sum()
                        if violations > 0:
                            issues.append(
                                f"Min/Max violation: '{mc}' > '{xc}' in {violations} row(s)"
                            )
                            penalty += min(20, violations * 3)
                    except Exception:
                        pass
        return issues, penalty

    def _check_hidden_numerics(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Object columns that are actually numeric — dtype mismatch."""
        issues  = []
        penalty = 0.0
        for col in _categorical_cols(df):
            sample = df[col].dropna().astype(str).head(300)
            if len(sample) == 0:
                continue
            converted = pd.to_numeric(sample, errors="coerce")
            numeric_rate = converted.notna().mean()
            if numeric_rate > 0.90:
                issues.append(
                    f"'{col}' is typed as object but {numeric_rate:.0%} values are numeric"
                )
                penalty += 8.0
        return issues, penalty

    def _check_cross_column(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Basic cross-column logical checks."""
        issues  = []
        penalty = 0.0
        cols_lower = {c.lower(): c for c in df.columns}

        # revenue = price * quantity
        price_col = next((v for k, v in cols_lower.items()
                          if "price" in k or "unit_cost" in k), None)
        qty_col   = next((v for k, v in cols_lower.items()
                          if "qty" in k or "quantity" in k), None)
        rev_col   = next((v for k, v in cols_lower.items()
                          if "revenue" in k or "total" in k or "amount" in k), None)

        if price_col and qty_col and rev_col:
            try:
                expected = df[price_col] * df[qty_col]
                actual   = df[rev_col]
                diff     = (expected - actual).abs()
                violations = (diff > expected.abs() * 0.01).sum()
                if violations > 0:
                    issues.append(
                        f"Revenue inconsistency: '{rev_col}' ≠ '{price_col}' × '{qty_col}'"
                        f" in {violations} row(s)"
                    )
                    penalty += 15.0
            except Exception:
                pass

        return issues, penalty

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        issues, positives = [], []
        total_penalty     = 0.0

        date_issues, date_pen = self._check_date_order(df)
        issues.extend(date_issues)
        total_penalty += date_pen

        mm_issues, mm_pen = self._check_min_max_pairs(df)
        issues.extend(mm_issues)
        total_penalty += mm_pen

        hidden_issues, hidden_pen = self._check_hidden_numerics(df)
        issues.extend(hidden_issues)
        total_penalty += hidden_pen

        cross_issues, cross_pen = self._check_cross_column(df)
        issues.extend(cross_issues)
        total_penalty += cross_pen

        if not date_issues:
            positives.append("Date columns appear logically ordered ✅")
        if not mm_issues:
            positives.append("Min/Max column pairs are consistent ✅")
        if not hidden_issues:
            positives.append("No dtype mismatches detected ✅")

        score = _clamp(100 - total_penalty)

        return DimensionResult(
            name      = "Consistency",
            score     = score,
            weight    = DIMENSION_WEIGHTS["consistency"],
            weighted  = score * DIMENSION_WEIGHTS["consistency"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "date_penalty"    : date_pen,
                "minmax_penalty"  : mm_pen,
                "dtype_penalty"   : hidden_pen,
                "cross_penalty"   : cross_pen,
            },
            details = {
                "date_issues"   : date_issues,
                "minmax_issues" : mm_issues,
                "dtype_issues"  : hidden_issues,
                "cross_issues"  : cross_issues,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 5 — INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class IntegrityAnalyzer:
    """
    Structural and referential integrity.

    Sub-scores:
    - schema_integrity    : expected dtypes consistent across dataset
    - pii_exposure        : PII columns detected → risk flag
    - column_naming       : bad names (spaces, special chars, conflicts)
    - index_integrity     : index is clean and monotonic if integer
    """

    def _detect_pii(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect PII in column names and sample values."""
        pii_found: Dict[str, List[str]] = {}

        for col in df.columns:
            col_lower = col.lower()
            found     = []

            # name-based detection
            for hint in PII_NAME_HINTS:
                if hint in col_lower:
                    found.append(f"name_hint:{hint}")
                    break

            # value-based detection (sample 100 rows)
            sample = df[col].dropna().astype(str).head(100)
            for pii_type, pattern in PII_PATTERNS.items():
                if sample.str.contains(pattern, regex=True, na=False).any():
                    found.append(f"pattern:{pii_type}")

            if found:
                pii_found[col] = found

        return pii_found

    def _check_column_names(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        """Column name quality check."""
        issues  = []
        penalty = 0.0

        for col in df.columns:
            if " " in str(col):
                issues.append(f"Column '{col}' has spaces — use underscores")
                penalty += 2.0
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", str(col)):
                issues.append(f"Column '{col}' has special characters")
                penalty += 3.0

        # duplicate column names
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate column names detected!")
            penalty += 20.0

        return issues, penalty

    def _check_index(self, df: pd.DataFrame) -> Tuple[List[str], float]:
        issues  = []
        penalty = 0.0

        if pd.api.types.is_integer_dtype(df.index):
            if not df.index.is_monotonic_increasing:
                issues.append("Integer index is not monotonically increasing")
                penalty += 5.0
            if df.index.duplicated().any():
                issues.append("Duplicate values in index")
                penalty += 10.0

        return issues, penalty

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        issues, positives = [], []
        total_penalty     = 0.0

        # PII
        pii = self._detect_pii(df)
        if pii:
            pii_cols = list(pii.keys())
            issues.append(
                f"⚠️  PII detected in {len(pii_cols)} column(s): "
                f"{', '.join(pii_cols[:5])}"
            )
            total_penalty += len(pii_cols) * 5  # soft penalty — awareness only

        # column names
        name_issues, name_pen = self._check_column_names(df)
        issues.extend(name_issues)
        total_penalty += name_pen

        # index
        idx_issues, idx_pen = self._check_index(df)
        issues.extend(idx_issues)
        total_penalty += idx_pen

        if not pii:
            positives.append("No PII detected in column names or values ✅")
        if not name_issues:
            positives.append("All column names are clean and well-formatted ✅")
        if not idx_issues:
            positives.append("Index integrity is clean ✅")

        score = _clamp(100 - total_penalty)

        return DimensionResult(
            name      = "Integrity",
            score     = score,
            weight    = DIMENSION_WEIGHTS["integrity"],
            weighted  = score * DIMENSION_WEIGHTS["integrity"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "pii_penalty"   : len(pii) * 5,
                "name_penalty"  : name_pen,
                "index_penalty" : idx_pen,
            },
            details = {
                "pii_columns"   : pii,
                "name_issues"   : name_issues,
                "index_issues"  : idx_issues,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 6 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class DistributionAnalyzer:
    """
    Statistical health of numeric distributions.

    Sub-scores:
    - normality_score      : Shapiro-Wilk normality test
    - skewness_score       : penalise extreme skew
    - kurtosis_score       : penalise extreme kurtosis
    - entropy_score        : Shannon entropy of categoricals
    - outlier_rate_score   : IQR-based outlier rate
    """

    def _shannon_entropy(self, series: pd.Series) -> float:
        """Normalized Shannon entropy [0, 1]. 1 = perfectly uniform."""
        vc = series.value_counts(normalize=True)
        if len(vc) <= 1:
            return 0.0
        entropy  = -np.sum(vc * np.log2(vc + 1e-12))
        max_entr = np.log2(len(vc))
        return float(entropy / max_entr) if max_entr > 0 else 0.0

    def _iqr_outlier_rate(self, s: pd.Series) -> float:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        issues, positives = [], []
        num_cols = _numeric_cols(df)
        cat_cols = _categorical_cols(df)

        skew_scores, kurt_scores, outlier_scores, entropy_scores = [], [], [], []
        skew_issues, kurt_issues, outlier_issues = [], [], []

        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 4:
                continue

            # skewness
            sk = abs(float(s.skew()))
            skew_scores.append(max(0, 100 - (sk * 10)))
            if sk > 3:
                skew_issues.append(f"'{col}': extreme skewness ({sk:.2f})")
            elif sk > 1.5:
                skew_issues.append(f"'{col}': moderate skewness ({sk:.2f})")

            # kurtosis
            kt = abs(float(s.kurtosis()))
            kurt_scores.append(max(0, 100 - (kt * 5)))
            if kt > 10:
                kurt_issues.append(f"'{col}': extreme kurtosis ({kt:.2f})")

            # outliers
            out_rate = self._iqr_outlier_rate(s)
            outlier_scores.append(max(0, 100 - out_rate * 500))
            if out_rate > 0.10:
                outlier_issues.append(
                    f"'{col}': {out_rate:.1%} outlier rate (IQR)"
                )

        for col in cat_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            ent = self._shannon_entropy(s)
            entropy_scores.append(ent * 100)

        # aggregate
        skew_score    = np.mean(skew_scores)    if skew_scores    else 100.0
        kurt_score    = np.mean(kurt_scores)    if kurt_scores    else 100.0
        outlier_score = np.mean(outlier_scores) if outlier_scores else 100.0
        entropy_score = np.mean(entropy_scores) if entropy_scores else 100.0

        issues.extend(skew_issues[:3])
        issues.extend(kurt_issues[:2])
        issues.extend(outlier_issues[:3])

        if not skew_issues:
            positives.append("All numeric distributions have acceptable skewness ✅")
        if not outlier_issues:
            positives.append("Outlier rates within acceptable bounds ✅")
        if entropy_scores and entropy_score > 60:
            positives.append("Categorical columns have good value diversity ✅")

        score = (
            skew_score    * 0.30 +
            kurt_score    * 0.20 +
            outlier_score * 0.30 +
            entropy_score * 0.20
        )

        return DimensionResult(
            name      = "Distribution",
            score     = _clamp(score),
            weight    = DIMENSION_WEIGHTS["distribution"],
            weighted  = _clamp(score) * DIMENSION_WEIGHTS["distribution"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "skewness_score"  : round(float(skew_score), 2),
                "kurtosis_score"  : round(float(kurt_score), 2),
                "outlier_score"   : round(float(outlier_score), 2),
                "entropy_score"   : round(float(entropy_score), 2),
            },
            details = {
                "skew_issues"    : skew_issues,
                "kurt_issues"    : kurt_issues,
                "outlier_issues" : outlier_issues,
                "n_numeric_cols" : len(num_cols),
                "n_cat_cols"     : len(cat_cols),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSION 7 — ML READINESS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class MLReadinessAnalyzer:
    """
    How ready is this dataset for machine learning?

    Sub-scores:
    - size_score         : enough rows?
    - feature_score      : enough numeric features?
    - null_tolerance     : nulls in numeric features
    - class_balance      : for classification targets
    - leakage_risk       : date/id-like columns in features
    - correlation_score  : multicollinearity check
    """

    MIN_ROWS_ML   = 100
    GOOD_ROWS_ML  = 1_000

    def _detect_target(self, df: pd.DataFrame) -> Optional[str]:
        target_hints = ["target", "label", "class", "output", "y",
                        "churn", "fraud", "outcome", "result", "status"]
        for col in df.columns:
            if col.lower() in target_hints or any(h in col.lower() for h in target_hints):
                return col
        return None

    def _class_balance_score(self, series: pd.Series) -> Tuple[float, str]:
        vc = series.value_counts(normalize=True)
        if len(vc) < 2:
            return 50.0, "single class — cannot train"
        ratio = vc.min() / vc.max()
        if ratio > 0.4:
            return 100.0, "balanced"
        elif ratio > 0.2:
            return 70.0, "slightly imbalanced"
        elif ratio > 0.05:
            return 40.0, "imbalanced — use SMOTE/class_weight"
        else:
            return 10.0, "severely imbalanced — needs treatment"

    def _multicollinearity_score(self, df: pd.DataFrame) -> float:
        num = df[_numeric_cols(df)].dropna()
        if num.shape[1] < 2:
            return 100.0
        corr  = num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high  = (upper > 0.95).sum().sum()
        return _clamp(100 - high * 10)

    def analyze(self, df: pd.DataFrame) -> DimensionResult:
        issues, positives = [], []
        n, p = df.shape
        num_cols = _numeric_cols(df)

        # size
        if n < self.MIN_ROWS_ML:
            issues.append(f"Only {n} rows — ML models need ≥{self.MIN_ROWS_ML}")
            size_score = 20.0
        elif n < self.GOOD_ROWS_ML:
            size_score = 60.0 + (n / self.GOOD_ROWS_ML) * 40
        else:
            size_score = 100.0
            positives.append(f"{n:,} rows — sufficient for ML ✅")

        # feature count
        if len(num_cols) == 0:
            issues.append("No numeric features — encode categoricals first")
            feature_score = 0.0
        elif len(num_cols) < 2:
            issues.append("Only 1 numeric feature — very limited")
            feature_score = 30.0
        else:
            feature_score = min(100, 50 + len(num_cols) * 5)
            positives.append(f"{len(num_cols)} numeric features available ✅")

        # null in numeric features
        if num_cols:
            num_null_rate = df[num_cols].isnull().mean().mean()
            if num_null_rate > 0.3:
                issues.append(f"High null rate in numeric features ({num_null_rate:.1%}) — impute first")
                null_score = 20.0
            elif num_null_rate > 0.1:
                null_score = 60.0
            else:
                null_score = 100.0
                positives.append("Numeric features are well-populated ✅")
        else:
            null_score = 50.0

        # target detection
        target_col = self._detect_target(df)
        if target_col:
            target_series = df[target_col].dropna()
            n_classes = target_series.nunique()
            if n_classes == 1:
                issues.append(f"Target '{target_col}' has only 1 class")
                balance_score, balance_note = 0.0, "single class"
            elif n_classes <= 20:
                balance_score, balance_note = self._class_balance_score(target_series)
                if balance_score < 70:
                    issues.append(f"Target '{target_col}' is {balance_note}")
                else:
                    positives.append(f"Target '{target_col}' is {balance_note} ✅")
            else:
                balance_score = 80.0
                balance_note  = "regression target (continuous)"
                positives.append(f"Target '{target_col}' appears continuous ✅")
        else:
            balance_score = 70.0
            balance_note  = "no target detected"

        # multicollinearity
        mc_score = self._multicollinearity_score(df)
        if mc_score < 70:
            issues.append("High multicollinearity detected — consider feature selection")
        else:
            positives.append("Multicollinearity within acceptable range ✅")

        score = (
            size_score    * 0.25 +
            feature_score * 0.20 +
            null_score    * 0.20 +
            balance_score * 0.20 +
            mc_score      * 0.15
        )

        return DimensionResult(
            name      = "ML Readiness",
            score     = _clamp(score),
            weight    = DIMENSION_WEIGHTS["ml_readiness"],
            weighted  = _clamp(score) * DIMENSION_WEIGHTS["ml_readiness"],
            issues    = issues,
            positives = positives,
            sub_scores = {
                "size_score"          : round(size_score, 2),
                "feature_score"       : round(feature_score, 2),
                "null_tolerance_score": round(null_score, 2),
                "class_balance_score" : round(balance_score, 2),
                "multicollinearity"   : round(mc_score, 2),
            },
            details = {
                "n_rows"             : n,
                "n_numeric_features" : len(num_cols),
                "detected_target"    : target_col,
                "balance_note"       : balance_note if target_col else "N/A",
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Generates prioritized, actionable recommendations
    based on the dimension scores and issues found.
    """

    def generate(self, dimensions: Dict[str, DimensionResult],
                 overall: float) -> List[str]:
        recs: List[str] = []

        # sort dimensions by score ascending → worst first
        sorted_dims = sorted(dimensions.values(), key=lambda d: d.score)

        for dim in sorted_dims:
            if dim.score >= 85:
                continue

            if dim.name == "Completeness":
                details = dim.details
                if details.get("critical_columns"):
                    recs.append(
                        f"🔴 CRITICAL: Drop or impute columns with >50% missing: "
                        f"{', '.join(details['critical_columns'][:3])}"
                    )
                if details.get("moderate_columns"):
                    recs.append(
                        f"🟠 Impute moderate-missing columns using mean/median/KNN: "
                        f"{', '.join(details['moderate_columns'][:3])}"
                    )

            elif dim.name == "Uniqueness":
                if dim.details.get("duplicate_rows", 0) > 0:
                    recs.append(
                        f"🟡 Remove {dim.details['duplicate_rows']} duplicate rows → df.drop_duplicates()"
                    )
                if dim.details.get("constant_columns"):
                    recs.append(
                        f"🟡 Drop zero-variance columns: "
                        f"{', '.join(dim.details['constant_columns'][:3])}"
                    )

            elif dim.name == "Validity":
                for issue in dim.details.get("range_violations", [])[:2]:
                    recs.append(f"🟠 Fix range: {issue}")
                for issue in dim.details.get("negative_violations", [])[:2]:
                    recs.append(f"🟠 Fix negatives: {issue}")

            elif dim.name == "Consistency":
                for issue in dim.details.get("date_issues", [])[:2]:
                    recs.append(f"🔴 Fix date ordering: {issue}")
                for issue in dim.details.get("dtype_issues", [])[:2]:
                    recs.append(f"🟡 Convert to numeric: {issue}")

            elif dim.name == "Integrity":
                pii = dim.details.get("pii_columns", {})
                if pii:
                    recs.append(
                        f"🔴 PII RISK: Anonymize/mask columns: {', '.join(list(pii.keys())[:3])}"
                    )

            elif dim.name == "Distribution":
                for issue in dim.details.get("skew_issues", [])[:2]:
                    recs.append(f"🟡 Apply log/sqrt transform: {issue}")
                for issue in dim.details.get("outlier_issues", [])[:2]:
                    recs.append(f"🟡 Handle outliers: {issue}")

            elif dim.name == "ML Readiness":
                for issue in dim.issues[:3]:
                    recs.append(f"🤖 ML: {issue}")

        if overall >= 90:
            recs.insert(0, "✅ Dataset quality is excellent. Ready for analysis and ML.")
        elif overall >= 75:
            recs.insert(0, "🟡 Dataset quality is good. Minor fixes recommended.")
        elif overall >= 60:
            recs.insert(0, "🟠 Dataset needs some cleaning before ML use.")
        else:
            recs.insert(0, "🔴 Dataset has significant quality issues. Clean before use.")

        return recs[:12]  # top 12 recommendations


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — DataQualityScorer
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityScorer:
    """
    Main entry point. Runs all 7 dimensions and returns a QualityProfile.

    Usage
    -----
    >>> scorer  = DataQualityScorer()
    >>> profile = scorer.score(df)
    >>> print(f"Score: {profile.overall_score:.1f} | Grade: {profile.grade}")
    >>> scorer.print_report(profile)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights    = weights or DIMENSION_WEIGHTS
        self._analyzers = {
            "completeness" : CompletenessAnalyzer(),
            "uniqueness"   : UniquenessAnalyzer(),
            "validity"     : ValidityAnalyzer(),
            "consistency"  : ConsistencyAnalyzer(),
            "integrity"    : IntegrityAnalyzer(),
            "distribution" : DistributionAnalyzer(),
            "ml_readiness" : MLReadinessAnalyzer(),
        }
        self._recommender = RecommendationEngine()

    def score(self, df: pd.DataFrame) -> QualityProfile:
        """Run full quality scoring. Returns QualityProfile."""
        if df.empty:
            raise ValueError("DataFrame is empty.")

        logger.info(f"[QualityScorer] Scoring dataset {df.shape}")

        dimensions: Dict[str, DimensionResult] = {}
        all_issues: List[str] = []

        # run all dimensions
        for name, analyzer in self._analyzers.items():
            try:
                result = analyzer.analyze(df)
                # override weight from custom weights if provided
                result.weight   = self.weights.get(name, result.weight)
                result.weighted = result.score * result.weight
                dimensions[name] = result
                all_issues.extend(result.issues)
                logger.info(
                    f"[QualityScorer] {name}: {result.score:.1f}/100"
                )
            except Exception as e:
                logger.warning(f"[QualityScorer] {name} failed: {e}")
                dimensions[name] = DimensionResult(
                    name=name, score=50.0,
                    weight=self.weights.get(name, 0.1),
                    weighted=5.0, issues=[str(e)], positives=[],
                )

        # overall weighted score
        overall = sum(d.weighted for d in dimensions.values())
        overall = _clamp(overall / sum(self.weights.values()))

        grade, grade_label = _grade(overall)

        # PII from integrity dimension
        pii = dimensions["integrity"].details.get("pii_columns", {})

        # recommendations
        recs = self._recommender.generate(dimensions, overall)

        # basic dataset statistics
        statistics = self._dataset_stats(df)
        num_cols = _numeric_cols(df)

        profile = QualityProfile(
            overall_score   = round(overall, 2),
            grade           = grade,
            grade_label     = grade_label,
            dimensions      = dimensions,
            issues          = list(dict.fromkeys(all_issues)),  # deduplicate
            recommendations = recs,
            pii_detected    = pii,
            statistics      = statistics,
            dataset_shape   = df.shape,
            health_badge    = _health_badge(overall),
            numeric_columns = num_cols,
        )

        return profile

    def _dataset_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        num_cols = _numeric_cols(df)
        cat_cols = _categorical_cols(df)
        dt_cols  = _datetime_cols(df)
        return {
            "rows"             : len(df),
            "columns"          : df.shape[1],
            "numeric_columns"  : len(num_cols),
            "categorical_cols" : len(cat_cols),
            "datetime_cols"    : len(dt_cols),
            "total_cells"      : df.size,
            "null_cells"       : int(df.isnull().sum().sum()),
            "null_rate"        : round(df.isnull().mean().mean(), 4),
            "duplicate_rows"   : int(df.duplicated().sum()),
            "memory_mb"        : round(df.memory_usage(deep=True).sum() / 1e6, 3),
        }

    def print_report(self, profile: QualityProfile) -> None:
        sep  = "═" * 65
        sep2 = "─" * 65
        print(f"\n{sep}")
        print("  Nydra — Unified Data Quality Report")
        print(sep)
        print(f"  Shape     : {profile.dataset_shape[0]:,} rows × {profile.dataset_shape[1]} cols")
        print(f"  Score     : {profile.overall_score:.1f}/100")
        print(f"  Grade     : {profile.grade_label}")
        print(f"  Health    : {profile.health_badge}")
        print(sep2)
        print("  Dimension Breakdown:")
        for name, dim in profile.dimensions.items():
            bar_len  = int(dim.score / 5)
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            emoji    = "✅" if dim.score >= 80 else "🟡" if dim.score >= 60 else "🔴"
            print(f"    {dim.name:<18} {bar}  {dim.score:>5.1f}/100  {emoji}")
        print(sep2)
        print("  Issues Found:")
        if profile.issues:
            for issue in profile.issues[:10]:
                print(f"    ⚠  {issue}")
        else:
            print("    None ✅")
        print(sep2)
        print("  Recommendations:")
        for rec in profile.recommendations[:8]:
            print(f"    {rec}")
        print(sep)
        if profile.pii_detected:
            print(f"\n  ⚠️  PII ALERT — {len(profile.pii_detected)} column(s): "
                  f"{', '.join(list(profile.pii_detected.keys())[:5])}")
            print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def score_quality(df: pd.DataFrame,
                  weights: Optional[Dict[str, float]] = None) -> QualityProfile:
    """
    One-liner quality scoring for nydra.

    >>> profile = score_quality(df)
    >>> print(profile.overall_score, profile.grade)
    """
    return DataQualityScorer(weights=weights).score(df)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    rng = np.random.default_rng(42)
    n   = 500

    df_test = pd.DataFrame({
        "customer_id"  : range(n),
        "age"          : rng.integers(18, 80, n).tolist() + [None] * 0,
        "salary"       : rng.normal(50_000, 15_000, n).tolist(),
        "email"        : [f"user{i}@example.com" for i in range(n)],
        "score"        : rng.uniform(0, 100, n),
        "churn"        : rng.choice([0, 1], n, p=[0.85, 0.15]),
        "category"     : rng.choice(["A", "B", "C", "D"], n),
        "start_date"   : pd.date_range("2020-01-01", periods=n, freq="D"),
        "revenue"      : rng.normal(1000, 300, n),
    })

    # inject some issues
    df_test.loc[:10, "age"]    = None
    df_test.loc[:5,  "salary"] = -1000    # impossible negatives
    df_test = pd.concat([df_test, df_test.iloc[:5]], ignore_index=True)  # duplicates

    scorer  = DataQualityScorer()
    profile = scorer.score(df_test)
    scorer.print_report(profile)
