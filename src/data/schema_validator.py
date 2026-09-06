"""
schema_validator.py — Data Schema Validator for Nydra v0.2.0

Validates a DataFrame against a user-defined schema.
Supports:
    - Type checking         (numeric, text, date, boolean)
    - Required fields       (no nulls allowed)
    - Uniqueness            (no duplicates in column)
    - Value ranges          (min / max for numeric)
    - Allowed values        (whitelist for categorical)
    - Regex patterns        (e.g. email, phone, ID format)
    - Custom rules          (callable functions)
    - Auto schema inference (generate schema from data)
"""

from __future__ import annotations

import re
import json
from typing import Any, Callable

import pandas as pd
import numpy as np


# ── Built-in regex patterns ───────────────────────────────────────────────────

PATTERNS = {
    "email":   r"^[\w\.-]+@[\w\.-]+\.\w{2,}$",
    "phone":   r"^\+?[\d\s\-\(\)]{7,15}$",
    "url":     r"^https?://[\w\.-]+\.[a-z]{2,}",
    "date_iso": r"^\d{4}-\d{2}-\d{2}$",
    "zipcode": r"^\d{5}(-\d{4})?$",
    "uuid":    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
}


# ── Schema field definition ───────────────────────────────────────────────────

class FieldSchema:
    """
    Defines validation rules for a single column.

    Args:
        dtype:          'numeric', 'text', 'date', 'boolean', 'any'
        required:       if True, no null values allowed
        unique:         if True, all values must be unique
        min_val:        minimum value (numeric only)
        max_val:        maximum value (numeric only)
        min_length:     minimum string length (text only)
        max_length:     maximum string length (text only)
        allowed_values: whitelist of accepted values
        forbidden_values: blacklist of rejected values
        pattern:        regex pattern name (see PATTERNS) or raw regex string
        custom:         callable(series) -> list[str] of error messages
        nullable_pct:   max allowed % of nulls (0.0 - 1.0), overrides required
    """

    def __init__(
        self,
        dtype:             str = "any",
        required:          bool = False,
        unique:            bool = False,
        min_val:           float | None = None,
        max_val:           float | None = None,
        min_length:        int | None = None,
        max_length:        int | None = None,
        allowed_values:    list | None = None,
        forbidden_values:  list | None = None,
        pattern:           str | None = None,
        custom:            Callable | None = None,
        nullable_pct:      float | None = None,
    ):
        self.dtype            = dtype
        self.required         = required
        self.unique           = unique
        self.min_val          = min_val
        self.max_val          = max_val
        self.min_length       = min_length
        self.max_length       = max_length
        self.allowed_values   = [str(v).strip().lower() for v in allowed_values] if allowed_values else None
        self.forbidden_values = [str(v).strip().lower() for v in forbidden_values] if forbidden_values else None
        self.pattern          = pattern
        self.custom           = custom
        self.nullable_pct     = nullable_pct

    def to_dict(self) -> dict:
        return {
            "dtype":           self.dtype,
            "required":        self.required,
            "unique":          self.unique,
            "min_val":         self.min_val,
            "max_val":         self.max_val,
            "min_length":      self.min_length,
            "max_length":      self.max_length,
            "allowed_values":  self.allowed_values,
            "forbidden_values":self.forbidden_values,
            "pattern":         self.pattern,
            "nullable_pct":    self.nullable_pct,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldSchema":
        d = {k: v for k, v in d.items() if k != "custom"}
        return cls(**d)


# ── Validator ─────────────────────────────────────────────────────────────────

def _validate_column(
    col: str,
    series: pd.Series,
    schema: FieldSchema,
    n_rows: int,
) -> dict[str, Any]:
    errors   : list[str] = []
    warnings : list[str] = []
    passed   : list[str] = []

    n_null = int(series.isna().sum())
    null_pct = n_null / max(n_rows, 1)

    # ── Required / nullable ───────────────────────────────────────────────────
    if schema.nullable_pct is not None:
        if null_pct > schema.nullable_pct:
            errors.append(
                f"Null rate {null_pct:.1%} exceeds allowed {schema.nullable_pct:.1%} "
                f"({n_null} values)."
            )
        else:
            passed.append(f"Null rate {null_pct:.1%} within limit.")
    elif schema.required:
        if n_null > 0:
            errors.append(f"Required column has {n_null} null value(s) ({null_pct:.1%}).")
        else:
            passed.append("No null values — required check passed.")

    # ── Uniqueness ────────────────────────────────────────────────────────────
    if schema.unique:
        n_dupes = int(series.duplicated().sum())
        if n_dupes > 0:
            dupe_vals = series[series.duplicated()].dropna().unique()[:3].tolist()
            errors.append(f"Uniqueness violated: {n_dupes} duplicate(s). Examples: {dupe_vals}")
        else:
            passed.append("All values unique.")

    # ── Type check ────────────────────────────────────────────────────────────
    non_null = series.dropna()
    if schema.dtype == "numeric":
        if not pd.api.types.is_numeric_dtype(series):
            # try casting
            try:
                pd.to_numeric(non_null, errors="raise")
                warnings.append("Column is stored as text but all values are numeric — consider casting.")
            except Exception:
                n_bad = int(pd.to_numeric(non_null, errors="coerce").isna().sum())
                errors.append(f"Expected numeric type — {n_bad} non-numeric value(s) found.")
        else:
            passed.append("Type is numeric — correct.")

    elif schema.dtype == "date":
        try:
            pd.to_datetime(non_null, errors="raise")
            passed.append("All values parseable as dates.")
        except Exception:
            n_bad = int(pd.to_datetime(non_null, errors="coerce").isna().sum())
            errors.append(f"Expected date type — {n_bad} unparseable value(s).")

    elif schema.dtype == "boolean":
        bool_vals = {"true","false","yes","no","1","0","y","n","t","f"}
        bad = non_null[~non_null.astype(str).str.lower().isin(bool_vals)]
        if len(bad) > 0:
            errors.append(f"Expected boolean — {len(bad)} invalid value(s): {bad.unique()[:3].tolist()}")
        else:
            passed.append("All values boolean-like.")

    # ── Range check ───────────────────────────────────────────────────────────
    if schema.min_val is not None or schema.max_val is not None:
        if pd.api.types.is_numeric_dtype(series):
            if schema.min_val is not None:
                below = int((non_null < schema.min_val).sum())
                if below > 0:
                    errors.append(f"{below} value(s) below minimum {schema.min_val}.")
                else:
                    passed.append(f"All values >= {schema.min_val}.")
            if schema.max_val is not None:
                above = int((non_null > schema.max_val).sum())
                if above > 0:
                    errors.append(f"{above} value(s) above maximum {schema.max_val}.")
                else:
                    passed.append(f"All values <= {schema.max_val}.")
        else:
            warnings.append("Range check skipped — column is not numeric.")

    # ── String length ─────────────────────────────────────────────────────────
    if schema.min_length is not None or schema.max_length is not None:
        lengths = non_null.astype(str).str.len()
        if schema.min_length is not None:
            short = int((lengths < schema.min_length).sum())
            if short > 0:
                errors.append(f"{short} value(s) shorter than min length {schema.min_length}.")
            else:
                passed.append(f"All values >= {schema.min_length} chars.")
        if schema.max_length is not None:
            long_ = int((lengths > schema.max_length).sum())
            if long_ > 0:
                errors.append(f"{long_} value(s) longer than max length {schema.max_length}.")
            else:
                passed.append(f"All values <= {schema.max_length} chars.")

    # ── Allowed values ────────────────────────────────────────────────────────
    if schema.allowed_values is not None:
        bad_vals = non_null[~non_null.astype(str).str.strip().str.lower().isin(schema.allowed_values)]
        if len(bad_vals) > 0:
            unique_bad = bad_vals.unique()[:5].tolist()
            errors.append(
                f"{len(bad_vals)} value(s) not in allowed list. "
                f"Examples: {unique_bad}. Allowed: {schema.allowed_values}"
            )
        else:
            passed.append(f"All values within allowed set ({len(schema.allowed_values)} options).")

    # ── Forbidden values ──────────────────────────────────────────────────────
    if schema.forbidden_values is not None:
        found = non_null[non_null.astype(str).str.strip().str.lower().isin(schema.forbidden_values)]
        if len(found) > 0:
            errors.append(f"{len(found)} forbidden value(s) found: {found.unique()[:5].tolist()}")
        else:
            passed.append("No forbidden values found.")

    # ── Pattern / regex ───────────────────────────────────────────────────────
    if schema.pattern is not None:
        regex = PATTERNS.get(schema.pattern, schema.pattern)
        try:
            bad = non_null[~non_null.astype(str).str.match(regex, na=False)]
            if len(bad) > 0:
                errors.append(
                    f"{len(bad)} value(s) do not match pattern '{schema.pattern}'. "
                    f"Examples: {bad.unique()[:3].tolist()}"
                )
            else:
                passed.append(f"All values match pattern '{schema.pattern}'.")
        except re.error as e:
            warnings.append(f"Invalid regex pattern: {e}")

    # ── Custom rule ───────────────────────────────────────────────────────────
    if schema.custom is not None:
        try:
            custom_errors = schema.custom(series)
            if custom_errors:
                errors.extend(custom_errors)
            else:
                passed.append("Custom rule passed.")
        except Exception as e:
            warnings.append(f"Custom rule error: {e}")

    status = "pass" if not errors else "fail"
    return {
        "column":   col,
        "status":   status,
        "errors":   errors,
        "warnings": warnings,
        "passed":   passed,
        "n_checks": len(errors) + len(warnings) + len(passed),
    }


# ── Auto schema inference ─────────────────────────────────────────────────────

def infer_schema(data: dict[str, Any]) -> dict[str, FieldSchema]:
    """
    Automatically infer a schema from a DataFrame.
    Useful as a starting point the user can then customize.
    """
    df     = data["df"]
    schema = {}

    for col in df.columns:
        series   = df[col]
        n_null   = int(series.isna().sum())
        n_unique = int(series.nunique())
        n_rows   = len(df)

        kwargs: dict[str, Any] = {}

        # dtype
        if pd.api.types.is_numeric_dtype(series):
            kwargs["dtype"]   = "numeric"
            kwargs["min_val"] = round(float(series.min()), 4)
            kwargs["max_val"] = round(float(series.max()), 4)
        else:
            try:
                pd.to_datetime(series.dropna(), errors="raise")
                kwargs["dtype"] = "date"
            except Exception:
                kwargs["dtype"] = "text"

        # required
        kwargs["required"] = n_null == 0

        # unique
        kwargs["unique"] = n_unique == n_rows and n_rows > 1

        # allowed values for low-cardinality text
        if kwargs.get("dtype") == "text" and n_unique <= 20:
            kwargs["allowed_values"] = series.dropna().astype(str).str.strip().unique().tolist()

        schema[col] = FieldSchema(**kwargs)

    return schema


def schema_to_dict(schema: dict[str, FieldSchema]) -> dict[str, dict]:
    return {col: fs.to_dict() for col, fs in schema.items()}


def schema_from_dict(d: dict[str, dict]) -> dict[str, FieldSchema]:
    return {col: FieldSchema.from_dict(rules) for col, rules in d.items()}


# ── Main validator ────────────────────────────────────────────────────────────

def validate_schema(
    data:   dict[str, Any],
    schema: dict[str, FieldSchema],
) -> dict[str, Any]:
    """
    Validate a Nydra data dict against a schema.

    Returns:
        {
          'passed':         int,
          'failed':         int,
          'warnings':       int,
          'total_columns':  int,
          'coverage':       float,   # % of df columns covered by schema
          'results':        list[dict],
          'summary':        str,
          'valid':          bool,
        }
    """
    df      = data["df"]
    n_rows  = len(df)
    results = []

    for col, rules in schema.items():
        if col not in df.columns:
            results.append({
                "column":   col,
                "status":   "missing",
                "errors":   [f"Column '{col}' defined in schema but not found in data."],
                "warnings": [],
                "passed":   [],
                "n_checks": 1,
            })
            continue
        results.append(_validate_column(col, df[col], rules, n_rows))

    # columns in data not covered by schema
    uncovered = [c for c in df.columns if c not in schema]

    n_passed   = sum(1 for r in results if r["status"] == "pass")
    n_failed   = sum(1 for r in results if r["status"] in ("fail", "missing"))
    n_warnings = sum(len(r["warnings"]) for r in results)
    coverage   = round(len(schema) / max(len(df.columns), 1) * 100, 1)

    valid   = n_failed == 0
    summary = (
        f"{'✓ VALID' if valid else '✗ INVALID'} — "
        f"{n_passed}/{len(results)} columns passed. "
        f"{n_failed} failed, {n_warnings} warning(s). "
        f"Schema covers {coverage}% of columns."
    )

    return {
        "passed":        n_passed,
        "failed":        n_failed,
        "warnings":      n_warnings,
        "total_columns": len(df.columns),
        "coverage":      coverage,
        "uncovered":     uncovered,
        "results":       results,
        "summary":       summary,
        "valid":         valid,
    }
