"""Adapter between the Nydra image pipeline and ``image_report.py``.

``image_report`` reads columns the pipeline never produces: ``quality_score``,
``issue_severity``, ``is_low_res``, ``is_low_quality`` (and a loader
``is_corrupted`` that stays False for unreadable files). Without them every
image scores 0.0 and gets grade F.

``prepare_report_frames`` returns copies of the two frames with those columns
added, so ``image_report.py`` itself stays untouched. Inputs are never mutated.

Thresholds are our own choices (the pipeline defines none):
  * low resolution : shorter side below LOW_RES_MIN_SIDE pixels
  * low quality    : score below LOW_QUALITY_BELOW
  * severity       : unreadable -> critical, score <45 -> high, <60 -> medium,
                     <75 -> low, exact duplicate -> at least low; no issue -> None
    (the bands follow image_report's own grade thresholds D/C/B)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOW_RES_MIN_SIDE = 224
LOW_QUALITY_BELOW = 60.0


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _severity(score: float, unreadable: bool, duplicate: bool):
    if unreadable:
        return "critical"
    sev = None
    if pd.notna(score):
        if score < 45:
            sev = "high"
        elif score < 60:
            sev = "medium"
        elif score < 75:
            sev = "low"
    if duplicate and sev is None:
        sev = "low"
    return sev


def prepare_report_frames(
    loader_df: pd.DataFrame,
    quality_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (loader_df2, quality_df2) ready for ``generate_report``."""
    if "file_path" not in loader_df.columns:
        raise ValueError("loader_df must contain a 'file_path' column.")

    loader2 = loader_df.copy().reset_index(drop=True)

    # --- one score per file --------------------------------------------------
    scores = pd.Series(np.nan, index=loader2.index, dtype="float64")
    if quality_df is not None and "file_path" in quality_df.columns:
        src = None
        if "quality_score" in quality_df.columns:
            src = "quality_score"
        elif "overall_score" in quality_df.columns:
            src = "overall_score"
        if src:
            q = quality_df[["file_path", src]].drop_duplicates("file_path")
            q = q.assign(**{src: pd.to_numeric(q[src], errors="coerce")})
            mapped = loader2["file_path"].map(q.set_index("file_path")[src])
            scores = pd.to_numeric(mapped, errors="coerce").astype("float64")

    # --- flags -----------------------------------------------------------------
    not_valid = ~_bool_col(loader2, "is_valid") if "is_valid" in loader2.columns \
        else pd.Series(False, index=loader2.index)
    scored = quality_df is not None and scores.notna().any()
    unreadable = _bool_col(loader2, "is_corrupted") | not_valid
    if scored:
        unreadable = unreadable | scores.isna()
    duplicate = _bool_col(loader2, "is_duplicate")

    if {"width", "height"} <= set(loader2.columns):
        w = pd.to_numeric(loader2["width"], errors="coerce")
        h = pd.to_numeric(loader2["height"], errors="coerce")
        low_res = (np.fmin(w, h) < LOW_RES_MIN_SIDE) & ~unreadable
        low_res = low_res.fillna(False)
    else:
        low_res = pd.Series(False, index=loader2.index)

    low_quality = (scores < LOW_QUALITY_BELOW).fillna(False) & ~unreadable

    # the loader never flags a broken jpg as corrupted; the report reads this column
    loader2["is_corrupted"] = unreadable.values

    severity = [
        _severity(s, u, d)
        for s, u, d in zip(scores.values, unreadable.values, duplicate.values)
    ]

    quality2 = pd.DataFrame({
        "file_path": loader2["file_path"].values,
        "quality_score": scores.values,
        "issue_severity": severity,
        "is_low_res": low_res.values,
        "is_low_quality": low_quality.values,
    })
    return loader2, quality2