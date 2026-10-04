"""Builds the Distributions tab payload: histogram + fit + skew + normality + transform per column."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MAX_COLS = 8
MAX_ROWS = 5000
BINS = 20


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _histogram(values: np.ndarray) -> Dict[str, Any]:
    counts, edges = np.histogram(values, bins=BINS)
    return {
        "counts": [int(c) for c in counts],
        "edges": [round(float(e), 6) for e in edges],
    }


def build_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    num = df.select_dtypes(include="number")
    def _is_id(c: str) -> bool:
        s = num[c].dropna()
        if len(s) == 0:
            return True
        unique_ratio = s.nunique() / len(s)
        name = str(c).lower()
        looks_like_id = name == "id" or name.endswith("_id") or name.endswith(" id")
        return looks_like_id or (unique_ratio > 0.98 and pd.api.types.is_integer_dtype(s))

    cols: List[str] = [
        c for c in num.columns
        if num[c].dropna().nunique() > 3 and not _is_id(c)
    ][:MAX_COLS]
    if not cols:
        return {"columns": [], "note": "No numeric columns with enough variation."}

    sample = num[cols]
    if len(sample) > MAX_ROWS:
        sample = sample.sample(MAX_ROWS, random_state=42)

    from src.data.advanced_stats import (
        NormalityTester, DistributionFitter,
        SkewnessKurtosisAnalyzer, TransformationAdvisor,
    )

    def run(cls, method):
        try:
            return getattr(cls(), method)(sample)
        except Exception as e:  # one analyzer failing must not kill the tab
            return {"_error": type(e).__name__}

    norm = run(NormalityTester, "test_all")
    fits = run(DistributionFitter, "fit_all")
    skew = run(SkewnessKurtosisAnalyzer, "analyze")
    trans = run(TransformationAdvisor, "advise")

    out: List[Dict[str, Any]] = []
    for c in cols:
        vals = sample[c].dropna().to_numpy(dtype=float)
        item: Dict[str, Any] = {"column": c, "n": int(len(vals)), "histogram": _histogram(vals)}

        n = norm.get(c) if isinstance(norm, dict) else None
        if n is not None and not isinstance(n, str):
            item["normality"] = {
                "is_normal": bool(getattr(n, "is_normal", False)),
                "confidence": getattr(n, "confidence", None),
                "verdict": getattr(n, "verdict", None),
                "shapiro_p": _f(getattr(n, "shapiro_p", None)),
            }

        d = fits.get(c) if isinstance(fits, dict) else None
        if d is not None and not isinstance(d, str):
            item["best_fit"] = {
                "distribution": d.best_distribution,
                "aic": _f(d.aic),
                "ks_p": _f(d.ks_p),
                "params": [_f(p) for p in (d.best_params or [])],
            }

        s = skew.get(c) if isinstance(skew, dict) else None
        if isinstance(s, dict):
            item["shape"] = {
                "skewness": _f(s.get("skewness")),
                "skewness_type": s.get("skewness_type"),
                "kurtosis_excess": _f(s.get("kurtosis_excess")),
                "kurtosis_type": s.get("kurtosis_type"),
                "possibly_bimodal": bool(s.get("possibly_bimodal", False)),
                "heavy_tail_ratio": _f(s.get("heavy_tail_ratio")),
            }

        t = trans.get(c) if isinstance(trans, dict) else None
        if isinstance(t, dict):
            br = t.get("best_result") or {}
            item["transform"] = {
                "needed": bool(t.get("needs_transformation", False)),
                "best": t.get("best_transform"),
                "skewness_after": _f(br.get("skewness_after")),
                "normality_p_after": _f(br.get("normality_p")),
            }
        out.append(item)

    return {"columns": out, "sampled_rows": int(len(sample))}