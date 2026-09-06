"""
bias_detector.py — Nydra v0.6.0
======================================
Advanced Bias Detection Pipeline — 8 blocks, 2000+ lines.

Detects 6 classes of bias across data, model predictions, and feature space:

  1. SensitiveFeatureAnalyzer      — profile sensitive attributes + proxy detection
  2. RepresentationBiasDetector    — group under/over-representation (4/5ths rule)
  3. StatisticalFairnessMetrics    — 8 fairness metrics (demographic parity, equalized odds…)
  4. HistoricalMeasurementBias     — label bias, measurement bias, sampling bias
  5. IntersectionalBiasDetector    — pairwise/3-way group combinations, Simpson's Paradox
  6. FeatureLevelBiasAnalyzer      — per-feature MI, KS-test, SHAP proxy chains
  7. BiasSeverityScorer            — composite 0-100 Bias Score + fix priorities
  8. BiasDetector Master           — one-call API + convenience functions

Version : 0.6.0
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import collections
import itertools
import json
import logging
import math
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("bias_detector")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    np = None           # type: ignore
    NUMPY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    pd = None           # type: ignore
    PANDAS_OK = False

try:
    from scipy.stats import (
        chi2_contingency, chi2, kruskal, ks_2samp,
        pointbiserialr, mannwhitneyu, permutation_test,
        fisher_exact,
    )
    from scipy.special import comb as scipy_comb
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        confusion_matrix, roc_auc_score,
        average_precision_score,
    )
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from fairlearn.metrics import MetricFrame, demographic_parity_difference
    FAIRLEARN_OK = True
except ImportError:
    FAIRLEARN_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "0.6.0"

# Severity labels
SEV_SEVERE      = "severe"
SEV_SIGNIFICANT = "significant"
SEV_MILD        = "mild"
SEV_FAIR        = "fair"

SEV_EMOJI = {
    SEV_SEVERE     : "❌",
    SEV_SIGNIFICANT: "⚠️",
    SEV_MILD       : "🟡",
    SEV_FAIR       : "✅",
}

SEV_COLOR = {
    SEV_SEVERE     : "#e74c3c",
    SEV_SIGNIFICANT: "#e67e22",
    SEV_MILD       : "#f1c40f",
    SEV_FAIR       : "#2ecc71",
}

# Fairness thresholds (industry standard)
FAIRNESS_THRESHOLDS = {
    "demographic_parity_diff"   : 0.10,
    "demographic_parity_ratio"  : 0.80,   # 4/5ths rule
    "equalized_odds_diff"       : 0.10,
    "equal_opportunity_diff"    : 0.10,
    "predictive_parity_diff"    : 0.10,
    "disparate_impact_ratio"    : 0.80,   # 4/5ths rule (EEOC)
    "treatment_equality_diff"   : 0.10,
    "calibration_diff"          : 0.05,
}

# Representation thresholds
REPRESENTATION_RATIO_THRESHOLD = 0.80   # 4/5ths rule
MIN_SAMPLE_SIZE_PER_GROUP = 30

# Proxy correlation thresholds
PROXY_CRAMERS_V_THRESHOLD       = 0.40
PROXY_POINT_BISERIAL_THRESHOLD  = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiasFinding:
    """Standardised bias finding across all detectors."""
    detector    : str
    attribute   : str           # sensitive attribute involved
    group       : str           # specific group value (or "all")
    bias_type   : str           # e.g. "representation", "demographic_parity"
    severity    : str           # severe / significant / mild / fair
    description : str
    evidence    : dict[str, Any]
    confidence  : float         # 0-1
    fix         : str
    metric_value: float = 0.0   # the raw metric number


@dataclass
class DetectorResult:
    """Standardised result from every detector."""
    detector_name : str
    findings      : list[BiasFinding] = field(default_factory=list)
    metrics       : dict[str, Any]    = field(default_factory=dict)
    runtime_sec   : float             = 0.0
    error         : str | None        = None

    def add(self, finding: BiasFinding) -> None:
        self.findings.append(finding)

    @property
    def severe_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEV_SEVERE)

    @property
    def significant_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEV_SIGNIFICANT)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — SensitiveFeatureAnalyzer
# Purpose : Profile sensitive columns + detect proxy features.
#
# Methods :
#   1. Value counts, entropy, majority dominance, missing rate
#   2. Binary vs multi-class classification of each sensitive feature
#   3. Cramér's V — categorical ↔ categorical correlation
#   4. Point-biserial — binary ↔ continuous correlation
#   5. Proxy feature detection: non-sensitive cols w/ high correlation
# ─────────────────────────────────────────────────────────────────────────────

class SensitiveFeatureAnalyzer:
    """
    Profiles each sensitive column and finds proxy features.

    A proxy feature is a non-sensitive column that is so highly
    correlated with a sensitive attribute that it effectively
    encodes sensitive information — causing indirect discrimination.

    Example: 'zip_code' can be a proxy for 'race' in lending.

    Metrics
    -------
    • Shannon entropy — measures diversity of the sensitive attribute
    • Majority dominance ratio — fraction belonging to largest group
    • Cramér's V — correlation between two categorical columns
    • Point-biserial r — correlation between binary and continuous column
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df   = df
        self.sens = sensitive_cols

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(self) -> dict[str, Any]:
        profiles : dict[str, dict] = {}
        proxies  : list[dict]      = []

        for col in self.sens:
            if col not in self.df.columns:
                continue
            profiles[col] = self._profile_column(col)

        # Proxy detection for every non-sensitive column
        non_sens = [c for c in self.df.columns if c not in self.sens]
        for sens_col in self.sens:
            if sens_col not in self.df.columns:
                continue
            for other_col in non_sens:
                proxy_info = self._check_proxy(sens_col, other_col)
                if proxy_info:
                    proxies.append(proxy_info)

        # Sort proxies by correlation strength
        proxies.sort(key=lambda p: -p["correlation"])

        return {
            "profiles"     : profiles,
            "proxy_features": proxies,
            "proxy_count"  : len(proxies),
        }

    # ── column profiling ──────────────────────────────────────────────────────

    def _profile_column(self, col: str) -> dict[str, Any]:
        series       = self.df[col].dropna()
        n_total      = len(self.df)
        n_valid      = len(series)
        value_counts = series.value_counts()
        n_unique     = len(value_counts)
        freqs        = (value_counts / n_valid).to_dict()

        entropy      = _shannon_entropy(list(freqs.values()))
        majority_dom = float(value_counts.iloc[0]) / n_valid if n_valid > 0 else 1.0
        missing_rate = (n_total - n_valid) / n_total

        is_binary    = n_unique == 2
        col_type     = "binary" if is_binary else (
                       "multi-class" if n_unique <= 20 else "continuous-like")

        # Small groups
        small_groups = {
            str(val): int(cnt)
            for val, cnt in value_counts.items()
            if cnt < MIN_SAMPLE_SIZE_PER_GROUP
        }

        return {
            "column"           : col,
            "type"             : col_type,
            "n_unique"         : n_unique,
            "n_valid"          : n_valid,
            "missing_rate"     : round(missing_rate, 4),
            "entropy"          : round(entropy, 4),
            "majority_dominance": round(majority_dom, 4),
            "value_counts"     : {str(k): int(v) for k, v in value_counts.items()},
            "frequencies"      : {str(k): round(float(v), 4) for k, v in freqs.items()},
            "small_groups"     : small_groups,
            "majority_group"   : str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "minority_group"   : str(value_counts.index[-1]) if len(value_counts) > 0 else None,
        }

    # ── proxy detection ───────────────────────────────────────────────────────

    def _check_proxy(self, sens_col: str, other_col: str) -> dict | None:
        """
        Check if other_col is a proxy for sens_col.
        Returns proxy info dict if correlation > threshold, else None.
        """
        try:
            sens_series  = self.df[sens_col].dropna()
            other_series = self.df[other_col].dropna()
            aligned      = pd.concat([sens_series, other_series], axis=1).dropna()
            if len(aligned) < 10:
                return None

            s = aligned[sens_col]
            o = aligned[other_col]

            # Both categorical → Cramér's V
            if _is_categorical(s) and _is_categorical(o):
                v = _cramers_v(s, o)
                method = "cramers_v"
                thresh = PROXY_CRAMERS_V_THRESHOLD
            # Binary sensitive + continuous other → point-biserial
            elif s.nunique() == 2 and _is_numeric(o):
                le  = LabelEncoder() if SKLEARN_OK else None
                s_enc = le.fit_transform(s.astype(str)) if le else s.astype(float)
                if SCIPY_OK:
                    corr, _ = pointbiserialr(s_enc, o.values)
                    v = abs(float(corr))
                else:
                    v = abs(_pearson_fallback(s_enc.astype(float), o.values.astype(float)))
                method = "point_biserial"
                thresh = PROXY_POINT_BISERIAL_THRESHOLD
            # Both numeric → Pearson
            elif _is_numeric(s) and _is_numeric(o):
                v = abs(_pearson_fallback(s.values.astype(float), o.values.astype(float)))
                method = "pearson"
                thresh = PROXY_POINT_BISERIAL_THRESHOLD
            else:
                return None

            if v >= thresh:
                return {
                    "sensitive_col": sens_col,
                    "proxy_col"    : other_col,
                    "correlation"  : round(float(v), 4),
                    "method"       : method,
                    "threshold"    : thresh,
                    "severity"     : SEV_SEVERE if v >= 0.70 else (
                                     SEV_SIGNIFICANT if v >= 0.55 else SEV_MILD),
                }
        except Exception as e:
            logger.debug(f"Proxy check {sens_col}↔{other_col}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — RepresentationBiasDetector
# Purpose : Detect under/over-represented groups.
#
# Methods :
#   1. Chi-squared test: observed vs expected group frequencies
#   2. Representation Ratio (80% / 4/5ths Rule — EEOC standard)
#   3. Effective Representation Score (cross-tab with target)
#   4. Sample size adequacy check (< 30 = unreliable)
#   5. Intersectional representation (pairwise group combinations)
# ─────────────────────────────────────────────────────────────────────────────

class RepresentationBiasDetector:
    """
    Checks whether sensitive groups are proportionally represented.

    The EEOC Four-Fifths Rule (80% Rule):
        If a group's selection rate < 80% of the highest group's rate,
        there is adverse impact → potential bias.

    Effective Representation:
        Checks representation within each target class separately.
        A group may be well-represented overall but absent from
        the positive class → effective under-representation.
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str | None = None,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df
        self.sens   = sensitive_cols
        self.target = target_col

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="RepresentationBiasDetector")

        for sens_col in self.sens:
            if sens_col not in self.df.columns:
                continue

            # 1. Chi-squared
            chi2_info = self._chi2_test(sens_col)

            # 2. Representation ratio (4/5ths rule)
            rep_ratios = self._representation_ratios(sens_col)

            # 3. Effective representation
            eff_rep = self._effective_representation(sens_col)

            # 4. Small groups
            small = self._small_group_check(sens_col)

            # ── findings ────────────────────────────────────────────────────
            if chi2_info.get("p_value", 1.0) < 0.05:
                sev = (SEV_SEVERE if chi2_info.get("effect_size", 0) > 0.3
                       else SEV_SIGNIFICANT if chi2_info.get("effect_size", 0) > 0.1
                       else SEV_MILD)
                result.add(BiasFinding(
                    detector    = "RepresentationBiasDetector",
                    attribute   = sens_col,
                    group       = "all",
                    bias_type   = "representation_imbalance",
                    severity    = sev,
                    description = (
                        f"Group frequencies in '{sens_col}' deviate significantly "
                        f"from uniform distribution "
                        f"(χ²={chi2_info['chi2_stat']:.2f}, "
                        f"p={chi2_info['p_value']:.4f}, "
                        f"Cramér's V={chi2_info['effect_size']:.3f})."
                    ),
                    evidence    = chi2_info,
                    confidence  = min(0.99, 1.0 - chi2_info.get("p_value", 0.05)),
                    fix         = (
                        f"Oversample minority groups in '{sens_col}' using SMOTE "
                        f"or stratified sampling. Consider collecting more data "
                        f"from under-represented groups."
                    ),
                    metric_value= chi2_info.get("effect_size", 0.0),
                ))

            for group_val, ratio_info in rep_ratios.items():
                ratio = ratio_info["representation_ratio"]
                if ratio < REPRESENTATION_RATIO_THRESHOLD:
                    sev = (SEV_SEVERE if ratio < 0.50 else
                           SEV_SIGNIFICANT if ratio < 0.65 else SEV_MILD)
                    result.add(BiasFinding(
                        detector    = "RepresentationBiasDetector",
                        attribute   = sens_col,
                        group       = str(group_val),
                        bias_type   = "four_fifths_rule_violation",
                        severity    = sev,
                        description = (
                            f"Group '{group_val}' in '{sens_col}' has representation "
                            f"ratio {ratio:.3f} (threshold: {REPRESENTATION_RATIO_THRESHOLD}). "
                            f"EEOC 4/5ths Rule violated — potential adverse impact."
                        ),
                        evidence    = ratio_info,
                        confidence  = 0.90,
                        fix         = (
                            f"Increase representation of '{group_val}' group. "
                            f"Use stratified sampling ensuring each group meets "
                            f"the 80% representation threshold."
                        ),
                        metric_value= ratio,
                    ))

            for grp_info in eff_rep:
                result.add(BiasFinding(
                    detector    = "RepresentationBiasDetector",
                    attribute   = sens_col,
                    group       = grp_info["group"],
                    bias_type   = "effective_underrepresentation",
                    severity    = SEV_SIGNIFICANT,
                    description = (
                        f"Group '{grp_info['group']}' is effectively absent from "
                        f"target class '{grp_info['target_class']}' "
                        f"(rate={grp_info['rate']:.3f} vs overall={grp_info['overall_rate']:.3f})."
                    ),
                    evidence    = grp_info,
                    confidence  = 0.80,
                    fix         = (
                        "Investigate data collection process for this group × class "
                        "combination. Ensure equal opportunity in labelling."
                    ),
                    metric_value= grp_info["rate"],
                ))

            for s_info in small:
                result.add(BiasFinding(
                    detector    = "RepresentationBiasDetector",
                    attribute   = sens_col,
                    group       = s_info["group"],
                    bias_type   = "insufficient_sample_size",
                    severity    = SEV_MILD,
                    description = (
                        f"Group '{s_info['group']}' has only {s_info['count']} samples "
                        f"(minimum recommended: {MIN_SAMPLE_SIZE_PER_GROUP}). "
                        f"Statistical tests for this group are unreliable."
                    ),
                    evidence    = s_info,
                    confidence  = 1.0,
                    fix         = (
                        f"Collect at least {MIN_SAMPLE_SIZE_PER_GROUP} samples for "
                        f"'{s_info['group']}' or merge with similar groups."
                    ),
                    metric_value= float(s_info["count"]),
                ))

            result.metrics[sens_col] = {
                "chi2_test"          : chi2_info,
                "representation_ratios": rep_ratios,
                "effective_representation": eff_rep,
                "small_groups"       : small,
            }

        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── methods ───────────────────────────────────────────────────────────────

    def _chi2_test(self, col: str) -> dict[str, Any]:
        """Chi-squared goodness-of-fit vs uniform expected distribution."""
        try:
            counts  = self.df[col].value_counts()
            n       = counts.sum()
            k       = len(counts)
            expected= [n / k] * k
            if SCIPY_OK:
                stat, pval = chi2_contingency(
                    [[o, e] for o, e in zip(counts.values, expected)]
                )[:2]
            else:
                stat = sum((o - e) ** 2 / max(e, 1)
                           for o, e in zip(counts.values, expected))
                df_  = k - 1
                pval = 1.0 - _chi2_cdf_approx(stat, df_)
            cramers = math.sqrt(stat / max(n * (k - 1), 1))
            return {
                "chi2_stat"  : round(float(stat), 4),
                "p_value"    : round(float(pval), 6),
                "effect_size": round(float(cramers), 4),
                "n_groups"   : k,
            }
        except Exception as e:
            return {"error": str(e)}

    def _representation_ratios(self, col: str) -> dict[str, dict]:
        """4/5ths rule per group."""
        counts = self.df[col].value_counts()
        n      = counts.sum()
        rates  = counts / n
        max_rate = float(rates.max())
        result: dict[str, dict] = {}
        for grp, cnt in counts.items():
            rate  = cnt / n
            ratio = float(rate) / max_rate if max_rate > 0 else 1.0
            result[str(grp)] = {
                "group"               : str(grp),
                "count"               : int(cnt),
                "frequency"           : round(float(rate), 4),
                "max_group_frequency" : round(max_rate, 4),
                "representation_ratio": round(ratio, 4),
                "passes_4_5ths_rule"  : ratio >= REPRESENTATION_RATIO_THRESHOLD,
            }
        return result

    def _effective_representation(self, col: str) -> list[dict]:
        """Check group presence within each target class."""
        if not self.target or self.target not in self.df.columns:
            return []
        issues: list[dict] = []
        target_classes = self.df[self.target].unique()
        overall_rates  = (self.df[col].value_counts(normalize=True)).to_dict()

        for cls in target_classes:
            subset = self.df[self.df[self.target] == cls]
            if len(subset) == 0:
                continue
            cls_rates = subset[col].value_counts(normalize=True).to_dict()
            for grp, overall_rate in overall_rates.items():
                cls_rate = cls_rates.get(grp, 0.0)
                # Flag if group is less than 30% of its expected rate in this class
                if cls_rate < overall_rate * 0.30 and overall_rate > 0.05:
                    issues.append({
                        "group"        : str(grp),
                        "target_class" : str(cls),
                        "rate"         : round(float(cls_rate), 4),
                        "overall_rate" : round(float(overall_rate), 4),
                        "ratio"        : round(float(cls_rate) / max(float(overall_rate), 1e-9), 4),
                    })
        return issues

    def _small_group_check(self, col: str) -> list[dict]:
        counts = self.df[col].value_counts()
        return [
            {"group": str(grp), "count": int(cnt)}
            for grp, cnt in counts.items()
            if cnt < MIN_SAMPLE_SIZE_PER_GROUP
        ]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — StatisticalFairnessMetrics
# Purpose : Compute 8 fairness metrics from model predictions.
#
# Metrics :
#   1. Demographic Parity Difference / Ratio
#   2. Equalized Odds Difference
#   3. Equal Opportunity Difference
#   4. Predictive Parity Difference
#   5. Disparate Impact Ratio (EEOC 4/5ths)
#   6. Treatment Equality (FP/FN ratio diff)
#   7. Calibration Difference
#   8. Per-group confusion matrix stats
#
# Falls back to data-level metrics if no predictions provided.
# Uses fairlearn.MetricFrame if available for multi-group extension.
# ─────────────────────────────────────────────────────────────────────────────

class StatisticalFairnessMetrics:
    """
    Computes the full suite of statistical fairness metrics.

    Requires model predictions (y_pred) and true labels (y_true).
    If predictions are unavailable, computes data-level proxies
    using the target label distribution per group.

    Multi-group support: uses fairlearn.MetricFrame when available,
    otherwise computes pairwise (majority vs each minority group).
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str,
        y_pred         : "np.ndarray | None" = None,
        y_prob         : "np.ndarray | None" = None,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df
        self.sens   = sensitive_cols
        self.target = target_col
        self.y_pred = y_pred
        self.y_prob = y_prob

    # ── public API ───────────────────────────────────────────────────────────

    def compute(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="StatisticalFairnessMetrics")

        if self.target not in self.df.columns:
            result.error = f"Target column '{self.target}' not found"
            return result

        y_true = self.df[self.target].values

        # Use predictions if available, else use label column as proxy
        y_pred_use = (self.y_pred if self.y_pred is not None
                      else y_true)

        for sens_col in self.sens:
            if sens_col not in self.df.columns:
                continue
            groups   = self.df[sens_col].values
            metrics  = {}

            # fairlearn MetricFrame (multi-group)
            if FAIRLEARN_OK and self.y_pred is not None:
                mf_metrics = self._fairlearn_metrics(
                    y_true, y_pred_use, groups, sens_col
                )
                metrics["fairlearn"] = mf_metrics

            # Manual per-group metrics
            manual = self._manual_metrics(
                y_true, y_pred_use, groups, sens_col
            )
            metrics["manual"] = manual

            # Calibration
            if self.y_prob is not None:
                cal = self._calibration_metrics(groups, y_true, sens_col)
                metrics["calibration"] = cal

            result.metrics[sens_col] = metrics

            # Generate findings from threshold violations
            self._generate_findings(result, sens_col, manual)

        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── fairlearn integration ──────────────────────────────────────────────────

    def _fairlearn_metrics(
        self,
        y_true : "np.ndarray",
        y_pred : "np.ndarray",
        groups : "np.ndarray",
        col    : str,
    ) -> dict:
        try:
            from fairlearn.metrics import (
                MetricFrame,
                selection_rate, true_positive_rate, false_positive_rate,
                demographic_parity_difference as dpd,
            )
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            mf = MetricFrame(
                metrics={
                    "selection_rate"   : selection_rate,
                    "tpr"              : true_positive_rate,
                    "fpr"              : false_positive_rate,
                    "accuracy"         : accuracy_score,
                },
                y_true=y_true, y_pred=y_pred,
                sensitive_features=groups,
            )
            return {
                "by_group"           : mf.by_group.to_dict() if hasattr(mf.by_group, "to_dict") else {},
                "overall"            : mf.overall.to_dict() if hasattr(mf.overall, "to_dict") else {},
                "demographic_parity_diff": float(mf.difference(method="between_groups")
                                                  .get("selection_rate", 0.0)),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── manual metrics ────────────────────────────────────────────────────────

    def _manual_metrics(
        self,
        y_true : "np.ndarray",
        y_pred : "np.ndarray",
        groups : "np.ndarray",
        col    : str,
    ) -> dict[str, Any]:
        """
        Compute all 8 fairness metrics manually.
        Handles binary classification.
        """
        unique_groups = [g for g in pd.Series(groups).unique()
                         if pd.notna(g)]
        if len(unique_groups) < 2:
            return {"error": "Need at least 2 groups"}

        # Per-group stats
        group_stats: dict[str, dict] = {}
        for grp in unique_groups:
            mask = (pd.Series(groups) == grp).values
            if mask.sum() < 5:
                continue
            yt = y_true[mask]
            yp = y_pred[mask]
            try:
                yt_bin = (yt == 1).astype(int)
                yp_bin = (yp == 1).astype(int)
            except Exception:
                yt_bin = yt.astype(int)
                yp_bin = yp.astype(int)

            tp = int(((yp_bin == 1) & (yt_bin == 1)).sum())
            fp = int(((yp_bin == 1) & (yt_bin == 0)).sum())
            fn = int(((yp_bin == 0) & (yt_bin == 1)).sum())
            tn = int(((yp_bin == 0) & (yt_bin == 0)).sum())
            n  = len(yt)

            tpr  = tp / max(tp + fn, 1)   # recall / sensitivity
            fpr  = fp / max(fp + tn, 1)
            ppv  = tp / max(tp + fp, 1)   # precision
            sel  = yp_bin.mean()          # selection rate
            tr_eq= fp / max(fn, 1)        # treatment equality ratio

            group_stats[str(grp)] = {
                "n"              : n,
                "selection_rate" : round(float(sel), 4),
                "tpr"            : round(tpr, 4),
                "fpr"            : round(fpr, 4),
                "ppv"            : round(ppv, 4),
                "treatment_eq"   : round(tr_eq, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }

        if not group_stats:
            return {"error": "Could not compute group stats"}

        # Find majority group (largest n)
        maj_grp = max(group_stats, key=lambda g: group_stats[g]["n"])
        maj     = group_stats[maj_grp]
        metrics : dict[str, Any] = {
            "majority_group": maj_grp,
            "group_stats"  : group_stats,
        }

        # ── 8 Fairness Metrics ────────────────────────────────────────────────
        diffs: dict[str, dict] = {}
        for grp, stats in group_stats.items():
            if grp == maj_grp:
                continue
            sel_diff = abs(maj["selection_rate"] - stats["selection_rate"])
            sel_ratio= (stats["selection_rate"] /
                        max(maj["selection_rate"], 1e-9))
            eqodds   = max(abs(maj["tpr"] - stats["tpr"]),
                           abs(maj["fpr"] - stats["fpr"]))
            eq_opp   = abs(maj["tpr"] - stats["tpr"])
            pred_par = abs(maj["ppv"] - stats["ppv"])
            di_ratio = (stats["selection_rate"] /
                        max(maj["selection_rate"], 1e-9))
            treat_eq = abs(maj["treatment_eq"] - stats["treatment_eq"])

            diffs[str(grp)] = {
                "demographic_parity_diff"   : round(sel_diff, 4),
                "demographic_parity_ratio"  : round(sel_ratio, 4),
                "equalized_odds_diff"       : round(eqodds, 4),
                "equal_opportunity_diff"    : round(eq_opp, 4),
                "predictive_parity_diff"    : round(pred_par, 4),
                "disparate_impact_ratio"    : round(di_ratio, 4),
                "treatment_equality_diff"   : round(treat_eq, 4),
            }
        metrics["pairwise_vs_majority"] = diffs
        return metrics

    def _calibration_metrics(
        self,
        groups : "np.ndarray",
        y_true : "np.ndarray",
        col    : str,
    ) -> dict[str, Any]:
        """
        Calibration: compare predicted probability to actual positive rate
        per group. Well-calibrated model → predicted prob ≈ actual rate.
        """
        if self.y_prob is None:
            return {}
        try:
            cal: dict[str, dict] = {}
            for grp in pd.Series(groups).unique():
                if pd.isna(grp):
                    continue
                mask     = (pd.Series(groups) == grp).values
                actual   = float(y_true[mask].mean())
                predicted= float(self.y_prob[mask].mean())
                cal[str(grp)] = {
                    "actual_positive_rate"   : round(actual, 4),
                    "mean_predicted_prob"    : round(predicted, 4),
                    "calibration_diff"       : round(abs(actual - predicted), 4),
                    "well_calibrated"        : abs(actual - predicted) < 0.05,
                }
            return cal
        except Exception:
            return {}

    # ── findings ─────────────────────────────────────────────────────────────

    def _generate_findings(
        self,
        result   : DetectorResult,
        sens_col : str,
        manual   : dict,
    ) -> None:
        pairwise = manual.get("pairwise_vs_majority", {})
        for grp, diff_dict in pairwise.items():
            for metric, value in diff_dict.items():
                threshold = FAIRNESS_THRESHOLDS.get(metric)
                if threshold is None:
                    continue
                # For ratio metrics: violation is below threshold
                is_ratio   = metric.endswith("_ratio")
                violated   = (value < threshold) if is_ratio else (value > threshold)
                if not violated:
                    continue
                sev = (SEV_SEVERE      if (is_ratio and value < 0.60) or
                                          (not is_ratio and value > 0.20)
                       else SEV_SIGNIFICANT if (is_ratio and value < 0.70) or
                                              (not is_ratio and value > 0.15)
                       else SEV_MILD)
                result.add(BiasFinding(
                    detector    = "StatisticalFairnessMetrics",
                    attribute   = sens_col,
                    group       = grp,
                    bias_type   = metric,
                    severity    = sev,
                    description = (
                        f"Fairness metric '{metric}' violated for group '{grp}' "
                        f"in '{sens_col}': value={value:.4f}, "
                        f"threshold={'>' if is_ratio else '<'}{threshold}."
                    ),
                    evidence    = {
                        "metric"          : metric,
                        "value"           : value,
                        "threshold"       : threshold,
                        "majority_group"  : manual.get("majority_group", ""),
                        "minority_group"  : grp,
                    },
                    confidence  = 0.90,
                    fix         = _fairness_fix(metric, grp, sens_col),
                    metric_value= float(value),
                ))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — HistoricalMeasurementBias
# Purpose : Detect bias baked into data before model training.
#
# Methods :
#   1. Historical bias: label positive rate per group + permutation test
#   2. Measurement bias: Kruskal-Wallis H-test per feature per group
#   3. Sampling bias: group proportions in train vs test splits
#   4. Confirmation bias: Cramér's V + MI + logistic coeff
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalMeasurementBias:
    """
    Detects bias inherent in the data collection and labelling process.

    Historical bias: systematic difference in target label rates
    across sensitive groups — the data reflects past discrimination.

    Measurement bias: features were measured differently across groups
    (different instruments, different conditions, different proxy accuracy).

    Sampling bias: certain groups are systematically over- or under-sampled
    relative to the true population.
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str | None = None,
        train_df       : "pd.DataFrame | None" = None,
        test_df        : "pd.DataFrame | None" = None,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df
        self.sens   = sensitive_cols
        self.target = target_col
        self.train  = train_df
        self.test   = test_df

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="HistoricalMeasurementBias")

        for sens_col in self.sens:
            if sens_col not in self.df.columns:
                continue

            # 1. Historical bias
            if self.target and self.target in self.df.columns:
                hist_findings = self._historical_bias(sens_col)
                for f in hist_findings:
                    result.add(f)

            # 2. Measurement bias
            meas_findings = self._measurement_bias(sens_col)
            for f in meas_findings:
                result.add(f)

            # 3. Sampling bias
            if self.train is not None and self.test is not None:
                samp_findings = self._sampling_bias(sens_col)
                for f in samp_findings:
                    result.add(f)

            # 4. Confirmation bias score
            if self.target and self.target in self.df.columns:
                conf_info = self._confirmation_bias(sens_col)
                result.metrics[f"{sens_col}_confirmation_bias"] = conf_info

        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: historical bias ─────────────────────────────────────────────

    def _historical_bias(self, sens_col: str) -> list[BiasFinding]:
        """
        Compare positive label rate across groups.
        Uses permutation test to confirm statistical significance.
        """
        findings: list[BiasFinding] = []
        target   = self.df[self.target]
        try:
            # Binary target only
            unique_tgt = target.nunique()
            if unique_tgt != 2:
                return []

            groups      = self.df[sens_col].dropna().unique()
            pos_rates: dict[str, float] = {}
            for grp in groups:
                mask     = self.df[sens_col] == grp
                grp_rate = float(target[mask].mean())
                pos_rates[str(grp)] = grp_rate

            if not pos_rates:
                return []

            max_rate = max(pos_rates.values())
            min_rate = min(pos_rates.values())
            rate_diff= max_rate - min_rate
            max_grp  = max(pos_rates, key=pos_rates.__getitem__)
            min_grp  = min(pos_rates, key=pos_rates.__getitem__)

            # Permutation test for significance
            p_value = self._permutation_test_label_rate(sens_col)

            if rate_diff > 0.05 and p_value < 0.05:
                sev = (SEV_SEVERE      if rate_diff > 0.25 else
                       SEV_SIGNIFICANT if rate_diff > 0.15 else SEV_MILD)
                findings.append(BiasFinding(
                    detector    = "HistoricalMeasurementBias",
                    attribute   = sens_col,
                    group       = min_grp,
                    bias_type   = "historical_label_bias",
                    severity    = sev,
                    description = (
                        f"Historical label bias in '{sens_col}': "
                        f"'{max_grp}' has positive rate {max_rate:.3f} vs "
                        f"'{min_grp}' at {min_rate:.3f} "
                        f"(diff={rate_diff:.3f}, p={p_value:.4f}). "
                        f"Labels reflect past discriminatory decisions."
                    ),
                    evidence    = {
                        "positive_rates"   : pos_rates,
                        "rate_difference"  : round(rate_diff, 4),
                        "p_value"          : round(p_value, 6),
                        "favoured_group"   : max_grp,
                        "disadvantaged_group": min_grp,
                    },
                    confidence  = min(0.99, 1.0 - p_value),
                    fix         = (
                        "Apply re-weighting or re-labelling to correct historical label bias. "
                        "Consider fairness-aware training objectives "
                        "(e.g., adversarial debiasing)."
                    ),
                    metric_value= rate_diff,
                ))
        except Exception as e:
            logger.debug(f"Historical bias error: {e}")
        return findings

    def _permutation_test_label_rate(self, sens_col: str, n_perm: int = 1000) -> float:
        """
        Permutation test: shuffle sensitive labels, recompute rate diff.
        p-value = fraction of permutations with diff >= observed diff.
        """
        try:
            groups      = self.df[sens_col].dropna()
            target      = self.df.loc[groups.index, self.target]
            unique_grps = groups.unique()
            if len(unique_grps) < 2:
                return 1.0

            # Observed difference
            rates    = {g: float(target[groups == g].mean()) for g in unique_grps}
            obs_diff = max(rates.values()) - min(rates.values())

            # Permutation
            target_arr = target.values.copy()
            count_ge   = 0
            rng = _rng()
            for _ in range(n_perm):
                perm   = rng.permutation(len(target_arr))
                t_perm = target_arr[perm]
                grp_arr= groups.values
                r      = {g: float(t_perm[grp_arr == g].mean())
                          for g in unique_grps if (grp_arr == g).sum() > 0}
                d = max(r.values(), default=0) - min(r.values(), default=0)
                if d >= obs_diff:
                    count_ge += 1

            return count_ge / n_perm
        except Exception:
            return 1.0

    # ── method 2: measurement bias ────────────────────────────────────────────

    def _measurement_bias(self, sens_col: str) -> list[BiasFinding]:
        """
        Kruskal-Wallis H-test per numeric feature across groups.
        Significant H → feature distribution differs by group
        → possible measurement bias.
        """
        if not SCIPY_OK:
            return []
        findings: list[BiasFinding] = []
        numeric_cols = [c for c in self.df.columns
                        if c != sens_col and c != self.target
                        and self.df[c].dtype.kind in ("i", "u", "f")]
        groups = self.df[sens_col].dropna().unique()
        if len(groups) < 2:
            return []

        for feat_col in numeric_cols[:30]:   # cap for speed
            try:
                group_data = [
                    self.df.loc[self.df[sens_col] == g, feat_col].dropna().values
                    for g in groups
                    if (self.df[sens_col] == g).sum() >= 5
                ]
                if len(group_data) < 2:
                    continue
                stat, pval = kruskal(*group_data)
                if pval < 0.01:
                    # Effect size: η² = (H - k + 1) / (n - k)
                    n = sum(len(d) for d in group_data)
                    k = len(group_data)
                    eta2 = max(0, (stat - k + 1) / max(n - k, 1))
                    if eta2 > 0.06:   # medium effect size
                        findings.append(BiasFinding(
                            detector    = "HistoricalMeasurementBias",
                            attribute   = sens_col,
                            group       = "all",
                            bias_type   = "measurement_bias",
                            severity    = SEV_SIGNIFICANT if eta2 > 0.14 else SEV_MILD,
                            description = (
                                f"Feature '{feat_col}' has significantly different "
                                f"distributions across '{sens_col}' groups "
                                f"(H={stat:.2f}, p={pval:.4f}, η²={eta2:.3f}). "
                                f"Possible measurement bias."
                            ),
                            evidence    = {
                                "feature"   : feat_col,
                                "H_stat"    : round(float(stat), 4),
                                "p_value"   : round(float(pval), 6),
                                "eta_squared": round(float(eta2), 4),
                            },
                            confidence  = min(0.99, 1.0 - pval),
                            fix         = (
                                f"Investigate how '{feat_col}' was collected across groups. "
                                f"Apply group-specific normalisation if measurement conditions differ."
                            ),
                            metric_value= float(eta2),
                        ))
            except Exception:
                pass
        return findings

    # ── method 3: sampling bias ───────────────────────────────────────────────

    def _sampling_bias(self, sens_col: str) -> list[BiasFinding]:
        """
        Compare group proportions in train vs test.
        Non-stratified splits → sampling bias.
        """
        findings: list[BiasFinding] = []
        try:
            if sens_col not in self.train.columns or sens_col not in self.test.columns:
                return []
            train_rates = (self.train[sens_col].value_counts(normalize=True)
                           .to_dict())
            test_rates  = (self.test[sens_col].value_counts(normalize=True)
                           .to_dict())
            for grp in set(list(train_rates.keys()) + list(test_rates.keys())):
                tr = train_rates.get(grp, 0.0)
                te = test_rates.get(grp, 0.0)
                drift = abs(tr - te)
                if drift > 0.05:
                    findings.append(BiasFinding(
                        detector    = "HistoricalMeasurementBias",
                        attribute   = sens_col,
                        group       = str(grp),
                        bias_type   = "sampling_bias",
                        severity    = SEV_SIGNIFICANT if drift > 0.10 else SEV_MILD,
                        description = (
                            f"Group '{grp}' has different representation in train "
                            f"({tr:.3f}) vs test ({te:.3f}) — drift={drift:.3f}. "
                            f"Split is not stratified."
                        ),
                        evidence    = {
                            "group"        : str(grp),
                            "train_rate"   : round(float(tr), 4),
                            "test_rate"    : round(float(te), 4),
                            "drift"        : round(float(drift), 4),
                        },
                        confidence  = 0.85,
                        fix         = (
                            f"Use stratified splitting: "
                            f"train_test_split(stratify=df['{sens_col}'])"
                        ),
                        metric_value= float(drift),
                    ))
        except Exception as e:
            logger.debug(f"Sampling bias error: {e}")
        return findings

    # ── method 4: confirmation bias ───────────────────────────────────────────

    def _confirmation_bias(self, sens_col: str) -> dict[str, Any]:
        """
        Multi-method confirmation bias score:
        Cramér's V, mutual information, logistic regression coefficient.
        """
        result: dict[str, Any] = {}
        try:
            s = self.df[sens_col].dropna()
            t = self.df.loc[s.index, self.target].dropna()
            aligned = pd.concat([s, t], axis=1).dropna()

            # Cramér's V
            v = _cramers_v(aligned[sens_col], aligned[self.target])
            result["cramers_v"] = round(float(v), 4)

            # Mutual information
            if SKLEARN_OK and NUMPY_OK:
                le = LabelEncoder()
                s_enc = le.fit_transform(aligned[sens_col].astype(str))
                t_enc = le.fit_transform(aligned[self.target].astype(str))
                mi = mutual_info_classif(
                    s_enc.reshape(-1, 1), t_enc, random_state=42
                )[0]
                result["mutual_information"] = round(float(mi), 4)

                # Logistic regression coefficient
                try:
                    lr = LogisticRegression(random_state=42, max_iter=200)
                    lr.fit(s_enc.reshape(-1, 1), t_enc)
                    result["logistic_coeff"] = round(float(abs(lr.coef_[0][0])), 4)
                except Exception:
                    pass

            # Overall confirmation bias score (0-1)
            cbs = (v * 0.4
                   + result.get("mutual_information", 0) * 0.3
                   + result.get("logistic_coeff", 0) * 0.3)
            result["confirmation_bias_score"] = round(float(min(cbs, 1.0)), 4)

        except Exception as e:
            result["error"] = str(e)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — IntersectionalBiasDetector
# Purpose : Detect bias only visible in combinations of sensitive attributes.
#
# Methods :
#   1. Pairwise (and 3-way) group combination construction
#   2. Positive label rate per intersectional group
#   3. Intersectional Bias Score = max deviation from overall rate
#   4. Cochran-Mantel-Haenszel test for stratified comparison
#   5. Simpson's Paradox detection
# ─────────────────────────────────────────────────────────────────────────────

class IntersectionalBiasDetector:
    """
    Finds bias that only emerges when combining multiple sensitive attributes.

    Example: gender alone shows no bias, age alone shows no bias,
    but young women have 40% lower loan approval rate → intersectional bias.

    Simpson's Paradox detection: a trend that appears in aggregate
    reverses within subgroups. Classic example: a treatment appears
    beneficial overall but harmful for every individual subgroup.
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str | None = None,
        max_order      : int = 2,   # 2=pairwise, 3=triplets
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df        = df
        self.sens      = sensitive_cols
        self.target    = target_col
        self.max_order = max_order

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="IntersectionalBiasDetector")

        if len(self.sens) < 2:
            result.error = "Need ≥ 2 sensitive columns for intersectional analysis"
            return result

        # Build intersectional groups
        groups_table = self._build_intersectional_table()
        result.metrics["intersectional_table"] = groups_table

        # Intersectional Bias Score
        if self.target and self.target in self.df.columns:
            ib_findings = self._intersectional_bias_score(groups_table)
            for f in ib_findings:
                result.add(f)

            # CMH test
            cmh_findings = self._cmh_test()
            for f in cmh_findings:
                result.add(f)

            # Simpson's Paradox
            simpson_findings = self._simpsons_paradox()
            for f in simpson_findings:
                result.add(f)

        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: intersectional table ───────────────────────────────────────

    def _build_intersectional_table(self) -> list[dict]:
        """
        Enumerate all combinations up to max_order and compute group stats.
        """
        table: list[dict] = []
        cols_to_combine = self.sens[:4]   # cap at 4 to avoid explosion

        for order in range(2, self.max_order + 1):
            for combo in itertools.combinations(cols_to_combine, order):
                try:
                    grp_df = self.df[list(combo)].dropna()
                    value_combos = grp_df.groupby(list(combo)).size().reset_index(name="count")
                    for _, row in value_combos.iterrows():
                        combo_key = " & ".join(
                            f"{c}={row[c]}" for c in combo
                        )
                        mask = pd.Series([True] * len(self.df))
                        for c in combo:
                            mask &= (self.df[c] == row[c])
                        n = int(mask.sum())
                        entry: dict[str, Any] = {
                            "combination"  : combo_key,
                            "attributes"   : list(combo),
                            "values"       : {c: str(row[c]) for c in combo},
                            "count"        : n,
                            "pct"          : round(n / max(len(self.df), 1) * 100, 2),
                        }
                        if (self.target and self.target in self.df.columns
                                and n >= 5):
                            pos_rate = float(
                                self.df.loc[mask, self.target].mean()
                            )
                            entry["positive_rate"] = round(pos_rate, 4)
                        table.append(entry)
                except Exception:
                    pass
        return table

    # ── method 2: intersectional bias score ───────────────────────────────────

    def _intersectional_bias_score(
        self, groups_table: list[dict]
    ) -> list[BiasFinding]:
        findings: list[BiasFinding] = []
        overall_rate = float(self.df[self.target].mean())
        entries_with_rate = [g for g in groups_table
                             if "positive_rate" in g and g["count"] >= 10]
        if not entries_with_rate:
            return []

        # Most disadvantaged group = lowest positive_rate
        worst  = min(entries_with_rate, key=lambda g: g["positive_rate"])
        best   = max(entries_with_rate, key=lambda g: g["positive_rate"])
        ib_score = abs(worst["positive_rate"] - overall_rate)

        if ib_score > 0.10:
            sev = (SEV_SEVERE      if ib_score > 0.25 else
                   SEV_SIGNIFICANT if ib_score > 0.15 else SEV_MILD)
            findings.append(BiasFinding(
                detector    = "IntersectionalBiasDetector",
                attribute   = " & ".join(self.sens[:2]),
                group       = worst["combination"],
                bias_type   = "intersectional_bias",
                severity    = sev,
                description = (
                    f"Intersectional bias detected. Most disadvantaged group: "
                    f"'{worst['combination']}' with positive rate {worst['positive_rate']:.3f} "
                    f"vs overall rate {overall_rate:.3f} "
                    f"(Intersectional Bias Score={ib_score:.3f}). "
                    f"Most advantaged: '{best['combination']}' "
                    f"({best['positive_rate']:.3f})."
                ),
                evidence    = {
                    "overall_rate"        : round(overall_rate, 4),
                    "worst_group"         : worst,
                    "best_group"          : best,
                    "intersectional_bias_score": round(ib_score, 4),
                    "n_groups_analysed"   : len(entries_with_rate),
                },
                confidence  = 0.85 if worst["count"] >= 30 else 0.60,
                fix         = (
                    f"Implement intersectional fairness constraints targeting "
                    f"'{worst['combination']}'. "
                    f"Collect additional data for this subgroup."
                ),
                metric_value= ib_score,
            ))
        return findings

    # ── method 3: Cochran-Mantel-Haenszel test ────────────────────────────────

    def _cmh_test(self) -> list[BiasFinding]:
        """
        CMH test for stratified group comparison.
        Tests whether the association between a sensitive attribute
        and outcome is consistent across strata defined by another attribute.
        """
        if len(self.sens) < 2 or not SCIPY_OK:
            return []
        findings: list[BiasFinding] = []
        try:
            s1, s2 = self.sens[0], self.sens[1]
            if s1 not in self.df or s2 not in self.df:
                return []
            target = self.df[self.target]
            strata = self.df[s2].dropna().unique()

            # Mantel-Haenszel common odds ratio approximation
            mh_num, mh_den = 0.0, 0.0
            chi2_sum = 0.0
            for stratum in strata:
                mask = self.df[s2] == stratum
                sub  = self.df[mask]
                if len(sub) < 5:
                    continue
                ct = pd.crosstab(sub[s1], target)
                if ct.shape != (2, 2):
                    continue
                a, b = float(ct.iloc[0, 1]), float(ct.iloc[0, 0])
                c, d = float(ct.iloc[1, 1]), float(ct.iloc[1, 0])
                n    = a + b + c + d
                mh_num += a * d / n
                mh_den += b * c / n
                # stratum-level chi2
                if SCIPY_OK:
                    try:
                        chi2_s, _, _, _ = chi2_contingency(ct)
                        chi2_sum += chi2_s
                    except Exception:
                        pass

            if mh_den == 0:
                return []
            common_or = mh_num / mh_den
            # CMH chi2 (simplified)
            p_approx = _chi2_cdf_approx(chi2_sum, len(strata))
            if abs(common_or - 1.0) > 0.30 and chi2_sum > 3.84:
                sev = SEV_SIGNIFICANT if abs(common_or - 1.0) > 0.50 else SEV_MILD
                findings.append(BiasFinding(
                    detector    = "IntersectionalBiasDetector",
                    attribute   = f"{s1} × {s2}",
                    group       = "stratified",
                    bias_type   = "cmh_stratified_bias",
                    severity    = sev,
                    description = (
                        f"Cochran-Mantel-Haenszel test: Association between "
                        f"'{s1}' and target is not homogeneous across '{s2}' strata. "
                        f"Common Odds Ratio = {common_or:.3f} "
                        f"(1.0 = no bias). CMH χ²={chi2_sum:.2f}."
                    ),
                    evidence    = {
                        "common_odds_ratio": round(common_or, 4),
                        "cmh_chi2"         : round(chi2_sum, 4),
                        "n_strata"         : len(strata),
                    },
                    confidence  = 0.75,
                    fix         = (
                        "The bias magnitude varies across subgroups. "
                        "Apply stratum-specific fairness interventions."
                    ),
                    metric_value= abs(common_or - 1.0),
                ))
        except Exception as e:
            logger.debug(f"CMH error: {e}")
        return findings

    # ── method 4: Simpson's Paradox ───────────────────────────────────────────

    def _simpsons_paradox(self) -> list[BiasFinding]:
        """
        Detect Simpson's Paradox: aggregate trend reverses in subgroups.

        Algorithm:
        1. Compute overall correlation between sensitive attribute and target.
        2. Compute per-stratum (second sensitive attribute) correlation.
        3. If sign flips in majority of strata → Simpson's Paradox.
        """
        if len(self.sens) < 2:
            return []
        findings: list[BiasFinding] = []
        s1, s2 = self.sens[0], self.sens[1]

        try:
            if s1 not in self.df or s2 not in self.df:
                return []
            target = self.df[self.target]

            # Encode s1 as binary (majority vs rest)
            s1_vals   = self.df[s1].dropna()
            maj_grp   = s1_vals.value_counts().index[0]
            s1_binary = (self.df[s1] == maj_grp).astype(int)

            aligned = pd.concat([s1_binary, self.df[s2], target], axis=1).dropna()
            aligned.columns = ["s1_bin", "s2", "target"]
            if len(aligned) < 20:
                return []

            # Overall correlation
            overall_corr = _pearson_fallback(
                aligned["s1_bin"].values.astype(float),
                aligned["target"].values.astype(float),
            )

            # Per-stratum correlation
            strata_corrs: dict[str, float] = {}
            for stratum in aligned["s2"].unique():
                sub = aligned[aligned["s2"] == stratum]
                if len(sub) < 10:
                    continue
                corr = _pearson_fallback(
                    sub["s1_bin"].values.astype(float),
                    sub["target"].values.astype(float),
                )
                strata_corrs[str(stratum)] = round(corr, 4)

            # Count sign reversals
            if not strata_corrs:
                return []
            reversals = sum(
                1 for c in strata_corrs.values()
                if (c > 0) != (overall_corr > 0)
            )
            reversal_rate = reversals / len(strata_corrs)

            if reversal_rate >= 0.60:
                findings.append(BiasFinding(
                    detector    = "IntersectionalBiasDetector",
                    attribute   = f"{s1} × {s2}",
                    group       = "all",
                    bias_type   = "simpsons_paradox",
                    severity    = SEV_SIGNIFICANT,
                    description = (
                        f"Simpson's Paradox detected between '{s1}' and target "
                        f"when stratified by '{s2}'. "
                        f"Aggregate correlation = {overall_corr:.3f}, "
                        f"but reverses in {reversals}/{len(strata_corrs)} strata. "
                        f"Aggregate statistics are misleading."
                    ),
                    evidence    = {
                        "overall_correlation" : round(overall_corr, 4),
                        "strata_correlations" : strata_corrs,
                        "reversal_count"      : reversals,
                        "reversal_rate"       : round(reversal_rate, 4),
                    },
                    confidence  = 0.80,
                    fix         = (
                        f"Always stratify analyses by '{s2}' when studying "
                        f"'{s1}'. Report subgroup-level results, not just aggregates."
                    ),
                    metric_value= reversal_rate,
                ))
        except Exception as e:
            logger.debug(f"Simpson's Paradox error: {e}")
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — FeatureLevelBiasAnalyzer
# Purpose : Check whether individual features carry or amplify bias.
#
# Methods :
#   1. Mutual information between features and sensitive attributes
#   2. KS-test (continuous) + Chi-squared (categorical) per feature per group
#   3. SHAP/permutation importance × MI proxy chains
#   4. Correlation network: feature × sensitive-attr matrix
# ─────────────────────────────────────────────────────────────────────────────

class FeatureLevelBiasAnalyzer:
    """
    Analyses individual features for bias-carrying potential.

    A feature can carry bias in two ways:
    1. Direct proxy: highly correlated with a sensitive attribute.
    2. Disparate distribution: the feature has very different
       statistical distributions across sensitive groups,
       causing the model to treat groups differently.

    Disparate distribution detection:
    - Continuous features: KS test (Kolmogorov-Smirnov)
    - Categorical features: Chi-squared test of independence
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str | None = None,
        y_pred         : "np.ndarray | None" = None,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df
        self.sens   = sensitive_cols
        self.target = target_col
        self.y_pred = y_pred

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="FeatureLevelBiasAnalyzer")

        non_sens = [c for c in self.df.columns
                    if c not in self.sens and c != self.target]
        if not non_sens:
            result.error = "No non-sensitive features to analyse"
            return result

        for sens_col in self.sens:
            if sens_col not in self.df.columns:
                continue

            # 1. MI per feature vs sensitive attribute
            mi_scores = self._mi_scores(non_sens, sens_col)

            # 2. Disparate distributions
            disp_findings = self._disparate_distributions(non_sens, sens_col)

            # 3. SHAP proxy chains
            shap_findings = self._shap_proxy_chain(non_sens, sens_col, mi_scores)

            # 4. Correlation network
            corr_net = self._correlation_network(non_sens, sens_col)

            result.metrics[f"{sens_col}_mi_scores"]     = mi_scores
            result.metrics[f"{sens_col}_corr_network"]  = corr_net

            for f in disp_findings + shap_findings:
                result.add(f)

        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: mutual information ─────────────────────────────────────────

    def _mi_scores(self, feat_cols: list[str], sens_col: str) -> dict[str, float]:
        if not SKLEARN_OK or not NUMPY_OK:
            return {}
        try:
            sub = self.df[feat_cols + [sens_col]].dropna()
            X   = sub[feat_cols].values.astype(float)
            le  = LabelEncoder()
            y   = le.fit_transform(sub[sens_col].astype(str))
            mi  = mutual_info_classif(X, y, random_state=42)
            max_mi = max(mi) if len(mi) > 0 else 1.0
            return {col: round(float(sc) / max(max_mi, 1e-9), 4)
                    for col, sc in zip(feat_cols, mi)}
        except Exception:
            return {}

    # ── method 2: disparate distributions ────────────────────────────────────

    def _disparate_distributions(
        self, feat_cols: list[str], sens_col: str
    ) -> list[BiasFinding]:
        findings: list[BiasFinding] = []
        groups  = self.df[sens_col].dropna().unique()
        if len(groups) < 2:
            return []

        # Get majority and minority groups
        counts  = self.df[sens_col].value_counts()
        maj_grp = counts.index[0]
        min_grp = counts.index[-1]

        for col in feat_cols[:25]:   # cap for speed
            try:
                if _is_numeric(self.df[col]) and SCIPY_OK:
                    # KS test
                    x1 = self.df.loc[self.df[sens_col] == maj_grp, col].dropna().values
                    x2 = self.df.loc[self.df[sens_col] == min_grp, col].dropna().values
                    if len(x1) < 5 or len(x2) < 5:
                        continue
                    stat, pval = ks_2samp(x1, x2)
                    if pval < 0.01 and stat > 0.15:
                        findings.append(BiasFinding(
                            detector    = "FeatureLevelBiasAnalyzer",
                            attribute   = sens_col,
                            group       = str(min_grp),
                            bias_type   = "disparate_feature_distribution",
                            severity    = SEV_SIGNIFICANT if stat > 0.35 else SEV_MILD,
                            description = (
                                f"Feature '{col}' has significantly different distributions "
                                f"between '{maj_grp}' and '{min_grp}' "
                                f"(KS={stat:.3f}, p={pval:.4f}). "
                                f"Model may learn group-specific patterns from this feature."
                            ),
                            evidence    = {
                                "feature"   : col,
                                "ks_stat"   : round(float(stat), 4),
                                "p_value"   : round(float(pval), 6),
                                "group_1"   : str(maj_grp),
                                "group_2"   : str(min_grp),
                                "mean_g1"   : round(float(x1.mean()), 4),
                                "mean_g2"   : round(float(x2.mean()), 4),
                            },
                            confidence  = min(0.99, 1.0 - pval),
                            fix         = (
                                f"Consider removing '{col}' or applying "
                                f"group-specific normalisation. "
                                f"Check whether the difference reflects real-world "
                                f"variation or measurement artifact."
                            ),
                            metric_value= float(stat),
                        ))
                elif _is_categorical(self.df[col]) and SCIPY_OK:
                    ct    = pd.crosstab(self.df[sens_col], self.df[col])
                    if ct.shape[0] < 2 or ct.shape[1] < 2:
                        continue
                    stat, pval, _, _ = chi2_contingency(ct)
                    v = _cramers_v(self.df[sens_col], self.df[col])
                    if pval < 0.01 and v > 0.15:
                        findings.append(BiasFinding(
                            detector    = "FeatureLevelBiasAnalyzer",
                            attribute   = sens_col,
                            group       = "all",
                            bias_type   = "disparate_feature_distribution",
                            severity    = SEV_SIGNIFICANT if v > 0.35 else SEV_MILD,
                            description = (
                                f"Categorical feature '{col}' distribution differs "
                                f"across '{sens_col}' groups "
                                f"(χ²={stat:.2f}, p={pval:.4f}, V={v:.3f})."
                            ),
                            evidence    = {
                                "feature"  : col,
                                "chi2_stat": round(float(stat), 4),
                                "p_value"  : round(float(pval), 6),
                                "cramers_v": round(float(v), 4),
                            },
                            confidence  = 0.85,
                            fix         = f"Investigate feature '{col}' for proxy bias.",
                            metric_value= float(v),
                        ))
            except Exception:
                pass
        return findings

    # ── method 3: SHAP proxy chain ────────────────────────────────────────────

    def _shap_proxy_chain(
        self,
        feat_cols : list[str],
        sens_col  : str,
        mi_scores : dict[str, float],
    ) -> list[BiasFinding]:
        """
        Double-flag features with BOTH high MI with sensitive attr
        AND high model importance (SHAP or permutation).
        """
        if not (SKLEARN_OK and NUMPY_OK and self.target
                and self.target in self.df.columns):
            return []
        findings: list[BiasFinding] = []
        try:
            sub = self.df[feat_cols + [self.target]].dropna()
            if len(sub) < 20:
                return []

            X   = sub[feat_cols].values.astype(float)
            le  = LabelEncoder()
            y   = le.fit_transform(sub[self.target].astype(str))

            model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            model.fit(X, y)
            importances = model.feature_importances_
            max_imp     = max(importances) if len(importances) > 0 else 1.0

            for i, col in enumerate(feat_cols):
                rel_imp = importances[i] / max(max_imp, 1e-9)
                rel_mi  = mi_scores.get(col, 0.0)
                # High importance AND high MI with sensitive attr = proxy chain
                if rel_imp > 0.15 and rel_mi > 0.40:
                    findings.append(BiasFinding(
                        detector    = "FeatureLevelBiasAnalyzer",
                        attribute   = sens_col,
                        group       = "all",
                        bias_type   = "shap_proxy_chain",
                        severity    = SEV_SEVERE if (rel_imp > 0.30 and rel_mi > 0.60)
                                      else SEV_SIGNIFICANT,
                        description = (
                            f"Feature '{col}' is both highly important for the model "
                            f"(relative importance={rel_imp:.3f}) AND a strong proxy for "
                            f"sensitive attribute '{sens_col}' (MI ratio={rel_mi:.3f}). "
                            f"This creates an indirect discrimination pathway."
                        ),
                        evidence    = {
                            "feature"            : col,
                            "relative_importance": round(rel_imp, 4),
                            "mi_ratio"           : round(rel_mi, 4),
                        },
                        confidence  = 0.85,
                        fix         = (
                            f"Remove '{col}' or apply adversarial debiasing to "
                            f"disentangle it from '{sens_col}'. "
                            f"Use SHAP interaction values to trace the discrimination path."
                        ),
                        metric_value= rel_imp * rel_mi,
                    ))
        except Exception as e:
            logger.debug(f"SHAP proxy chain error: {e}")
        return findings

    # ── method 4: correlation network ─────────────────────────────────────────

    def _correlation_network(
        self, feat_cols: list[str], sens_col: str
    ) -> dict[str, float]:
        """
        Compute correlation of each feature with the sensitive attribute.
        Returns a dict usable as a network adjacency list.
        """
        result: dict[str, float] = {}
        s = self.df[sens_col]
        is_bin = s.nunique() == 2

        for col in feat_cols[:30]:
            try:
                o = self.df[col]
                if _is_numeric(o) and is_bin:
                    le   = LabelEncoder() if SKLEARN_OK else None
                    s_enc= (le.fit_transform(s.astype(str)) if le
                            else s.astype(float))
                    aligned = pd.concat(
                        [pd.Series(s_enc, index=s.index), o], axis=1
                    ).dropna()
                    if len(aligned) < 5:
                        continue
                    if SCIPY_OK:
                        corr, _ = pointbiserialr(
                            aligned.iloc[:, 0].values,
                            aligned.iloc[:, 1].values,
                        )
                    else:
                        corr = _pearson_fallback(
                            aligned.iloc[:, 0].values.astype(float),
                            aligned.iloc[:, 1].values.astype(float),
                        )
                    result[col] = round(abs(float(corr)), 4)
                elif _is_categorical(o) and _is_categorical(s):
                    v = _cramers_v(s, o)
                    result[col] = round(float(v), 4)
            except Exception:
                pass
        return dict(sorted(result.items(), key=lambda x: -x[1])[:20])


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — BiasSeverityScorer
# Purpose : Aggregate all findings into a composite Bias Score (0-100).
#
# Weights :
#   Representation      20%
#   Statistical Fairness 30%
#   Historical Bias      20%
#   Intersectional       15%
#   Feature-Level        15%
#
# Output  : score, grade, per-attribute severity, ordered fix priority list
# ─────────────────────────────────────────────────────────────────────────────

class BiasSeverityScorer:
    """
    Aggregates all bias findings into a single Bias Score (0-100).

    Higher score = more biased (opposite of quality score convention).
    0   = completely fair (unlikely)
    100 = maximally biased (all metrics maximally violated)

    Alternatively: Fairness Score = 100 - Bias Score
    """

    WEIGHTS = {
        "RepresentationBiasDetector"       : 0.20,
        "StatisticalFairnessMetrics"       : 0.30,
        "HistoricalMeasurementBias"        : 0.20,
        "IntersectionalBiasDetector"       : 0.15,
        "FeatureLevelBiasAnalyzer"         : 0.15,
    }

    SEV_SCORE = {
        SEV_SEVERE     : 25.0,
        SEV_SIGNIFICANT: 15.0,
        SEV_MILD       : 7.0,
        SEV_FAIR       : 0.0,
    }

    BIAS_GRADES = [
        (0,  10,  SEV_FAIR,       "Fair ✅",           "No significant bias detected"),
        (10, 25,  SEV_MILD,       "Mild Bias 🟡",      "Minor bias — monitor closely"),
        (25, 50,  SEV_SIGNIFICANT,"Significant Bias ⚠️","Bias confirmed — intervention needed"),
        (50, 100, SEV_SEVERE,     "Severe Bias ❌",     "Severe bias — model should not be deployed"),
    ]

    def __init__(self, detector_results: list[DetectorResult]) -> None:
        self.results = detector_results

    # ── public API ───────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        all_findings = [f for r in self.results for f in r.findings]

        # Per-detector bias contribution
        det_scores: dict[str, float] = {}
        for result in self.results:
            penalty = sum(
                self.SEV_SCORE.get(f.severity, 0) * f.confidence
                for f in result.findings
            )
            weight = self.WEIGHTS.get(result.detector_name, 0.10)
            det_scores[result.detector_name] = min(100.0, penalty)

        # Weighted composite bias score
        total_weight = sum(
            self.WEIGHTS.get(r.detector_name, 0.10)
            for r in self.results
        )
        bias_score = sum(
            det_scores.get(r.detector_name, 0.0)
            * self.WEIGHTS.get(r.detector_name, 0.10)
            for r in self.results
        ) / max(total_weight, 1e-9)

        bias_score = min(100.0, max(0.0, bias_score))
        fairness_score = round(100.0 - bias_score, 2)
        grade, label, description = self._grade(bias_score)

        # Per-attribute severity
        attr_severity = self._per_attribute_severity(all_findings)

        # Fix priority list
        fix_priority = self._fix_priority(all_findings)

        # Severity distribution
        sev_dist = dict(collections.Counter(f.severity for f in all_findings))

        return {
            "bias_score"        : round(bias_score, 2),
            "fairness_score"    : fairness_score,
            "grade"             : grade,
            "label"             : label,
            "description"       : description,
            "severity_distribution": sev_dist,
            "per_detector_scores"  : {k: round(v, 2) for k, v in det_scores.items()},
            "per_attribute_severity": attr_severity,
            "fix_priority"      : fix_priority,
            "total_findings"    : len(all_findings),
            "severe_count"      : sum(1 for f in all_findings if f.severity == SEV_SEVERE),
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _grade(self, score: float) -> tuple[str, str, str]:
        for lo, hi, grade, label, desc in self.BIAS_GRADES:
            if lo <= score < hi:
                return grade, label, desc
        return SEV_SEVERE, "Severe Bias ❌", "Severe bias detected"

    def _per_attribute_severity(
        self, findings: list[BiasFinding]
    ) -> dict[str, dict]:
        attr_map: dict[str, list[BiasFinding]] = collections.defaultdict(list)
        for f in findings:
            attr_map[f.attribute].append(f)

        result: dict[str, dict] = {}
        for attr, attr_findings in attr_map.items():
            sev_counts = collections.Counter(f.severity for f in attr_findings)
            worst = min(
                attr_findings,
                key=lambda f: [SEV_SEVERE, SEV_SIGNIFICANT, SEV_MILD, SEV_FAIR].index(f.severity),
                default=None,
            )
            result[attr] = {
                "total_findings"   : len(attr_findings),
                "severity_counts"  : dict(sev_counts),
                "worst_severity"   : worst.severity if worst else SEV_FAIR,
                "worst_bias_type"  : worst.bias_type if worst else "none",
            }
        return result

    def _fix_priority(self, findings: list[BiasFinding]) -> list[dict]:
        """Order fixes: severe → significant → mild, by confidence."""
        SEV_IDX = {SEV_SEVERE: 0, SEV_SIGNIFICANT: 1, SEV_MILD: 2, SEV_FAIR: 3}
        sorted_findings = sorted(
            findings,
            key=lambda f: (SEV_IDX.get(f.severity, 3), -f.confidence),
        )
        seen: set[str] = set()
        priority: list[dict] = []
        for f in sorted_findings:
            key = f"{f.bias_type}:{f.attribute}:{f.group}"
            if key in seen:
                continue
            seen.add(key)
            priority.append({
                "rank"      : len(priority) + 1,
                "attribute" : f.attribute,
                "group"     : f.group,
                "bias_type" : f.bias_type,
                "severity"  : f.severity,
                "emoji"     : SEV_EMOJI.get(f.severity, ""),
                "fix"       : f.fix,
                "confidence": f.confidence,
            })
        return priority[:20]   # top-20 fix items


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — BiasDetector MASTER CLASS + CONVENIENCE FUNCTIONS
# Purpose : One-call API that wires all 7 blocks.
# ─────────────────────────────────────────────────────────────────────────────

class BiasDetector:
    """
    Master bias detection class — runs all 6 detectors + scorer.

    Quick start
    -----------
    bd     = BiasDetector(df, sensitive_cols=["gender","age_group"], target_col="loan_approved")
    report = bd.detect()
    bd.export_report("reports/")

    With model predictions
    ----------------------
    bd = BiasDetector(
        df, sensitive_cols=["gender"], target_col="outcome",
        y_pred=model.predict(X), y_prob=model.predict_proba(X)[:,1]
    )
    report = bd.detect()

    Convenience shortcuts
    ---------------------
    bd.quick_fairness()          → fairness metrics only
    bd.find_proxies()            → proxy feature list
    bd.get_bias_score()          → single 0-100 score
    """

    def __init__(
        self,
        df             : "pd.DataFrame",
        sensitive_cols : list[str],
        target_col     : str | None    = None,
        y_pred         : "np.ndarray | None" = None,
        y_prob         : "np.ndarray | None" = None,
        train_df       : "pd.DataFrame | None" = None,
        test_df        : "pd.DataFrame | None" = None,
        max_intersect_order: int = 2,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df        = df
        self.sens      = sensitive_cols
        self.target    = target_col
        self.y_pred    = y_pred
        self.y_prob    = y_prob
        self.train_df  = train_df
        self.test_df   = test_df
        self.max_order = max_intersect_order
        self._results : list[DetectorResult] = []
        self._score   : dict[str, Any]       = {}
        self._profile : dict[str, Any]       = {}

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self) -> dict[str, Any]:
        """Run all detectors and return complete report dict."""
        t0 = time.perf_counter()

        # Block 1: Profile
        analyzer     = SensitiveFeatureAnalyzer(self.df, self.sens)
        self._profile= analyzer.analyze()

        # Block 2: Representation
        rep_result = self._run_safe(
            "RepresentationBiasDetector",
            lambda: RepresentationBiasDetector(
                self.df, self.sens, self.target
            ).detect(),
        )
        self._results.append(rep_result)

        # Block 3: Statistical Fairness
        if self.target:
            fair_result = self._run_safe(
                "StatisticalFairnessMetrics",
                lambda: StatisticalFairnessMetrics(
                    self.df, self.sens, self.target,
                    self.y_pred, self.y_prob,
                ).compute(),
            )
            self._results.append(fair_result)

        # Block 4: Historical + Measurement
        hist_result = self._run_safe(
            "HistoricalMeasurementBias",
            lambda: HistoricalMeasurementBias(
                self.df, self.sens, self.target,
                self.train_df, self.test_df,
            ).detect(),
        )
        self._results.append(hist_result)

        # Block 5: Intersectional
        if len(self.sens) >= 2:
            inter_result = self._run_safe(
                "IntersectionalBiasDetector",
                lambda: IntersectionalBiasDetector(
                    self.df, self.sens, self.target,
                    self.max_order,
                ).detect(),
            )
            self._results.append(inter_result)

        # Block 6: Feature-Level
        feat_result = self._run_safe(
            "FeatureLevelBiasAnalyzer",
            lambda: FeatureLevelBiasAnalyzer(
                self.df, self.sens, self.target, self.y_pred,
            ).analyze(),
        )
        self._results.append(feat_result)

        # Block 7: Score
        self._score = BiasSeverityScorer(self._results).compute()

        return {
            "bias_score"    : self._score,
            "profile"       : self._profile,
            "findings"      : self._all_findings_dicts(),
            "per_detector"  : {r.detector_name: r.metrics for r in self._results},
            "runtime_sec"   : round(time.perf_counter() - t0, 3),
        }

    def quick_fairness(self) -> dict[str, Any]:
        """Block 3 only — requires target_col."""
        if not self.target:
            return {"error": "target_col required for fairness metrics"}
        result = StatisticalFairnessMetrics(
            self.df, self.sens, self.target, self.y_pred, self.y_prob
        ).compute()
        return result.metrics

    def find_proxies(self) -> list[dict]:
        """Block 1 proxy detection only."""
        return SensitiveFeatureAnalyzer(self.df, self.sens).analyze()["proxy_features"]

    def get_bias_score(self) -> float:
        """Returns single bias score 0-100. Runs full detection if needed."""
        if not self._score:
            self.detect()
        return float(self._score.get("bias_score", 0.0))

    def summary(self) -> str:
        """Text summary of bias findings."""
        if not self._score:
            self.detect()
        s     = self._score
        lines = [
            "=" * 60,
            f"  Nydra — Bias Score: {s.get('bias_score', 0):.1f}/100",
            f"  Fairness Score: {s.get('fairness_score', 100):.1f}/100",
            f"  Grade: {s.get('label', '?')}",
            f"  Total Findings: {s.get('total_findings', 0)}",
            "=" * 60,
        ]
        sev_dist = s.get("severity_distribution", {})
        for sev in [SEV_SEVERE, SEV_SIGNIFICANT, SEV_MILD, SEV_FAIR]:
            count = sev_dist.get(sev, 0)
            if count > 0:
                lines.append(f"  {SEV_EMOJI.get(sev,'')} {sev.title():14s}: {count}")
        lines.append("=" * 60)
        top_fixes = s.get("fix_priority", [])[:3]
        if top_fixes:
            lines.append("  Top 3 Fixes:")
            for fix in top_fixes:
                lines.append(
                    f"    {fix['rank']}. [{fix['severity'].upper()}] "
                    f"{fix['attribute']} — {fix['bias_type']}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def export_report(
        self,
        output_dir: str | Path = "reports",
        formats   : list[str]  = ("md", "json", "html"),
        prefix    : str        = "bias_report",
    ) -> dict[str, Path]:
        """Generate and save bias report in specified formats."""
        if not self._score:
            self.detect()
        reporter = BiasReporter(
            score_result     = self._score,
            detector_results = self._results,
            profile          = self._profile,
            dataset_info     = {
                "rows"  : len(self.df),
                "cols"  : len(self.df.columns),
                "target": self.target,
                "sensitive_cols": self.sens,
            },
        )
        return reporter.export(output_dir=output_dir, formats=formats, prefix=prefix)

    # ── internals ─────────────────────────────────────────────────────────────

    def _all_findings_dicts(self) -> list[dict]:
        return [
            {
                "detector"   : f.detector,
                "attribute"  : f.attribute,
                "group"      : f.group,
                "bias_type"  : f.bias_type,
                "severity"   : f.severity,
                "description": f.description,
                "fix"        : f.fix,
                "confidence" : f.confidence,
                "metric_value": f.metric_value,
            }
            for r in self._results for f in r.findings
        ]

    @staticmethod
    def _run_safe(name: str, fn: Callable) -> DetectorResult:
        try:
            return fn()
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return DetectorResult(detector_name=name, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# BIAS REPORTER — Markdown + JSON + HTML
# ─────────────────────────────────────────────────────────────────────────────

class BiasReporter:
    """Generates multi-format bias reports from BiasDetector output."""

    def __init__(
        self,
        score_result    : dict[str, Any],
        detector_results: list[DetectorResult],
        profile         : dict[str, Any],
        dataset_info    : dict[str, Any] | None = None,
    ) -> None:
        self.score    = score_result
        self.results  = detector_results
        self.profile  = profile
        self.info     = dataset_info or {}
        self.findings = sorted(
            [f for r in detector_results for f in r.findings],
            key=lambda f: [SEV_SEVERE, SEV_SIGNIFICANT, SEV_MILD, SEV_FAIR].index(f.severity),
        )

    # ── export ────────────────────────────────────────────────────────────────

    def export(
        self,
        output_dir: str | Path = "reports",
        formats   : list[str]  = ("md", "json", "html"),
        prefix    : str        = "bias_report",
    ) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out / f"{prefix}_{ts}"
        paths: dict[str, Path] = {}

        if "md" in formats:
            p = base.with_suffix(".md")
            p.write_text(self._to_markdown(), encoding="utf-8")
            paths["md"] = p
        if "json" in formats:
            p = base.with_suffix(".json")
            p.write_text(self._to_json(), encoding="utf-8")
            paths["json"] = p
        if "html" in formats:
            p = base.with_suffix(".html")
            p.write_text(self._to_html(), encoding="utf-8")
            paths["html"] = p
        return paths

    # ── Markdown ──────────────────────────────────────────────────────────────

    def _to_markdown(self) -> str:
        s     = self.score
        grade = s.get("label", "?")
        bs    = s.get("bias_score", 0)
        fs    = s.get("fairness_score", 100)

        rows  = "\n".join(
            f"| {SEV_EMOJI.get(f.severity,'')} {f.severity.title()} "
            f"| `{f.attribute}` | {f.group} "
            f"| {f.bias_type} "
            f"| {f.description[:80]}… "
            f"| {round(f.confidence*100)}% |"
            for f in self.findings[:30]
        )
        fixes = "\n".join(
            f"| {p['rank']} | {SEV_EMOJI.get(p['severity'],'')} {p['severity'].title()} "
            f"| `{p['attribute']}` | {p['bias_type']} | {p['fix'][:80]} |"
            for p in s.get("fix_priority", [])[:10]
        )

        return f"""# 🩺 Nydra — Bias Detection Report

> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> Nydra v{VERSION}

---

## 🏆 Bias Score

**Grade: {grade}**
Bias Score: **{bs}/100** (lower = fairer)
Fairness Score: **{fs}/100** (higher = fairer)
Total Findings: **{s.get('total_findings', 0)}**

---

## ⚠️ Findings

| Severity | Attribute | Group | Bias Type | Description | Confidence |
|----------|-----------|-------|-----------|-------------|-----------|
{rows or '| ✅ No bias detected | — | — | — | — | — |'}

---

## 💡 Fix Priority

| # | Severity | Attribute | Bias Type | Recommended Fix |
|---|----------|-----------|-----------|-----------------|
{fixes or '| — | ✅ No fixes required | — | — | — |'}

---

*Built with ❤️ by Nydra Team*
"""

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _to_json(self) -> str:
        payload = {
            "report_generated"  : datetime.now().isoformat(),
            "nydra_version": VERSION,
            "dataset_info"      : self.info,
            "score"             : self.score,
            "profile"           : self.profile,
            "findings"          : [
                {
                    "detector"   : f.detector,
                    "attribute"  : f.attribute,
                    "group"      : f.group,
                    "bias_type"  : f.bias_type,
                    "severity"   : f.severity,
                    "description": f.description,
                    "evidence"   : f.evidence,
                    "fix"        : f.fix,
                    "confidence" : f.confidence,
                    "metric_value": f.metric_value,
                }
                for f in self.findings
            ],
        }
        return json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False)

    # ── HTML ──────────────────────────────────────────────────────────────────

    def _to_html(self) -> str:
        s       = self.score
        bs      = s.get("bias_score", 0)
        fs      = s.get("fairness_score", 100)
        grade   = s.get("label", "?")
        grade_k = s.get("grade", SEV_FAIR)
        color   = SEV_COLOR.get(grade_k, "#9099b0")
        n_f     = len(self.findings)

        finding_rows = "".join(
            f'<tr><td class="sev-{f.severity}">'
            f'{SEV_EMOJI.get(f.severity,"")} {f.severity.title()}</td>'
            f'<td><strong>{f.attribute}</strong></td>'
            f'<td>{f.group}</td>'
            f'<td>{f.bias_type}</td>'
            f'<td>{f.description[:100]}…</td>'
            f'<td>{round(f.confidence*100)}%</td></tr>'
            for f in self.findings[:40]
        )
        fix_rows = "".join(
            f'<tr><td>{p["rank"]}</td>'
            f'<td class="sev-{p["severity"]}">{p["emoji"]} {p["severity"].title()}</td>'
            f'<td><code>{p["attribute"]}</code></td>'
            f'<td>{p["bias_type"]}</td>'
            f'<td>{p["fix"][:100]}</td></tr>'
            for p in s.get("fix_priority", [])[:10]
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nydra — Bias Report</title>
<style>
  :root{{--bg:#0f1117;--surf:#1e2130;--bord:#2d3250;
        --txt:#e8eaf0;--muted:#9099b0;--accent:#4f8ef7;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--txt);
       font-family:'Segoe UI',system-ui,sans-serif;}}
  .container{{max-width:1100px;margin:auto;padding:32px 20px;}}
  h1{{font-size:2rem;margin-bottom:4px;}}
  h2{{font-size:1.2rem;color:var(--accent);margin:24px 0 10px;}}
  .meta{{color:var(--muted);font-size:0.85rem;margin-bottom:24px;}}
  .badge{{display:inline-block;padding:6px 20px;border-radius:8px;
          font-size:2rem;font-weight:900;color:{color};border:3px solid {color};}}
  .card{{background:var(--surf);border:1px solid var(--bord);
         border-radius:12px;padding:20px;margin-bottom:16px;}}
  .grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}}
  .stat{{text-align:center;padding:12px;}}
  .stat-val{{font-size:1.6rem;font-weight:700;color:var(--accent);}}
  .stat-lbl{{font-size:0.8rem;color:var(--muted);}}
  table{{width:100%;border-collapse:collapse;font-size:0.83rem;}}
  th{{background:#252840;padding:8px 12px;text-align:left;color:var(--muted);}}
  td{{padding:8px 12px;border-top:1px solid var(--bord);}}
  tr:hover td{{background:#252840;}}
  .sev-severe{{color:#e74c3c;}}.sev-significant{{color:#e67e22;}}
  .sev-mild{{color:#f1c40f;}}  .sev-fair{{color:#2ecc71;}}
  details summary{{cursor:pointer;font-weight:600;color:var(--accent);}}
  code{{background:#252840;padding:2px 6px;border-radius:4px;font-size:0.85em;}}
</style>
</head>
<body>
<div class="container">
  <h1>🩺 Nydra — Bias Report</h1>
  <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
     Nydra v{VERSION} &nbsp;|&nbsp;
     {self.info.get('rows', '?'):,} rows analysed</p>

  <div class="card" style="display:flex;align-items:center;gap:32px;flex-wrap:wrap">
    <div class="badge">{grade}</div>
    <div>
      <p>Bias Score: <strong style="color:{color}">{bs}/100</strong></p>
      <p>Fairness Score: <strong style="color:#2ecc71">{fs}/100</strong></p>
      <p style="color:var(--muted)">Total Findings: <strong>{n_f}</strong></p>
    </div>
  </div>

  <div class="card grid-3">
    <div class="stat"><div class="stat-val" style="color:{color}">{bs:.1f}</div>
      <div class="stat-lbl">Bias Score</div></div>
    <div class="stat"><div class="stat-val">{n_f}</div>
      <div class="stat-lbl">Total Findings</div></div>
    <div class="stat"><div class="stat-val" style="color:#e74c3c">
      {s.get('severe_count',0)}</div>
      <div class="stat-lbl">Severe Issues</div></div>
  </div>

  <details open>
    <summary><h2 style="display:inline">⚠️ Findings ({n_f})</h2></summary>
    <div class="card" style="margin-top:12px">
      <table>
        <tr><th>Severity</th><th>Attribute</th><th>Group</th>
            <th>Bias Type</th><th>Description</th><th>Confidence</th></tr>
        {finding_rows or
         '<tr><td colspan="6" style="color:#2ecc71">✅ No bias detected!</td></tr>'}
      </table>
    </div>
  </details>

  <details open>
    <summary><h2 style="display:inline">💡 Fix Priority</h2></summary>
    <div class="card" style="margin-top:12px">
      <table>
        <tr><th>#</th><th>Severity</th><th>Attribute</th>
            <th>Bias Type</th><th>Recommended Fix</th></tr>
        {fix_rows or
         '<tr><td colspan="5" style="color:#2ecc71">✅ No fixes required</td></tr>'}
      </table>
    </div>
  </details>

  <p class="meta" style="margin-top:40px;text-align:center">
    Built with ❤️ by
    <a href="https://github.com/Denterio1/Nydra"
       style="color:var(--accent)">Nydra Team</a>
  </p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect(
    df             : "pd.DataFrame",
    sensitive_cols : list[str],
    target_col     : str | None = None,
    y_pred         : "np.ndarray | None" = None,
    y_prob         : "np.ndarray | None" = None,
) -> dict[str, Any]:
    """Full bias detection in one call."""
    return BiasDetector(
        df=df, sensitive_cols=sensitive_cols,
        target_col=target_col, y_pred=y_pred, y_prob=y_prob,
    ).detect()


def quick_fairness(
    df             : "pd.DataFrame",
    sensitive_cols : list[str],
    target_col     : str,
    y_pred         : "np.ndarray | None" = None,
    y_prob         : "np.ndarray | None" = None,
) -> dict[str, Any]:
    """Fairness metrics only — fastest check."""
    return BiasDetector(
        df=df, sensitive_cols=sensitive_cols,
        target_col=target_col, y_pred=y_pred, y_prob=y_prob,
    ).quick_fairness()


def find_proxies(
    df             : "pd.DataFrame",
    sensitive_cols : list[str],
) -> list[dict]:
    """Proxy feature detection only."""
    return BiasDetector(df=df, sensitive_cols=sensitive_cols).find_proxies()


def get_bias_score(
    df             : "pd.DataFrame",
    sensitive_cols : list[str],
    target_col     : str | None = None,
) -> float:
    """Single bias score 0-100."""
    return BiasDetector(
        df=df, sensitive_cols=sensitive_cols, target_col=target_col
    ).get_bias_score()


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_categorical(s: "pd.Series") -> bool:
    return s.dtype.kind in ("O", "U", "S") or str(s.dtype) == "category"


def _is_numeric(s: "pd.Series") -> bool:
    return s.dtype.kind in ("i", "u", "f")


def _cramers_v(x: "pd.Series", y: "pd.Series") -> float:
    """Cramér's V correlation between two categorical Series."""
    try:
        ct = pd.crosstab(x, y)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return 0.0
        if SCIPY_OK:
            chi2_stat, _, _, _ = chi2_contingency(ct)
        else:
            n  = ct.values.sum()
            expected = (ct.sum(axis=1).values.reshape(-1, 1) *
                        ct.sum(axis=0).values.reshape(1, -1)) / n
            chi2_stat = float(((ct.values - expected) ** 2 / (expected + 1e-9)).sum())
        n   = ct.values.sum()
        r, k= ct.shape
        v   = math.sqrt(chi2_stat / max(n * (min(r, k) - 1), 1))
        return float(min(v, 1.0))
    except Exception:
        return 0.0


def _shannon_entropy(freqs: list[float]) -> float:
    """Shannon entropy of a probability distribution."""
    return -sum(p * math.log2(p + 1e-12) for p in freqs if p > 0)


def _pearson_fallback(x: "np.ndarray", y: "np.ndarray") -> float:
    """Pure-Python Pearson correlation — no scipy/numpy needed."""
    try:
        n  = len(x)
        if n < 2:
            return 0.0
        mx, my = sum(x) / n, sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
        return num / (dx * dy + 1e-9)
    except Exception:
        return 0.0


def _chi2_cdf_approx(x: float, df: int) -> float:
    """Very rough chi2 CDF approximation (Wilson-Hilferty) — fallback only."""
    try:
        if df <= 0:
            return 0.0
        z = ((x / df) ** (1/3) - (1 - 2/(9*df))) / math.sqrt(2/(9*df))
        # Normal CDF approximation
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    except Exception:
        return 0.5


def _fairness_fix(metric: str, group: str, attr: str) -> str:
    """Return a targeted fix recommendation for a fairness metric violation."""
    fixes = {
        "demographic_parity_diff"  : (
            f"Apply post-processing threshold adjustment for '{group}' in '{attr}' "
            f"to equalise selection rates. Use Fairlearn's ThresholdOptimizer."
        ),
        "demographic_parity_ratio" : (
            f"Increase selection rate for '{group}'. Consider re-weighting training "
            f"samples or using in-processing fairness constraints."
        ),
        "equalized_odds_diff"      : (
            f"Equalise TPR and FPR for '{group}' in '{attr}' using "
            f"equalized odds post-processing or adversarial debiasing."
        ),
        "equal_opportunity_diff"   : (
            f"Improve TPR for '{group}' in '{attr}'. "
            f"Use opportunity-aware oversampling or cost-sensitive learning."
        ),
        "predictive_parity_diff"   : (
            f"Calibrate model separately for '{group}' to achieve equal PPV. "
            f"Check if base rates differ and apply Platt scaling per group."
        ),
        "disparate_impact_ratio"   : (
            f"Disparate Impact for '{group}' violates EEOC 4/5ths rule. "
            f"Apply disparate impact remover preprocessing or re-weight samples."
        ),
        "treatment_equality_diff"  : (
            f"FP/FN error ratio differs for '{group}' in '{attr}'. "
            f"Adjust decision threshold per group to equalise error costs."
        ),
        "calibration_diff"         : (
            f"Model is miscalibrated for '{group}'. Apply group-specific "
            f"probability calibration (isotonic regression or Platt scaling)."
        ),
    }
    return fixes.get(metric, f"Review {metric} for group '{group}' in '{attr}'.")


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf/numpy types for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if NUMPY_OK:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _rng() -> "np.random.Generator":
    if NUMPY_OK:
        return np.random.default_rng(42)
    import random
    class _FallbackRNG:
        def permutation(self, n):
            lst = list(range(n))
            random.shuffle(lst)
            import types
            arr = types.SimpleNamespace()
            arr._lst = lst
            return lst
    return _FallbackRNG()
