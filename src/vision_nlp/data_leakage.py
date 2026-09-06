"""
data_leakage.py — Nydra v0.6.0
======================================
Advanced Data Leakage Detection Pipeline.

Detects 7 classes of data leakage that silently destroy ML model validity:

  1. TargetLeakageDetector         — features encoding future/direct target info
  2. TrainTestOverlapDetector      — duplicate/near-duplicate rows across splits
  3. TemporalLeakageDetector       — future data used in training (look-ahead bias)
  4. GroupLeakageDetector          — same entity (user/patient) in train + test
  5. PreprocessingLeakageDetector  — scaler/encoder fitted before split
  6. NearDuplicateDetector         — SimHash + MinHash LSH cluster analysis
  7. FeatureCorrelationLeakageDetector — VIF + proxy feature + redundancy
  8. LeakageScorer                 — composite 0-100 score + severity ratings
  9. LeakageReporter               — Markdown + JSON + HTML reports
  10. LeakageOrchestrator          — master one-call API

Author  : Nydra Team
Project : https://github.com/Denterio1/Nydra
Version : 0.6.0
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import collections
import hashlib
import itertools
import json
import logging
import math
import os
import re
import struct
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("data_leakage")

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
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from scipy.stats import pearsonr, spearmanr, chi2_contingency, kendalltau
    from scipy.spatial.distance import cosine as cosine_dist, hamming as hamming_dist
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "0.6.0"

# ── Severity levels ──────────────────────────────────────────────────────────
SEVERITY_CRITICAL = "critical"
SEVERITY_HIGH     = "high"
SEVERITY_MEDIUM   = "medium"
SEVERITY_LOW      = "low"
SEVERITY_INFO     = "info"

SEVERITY_ORDER = [SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW, SEVERITY_INFO]

SEVERITY_EMOJI = {
    SEVERITY_CRITICAL: "🔴",
    SEVERITY_HIGH    : "🟠",
    SEVERITY_MEDIUM  : "🟡",
    SEVERITY_LOW     : "🟢",
    SEVERITY_INFO    : "ℹ️",
}

SEVERITY_SCORE_PENALTY = {
    SEVERITY_CRITICAL: 25,
    SEVERITY_HIGH    : 15,
    SEVERITY_MEDIUM  : 8,
    SEVERITY_LOW     : 3,
    SEVERITY_INFO    : 1,
}

# ── Detection thresholds ─────────────────────────────────────────────────────
@dataclass
class LeakageConfig:
    """
    Central configuration object — tune all detection thresholds here.
    Pass a custom instance to LeakageOrchestrator to override defaults.
    """
    # TargetLeakage
    target_mi_threshold        : float = 0.85   # mutual information ratio vs target
    target_corr_threshold      : float = 0.95   # Pearson/Spearman correlation
    target_permutation_auc_drop: float = 0.10   # AUC drop after permutation
    target_shap_ratio          : float = 0.80   # single feature SHAP dominance

    # TrainTestOverlap
    overlap_exact_threshold    : float = 0.01   # > 1% exact duplicates = high
    overlap_near_threshold     : float = 0.05   # > 5% near-duplicates = high
    minhash_num_perm           : int   = 128    # MinHash permutations
    minhash_threshold          : float = 0.80   # Jaccard threshold for near-dup

    # TemporalLeakage
    temporal_lookahead_rows    : int   = 1      # rows ahead = leakage
    temporal_roll_window       : int   = 7      # rolling window to check
    future_label_gap_hours     : float = 0.0    # acceptable label delay

    # GroupLeakage
    group_overlap_threshold    : float = 0.02   # > 2% shared entities = high
    k_anonymity_k              : int   = 5      # minimum group size

    # PreprocessingLeakage
    preprocessing_mean_tol     : float = 1e-6   # tolerance for mean equality
    preprocessing_std_tol      : float = 1e-6   # tolerance for std equality

    # NearDuplicate
    simhash_bits               : int   = 64     # SimHash fingerprint bits
    simhash_hamming_threshold  : int   = 3      # max bit flips = near-dup
    near_dup_sample_size       : int   = 10_000 # max rows for LSH scan

    # FeatureCorrelation
    vif_threshold              : float = 10.0   # VIF > 10 = multicollinearity
    proxy_corr_threshold       : float = 0.98   # feature-feature correlation
    redundancy_cluster_threshold: float = 0.85  # cluster cutoff for redundancy

    # LeakageScorer
    max_penalty                : float = 100.0  # cap for composite penalty
    confidence_sample_size     : int   = 5_000  # rows for confidence estimation


# Global default config instance
DEFAULT_CONFIG = LeakageConfig()


# ── Known entity-ID column patterns ─────────────────────────────────────────
ENTITY_ID_PATTERNS = re.compile(
    r"(user|patient|customer|client|person|subject|employee|"
    r"account|session|device|order|transaction|visit|id|uid|uuid|"
    r"member|respondent|participant)[\s_\-]*(id|key|code|num|number|uuid)?$",
    re.IGNORECASE,
)

# ── Known temporal column patterns ───────────────────────────────────────────
TEMPORAL_PATTERNS = re.compile(
    r"(date|time|timestamp|datetime|created|updated|recorded|"
    r"event_time|log_time|start|end|year|month|day|hour|week|period|"
    r"at$|_at$|_on$|_date$|_time$|_dt$)",
    re.IGNORECASE,
)

# ── Common target/label column patterns ─────────────────────────────────────
TARGET_PATTERNS = re.compile(
    r"(target|label|y|output|outcome|response|churn|fraud|"
    r"default|survived|diagnosis|class|category|result|"
    r"price|revenue|sales|score|rating|value)$",
    re.IGNORECASE,
)

# ── Preprocessing class patterns ─────────────────────────────────────────────
SCALER_CLASSES = {"StandardScaler", "MinMaxScaler", "RobustScaler",
                  "Normalizer", "MaxAbsScaler", "PowerTransformer",
                  "QuantileTransformer"}
ENCODER_CLASSES = {"LabelEncoder", "OrdinalEncoder", "OneHotEncoder",
                   "TargetEncoder", "BinaryEncoder"}
IMPUTER_CLASSES = {"SimpleImputer", "KNNImputer", "IterativeImputer"}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeakageFinding:
    """Single leakage finding — standardised across all detectors."""
    detector   : str                    # which detector found this
    feature    : str                    # affected column / feature name
    severity   : str                    # critical / high / medium / low / info
    description: str                    # human-readable explanation
    evidence   : dict[str, Any]         # quantitative evidence dict
    confidence : float                  # 0-1 confidence in this finding
    fix        : str                    # recommended remediation action
    is_confirmed: bool = True           # False = potential false positive


@dataclass
class DetectorResult:
    """Standardised result object returned by every detector."""
    detector_name : str
    findings      : list[LeakageFinding] = field(default_factory=list)
    summary       : dict[str, Any]       = field(default_factory=dict)
    runtime_sec   : float                = 0.0
    error         : str | None           = None

    def add(self, finding: LeakageFinding) -> None:
        self.findings.append(finding)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == SEVERITY_HIGH)

    @property
    def total_penalty(self) -> float:
        return sum(SEVERITY_SCORE_PENALTY.get(f.severity, 0) for f in self.findings)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1 — TargetLeakageDetector
# Purpose : Detect features that directly encode target information.
#
# Methods :
#   1. Mutual Information ratio vs target (sklearn)
#   2. Pearson / Spearman correlation threshold
#   3. Permutation importance — trains RF, permutes each feature,
#      measures AUC / R² drop. Large drop → feature is critical for prediction
#      → potential proxy for target.
#   4. SHAP-based dominance — if one feature accounts for > 80% of SHAP
#      value mass, it's a near-perfect predictor = leakage.
#   5. Post-split feature drift check — if a feature's distribution changes
#      dramatically between train and test it may encode split info.
# ─────────────────────────────────────────────────────────────────────────────

class TargetLeakageDetector:
    """
    Identifies features that are suspiciously predictive of the target.

    Heuristic: a legitimate feature should not be able to perfectly
    predict the target by itself. If it can, it likely encodes
    future / direct target information = leakage.

    Supports classification and regression targets (auto-detected).
    """

    def __init__(
        self,
        df         : "pd.DataFrame",
        target_col : str,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required for TargetLeakageDetector")
        self.df         = df.copy()
        self.target_col = target_col
        self.cfg        = config
        self.task_type  = self._detect_task()

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="TargetLeakageDetector")

        if self.target_col not in self.df.columns:
            result.error = f"Target column '{self.target_col}' not found in DataFrame"
            return result

        feature_cols = [c for c in self.df.columns if c != self.target_col]
        numeric_cols = self._numeric_features(feature_cols)

        if not numeric_cols:
            result.error = "No numeric features found for leakage detection"
            return result

        X, y = self._prepare_Xy(numeric_cols)
        if X is None or y is None:
            result.error = "Could not prepare feature matrix"
            return result

        # 1. Mutual Information
        mi_scores = self._mutual_information(X, y, numeric_cols)

        # 2. Correlation analysis
        corr_scores = self._correlation_analysis(numeric_cols, y)

        # 3. Permutation importance
        perm_scores = self._permutation_importance(X, y, numeric_cols)

        # 4. SHAP dominance (optional)
        shap_scores = self._shap_dominance(X, y, numeric_cols)

        # 5. Post-split drift
        drift_flags = self._post_split_drift(numeric_cols)

        # ── Combine evidence ──────────────────────────────────────────────
        for col in numeric_cols:
            evidence = {
                "mutual_information"      : round(mi_scores.get(col, 0.0), 4),
                "correlation_with_target" : round(corr_scores.get(col, 0.0), 4),
                "permutation_auc_drop"    : round(perm_scores.get(col, 0.0), 4),
                "shap_dominance"          : round(shap_scores.get(col, 0.0), 4),
                "distribution_drift"      : drift_flags.get(col, False),
            }
            severity, confidence, desc, fix = self._classify(col, evidence)
            if severity != SEVERITY_INFO:
                result.add(LeakageFinding(
                    detector    = "TargetLeakageDetector",
                    feature     = col,
                    severity    = severity,
                    description = desc,
                    evidence    = evidence,
                    confidence  = confidence,
                    fix         = fix,
                ))

        result.findings.sort(key=lambda f: (SEVERITY_ORDER.index(f.severity),
                                            -f.evidence.get("mutual_information", 0)))
        result.summary = {
            "features_checked"    : len(numeric_cols),
            "leakage_found"       : len(result.findings),
            "critical_count"      : result.critical_count,
            "high_count"          : result.high_count,
            "task_type"           : self.task_type,
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── task type ─────────────────────────────────────────────────────────────

    def _detect_task(self) -> str:
        if not PANDAS_OK:
            return "unknown"
        col = self.df.get(self.target_col)
        if col is None:
            return "unknown"
        n_unique = col.nunique()
        if n_unique <= 20 or col.dtype in ("object", "category", "bool"):
            return "classification"
        return "regression"

    # ── feature prep ─────────────────────────────────────────────────────────

    def _numeric_features(self, cols: list[str]) -> list[str]:
        return [c for c in cols
                if self.df[c].dtype.kind in ("i", "u", "f")
                and self.df[c].nunique() > 1]

    def _prepare_Xy(
        self, numeric_cols: list[str]
    ) -> tuple["np.ndarray | None", "np.ndarray | None"]:
        if not (SKLEARN_OK and NUMPY_OK):
            return None, None
        try:
            sub = self.df[numeric_cols + [self.target_col]].dropna()
            if len(sub) < 10:
                return None, None
            X = sub[numeric_cols].values.astype(float)
            y = sub[self.target_col].values
            if self.task_type == "classification":
                le = LabelEncoder()
                y  = le.fit_transform(y.astype(str))
            else:
                y = y.astype(float)
            return X, y
        except Exception as e:
            logger.debug(f"_prepare_Xy error: {e}")
            return None, None

    # ── method 1: mutual information ─────────────────────────────────────────

    def _mutual_information(
        self, X: "np.ndarray", y: "np.ndarray", cols: list[str]
    ) -> dict[str, float]:
        if not SKLEARN_OK:
            return {}
        try:
            if self.task_type == "classification":
                scores = mutual_info_classif(X, y, random_state=42)
            else:
                scores = mutual_info_regression(X, y, random_state=42)
            max_mi = max(scores) if len(scores) > 0 else 1.0
            # Normalise to 0-1 relative to the most informative feature
            return {col: float(sc) / max(max_mi, 1e-9)
                    for col, sc in zip(cols, scores)}
        except Exception as e:
            logger.debug(f"MI error: {e}")
            return {}

    # ── method 2: correlation ────────────────────────────────────────────────

    def _correlation_analysis(
        self, cols: list[str], y: "np.ndarray"
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for col in cols:
            try:
                x_vals = self.df[col].dropna().values.astype(float)
                # Align lengths
                min_len = min(len(x_vals), len(y))
                x_vals  = x_vals[:min_len]
                y_vals  = y[:min_len]
                if SCIPY_OK:
                    corr, _ = spearmanr(x_vals, y_vals)
                    result[col] = abs(float(corr)) if not math.isnan(float(corr)) else 0.0
                else:
                    result[col] = _pearson_fallback(x_vals, y_vals)
            except Exception:
                result[col] = 0.0
        return result

    # ── method 3: permutation importance ─────────────────────────────────────

    def _permutation_importance(
        self, X: "np.ndarray", y: "np.ndarray", cols: list[str]
    ) -> dict[str, float]:
        """
        Train a quick RandomForest, then permute each feature one-by-one.
        Measure how much the AUC / R² drops.
        Large drop → feature is critical to prediction → potential leakage.
        """
        if not SKLEARN_OK:
            return {}
        try:
            sample_size = min(len(X), 3000)
            idx = np.random.choice(len(X), sample_size, replace=False)
            Xs, ys = X[idx], y[idx]

            if self.task_type == "classification":
                model = RandomForestClassifier(
                    n_estimators=50, max_depth=6, random_state=42, n_jobs=-1
                )
                model.fit(Xs, ys)
                n_classes = len(np.unique(ys))
                if n_classes == 2:
                    base_score = roc_auc_score(ys, model.predict_proba(Xs)[:, 1])
                else:
                    base_score = model.score(Xs, ys)
            else:
                model = RandomForestRegressor(
                    n_estimators=50, max_depth=6, random_state=42, n_jobs=-1
                )
                model.fit(Xs, ys)
                base_score = r2_score(ys, model.predict(Xs))

            result: dict[str, float] = {}
            for i, col in enumerate(cols):
                Xs_perm = Xs.copy()
                np.random.shuffle(Xs_perm[:, i])
                if self.task_type == "classification":
                    n_classes = len(np.unique(ys))
                    if n_classes == 2:
                        perm_score = roc_auc_score(
                            ys, model.predict_proba(Xs_perm)[:, 1]
                        )
                    else:
                        perm_score = model.score(Xs_perm, ys)
                else:
                    perm_score = r2_score(ys, model.predict(Xs_perm))
                drop = max(0.0, base_score - perm_score)
                result[col] = float(drop)

            return result
        except Exception as e:
            logger.debug(f"Permutation importance error: {e}")
            return {}

    # ── method 4: SHAP dominance ─────────────────────────────────────────────

    def _shap_dominance(
        self, X: "np.ndarray", y: "np.ndarray", cols: list[str]
    ) -> dict[str, float]:
        """
        Compute SHAP values with TreeExplainer.
        If one feature accounts for > 80% of total |SHAP| mass → dominance flag.
        Returns per-feature dominance ratio 0-1.
        """
        if not (SHAP_OK and SKLEARN_OK and NUMPY_OK):
            return {}
        try:
            sample_size = min(len(X), 2000)
            idx = np.random.choice(len(X), sample_size, replace=False)
            Xs, ys = X[idx], y[idx]

            if self.task_type == "classification":
                model = DecisionTreeClassifier(max_depth=8, random_state=42)
            else:
                model = DecisionTreeRegressor(max_depth=8, random_state=42)
            model.fit(Xs, ys)

            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(Xs)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]      # binary: take positive class

            total_shap = np.abs(shap_vals).sum()
            if total_shap == 0:
                return {}
            per_feature = np.abs(shap_vals).mean(axis=0)
            ratios      = per_feature / total_shap
            return {col: float(ratios[i]) for i, col in enumerate(cols)}
        except Exception as e:
            logger.debug(f"SHAP error: {e}")
            return {}

    # ── method 5: post-split distribution drift ───────────────────────────────

    def _post_split_drift(self, cols: list[str]) -> dict[str, bool]:
        """
        Randomly split df 80/20, compare feature means/stds.
        Large drift between random splits is unusual and may indicate
        the feature encodes split-related information.
        """
        if not NUMPY_OK:
            return {}
        try:
            n   = len(self.df)
            idx = np.random.permutation(n)
            train_idx = idx[:int(n * 0.8)]
            test_idx  = idx[int(n * 0.8):]
            flags: dict[str, bool] = {}
            for col in cols:
                vals     = self.df[col].dropna().values.astype(float)
                if len(vals) < 20:
                    flags[col] = False
                    continue
                train_v  = vals[train_idx[train_idx < len(vals)]]
                test_v   = vals[test_idx[test_idx < len(vals)]]
                if len(train_v) < 5 or len(test_v) < 5:
                    flags[col] = False
                    continue
                mean_drift = abs(train_v.mean() - test_v.mean()) / (abs(vals.mean()) + 1e-9)
                std_drift  = abs(train_v.std()  - test_v.std())  / (vals.std() + 1e-9)
                flags[col] = bool(mean_drift > 0.5 or std_drift > 0.5)
            return flags
        except Exception:
            return {}

    # ── severity classification ───────────────────────────────────────────────

    def _classify(
        self, col: str, ev: dict[str, Any]
    ) -> tuple[str, float, str, str]:
        mi   = ev.get("mutual_information", 0.0)
        corr = ev.get("correlation_with_target", 0.0)
        perm = ev.get("permutation_auc_drop", 0.0)
        shap = ev.get("shap_dominance", 0.0)
        drift= ev.get("distribution_drift", False)

        # Count strong signals
        signals = sum([
            mi   >= self.cfg.target_mi_threshold,
            corr >= self.cfg.target_corr_threshold,
            perm >= self.cfg.target_permutation_auc_drop,
            shap >= self.cfg.target_shap_ratio,
        ])
        confidence = round(min(1.0, signals / 4.0 + 0.2 * drift), 4)

        if signals >= 3:
            severity    = SEVERITY_CRITICAL
            description = (
                f"Feature '{col}' is almost certainly a direct proxy for the target. "
                f"MI={mi:.3f}, corr={corr:.3f}, perm_drop={perm:.3f}, SHAP={shap:.3f}. "
                f"This feature likely encodes future or target information."
            )
            fix = (
                f"Remove '{col}' from the feature set immediately. "
                f"Verify its data lineage — was it computed after the target was observed?"
            )
        elif signals == 2:
            severity    = SEVERITY_HIGH
            description = (
                f"Feature '{col}' shows strong signs of target leakage "
                f"(MI={mi:.3f}, corr={corr:.3f}). Investigate its data origin."
            )
            fix = (
                f"Investigate '{col}' data pipeline. Check if its values are derived "
                f"from or correlated with future observations of the target."
            )
        elif signals == 1:
            severity    = SEVERITY_MEDIUM
            description = (
                f"Feature '{col}' shows mild leakage signals (MI={mi:.3f}). "
                f"May be a legitimate strong predictor — validate manually."
            )
            fix = (
                f"Review '{col}' domain meaning. If it's computed from future data, "
                f"remove it. Otherwise document it as a strong legitimate predictor."
            )
        elif drift:
            severity    = SEVERITY_LOW
            description = (
                f"Feature '{col}' shows unusual distribution drift between random splits, "
                f"which may indicate implicit leakage."
            )
            fix = f"Monitor '{col}' across different time-based splits for consistency."
        else:
            return SEVERITY_INFO, 0.0, "", ""

        return severity, confidence, description, fix


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 — TrainTestOverlapDetector
# Purpose : Find rows shared between train and test splits.
#
# Methods :
#   1. Exact row hash (SHA256 per row → compare sets)
#   2. Near-duplicate via MinHash + LSH (Jaccard similarity)
#   3. Column-subset overlap (same values in key columns)
#   4. Overlap ratio per feature
#   5. Stratified overlap analysis (overlap rate per class)
# ─────────────────────────────────────────────────────────────────────────────

class TrainTestOverlapDetector:
    """
    Detects rows duplicated or near-duplicated between train and test sets.

    This is one of the most common and dangerous forms of leakage:
    if test rows appear in training, reported metrics are inflated.

    Methods
    -------
    1. Exact SHA256 hash comparison — O(n) per set
    2. MinHash LSH — probabilistic near-duplicate detection
       (Jaccard similarity > threshold → near-duplicate)
    3. Key-column subset overlap — checks entity IDs and timestamps
    4. Per-feature overlap ratio — which columns drive the overlap?
    5. Stratified overlap — does overlap concentrate in one class?
    """

    def __init__(
        self,
        train_df   : "pd.DataFrame",
        test_df    : "pd.DataFrame",
        target_col : str | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required for TrainTestOverlapDetector")
        self.train  = train_df.copy()
        self.test   = test_df.copy()
        self.target = target_col
        self.cfg    = config

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="TrainTestOverlapDetector")

        n_train, n_test = len(self.train), len(self.test)
        if n_train == 0 or n_test == 0:
            result.error = "Empty train or test DataFrame"
            return result

        # 1. Exact overlap
        exact_count, exact_train_idx, exact_test_idx = self._exact_overlap()
        exact_ratio = exact_count / max(n_test, 1)

        # 2. Near-duplicate overlap
        near_pairs = self._near_duplicate_overlap()
        near_ratio  = len(near_pairs) / max(n_test, 1)

        # 3. Column-subset overlap
        col_overlap = self._column_subset_overlap()

        # 4. Per-feature overlap ratio
        feature_overlap = self._feature_overlap_ratio()

        # 5. Stratified overlap
        strat_overlap = self._stratified_overlap(exact_train_idx, exact_test_idx)

        # ── Generate findings ────────────────────────────────────────────────
        if exact_count > 0:
            sev = (SEVERITY_CRITICAL if exact_ratio > self.cfg.overlap_exact_threshold
                   else SEVERITY_HIGH)
            result.add(LeakageFinding(
                detector    = "TrainTestOverlapDetector",
                feature     = "__all_features__",
                severity    = sev,
                description = (
                    f"{exact_count} exact duplicate rows found between train and test "
                    f"({exact_ratio*100:.2f}% of test set). These rows will inflate "
                    f"all reported metrics."
                ),
                evidence    = {
                    "exact_duplicate_count": exact_count,
                    "exact_overlap_ratio"  : round(exact_ratio, 4),
                    "example_train_indices": exact_train_idx[:5],
                    "example_test_indices" : exact_test_idx[:5],
                },
                confidence  = 1.0,   # exact match = certain
                fix         = (
                    "Re-split your data ensuring no row appears in both train and test. "
                    "Use sklearn train_test_split with shuffle=True on the original dataset, "
                    "then verify with this detector."
                ),
            ))

        if len(near_pairs) > 0:
            sev = (SEVERITY_HIGH if near_ratio > self.cfg.overlap_near_threshold
                   else SEVERITY_MEDIUM)
            result.add(LeakageFinding(
                detector    = "TrainTestOverlapDetector",
                feature     = "__near_duplicate__",
                severity    = sev,
                description = (
                    f"{len(near_pairs)} near-duplicate row pairs detected between splits "
                    f"(Jaccard ≥ {self.cfg.minhash_threshold}). "
                    f"Near-duplicates cause optimistic metric estimates."
                ),
                evidence    = {
                    "near_duplicate_pairs" : len(near_pairs),
                    "near_overlap_ratio"   : round(near_ratio, 4),
                    "example_pairs"        : near_pairs[:5],
                    "similarity_threshold" : self.cfg.minhash_threshold,
                },
                confidence  = 0.85,
                fix         = (
                    "Deduplicate your dataset before splitting. "
                    "Use hash-based deduplication or MinHash clustering "
                    "to remove near-duplicate rows."
                ),
            ))

        # Column-subset leakage
        for col_info in col_overlap:
            result.add(LeakageFinding(
                detector    = "TrainTestOverlapDetector",
                feature     = col_info["column"],
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"Column '{col_info['column']}' has {col_info['overlap_count']} "
                    f"overlapping values between train and test "
                    f"({col_info['overlap_ratio']*100:.1f}%). "
                    f"If this is an entity ID, it means the same entity is in both splits."
                ),
                evidence    = col_info,
                confidence  = 0.70,
                fix         = (
                    f"If '{col_info['column']}' is an entity identifier, use group-based "
                    f"splitting (GroupShuffleSplit) to ensure each entity is in only one split."
                ),
            ))

        result.summary = {
            "train_size"           : n_train,
            "test_size"            : n_test,
            "exact_duplicates"     : exact_count,
            "near_duplicates"      : len(near_pairs),
            "column_overlaps"      : len(col_overlap),
            "feature_overlap_ratios": feature_overlap,
            "stratified_overlap"   : strat_overlap,
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: exact SHA256 hash ───────────────────────────────────────────

    def _exact_overlap(self) -> tuple[int, list[int], list[int]]:
        """Hash each row → find matching hashes between train and test."""
        train_hashes = self._hash_dataframe(self.train)
        test_hashes  = self._hash_dataframe(self.test)

        train_hash_map: dict[str, list[int]] = collections.defaultdict(list)
        for i, h in enumerate(train_hashes):
            train_hash_map[h].append(i)

        exact_train: list[int] = []
        exact_test : list[int] = []
        for j, h in enumerate(test_hashes):
            if h in train_hash_map:
                exact_train.extend(train_hash_map[h])
                exact_test.append(j)

        return len(exact_test), exact_train, exact_test

    @staticmethod
    def _hash_dataframe(df: "pd.DataFrame") -> list[str]:
        hashes = []
        for _, row in df.iterrows():
            row_str = "|".join(str(v) for v in row.values)
            hashes.append(hashlib.sha256(row_str.encode()).hexdigest())
        return hashes

    # ── method 2: MinHash LSH near-duplicate ─────────────────────────────────

    def _near_duplicate_overlap(self) -> list[dict]:
        """
        MinHash approximation of Jaccard similarity between rows.
        Each row is treated as a set of (col, value_bin) tokens.
        """
        n_perm = self.cfg.minhash_num_perm
        thresh = self.cfg.minhash_threshold
        sample = self.cfg.near_dup_sample_size

        try:
            train_sample = self.train.head(sample)
            test_sample  = self.test.head(sample)

            train_sigs = [self._minhash(row, n_perm)
                          for _, row in train_sample.iterrows()]
            test_sigs  = [self._minhash(row, n_perm)
                          for _, row in test_sample.iterrows()]

            pairs: list[dict] = []
            for j, test_sig in enumerate(test_sigs):
                for i, train_sig in enumerate(train_sigs):
                    sim = self._jaccard_from_minhash(train_sig, test_sig)
                    if sim >= thresh:
                        pairs.append({
                            "train_idx" : i,
                            "test_idx"  : j,
                            "similarity": round(sim, 4),
                        })
                        break   # one match per test row is enough
            return pairs
        except Exception as e:
            logger.debug(f"MinHash error: {e}")
            return []

    @staticmethod
    def _minhash(row: "pd.Series", n_perm: int) -> list[int]:
        """Compute MinHash signature for a pandas row."""
        tokens = set()
        for col, val in row.items():
            if pd.notna(val):
                tokens.add(f"{col}={val}")
        if not tokens:
            return [0] * n_perm

        sig = [float("inf")] * n_perm
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for seed in range(n_perm):
                # Universal hash: (a*h + b) mod large_prime
                a = seed * 2654435761 & 0xFFFFFFFF
                v = (a * (h & 0xFFFFFFFF) + seed) & 0xFFFFFFFF
                if v < sig[seed]:
                    sig[seed] = v
        return [int(s) for s in sig]

    @staticmethod
    def _jaccard_from_minhash(sig1: list[int], sig2: list[int]) -> float:
        if not sig1 or not sig2 or len(sig1) != len(sig2):
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    # ── method 3: column-subset overlap ──────────────────────────────────────

    def _column_subset_overlap(self) -> list[dict]:
        """
        For each column that looks like an entity ID or key,
        measure value overlap between train and test.
        """
        overlaps: list[dict] = []
        for col in self.train.columns:
            if not ENTITY_ID_PATTERNS.search(col):
                continue
            train_vals = set(self.train[col].dropna().astype(str))
            test_vals  = set(self.test[col].dropna().astype(str))
            common     = train_vals & test_vals
            if not common:
                continue
            ratio = len(common) / max(len(test_vals), 1)
            if ratio > 0.05:    # > 5% entity overlap is suspicious
                overlaps.append({
                    "column"        : col,
                    "overlap_count" : len(common),
                    "overlap_ratio" : round(ratio, 4),
                    "train_unique"  : len(train_vals),
                    "test_unique"   : len(test_vals),
                    "example_values": list(common)[:5],
                })
        return overlaps

    # ── method 4: per-feature overlap ratio ───────────────────────────────────

    def _feature_overlap_ratio(self) -> dict[str, float]:
        """For each numeric column, measure value-range overlap."""
        result: dict[str, float] = {}
        for col in self.train.columns:
            if self.train[col].dtype.kind not in ("i", "u", "f"):
                continue
            try:
                tr_min, tr_max = self.train[col].min(), self.train[col].max()
                te_min, te_max = self.test[col].min(),  self.test[col].max()
                tr_range = tr_max - tr_min
                if tr_range == 0:
                    continue
                overlap_lo = max(tr_min, te_min)
                overlap_hi = min(tr_max, te_max)
                if overlap_lo > overlap_hi:
                    result[col] = 0.0
                else:
                    result[col] = round((overlap_hi - overlap_lo) / tr_range, 4)
            except Exception:
                pass
        return result

    # ── method 5: stratified overlap ──────────────────────────────────────────

    def _stratified_overlap(
        self, train_idx: list[int], test_idx: list[int]
    ) -> dict[str, Any]:
        if not self.target or self.target not in self.train.columns:
            return {}
        if not train_idx:
            return {"overlap_per_class": {}}
        try:
            classes = self.train[self.target].unique()
            class_counts: dict[str, int] = {}
            for cls in classes:
                train_class_idx = set(
                    self.train.index[self.train[self.target] == cls].tolist()
                )
                overlap_in_class = sum(1 for i in train_idx if i in train_class_idx)
                class_counts[str(cls)] = overlap_in_class
            return {"overlap_per_class": class_counts}
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3 — TemporalLeakageDetector
# Purpose : Detect look-ahead bias and future data in training.
#
# Methods :
#   1. Time column auto-detection
#   2. Future-in-train scan (test timestamps in train)
#   3. Look-ahead feature detection (rolling windows using future rows)
#   4. Rolling window leakage analysis
#   5. Event leakage (labels assigned before event time)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalLeakageDetector:
    """
    Detects time-related leakage — the silent killer of time-series ML.

    Look-ahead bias: the model learns from the future.
    Common causes:
      - Rolling means computed including future rows
      - Test data time range overlaps with training range
      - Labels assigned before the labelled event occurred

    Requires at least one datetime or timestamp column.
    """

    def __init__(
        self,
        df         : "pd.DataFrame",
        time_col   : str | None = None,
        target_col : str | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df         = df.copy()
        self.target_col = target_col
        self.cfg        = config
        self.time_col   = time_col or self._auto_detect_time_col()

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="TemporalLeakageDetector")

        if self.time_col is None:
            result.error = (
                "No time/date column detected. "
                "Pass time_col= explicitly or ensure a datetime column exists."
            )
            return result

        if self.time_col not in self.df.columns:
            result.error = f"Time column '{self.time_col}' not found"
            return result

        time_series = self._parse_time_column()
        if time_series is None:
            result.error = f"Could not parse '{self.time_col}' as datetime"
            return result

        # 1. Sort check — is data already sorted by time?
        is_sorted = bool(time_series.is_monotonic_increasing)

        # 2. Future-in-train scan
        future_leakage = self._future_in_train_scan(time_series)

        # 3. Look-ahead features
        lookahead_cols = self._lookahead_feature_detection(time_series)

        # 4. Rolling window analysis
        roll_leakage = self._rolling_window_leakage(time_series)

        # 5. Event leakage
        event_leakage = self._event_label_leakage(time_series)

        # ── generate findings ────────────────────────────────────────────────
        if not is_sorted:
            result.add(LeakageFinding(
                detector    = "TemporalLeakageDetector",
                feature     = self.time_col,
                severity    = SEVERITY_HIGH,
                description = (
                    f"Data is not sorted by time column '{self.time_col}'. "
                    f"Any train/test split by row position (e.g. first 80%) "
                    f"will mix future and past data — classic look-ahead bias."
                ),
                evidence    = {"is_sorted": False, "time_col": self.time_col},
                confidence  = 0.95,
                fix         = (
                    f"Sort the dataset by '{self.time_col}' before splitting. "
                    f"Use time-based split: train = before cutoff, test = after."
                ),
            ))

        if future_leakage["overlap_count"] > 0:
            result.add(LeakageFinding(
                detector    = "TemporalLeakageDetector",
                feature     = self.time_col,
                severity    = SEVERITY_CRITICAL,
                description = (
                    f"{future_leakage['overlap_count']} rows in the training period "
                    f"have timestamps that fall within the test period. "
                    f"This is direct temporal leakage."
                ),
                evidence    = future_leakage,
                confidence  = 0.99,
                fix         = (
                    "Use a strict temporal split: all training data must precede "
                    "all test data in time. Remove any rows violating this."
                ),
            ))

        for col_info in lookahead_cols:
            result.add(LeakageFinding(
                detector    = "TemporalLeakageDetector",
                feature     = col_info["column"],
                severity    = SEVERITY_HIGH,
                description = (
                    f"Feature '{col_info['column']}' may be computed using future rows. "
                    f"Detected lag-0 autocorrelation pattern consistent with look-ahead "
                    f"in rolling computations."
                ),
                evidence    = col_info,
                confidence  = 0.75,
                fix         = (
                    f"Recompute '{col_info['column']}' using only past data. "
                    f"Use df.shift(1) before rolling to exclude the current row."
                ),
            ))

        for roll_info in roll_leakage:
            result.add(LeakageFinding(
                detector    = "TemporalLeakageDetector",
                feature     = roll_info["column"],
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"Rolling feature '{roll_info['column']}' shows values at time t "
                    f"that correlate unusually highly with t+1 values, "
                    f"suggesting future data inclusion."
                ),
                evidence    = roll_info,
                confidence  = 0.65,
                fix         = (
                    f"Verify rolling window for '{roll_info['column']}' uses "
                    f"closed='left' to exclude the current timestamp."
                ),
            ))

        if event_leakage.get("leakage_detected"):
            result.add(LeakageFinding(
                detector    = "TemporalLeakageDetector",
                feature     = self.target_col or "target",
                severity    = SEVERITY_CRITICAL,
                description = (
                    f"Labels appear to be assigned before the corresponding event time "
                    f"in {event_leakage.get('count', 0)} rows. "
                    f"The model will learn from future label information."
                ),
                evidence    = event_leakage,
                confidence  = 0.90,
                fix         = (
                    "Ensure labels are assigned only after the event completes. "
                    "Add a temporal gap between feature observation and label assignment."
                ),
            ))

        result.summary = {
            "time_column"     : self.time_col,
            "data_sorted"     : is_sorted,
            "time_range"      : {
                "min": str(time_series.min()),
                "max": str(time_series.max()),
            },
            "future_overlap"  : future_leakage.get("overlap_count", 0),
            "lookahead_cols"  : len(lookahead_cols),
            "rolling_issues"  : len(roll_leakage),
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _auto_detect_time_col(self) -> str | None:
        if not PANDAS_OK:
            return None
        for col in self.df.columns:
            if TEMPORAL_PATTERNS.search(col):
                return col
            if hasattr(self.df[col], "dtype"):
                if "datetime" in str(self.df[col].dtype):
                    return col
        return None

    def _parse_time_column(self) -> "pd.Series | None":
        try:
            return pd.to_datetime(self.df[self.time_col], errors="coerce")
        except Exception:
            return None

    def _future_in_train_scan(self, ts: "pd.Series") -> dict[str, Any]:
        """
        Simulate an 80/20 time-based split.
        Check how many 'train' rows have timestamps after the split point.
        """
        try:
            n         = len(ts.dropna())
            split_idx = int(n * 0.8)
            sorted_ts = ts.dropna().sort_values()
            split_time= sorted_ts.iloc[split_idx] if split_idx < len(sorted_ts) else sorted_ts.max()
            train_mask= ts <= split_time
            overlap   = self.df.loc[~train_mask].index.tolist()
            return {
                "split_time"   : str(split_time),
                "overlap_count": len(overlap),
                "overlap_idx"  : overlap[:10],
            }
        except Exception:
            return {"overlap_count": 0}

    def _lookahead_feature_detection(self, ts: "pd.Series") -> list[dict]:
        """
        Detects columns whose values correlate strongly with future target values.
        Proxy for rolling-window look-ahead.
        """
        if not (SKLEARN_OK and self.target_col and self.target_col in self.df.columns):
            return []
        suspicious: list[dict] = []
        target = self.df[self.target_col]
        if target.dtype.kind not in ("i", "u", "f"):
            return []
        target_shifted = target.shift(-1)   # t+1 target
        for col in self.df.columns:
            if col in (self.time_col, self.target_col):
                continue
            if self.df[col].dtype.kind not in ("i", "u", "f"):
                continue
            try:
                aligned = pd.concat([self.df[col], target_shifted], axis=1).dropna()
                if len(aligned) < 10:
                    continue
                if SCIPY_OK:
                    corr, pval = pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                else:
                    corr = _pearson_fallback(
                        aligned.iloc[:, 0].values,
                        aligned.iloc[:, 1].values,
                    )
                    pval = 0.05
                if abs(corr) > 0.80 and pval < 0.01:
                    suspicious.append({
                        "column"            : col,
                        "future_correlation": round(float(corr), 4),
                        "p_value"           : round(float(pval), 6),
                    })
            except Exception:
                pass
        return suspicious

    def _rolling_window_leakage(self, ts: "pd.Series") -> list[dict]:
        """
        Look for columns that appear to be rolling statistics.
        Checks if the value at time t is equal to the mean/max of t and t+1.
        """
        roll_cols = [c for c in self.df.columns
                     if re.search(r"(roll|rolling|window|moving|ma_|avg_|mean_|sum_)",
                                  c, re.IGNORECASE)]
        issues: list[dict] = []
        for col in roll_cols:
            if self.df[col].dtype.kind not in ("i", "u", "f"):
                continue
            try:
                vals       = self.df[col].dropna().values.astype(float)
                shift1     = vals[1:]
                vals_trim  = vals[:-1]
                if len(vals_trim) < 5:
                    continue
                if SCIPY_OK:
                    corr, _ = pearsonr(vals_trim, shift1)
                else:
                    corr = _pearson_fallback(vals_trim, shift1)
                if abs(corr) > 0.99:   # suspiciously high auto-correlation
                    issues.append({
                        "column"      : col,
                        "lag1_autocorr": round(float(corr), 4),
                    })
            except Exception:
                pass
        return issues

    def _event_label_leakage(self, ts: "pd.Series") -> dict[str, Any]:
        """
        Check if label timestamps precede the event time.
        Requires a secondary 'event_time' column.
        """
        event_col = None
        for col in self.df.columns:
            if re.search(r"(event|occur|happen|start|trigger)[\s_]*(time|date|at)?$",
                         col, re.IGNORECASE):
                event_col = col
                break
        if event_col is None:
            return {"leakage_detected": False}
        try:
            event_ts = pd.to_datetime(self.df[event_col], errors="coerce")
            leakage_mask = ts < event_ts    # label time before event time
            count = int(leakage_mask.sum())
            return {
                "leakage_detected": count > 0,
                "count"           : count,
                "event_column"    : event_col,
                "label_column"    : self.time_col,
                "pct"             : round(count / max(len(self.df), 1) * 100, 2),
            }
        except Exception:
            return {"leakage_detected": False}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4 — GroupLeakageDetector
# Purpose : Detect same entity appearing in both train and test.
#
# Methods :
#   1. Entity ID column auto-detection
#   2. Group overlap ratio across splits
#   3. Group contamination ratio (% of entity's rows in test also in train)
#   4. k-anonymity check (groups smaller than k are identifiable)
#   5. Intra-group target leakage (same entity, different label)
# ─────────────────────────────────────────────────────────────────────────────

class GroupLeakageDetector:
    """
    Detects group/entity contamination between train and test sets.

    If the same user, patient, or customer appears in both train and test,
    the model memorises entity-specific patterns rather than generalising.
    This inflates test metrics and fails in real deployment.
    """

    def __init__(
        self,
        train_df   : "pd.DataFrame",
        test_df    : "pd.DataFrame",
        group_cols : list[str] | None = None,
        target_col : str | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.train      = train_df.copy()
        self.test       = test_df.copy()
        self.group_cols = group_cols or self._detect_group_cols()
        self.target_col = target_col
        self.cfg        = config

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="GroupLeakageDetector")

        if not self.group_cols:
            result.error = (
                "No entity/group ID columns detected. "
                "Pass group_cols=['user_id', ...] explicitly."
            )
            return result

        all_overlap_info: list[dict] = []
        for col in self.group_cols:
            if col not in self.train.columns or col not in self.test.columns:
                continue
            info = self._analyse_group_column(col)
            all_overlap_info.append(info)

            if info["overlap_ratio"] > self.cfg.group_overlap_threshold:
                sev = (SEVERITY_CRITICAL
                       if info["overlap_ratio"] > 0.10 else SEVERITY_HIGH)
                result.add(LeakageFinding(
                    detector    = "GroupLeakageDetector",
                    feature     = col,
                    severity    = sev,
                    description = (
                        f"Group column '{col}': {info['overlap_count']} entities "
                        f"appear in both train and test "
                        f"({info['overlap_ratio']*100:.1f}% of test entities). "
                        f"The model can memorise entity behaviour."
                    ),
                    evidence    = info,
                    confidence  = 0.95,
                    fix         = (
                        f"Use GroupShuffleSplit or GroupKFold with groups='{col}' "
                        f"to ensure each entity is in exactly one split."
                    ),
                ))

        # k-anonymity check
        k_issues = self._k_anonymity_check()
        for issue in k_issues:
            result.add(LeakageFinding(
                detector    = "GroupLeakageDetector",
                feature     = issue["column"],
                severity    = SEVERITY_LOW,
                description = (
                    f"Column '{issue['column']}' has {issue['small_group_count']} groups "
                    f"with fewer than k={self.cfg.k_anonymity_k} members. "
                    f"Small groups are re-identifiable."
                ),
                evidence    = issue,
                confidence  = 0.70,
                fix         = (
                    f"Apply k-anonymisation: merge groups smaller than {self.cfg.k_anonymity_k} "
                    f"into an 'Other' category."
                ),
            ))

        # Intra-group target consistency
        if self.target_col:
            intra_issues = self._intra_group_target_leakage()
            for issue in intra_issues:
                result.add(LeakageFinding(
                    detector    = "GroupLeakageDetector",
                    feature     = issue["column"],
                    severity    = SEVERITY_MEDIUM,
                    description = (
                        f"Same entity in '{issue['column']}' has inconsistent target labels "
                        f"across train and test ({issue['inconsistent_count']} entities). "
                        f"May cause label leakage."
                    ),
                    evidence    = issue,
                    confidence  = 0.75,
                    fix         = (
                        "Investigate why the same entity has different labels across splits. "
                        "Consider using the last known label per entity."
                    ),
                ))

        result.summary = {
            "group_columns_checked": self.group_cols,
            "overlap_info"         : all_overlap_info,
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── helpers ───────────────────────────────────────────────────────────────

    def _detect_group_cols(self) -> list[str]:
        return [c for c in self.train.columns if ENTITY_ID_PATTERNS.search(c)]

    def _analyse_group_column(self, col: str) -> dict[str, Any]:
        train_groups = set(self.train[col].dropna().astype(str))
        test_groups  = set(self.test[col].dropna().astype(str))
        common       = train_groups & test_groups
        overlap_ratio= len(common) / max(len(test_groups), 1)

        # Contamination: for each shared entity, what fraction of their
        # test rows had the entity represented in train?
        contaminated_rows = int(
            self.test[col].astype(str).isin(common).sum()
        )
        contamination_ratio = contaminated_rows / max(len(self.test), 1)

        return {
            "column"             : col,
            "train_unique"       : len(train_groups),
            "test_unique"        : len(test_groups),
            "overlap_count"      : len(common),
            "overlap_ratio"      : round(overlap_ratio, 4),
            "contaminated_rows"  : contaminated_rows,
            "contamination_ratio": round(contamination_ratio, 4),
            "example_entities"   : list(common)[:5],
        }

    def _k_anonymity_check(self) -> list[dict]:
        k       = self.cfg.k_anonymity_k
        issues: list[dict] = []
        full_df = pd.concat([self.train, self.test], ignore_index=True)
        for col in self.group_cols:
            if col not in full_df.columns:
                continue
            counts = full_df[col].value_counts()
            small  = counts[counts < k]
            if len(small) > 0:
                issues.append({
                    "column"           : col,
                    "small_group_count": len(small),
                    "k"                : k,
                    "smallest_group"   : int(small.min()),
                    "example_groups"   : small.head(5).index.tolist(),
                })
        return issues

    def _intra_group_target_leakage(self) -> list[dict]:
        if not self.target_col:
            return []
        issues: list[dict] = []
        combined = pd.concat([
            self.train[[col for col in self.group_cols
                        if col in self.train.columns] + [self.target_col]],
            self.test[[col for col in self.group_cols
                       if col in self.test.columns] + [self.target_col]],
        ], ignore_index=True)

        for col in self.group_cols:
            if col not in combined.columns:
                continue
            entity_labels = combined.groupby(col)[self.target_col].nunique()
            inconsistent  = entity_labels[entity_labels > 1]
            if len(inconsistent) > 0:
                issues.append({
                    "column"            : col,
                    "inconsistent_count": len(inconsistent),
                    "example_entities"  : inconsistent.head(5).index.tolist(),
                })
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5 — PreprocessingLeakageDetector
# Purpose : Detect scalers / encoders fitted on the full dataset before split.
#
# Methods :
#   1. StandardScaler mean/std comparison (full vs train subset)
#   2. MinMaxScaler min/max comparison
#   3. Feature selection leakage (selection done on full dataset)
#   4. Imputer leakage (imputer mean fitted before split)
#   5. Pipeline order analysis (if sklearn Pipeline is provided)
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingLeakageDetector:
    """
    Detects the common mistake of fitting preprocessors on the full dataset
    before splitting into train and test.

    This leaks test statistics into the training process:
    the model indirectly knows the test distribution.

    Approach
    --------
    1. If a fitted sklearn object is provided → inspect its stored statistics
       against the training set to detect if full-data statistics were used.
    2. If no fitted object → compare train statistics to full-data statistics;
       if they match closely, preprocessor was fitted on full data.
    3. Inspect sklearn Pipeline objects for incorrect operation order.
    """

    def __init__(
        self,
        full_df    : "pd.DataFrame",
        train_df   : "pd.DataFrame",
        test_df    : "pd.DataFrame" | None = None,
        fitted_objs: list[Any] | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.full   = full_df.copy()
        self.train  = train_df.copy()
        self.test   = test_df.copy() if test_df is not None else None
        self.fitted = fitted_objs or []
        self.cfg    = config

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="PreprocessingLeakageDetector")

        numeric_cols = [c for c in self.full.columns
                        if self.full[c].dtype.kind in ("i", "u", "f")]
        if not numeric_cols:
            result.error = "No numeric columns for preprocessing leakage check"
            return result

        # 1. Check fitted objects
        for obj in self.fitted:
            obj_findings = self._inspect_fitted_object(obj, numeric_cols)
            result.findings.extend(obj_findings)

        # 2. Statistical comparison: full vs train
        stat_findings = self._statistical_comparison(numeric_cols)
        result.findings.extend(stat_findings)

        # 3. Feature selection leakage
        fs_findings = self._feature_selection_leakage()
        result.findings.extend(fs_findings)

        # 4. Imputer leakage
        imp_findings = self._imputer_leakage(numeric_cols)
        result.findings.extend(imp_findings)

        # 5. Pipeline order
        for obj in self.fitted:
            pipe_findings = self._pipeline_order_check(obj)
            result.findings.extend(pipe_findings)

        result.summary = {
            "numeric_cols_checked": len(numeric_cols),
            "fitted_objects"      : len(self.fitted),
            "findings"            : len(result.findings),
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: inspect fitted sklearn object ───────────────────────────────

    def _inspect_fitted_object(
        self, obj: Any, numeric_cols: list[str]
    ) -> list[LeakageFinding]:
        findings: list[LeakageFinding] = []
        obj_name = type(obj).__name__

        if obj_name not in SCALER_CLASSES:
            return findings

        if not SKLEARN_OK or not NUMPY_OK:
            return findings

        try:
            # Get stored statistics
            stored_mean = getattr(obj, "mean_",   None)
            stored_std  = getattr(obj, "scale_",  None)
            stored_min  = getattr(obj, "data_min_", None)
            stored_max  = getattr(obj, "data_max_", None)

            if stored_mean is not None:
                # Compare stored mean to full-dataset mean vs train-only mean
                full_means  = self.full[numeric_cols].mean().values
                train_means = self.train[numeric_cols].mean().values
                n = min(len(stored_mean), len(full_means), len(train_means))

                full_diff  = float(np.abs(stored_mean[:n] - full_means[:n]).mean())
                train_diff = float(np.abs(stored_mean[:n] - train_means[:n]).mean())

                if full_diff < self.cfg.preprocessing_mean_tol * 100:
                    findings.append(LeakageFinding(
                        detector    = "PreprocessingLeakageDetector",
                        feature     = f"{obj_name}.mean_",
                        severity    = SEVERITY_CRITICAL,
                        description = (
                            f"Fitted {obj_name} statistics match FULL dataset statistics "
                            f"(mean diff from full={full_diff:.2e}, from train={train_diff:.2e}). "
                            f"The scaler was fitted BEFORE the train/test split — "
                            f"test statistics have leaked into training."
                        ),
                        evidence    = {
                            "object_type"    : obj_name,
                            "mean_diff_full" : full_diff,
                            "mean_diff_train": train_diff,
                        },
                        confidence  = 0.98,
                        fix         = (
                            "Fit the scaler ONLY on training data: "
                            "scaler.fit(X_train), then scaler.transform(X_test)."
                        ),
                    ))
        except Exception as e:
            logger.debug(f"Scaler inspection error: {e}")

        return findings

    # ── method 2: statistical comparison ─────────────────────────────────────

    def _statistical_comparison(self, numeric_cols: list[str]) -> list[LeakageFinding]:
        """
        Compare full-dataset statistics to train-only statistics.
        If they are suspiciously close, the train set IS the full dataset
        → no split was applied → test data leaked into training.
        """
        if not NUMPY_OK:
            return []
        findings: list[LeakageFinding] = []
        cols = [c for c in numeric_cols
                if c in self.full.columns and c in self.train.columns]
        if not cols:
            return []

        full_means  = self.full[cols].mean()
        train_means = self.train[cols].mean()
        full_stds   = self.full[cols].std()
        train_stds  = self.train[cols].std()

        # Relative difference
        mean_rel_diff = ((full_means - train_means).abs() / (full_means.abs() + 1e-9)).mean()
        std_rel_diff  = ((full_stds  - train_stds).abs()  / (full_stds.abs() + 1e-9)).mean()

        # If train ≈ full (rel diff < 1%) → train = full → no split
        if float(mean_rel_diff) < 0.01 and float(std_rel_diff) < 0.01:
            size_ratio = len(self.train) / max(len(self.full), 1)
            if size_ratio > 0.99:   # train is 99%+ of full data
                findings.append(LeakageFinding(
                    detector    = "PreprocessingLeakageDetector",
                    feature     = "__statistics__",
                    severity    = SEVERITY_HIGH,
                    description = (
                        f"Training set statistics are nearly identical to full dataset "
                        f"statistics (mean_rel_diff={float(mean_rel_diff):.4f}). "
                        f"Training set is {size_ratio*100:.1f}% of full data — "
                        f"preprocessors fitted on full data before split."
                    ),
                    evidence    = {
                        "mean_rel_diff" : float(mean_rel_diff),
                        "std_rel_diff"  : float(std_rel_diff),
                        "size_ratio"    : round(size_ratio, 4),
                    },
                    confidence  = 0.85,
                    fix         = (
                        "Split your data FIRST, then fit all preprocessors on train only."
                    ),
                ))
        return findings

    # ── method 3: feature selection leakage ───────────────────────────────────

    def _feature_selection_leakage(self) -> list[LeakageFinding]:
        """
        Heuristic: if the number of columns in train << full dataset columns,
        feature selection was likely performed. Check if it was done on full data.
        (We can't know for certain without a pipeline, but we flag the possibility.)
        """
        findings: list[LeakageFinding] = []
        full_cols  = set(self.full.columns)
        train_cols = set(self.train.columns)
        removed    = full_cols - train_cols

        if len(removed) > 0 and len(removed) < len(full_cols) * 0.5:
            findings.append(LeakageFinding(
                detector    = "PreprocessingLeakageDetector",
                feature     = "__feature_selection__",
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"{len(removed)} columns present in full data are absent from train. "
                    f"If feature selection was performed using the full dataset's target "
                    f"correlation, test information may have influenced feature selection."
                ),
                evidence    = {
                    "removed_cols"   : list(removed)[:10],
                    "removed_count"  : len(removed),
                },
                confidence  = 0.55,
                fix         = (
                    "Perform feature selection exclusively on training data. "
                    "Wrap selection in a Pipeline to prevent leakage."
                ),
            ))
        return findings

    # ── method 4: imputer leakage ──────────────────────────────────────────────

    def _imputer_leakage(self, numeric_cols: list[str]) -> list[LeakageFinding]:
        """
        Check if imputed means in train match full-dataset means.
        If train has NaN replaced with the full-dataset mean → imputer was
        fitted before split.
        """
        findings: list[LeakageFinding] = []
        cols = [c for c in numeric_cols
                if c in self.full.columns and c in self.train.columns
                and self.full[c].isna().sum() > 0]
        if not cols:
            return []

        full_mean  = self.full[cols].mean()
        train_mean = self.train[cols].mean()
        # If there are NO NaN in train but there were in full, imputation happened
        # Check if the imputed value matches full mean (= pre-split imputer)
        for col in cols:
            if self.train[col].isna().sum() == 0 and self.full[col].isna().sum() > 0:
                # Column was imputed in train — check if fill value = full mean
                try:
                    # Most common imputation value = mode / mean of non-null in train
                    unique_non_null = self.train[col].dropna().unique()
                    full_col_mean   = float(full_mean[col])
                    for val in unique_non_null:
                        if abs(float(val) - full_col_mean) < self.cfg.preprocessing_mean_tol:
                            findings.append(LeakageFinding(
                                detector    = "PreprocessingLeakageDetector",
                                feature     = col,
                                severity    = SEVERITY_MEDIUM,
                                description = (
                                    f"Column '{col}' was imputed with value {val:.4f} "
                                    f"which matches the full dataset mean ({full_col_mean:.4f}). "
                                    f"Imputer was likely fitted before the train/test split."
                                ),
                                evidence    = {
                                    "imputed_value"  : float(val),
                                    "full_mean"      : float(full_col_mean),
                                    "train_mean"     : float(train_mean[col]),
                                },
                                confidence  = 0.70,
                                fix         = (
                                    f"Refit the imputer on train data only: "
                                    f"imputer.fit(X_train), then imputer.transform(X_test)."
                                ),
                            ))
                            break
                except Exception:
                    pass
        return findings

    # ── method 5: pipeline order check ────────────────────────────────────────

    def _pipeline_order_check(self, obj: Any) -> list[LeakageFinding]:
        """
        If a sklearn Pipeline is provided, check that fit is called on train,
        and that target-dependent steps (e.g., SelectKBest) come after split.
        """
        findings: list[LeakageFinding] = []
        if not SKLEARN_OK:
            return findings
        try:
            from sklearn.pipeline import Pipeline as SKPipeline
            if not isinstance(obj, SKPipeline):
                return findings

            step_names = [name for name, _ in obj.steps]
            selectors  = {"SelectKBest", "SelectFromModel", "RFECV", "RFE",
                          "VarianceThreshold", "SelectPercentile"}
            for i, (name, estimator) in enumerate(obj.steps):
                est_name = type(estimator).__name__
                if est_name in selectors and i == 0:
                    findings.append(LeakageFinding(
                        detector    = "PreprocessingLeakageDetector",
                        feature     = f"Pipeline.{name}",
                        severity    = SEVERITY_HIGH,
                        description = (
                            f"Feature selector '{est_name}' is the first step in the Pipeline. "
                            f"If pipeline.fit() is called on full data, selection leaks test info."
                        ),
                        evidence    = {"step": name, "estimator": est_name, "position": i},
                        confidence  = 0.80,
                        fix         = (
                            "Call pipeline.fit(X_train, y_train) ONLY. "
                            "Never call pipeline.fit on the full dataset."
                        ),
                    ))
        except Exception:
            pass
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6 — NearDuplicateDetector
# Purpose : Find near-duplicate rows within a single dataset.
#
# Methods :
#   1. SimHash per row — 64-bit fingerprint, Hamming distance threshold
#   2. MinHash LSH — probabilistic Jaccard similarity clusters
#   3. Feature-space cosine similarity (PCA-reduced if needed)
#   4. Duplicate cluster analysis (which rows cluster together?)
#   5. Contamination rate per class
# ─────────────────────────────────────────────────────────────────────────────

class NearDuplicateDetector:
    """
    Finds near-duplicate rows within a single DataFrame.

    Unlike TrainTestOverlapDetector (which needs two DataFrames),
    this class operates on ONE DataFrame and identifies clusters of
    near-identical rows — often caused by data collection errors,
    data augmentation leakage, or copy-paste data entry.

    Methods
    -------
    1. SimHash (64-bit): fast, deterministic, Hamming-based
    2. MinHash LSH: probabilistic, Jaccard-based, scalable
    3. Cosine similarity in feature space (sklearn/scipy)
    4. Cluster analysis of duplicate groups
    5. Per-class contamination rate
    """

    def __init__(
        self,
        df         : "pd.DataFrame",
        target_col : str | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df.copy()
        self.target = target_col
        self.cfg    = config

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="NearDuplicateDetector")

        n = len(self.df)
        if n == 0:
            result.error = "Empty DataFrame"
            return result

        sample = self.df.head(self.cfg.near_dup_sample_size)

        # 1. SimHash
        simhash_clusters = self._simhash_clusters(sample)

        # 2. MinHash LSH clusters
        minhash_clusters = self._minhash_clusters(sample)

        # 3. Cosine similarity clusters
        cosine_clusters = self._cosine_clusters(sample)

        # 4. Merge cluster findings
        all_clusters = self._merge_clusters(
            simhash_clusters, minhash_clusters, cosine_clusters
        )

        total_near_dup = sum(len(c["members"]) - 1 for c in all_clusters)
        dup_rate = total_near_dup / max(n, 1)

        if all_clusters:
            sev = (SEVERITY_HIGH if dup_rate > 0.10 else
                   SEVERITY_MEDIUM if dup_rate > 0.03 else SEVERITY_LOW)
            result.add(LeakageFinding(
                detector    = "NearDuplicateDetector",
                feature     = "__rows__",
                severity    = sev,
                description = (
                    f"Found {len(all_clusters)} near-duplicate clusters "
                    f"containing {total_near_dup} extra rows "
                    f"({dup_rate*100:.2f}% of dataset). "
                    f"Near-duplicates cause train/test contamination when split."
                ),
                evidence    = {
                    "cluster_count"    : len(all_clusters),
                    "near_dup_rows"    : total_near_dup,
                    "near_dup_rate"    : round(dup_rate, 4),
                    "example_clusters" : all_clusters[:3],
                },
                confidence  = 0.85,
                fix         = (
                    "Deduplicate the dataset before any splitting. "
                    "Keep one representative row per cluster and remove the rest."
                ),
            ))

        # 5. Per-class contamination
        class_contamination = self._class_contamination(all_clusters)
        if class_contamination:
            result.add(LeakageFinding(
                detector    = "NearDuplicateDetector",
                feature     = self.target or "target",
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"Near-duplicate rows span multiple class labels "
                    f"in {len(class_contamination)} clusters. "
                    f"This is a sign of mislabelling or augmentation leakage."
                ),
                evidence    = {"multi_class_clusters": class_contamination[:5]},
                confidence  = 0.75,
                fix         = (
                    "Review and correct labels for rows in multi-class clusters."
                ),
            ))

        result.summary = {
            "rows_checked"    : len(sample),
            "clusters_found"  : len(all_clusters),
            "near_dup_rate"   : round(dup_rate, 4),
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: SimHash ────────────────────────────────────────────────────

    def _simhash_clusters(self, df: "pd.DataFrame") -> list[dict]:
        hashes = [_simhash_row(row, self.cfg.simhash_bits)
                  for _, row in df.iterrows()]
        threshold = self.cfg.simhash_hamming_threshold
        clusters: list[dict] = []
        assigned = set()
        for i in range(len(hashes)):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(i + 1, len(hashes)):
                if j in assigned:
                    continue
                if _hamming_distance(hashes[i], hashes[j]) <= threshold:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) > 1:
                assigned.add(i)
                clusters.append({
                    "method"  : "simhash",
                    "members" : cluster,
                    "size"    : len(cluster),
                })
        return clusters

    # ── method 2: MinHash LSH ────────────────────────────────────────────────

    def _minhash_clusters(self, df: "pd.DataFrame") -> list[dict]:
        n_perm = self.cfg.minhash_num_perm
        thresh = self.cfg.minhash_threshold
        sigs   = [TrainTestOverlapDetector._minhash(row, n_perm)
                  for _, row in df.iterrows()]
        clusters: list[dict] = []
        assigned = set()
        for i in range(len(sigs)):
            if i in assigned:
                continue
            cluster = [i]
            for j in range(i + 1, min(len(sigs), i + 200)):  # local window
                if j in assigned:
                    continue
                sim = TrainTestOverlapDetector._jaccard_from_minhash(sigs[i], sigs[j])
                if sim >= thresh:
                    cluster.append(j)
                    assigned.add(j)
            if len(cluster) > 1:
                assigned.add(i)
                clusters.append({
                    "method"  : "minhash",
                    "members" : cluster,
                    "size"    : len(cluster),
                })
        return clusters

    # ── method 3: cosine similarity ───────────────────────────────────────────

    def _cosine_clusters(self, df: "pd.DataFrame") -> list[dict]:
        if not SKLEARN_OK or not NUMPY_OK:
            return []
        numeric_cols = [c for c in df.columns
                        if df[c].dtype.kind in ("i", "u", "f")]
        if len(numeric_cols) < 2:
            return []
        try:
            X = df[numeric_cols].fillna(0).values.astype(float)
            # PCA reduce to 50 dims max for speed
            if X.shape[1] > 50:
                pca = PCA(n_components=50, random_state=42)
                X   = pca.fit_transform(X)
            X_normed = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            sim_matrix = X_normed @ X_normed.T
            clusters: list[dict] = []
            assigned = set()
            for i in range(len(X)):
                if i in assigned:
                    continue
                similar = [j for j in range(i+1, len(X))
                           if j not in assigned
                           and sim_matrix[i, j] > 0.99]
                if similar:
                    cluster = [i] + similar
                    for j in similar:
                        assigned.add(j)
                    assigned.add(i)
                    clusters.append({
                        "method"  : "cosine",
                        "members" : cluster,
                        "size"    : len(cluster),
                    })
            return clusters
        except Exception:
            return []

    @staticmethod
    def _merge_clusters(
        *cluster_lists: list[dict],
    ) -> list[dict]:
        """Merge clusters from different methods by union-find."""
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), x)
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for cluster_list in cluster_lists:
            for cluster in cluster_list:
                members = cluster["members"]
                for j in range(1, len(members)):
                    union(members[0], members[j])

        groups: dict[int, list[int]] = collections.defaultdict(list)
        all_members = set()
        for cluster_list in cluster_lists:
            for cluster in cluster_list:
                for m in cluster["members"]:
                    all_members.add(m)
        for m in all_members:
            groups[find(m)].append(m)

        return [
            {"method": "merged", "members": sorted(v), "size": len(v)}
            for v in groups.values() if len(v) > 1
        ]

    def _class_contamination(self, clusters: list[dict]) -> list[dict]:
        if not self.target or self.target not in self.df.columns:
            return []
        multi: list[dict] = []
        for cluster in clusters:
            idxs   = cluster["members"]
            labels = self.df.iloc[idxs][self.target].unique()
            if len(labels) > 1:
                multi.append({
                    "members": idxs,
                    "labels" : [str(l) for l in labels],
                })
        return multi


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7 — FeatureCorrelationLeakageDetector
# Purpose : Detect multicollinearity and proxy features.
#
# Methods :
#   1. Pairwise Pearson/Spearman correlation matrix
#   2. VIF (Variance Inflation Factor) — detects linear combinations
#   3. Proxy feature detection (one feature perfectly predicts another)
#   4. Redundant feature clustering (agglomerative by correlation)
#   5. Lagged correlation (feature at time t-1 predicts feature at t)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureCorrelationLeakageDetector:
    """
    Detects features that are linear combinations of or proxies for other features.

    VIF Analysis
    ------------
    VIF for feature X_i = 1 / (1 - R²), where R² is obtained by
    regressing X_i on all other features.
    VIF > 10 → severe multicollinearity (feature is almost redundant).
    VIF > 5  → moderate multicollinearity.

    Proxy Detection
    ---------------
    A feature A is a proxy for B if corr(A, B) > threshold (default 0.98).
    This means A and B carry essentially the same information → one is redundant.
    If one was engineered from a target-correlated feature, it constitutes leakage.

    Lagged Correlation
    ------------------
    For time-series: if X[t] correlates with X[t+k] very highly,
    rolling or lag features may encode future information.
    """

    def __init__(
        self,
        df         : "pd.DataFrame",
        target_col : str | None = None,
        config     : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas required")
        self.df     = df.copy()
        self.target = target_col
        self.cfg    = config

    # ── public API ───────────────────────────────────────────────────────────

    def detect(self) -> DetectorResult:
        t0     = time.perf_counter()
        result = DetectorResult(detector_name="FeatureCorrelationLeakageDetector")

        numeric_cols = [c for c in self.df.columns
                        if c != self.target
                        and self.df[c].dtype.kind in ("i", "u", "f")
                        and self.df[c].nunique() > 1]
        if len(numeric_cols) < 2:
            result.error = "Need at least 2 numeric features"
            return result

        # 1. Pairwise correlation
        corr_pairs = self._pairwise_correlation(numeric_cols)

        # 2. VIF
        vif_scores = self._compute_vif(numeric_cols)

        # 3. Proxy detection
        proxy_pairs = self._proxy_detection(corr_pairs)

        # 4. Redundant clusters
        clusters = self._redundant_clusters(numeric_cols, corr_pairs)

        # 5. Lagged correlation
        lagged = self._lagged_correlation(numeric_cols)

        # ── generate findings ────────────────────────────────────────────────

        # High-correlation pairs
        high_corr = [(a, b, c) for a, b, c in corr_pairs
                     if abs(c) >= self.cfg.proxy_corr_threshold]
        if high_corr:
            result.add(LeakageFinding(
                detector    = "FeatureCorrelationLeakageDetector",
                feature     = ", ".join(f"{a}↔{b}" for a, b, _ in high_corr[:3]),
                severity    = SEVERITY_HIGH,
                description = (
                    f"{len(high_corr)} feature pairs have correlation ≥ "
                    f"{self.cfg.proxy_corr_threshold}. "
                    f"Near-perfect correlation suggests one feature is a linear "
                    f"transformation of the other — potential proxy leakage."
                ),
                evidence    = {
                    "high_corr_pairs": [
                        {"feat_a": a, "feat_b": b, "correlation": round(c, 4)}
                        for a, b, c in high_corr[:10]
                    ],
                    "count": len(high_corr),
                },
                confidence  = 0.85,
                fix         = (
                    "Remove one feature from each highly-correlated pair. "
                    "Keep the one with more direct business meaning."
                ),
            ))

        # High VIF features
        high_vif = {col: vif for col, vif in vif_scores.items()
                    if vif > self.cfg.vif_threshold}
        if high_vif:
            sev = SEVERITY_CRITICAL if any(v > 50 for v in high_vif.values()) else SEVERITY_HIGH
            result.add(LeakageFinding(
                detector    = "FeatureCorrelationLeakageDetector",
                feature     = ", ".join(list(high_vif.keys())[:5]),
                severity    = sev,
                description = (
                    f"{len(high_vif)} features have VIF > {self.cfg.vif_threshold}. "
                    f"High VIF means the feature is nearly a linear combination "
                    f"of other features — severe multicollinearity."
                ),
                evidence    = {
                    "high_vif_features": {
                        col: round(vif, 2)
                        for col, vif in sorted(high_vif.items(),
                                               key=lambda x: -x[1])
                    },
                    "vif_threshold": self.cfg.vif_threshold,
                },
                confidence  = 0.90,
                fix         = (
                    "Remove high-VIF features or apply PCA to collapse "
                    "correlated feature groups into orthogonal components."
                ),
            ))

        # Proxy pairs
        for proxy_info in proxy_pairs[:10]:
            result.add(LeakageFinding(
                detector    = "FeatureCorrelationLeakageDetector",
                feature     = proxy_info["feature"],
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"Feature '{proxy_info['feature']}' appears to be a proxy for "
                    f"'{proxy_info['proxy_of']}' "
                    f"(correlation={proxy_info['correlation']:.4f}). "
                    f"If '{proxy_info['proxy_of']}' encodes target info, "
                    f"so does '{proxy_info['feature']}'."
                ),
                evidence    = proxy_info,
                confidence  = 0.70,
                fix         = (
                    f"Verify whether '{proxy_info['feature']}' was engineered "
                    f"from '{proxy_info['proxy_of']}'. If so, remove one."
                ),
            ))

        # Redundant clusters
        for cluster in clusters:
            if len(cluster["features"]) > 2:
                result.add(LeakageFinding(
                    detector    = "FeatureCorrelationLeakageDetector",
                    feature     = cluster["features"][0],
                    severity    = SEVERITY_LOW,
                    description = (
                        f"Redundant feature cluster of {len(cluster['features'])} features: "
                        f"{cluster['features'][:5]}. "
                        f"High mutual correlation — keep only one representative."
                    ),
                    evidence    = cluster,
                    confidence  = 0.70,
                    fix         = "Keep one feature from this cluster; remove the rest.",
                ))

        # Lagged correlations
        for lag_info in lagged:
            result.add(LeakageFinding(
                detector    = "FeatureCorrelationLeakageDetector",
                feature     = lag_info["feature"],
                severity    = SEVERITY_MEDIUM,
                description = (
                    f"Feature '{lag_info['feature']}' has lag-{lag_info['lag']} "
                    f"autocorrelation of {lag_info['autocorr']:.4f}. "
                    f"If lag features were created without proper temporal splitting, "
                    f"this constitutes temporal leakage."
                ),
                evidence    = lag_info,
                confidence  = 0.65,
                fix         = (
                    "Verify lag features are computed using only past data. "
                    "Apply temporal cross-validation for time-series models."
                ),
            ))

        result.summary = {
            "features_checked"  : len(numeric_cols),
            "high_corr_pairs"   : len(high_corr),
            "high_vif_features" : len(high_vif),
            "proxy_pairs"       : len(proxy_pairs),
            "redundant_clusters": len(clusters),
            "lagged_issues"     : len(lagged),
            "full_corr_matrix"  : self._corr_matrix_dict(numeric_cols),
            "vif_scores"        : {col: round(v, 2) for col, v in vif_scores.items()},
        }
        result.runtime_sec = round(time.perf_counter() - t0, 3)
        return result

    # ── method 1: pairwise correlation ───────────────────────────────────────

    def _pairwise_correlation(
        self, cols: list[str]
    ) -> list[tuple[str, str, float]]:
        pairs: list[tuple[str, str, float]] = []
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                try:
                    xa = self.df[a].dropna().values.astype(float)
                    xb = self.df[b].dropna().values.astype(float)
                    n  = min(len(xa), len(xb))
                    xa, xb = xa[:n], xb[:n]
                    if n < 5:
                        continue
                    if SCIPY_OK:
                        corr, _ = spearmanr(xa, xb)
                    else:
                        corr = _pearson_fallback(xa, xb)
                    if not math.isnan(float(corr)):
                        pairs.append((a, b, float(corr)))
                except Exception:
                    pass
        pairs.sort(key=lambda x: -abs(x[2]))
        return pairs

    # ── method 2: VIF ────────────────────────────────────────────────────────

    def _compute_vif(self, cols: list[str]) -> dict[str, float]:
        """
        VIF_i = 1 / (1 - R²_i)
        where R²_i is from regressing feature i on all other features.
        VIF > 10 = severe multicollinearity.
        """
        if not SKLEARN_OK or not NUMPY_OK:
            return {}
        vif: dict[str, float] = {}
        sub = self.df[cols].dropna()
        if len(sub) < len(cols) + 5:
            return {}
        X = sub.values.astype(float)
        for i, col in enumerate(cols):
            try:
                y_i   = X[:, i]
                X_rest= np.delete(X, i, axis=1)
                model = LinearRegression()
                model.fit(X_rest, y_i)
                y_pred= model.predict(X_rest)
                ss_res= float(np.sum((y_i - y_pred) ** 2))
                ss_tot= float(np.sum((y_i - y_i.mean()) ** 2))
                r2    = 1 - ss_res / max(ss_tot, 1e-9)
                vif_i = 1.0 / max(1.0 - r2, 1e-9)
                vif[col] = float(vif_i)
            except Exception:
                vif[col] = 1.0
        return vif

    # ── method 3: proxy detection ─────────────────────────────────────────────

    def _proxy_detection(
        self, corr_pairs: list[tuple[str, str, float]]
    ) -> list[dict]:
        threshold = self.cfg.proxy_corr_threshold
        proxies: list[dict] = []
        for a, b, corr in corr_pairs:
            if abs(corr) >= threshold:
                # Determine which is the "proxy" (usually the one with lower MI with target)
                proxies.append({
                    "feature"    : a,
                    "proxy_of"   : b,
                    "correlation": round(abs(corr), 4),
                    "direction"  : "positive" if corr > 0 else "negative",
                })
        return proxies

    # ── method 4: redundant clusters ─────────────────────────────────────────

    def _redundant_clusters(
        self, cols: list[str], corr_pairs: list[tuple[str, str, float]]
    ) -> list[dict]:
        """
        Build adjacency graph where edges are high-correlation pairs.
        Connected components = redundant feature clusters.
        """
        threshold = self.cfg.redundancy_cluster_threshold
        adj: dict[str, set[str]] = {c: set() for c in cols}
        for a, b, corr in corr_pairs:
            if abs(corr) >= threshold:
                adj[a].add(b)
                adj[b].add(a)

        visited: set[str] = set()
        clusters: list[dict] = []
        for col in cols:
            if col in visited or not adj[col]:
                continue
            # BFS
            queue   = collections.deque([col])
            cluster = []
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                queue.extend(adj[node] - visited)
            if len(cluster) > 1:
                clusters.append({
                    "features"   : cluster,
                    "size"       : len(cluster),
                    "threshold"  : threshold,
                })
        return clusters

    # ── method 5: lagged correlation ──────────────────────────────────────────

    def _lagged_correlation(
        self, cols: list[str], max_lag: int = 5
    ) -> list[dict]:
        issues: list[dict] = []
        for col in cols:
            vals = self.df[col].dropna().values.astype(float)
            if len(vals) < max_lag + 5:
                continue
            for lag in range(1, max_lag + 1):
                x = vals[:-lag]
                y = vals[lag:]
                try:
                    if SCIPY_OK:
                        corr, _ = pearsonr(x, y)
                    else:
                        corr = _pearson_fallback(x, y)
                    if abs(corr) > 0.97:
                        issues.append({
                            "feature" : col,
                            "lag"     : lag,
                            "autocorr": round(float(corr), 4),
                        })
                        break   # report first high-autocorr lag only
                except Exception:
                    pass
        return issues

    def _corr_matrix_dict(self, cols: list[str]) -> dict[str, dict[str, float]]:
        """Return pairwise correlation as nested dict for JSON serialisation."""
        try:
            cm = self.df[cols].corr(method="spearman")
            return {
                row: {col: round(float(val), 4) for col, val in cm[row].items()}
                for row in cm.index
            }
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 8 — LeakageScorer
# Purpose : Aggregate all findings into a composite 0-100 leakage score.
#
# Logic   :
#   Base score = 100 (no leakage).
#   Each finding subtracts a severity-weighted penalty.
#   Penalties are capped at 100 total.
#   Confidence multiplier: low-confidence findings subtract less.
#   Final score = max(0, 100 - total_penalty).
# ─────────────────────────────────────────────────────────────────────────────

class LeakageScorer:
    """
    Converts all DetectorResult findings into a composite Leakage Risk Score.

    Score = 100 means no leakage detected.
    Score = 0   means severe, confirmed leakage across multiple dimensions.

    Formula
    -------
    For each finding:
        penalty = SEVERITY_SCORE_PENALTY[severity] × confidence × weight

    where weight = 1.0 if is_confirmed else 0.5

    Final score = max(0, min(100, 100 - sum(penalties)))

    Grade
    -----
    90-100 : Safe ✅
    70-89  : Low Risk 🟡
    50-69  : Moderate Risk 🟠
    30-49  : High Risk 🔴
    0-29   : Critical Risk ⛔
    """

    GRADES = [
        (90, "Safe",          "✅ No significant leakage detected"),
        (70, "Low Risk",      "🟡 Minor leakage signals — investigate"),
        (50, "Moderate Risk", "🟠 Leakage likely — review flagged features"),
        (30, "High Risk",     "🔴 Confirmed leakage — re-engineer features"),
        (0,  "Critical Risk", "⛔ Severe leakage — model results are invalid"),
    ]

    def __init__(
        self,
        results: list[DetectorResult],
        config : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        self.results = results
        self.cfg     = config

    # ── public API ───────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        all_findings = [f for r in self.results for f in r.findings]

        # Per-finding penalty
        penalties: list[dict] = []
        total_penalty = 0.0
        for finding in all_findings:
            base_penalty = SEVERITY_SCORE_PENALTY.get(finding.severity, 1)
            weight       = 1.0 if finding.is_confirmed else 0.5
            penalty      = base_penalty * finding.confidence * weight
            total_penalty += penalty
            penalties.append({
                "detector"   : finding.detector,
                "feature"    : finding.feature,
                "severity"   : finding.severity,
                "confidence" : finding.confidence,
                "penalty"    : round(penalty, 2),
            })

        total_penalty = min(total_penalty, self.cfg.max_penalty)
        score         = max(0.0, 100.0 - total_penalty)
        grade, label  = self._grade(score)

        # Per-detector breakdown
        per_detector: dict[str, dict] = {}
        for result in self.results:
            if result.error:
                per_detector[result.detector_name] = {"error": result.error}
                continue
            det_penalty = sum(
                p["penalty"] for p in penalties
                if p["detector"] == result.detector_name
            )
            per_detector[result.detector_name] = {
                "findings"     : len(result.findings),
                "critical"     : result.critical_count,
                "high"         : result.high_count,
                "penalty"      : round(det_penalty, 2),
                "runtime_sec"  : result.runtime_sec,
            }

        # Severity distribution
        sev_dist: dict[str, int] = collections.Counter(
            f.severity for f in all_findings
        )

        # False positive estimation
        fp_rate = self._estimate_false_positive_rate(all_findings)

        return {
            "leakage_risk_score"  : round(score, 2),
            "grade"               : grade,
            "label"               : label,
            "total_penalty"       : round(total_penalty, 2),
            "total_findings"      : len(all_findings),
            "severity_distribution": dict(sev_dist),
            "per_detector"        : per_detector,
            "penalty_breakdown"   : penalties,
            "estimated_fp_rate"   : round(fp_rate, 4),
            "confidence_adjusted" : True,
        }

    @classmethod
    def _grade(cls, score: float) -> tuple[str, str]:
        for thresh, grade, label in cls.GRADES:
            if score >= thresh:
                return grade, label
        return "Critical Risk", "⛔ Severe leakage"

    @staticmethod
    def _estimate_false_positive_rate(findings: list[LeakageFinding]) -> float:
        """
        Estimate fraction of findings that may be false positives.
        Low-confidence + medium/low severity → higher FP probability.
        """
        if not findings:
            return 0.0
        fp_scores = []
        for f in findings:
            # FP probability = (1 - confidence) * severity_factor
            sev_factor = {
                SEVERITY_CRITICAL: 0.05,
                SEVERITY_HIGH    : 0.10,
                SEVERITY_MEDIUM  : 0.25,
                SEVERITY_LOW     : 0.40,
                SEVERITY_INFO    : 0.60,
            }.get(f.severity, 0.30)
            fp_scores.append((1.0 - f.confidence) * sev_factor)
        return sum(fp_scores) / len(fp_scores)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 9 — LeakageReporter
# Purpose : Generate human-readable reports from all findings.
#
# Formats :
#   • Markdown — GitHub/Notion ready
#   • JSON     — machine-readable, stable schema
#   • HTML     — styled, self-contained, no external deps
# ─────────────────────────────────────────────────────────────────────────────

class LeakageReporter:
    """
    Generates multi-format leakage reports.

    Takes the output of LeakageScorer + all DetectorResults
    and renders them as Markdown, JSON, and HTML.

    Features
    --------
    • Severity-sorted issue table
    • Per-feature issue list with fix recommendations
    • Score gauge and grade badge
    • Collapsible HTML sections
    • Stable JSON schema for downstream consumption
    """

    def __init__(
        self,
        score_result  : dict[str, Any],
        detector_results: list[DetectorResult],
        dataset_info  : dict[str, Any] | None = None,
    ) -> None:
        self.score    = score_result
        self.results  = detector_results
        self.info     = dataset_info or {}
        self.all_findings: list[LeakageFinding] = [
            f for r in detector_results for f in r.findings
        ]
        self.all_findings.sort(
            key=lambda f: SEVERITY_ORDER.index(f.severity)
        )

    # ── public API ───────────────────────────────────────────────────────────

    def to_markdown(self) -> str:
        lines = [
            "# 🩺 Nydra — Data Leakage Report\n",
            f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> Nydra v{VERSION}\n",
            "---\n",
            self._md_score_section(),
            self._md_findings_table(),
            self._md_per_detector(),
            self._md_recommendations(),
            "---\n",
            "*Built with ❤️ by Nydra Team — Algeria 🇩🇿*\n",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "report_generated"  : datetime.now().isoformat(),
            "nydra_version": VERSION,
            "dataset_info"      : self.info,
            "score"             : self.score,
            "findings"          : [
                {
                    "detector"   : f.detector,
                    "feature"    : f.feature,
                    "severity"   : f.severity,
                    "description": f.description,
                    "evidence"   : f.evidence,
                    "confidence" : f.confidence,
                    "fix"        : f.fix,
                    "confirmed"  : f.is_confirmed,
                }
                for f in self.all_findings
            ],
            "per_detector_summary": {
                r.detector_name: {
                    "findings"   : len(r.findings),
                    "runtime_sec": r.runtime_sec,
                    "error"      : r.error,
                }
                for r in self.results
            },
        }
        return json.dumps(_sanitize_for_json(payload), indent=2, ensure_ascii=False)

    def to_html(self) -> str:
        score_val = float(self.score.get("leakage_risk_score", 100))
        grade     = self.score.get("grade", "?")
        label     = self.score.get("label", "")
        color     = self._grade_color(grade)
        n_findings= len(self.all_findings)
        n_critical= sum(1 for f in self.all_findings if f.severity == SEVERITY_CRITICAL)

        findings_rows = "".join(
            f'<tr>'
            f'<td class="sev-{f.severity}">'
            f'{SEVERITY_EMOJI.get(f.severity,"")} {f.severity.title()}</td>'
            f'<td><strong>{f.feature}</strong></td>'
            f'<td>{f.detector.replace("Detector","")}</td>'
            f'<td>{f.description[:120]}…</td>'
            f'<td>{round(f.confidence*100)}%</td>'
            f'</tr>'
            for f in self.all_findings[:50]
        )

        det_rows = "".join(
            f'<tr><td>{r.detector_name.replace("Detector","")}</td>'
            f'<td>{len(r.findings)}</td>'
            f'<td>{r.critical_count}</td>'
            f'<td>{r.high_count}</td>'
            f'<td>{r.runtime_sec}s</td>'
            f'<td>{"✅" if not r.error else "❌ "+str(r.error)[:40]}</td></tr>'
            for r in self.results
        )

        rec_items = "".join(
            f'<li class="sev-{f.severity}"><strong>{f.feature}</strong>: {f.fix}</li>'
            for f in self.all_findings
            if f.severity in (SEVERITY_CRITICAL, SEVERITY_HIGH)
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Nydra — Leakage Report</title>
<style>
  :root{{--bg:#0f1117;--surf:#1e2130;--bord:#2d3250;
        --txt:#e8eaf0;--muted:#9099b0;--accent:#4f8ef7;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--txt);
       font-family:'Segoe UI',system-ui,sans-serif;}}
  .container{{max-width:1100px;margin:auto;padding:32px 20px;}}
  h1{{font-size:2rem;margin-bottom:4px;}}
  h2{{font-size:1.2rem;color:var(--accent);margin:28px 0 10px;}}
  .meta{{color:var(--muted);font-size:0.85rem;margin-bottom:24px;}}
  .badge{{display:inline-block;padding:6px 20px;border-radius:8px;
          font-size:2.5rem;font-weight:900;color:{color};
          border:3px solid {color};}}
  .card{{background:var(--surf);border:1px solid var(--bord);
         border-radius:12px;padding:20px;margin-bottom:18px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  .stat{{text-align:center;padding:12px;}}
  .stat-val{{font-size:1.8rem;font-weight:700;color:var(--accent);}}
  .stat-lbl{{font-size:0.8rem;color:var(--muted);}}
  table{{width:100%;border-collapse:collapse;font-size:0.85rem;}}
  th{{background:#252840;padding:8px 12px;text-align:left;color:var(--muted);}}
  td{{padding:8px 12px;border-top:1px solid var(--bord);}}
  tr:hover td{{background:#252840;}}
  .sev-critical{{color:#e74c3c;}} .sev-high{{color:#e67e22;}}
  .sev-medium{{color:#f1c40f;}}   .sev-low{{color:#2ecc71;}}
  details summary{{cursor:pointer;font-weight:600;color:var(--accent);}}
  ul.recs{{list-style:none;padding:0;}}
  ul.recs li{{padding:8px 0;border-bottom:1px solid var(--bord);font-size:0.9rem;}}
</style>
</head>
<body>
<div class="container">
  <h1>🩺 Nydra — Leakage Report</h1>
  <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp;
     Nydra v{VERSION}</p>

  <div class="card">
    <div style="display:flex;align-items:center;gap:32px;flex-wrap:wrap">
      <div class="badge">{grade}</div>
      <div>
        <h2 style="margin:0">{label}</h2>
        <p style="color:var(--muted)">Leakage Risk Score:
          <strong style="color:{color}">{score_val}/100</strong></p>
        <p style="color:var(--muted)">Total Findings: <strong>{n_findings}</strong>
          &nbsp;|&nbsp; Critical: <strong style="color:#e74c3c">{n_critical}</strong></p>
      </div>
    </div>
  </div>

  <div class="card grid-2">
    <div class="stat"><div class="stat-val">{n_findings}</div>
      <div class="stat-lbl">Total Findings</div></div>
    <div class="stat"><div class="stat-val" style="color:#e74c3c">{n_critical}</div>
      <div class="stat-lbl">Critical Issues</div></div>
    <div class="stat"><div class="stat-val">{len(self.results)}</div>
      <div class="stat-lbl">Detectors Run</div></div>
    <div class="stat">
      <div class="stat-val">{round(self.score.get('estimated_fp_rate',0)*100)}%</div>
      <div class="stat-lbl">Est. False Positive Rate</div></div>
  </div>

  <details open>
    <summary><h2 style="display:inline">⚠️ All Findings ({n_findings})</h2></summary>
    <div class="card" style="margin-top:12px">
      <table>
        <tr><th>Severity</th><th>Feature</th><th>Detector</th>
            <th>Description</th><th>Confidence</th></tr>
        {findings_rows or '<tr><td colspan="5" style="color:#2ecc71">✅ No leakage found!</td></tr>'}
      </table>
    </div>
  </details>

  <details open>
    <summary><h2 style="display:inline">🔬 Detector Summary</h2></summary>
    <div class="card" style="margin-top:12px">
      <table>
        <tr><th>Detector</th><th>Findings</th><th>Critical</th>
            <th>High</th><th>Runtime</th><th>Status</th></tr>
        {det_rows}
      </table>
    </div>
  </details>

  <details open>
    <summary><h2 style="display:inline">💡 Recommendations</h2></summary>
    <div class="card" style="margin-top:12px">
      <ul class="recs">
        {rec_items or '<li style="color:#2ecc71">✅ No critical fixes required</li>'}
      </ul>
    </div>
  </details>

  <p class="meta" style="margin-top:40px;text-align:center">
    Built with ❤️ by <a href="https://github.com/Denterio1/Nydra"
    style="color:var(--accent)">Nydra Team</a> — Algeria 🇩🇿
  </p>
</div>
</body>
</html>"""

    def export(
        self,
        output_dir: str | Path = "reports",
        prefix    : str = "leakage_report",
        formats   : list[str] = ("md", "json", "html"),
    ) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        base  = out / f"{prefix}_{ts}"
        paths: dict[str, Path] = {}

        if "md" in formats:
            p = base.with_suffix(".md")
            p.write_text(self.to_markdown(), encoding="utf-8")
            paths["md"] = p

        if "json" in formats:
            p = base.with_suffix(".json")
            p.write_text(self.to_json(), encoding="utf-8")
            paths["json"] = p

        if "html" in formats:
            p = base.with_suffix(".html")
            p.write_text(self.to_html(), encoding="utf-8")
            paths["html"] = p

        return paths

    # ── section helpers ───────────────────────────────────────────────────────

    def _md_score_section(self) -> str:
        s     = self.score
        grade = s.get("grade", "?")
        label = s.get("label", "")
        score = s.get("leakage_risk_score", 100)
        return (
            f"## 🏆 Leakage Risk Score\n\n"
            f"**Grade: `{grade}` — {label}**  \n"
            f"Score: **{score}/100**  \n"
            f"Total Findings: **{s.get('total_findings', 0)}**  \n"
            f"Estimated False Positive Rate: **{round(s.get('estimated_fp_rate',0)*100)}%**\n\n"
            f"---\n"
        )

    def _md_findings_table(self) -> str:
        if not self.all_findings:
            return "## ⚠️ Findings\n\n✅ No leakage detected!\n\n---\n"
        rows = [
            "## ⚠️ Findings\n",
            "| Severity | Feature | Detector | Description | Confidence |",
            "|----------|---------|---------|-------------|-----------|",
        ]
        for f in self.all_findings[:50]:
            emoji = SEVERITY_EMOJI.get(f.severity, "")
            rows.append(
                f"| {emoji} {f.severity.title()} | `{f.feature}` "
                f"| {f.detector.replace('Detector','')} "
                f"| {f.description[:80]}… "
                f"| {round(f.confidence*100)}% |"
            )
        return "\n".join(rows) + "\n\n---\n"

    def _md_per_detector(self) -> str:
        lines = ["## 🔬 Per-Detector Summary\n",
                 "| Detector | Findings | Critical | High | Runtime |",
                 "|---------|---------|---------|------|---------|"]
        for r in self.results:
            lines.append(
                f"| {r.detector_name.replace('Detector','')} "
                f"| {len(r.findings)} "
                f"| {r.critical_count} "
                f"| {r.high_count} "
                f"| {r.runtime_sec}s |"
            )
        return "\n".join(lines) + "\n\n---\n"

    def _md_recommendations(self) -> str:
        critical_high = [f for f in self.all_findings
                         if f.severity in (SEVERITY_CRITICAL, SEVERITY_HIGH)]
        if not critical_high:
            return "## 💡 Recommendations\n\n✅ No critical fixes required.\n\n---\n"
        lines = ["## 💡 Recommendations (Priority Order)\n",
                 "| # | Feature | Severity | Action |",
                 "|---|---------|---------|--------|"]
        for i, f in enumerate(critical_high, 1):
            emoji = SEVERITY_EMOJI.get(f.severity, "")
            lines.append(
                f"| {i} | `{f.feature}` | {emoji} {f.severity.title()} "
                f"| {f.fix[:100]} |"
            )
        return "\n".join(lines) + "\n\n---\n"

    @staticmethod
    def _grade_color(grade: str) -> str:
        return {
            "Safe"         : "#2ecc71",
            "Low Risk"     : "#f1c40f",
            "Moderate Risk": "#e67e22",
            "High Risk"    : "#e74c3c",
            "Critical Risk": "#8e44ad",
        }.get(grade, "#9099b0")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 10 — LeakageOrchestrator
# Purpose : Master class — runs all detectors, one-call API.
#
# Usage   :
#   orch = LeakageOrchestrator(df, target_col="churn",
#                              train_df=train, test_df=test)
#   result = orch.run()
#   print(result["score"])
#   orch.export_report("reports/")
# ─────────────────────────────────────────────────────────────────────────────

class LeakageOrchestrator:
    """
    Runs all 7 leakage detectors, scores results, and generates reports.

    Usage (minimal)
    ---------------
    orch = LeakageOrchestrator(df=my_df, target_col="y")
    result = orch.run()

    Usage (with splits)
    -------------------
    orch = LeakageOrchestrator(
        df         = full_df,
        target_col = "churn",
        train_df   = train_df,
        test_df    = test_df,
        time_col   = "event_date",
    )
    result = orch.run()
    paths  = orch.export_report("reports/", formats=["html","json","md"])
    """

    def __init__(
        self,
        df          : "pd.DataFrame",
        target_col  : str | None = None,
        train_df    : "pd.DataFrame | None" = None,
        test_df     : "pd.DataFrame | None" = None,
        time_col    : str | None = None,
        group_cols  : list[str] | None = None,
        fitted_objs : list[Any] | None = None,
        config      : LeakageConfig = DEFAULT_CONFIG,
    ) -> None:
        if not PANDAS_OK:
            raise ImportError("pandas is required for LeakageOrchestrator")
        self.df          = df
        self.target_col  = target_col or self._auto_detect_target()
        self.train_df    = train_df
        self.test_df     = test_df
        self.time_col    = time_col
        self.group_cols  = group_cols
        self.fitted_objs = fitted_objs or []
        self.cfg         = config
        self._results   : list[DetectorResult] = []
        self._score     : dict[str, Any] = {}

    # ── public API ───────────────────────────────────────────────────────────

    def run(
        self,
        detectors: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run all (or specified) detectors.

        Parameters
        ----------
        detectors : list of detector names to run, or None for all.
                    Options: "target","overlap","temporal","group",
                             "preprocessing","near_duplicate","correlation"

        Returns
        -------
        Full result dict with score, findings, and per-detector summaries.
        """
        run_all = detectors is None
        self._results = []

        # 1. TargetLeakage
        if (run_all or "target" in detectors) and self.target_col:
            result = self._run_safe(
                "TargetLeakageDetector",
                lambda: TargetLeakageDetector(
                    self.df, self.target_col, self.cfg
                ).detect(),
            )
            self._results.append(result)

        # 2. TrainTestOverlap
        if (run_all or "overlap" in detectors):
            if self.train_df is not None and self.test_df is not None:
                result = self._run_safe(
                    "TrainTestOverlapDetector",
                    lambda: TrainTestOverlapDetector(
                        self.train_df, self.test_df,
                        self.target_col, self.cfg
                    ).detect(),
                )
                self._results.append(result)
            else:
                self._results.append(DetectorResult(
                    detector_name="TrainTestOverlapDetector",
                    error="train_df and test_df required"
                ))

        # 3. TemporalLeakage
        if run_all or "temporal" in detectors:
            result = self._run_safe(
                "TemporalLeakageDetector",
                lambda: TemporalLeakageDetector(
                    self.df, self.time_col, self.target_col, self.cfg
                ).detect(),
            )
            self._results.append(result)

        # 4. GroupLeakage
        if run_all or "group" in detectors:
            if self.train_df is not None and self.test_df is not None:
                result = self._run_safe(
                    "GroupLeakageDetector",
                    lambda: GroupLeakageDetector(
                        self.train_df, self.test_df,
                        self.group_cols, self.target_col, self.cfg
                    ).detect(),
                )
            else:
                result = DetectorResult(
                    detector_name="GroupLeakageDetector",
                    error="train_df and test_df required"
                )
            self._results.append(result)

        # 5. PreprocessingLeakage
        if run_all or "preprocessing" in detectors:
            if self.train_df is not None:
                result = self._run_safe(
                    "PreprocessingLeakageDetector",
                    lambda: PreprocessingLeakageDetector(
                        self.df, self.train_df, self.test_df,
                        self.fitted_objs, self.cfg
                    ).detect(),
                )
                self._results.append(result)
            else:
                self._results.append(DetectorResult(
                    detector_name="PreprocessingLeakageDetector",
                    error="train_df required"
                ))

        # 6. NearDuplicate
        if run_all or "near_duplicate" in detectors:
            result = self._run_safe(
                "NearDuplicateDetector",
                lambda: NearDuplicateDetector(
                    self.df, self.target_col, self.cfg
                ).detect(),
            )
            self._results.append(result)

        # 7. FeatureCorrelationLeakage
        if run_all or "correlation" in detectors:
            result = self._run_safe(
                "FeatureCorrelationLeakageDetector",
                lambda: FeatureCorrelationLeakageDetector(
                    self.df, self.target_col, self.cfg
                ).detect(),
            )
            self._results.append(result)

        # Score all findings
        self._score = LeakageScorer(self._results, self.cfg).compute()

        return {
            "score"           : self._score,
            "detector_results": {r.detector_name: r.summary for r in self._results},
            "all_findings"    : [
                {
                    "detector"  : f.detector,
                    "feature"   : f.feature,
                    "severity"  : f.severity,
                    "description": f.description,
                    "fix"       : f.fix,
                    "confidence": f.confidence,
                }
                for r in self._results for f in r.findings
            ],
        }

    def export_report(
        self,
        output_dir: str | Path = "reports",
        formats   : list[str] = ("md", "json", "html"),
        prefix    : str = "leakage_report",
    ) -> dict[str, Path]:
        """Generate and save reports in specified formats."""
        if not self._score:
            self.run()
        reporter = LeakageReporter(
            score_result     = self._score,
            detector_results = self._results,
            dataset_info     = {
                "rows"   : len(self.df),
                "cols"   : len(self.df.columns),
                "target" : self.target_col,
            },
        )
        return reporter.export(output_dir=output_dir, formats=formats, prefix=prefix)

    def summary(self) -> str:
        """Quick text summary of leakage findings."""
        if not self._score:
            self.run()
        score = self._score.get("leakage_risk_score", 100)
        grade = self._score.get("grade", "?")
        label = self._score.get("label", "")
        n     = self._score.get("total_findings", 0)
        lines = [
            f"{'='*60}",
            f"  Nydra — Leakage Risk Score: {score}/100",
            f"  Grade: {grade} — {label}",
            f"  Total Findings: {n}",
            f"{'='*60}",
        ]
        sev_dist = self._score.get("severity_distribution", {})
        for sev in SEVERITY_ORDER:
            count = sev_dist.get(sev, 0)
            if count > 0:
                emoji = SEVERITY_EMOJI.get(sev, "")
                lines.append(f"  {emoji} {sev.title():10s}: {count}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)

    # ── internals ─────────────────────────────────────────────────────────────

    def _auto_detect_target(self) -> str | None:
        if not PANDAS_OK:
            return None
        for col in self.df.columns:
            if TARGET_PATTERNS.search(col):
                return col
        return None

    @staticmethod
    def _run_safe(name: str, fn: Callable) -> DetectorResult:
        try:
            return fn()
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            return DetectorResult(detector_name=name, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def detect_leakage(
    df          : "pd.DataFrame",
    target_col  : str | None = None,
    train_df    : "pd.DataFrame | None" = None,
    test_df     : "pd.DataFrame | None" = None,
    time_col    : str | None = None,
    config      : LeakageConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """
    Full leakage detection in one call.

    Parameters
    ----------
    df         : full dataset DataFrame
    target_col : target/label column name
    train_df   : training split (optional but recommended)
    test_df    : test split (optional but recommended)
    time_col   : datetime column (optional, auto-detected)
    config     : LeakageConfig instance for custom thresholds

    Returns
    -------
    dict with score, findings, and per-detector results
    """
    orch = LeakageOrchestrator(
        df=df, target_col=target_col,
        train_df=train_df, test_df=test_df,
        time_col=time_col, config=config,
    )
    return orch.run()


def check_target_leakage(
    df        : "pd.DataFrame",
    target_col: str,
    config    : LeakageConfig = DEFAULT_CONFIG,
) -> DetectorResult:
    """Run only TargetLeakageDetector — fastest single check."""
    return TargetLeakageDetector(df, target_col, config).detect()


def find_train_test_overlap(
    train_df: "pd.DataFrame",
    test_df : "pd.DataFrame",
    config  : LeakageConfig = DEFAULT_CONFIG,
) -> DetectorResult:
    """Run only TrainTestOverlapDetector."""
    return TrainTestOverlapDetector(train_df, test_df, config=config).detect()


def full_leakage_report(
    df         : "pd.DataFrame",
    output_dir : str | Path = "reports",
    target_col : str | None = None,
    train_df   : "pd.DataFrame | None" = None,
    test_df    : "pd.DataFrame | None" = None,
    formats    : list[str] = ("md", "json", "html"),
    config     : LeakageConfig = DEFAULT_CONFIG,
) -> dict[str, Path]:
    """
    One-call: run all detectors + export full report.

    Returns
    -------
    dict mapping format → Path to generated file
    """
    orch = LeakageOrchestrator(
        df=df, target_col=target_col,
        train_df=train_df, test_df=test_df,
        config=config,
    )
    orch.run()
    return orch.export_report(output_dir=output_dir, formats=formats)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_fallback(x: "np.ndarray", y: "np.ndarray") -> float:
    """Pure-Python Pearson correlation — no scipy needed."""
    try:
        n  = len(x)
        if n < 2:
            return 0.0
        mx = sum(x) / n
        my = sum(y) / n
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
        return num / (dx * dy + 1e-9)
    except Exception:
        return 0.0


def _simhash_row(row: "pd.Series", bits: int = 64) -> int:
    """
    SimHash of a pandas row.
    Produces a `bits`-bit integer fingerprint.
    """
    v = [0.0] * bits
    for col, val in row.items():
        if not (PANDAS_OK and pd.isna(val)):
            token = f"{col}:{val}"
            h     = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(bits):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    xor = a ^ b
    count = 0
    while xor:
        count += xor & 1
        xor >>= 1
    return count


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN / Inf / numpy types for json.dumps."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if NUMPY_OK:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
