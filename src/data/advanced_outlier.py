"""
Nydra — src/data/advanced_outlier.py
==========================================
Advanced Outlier Detection Engine
Covers: Statistical + ML + Ensemble + SmartDetector + OutlierReport


Version : 0.5.0
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings("ignore")
logger = logging.getLogger("nydra.outlier")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SMALL_DATASET_THRESHOLD   = 100     # rows below → prefer statistical
MEDIUM_DATASET_THRESHOLD  = 1_000   # rows below → balanced
HIGH_DIM_THRESHOLD        = 10      # cols above → prefer ML
MIN_SAMPLES_ML            = 20      # minimum rows for ML methods
CONTAMINATION_DEFAULT     = 0.05    # assumed outlier rate


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OutlierResult:
    """Container returned by every detector."""
    method        : str
    column        : Optional[str]          # None → multivariate
    outlier_mask  : pd.Series             # boolean
    outlier_count : int
    outlier_rate  : float
    scores        : Optional[pd.Series]   # raw anomaly scores
    thresholds    : Dict[str, float]
    details       : Dict[str, Any] = field(default_factory=dict)
    warnings      : List[str]      = field(default_factory=list)


@dataclass
class EnsembleResult:
    """Aggregated result from multiple methods."""
    votes_df      : pd.DataFrame           # (n_rows, n_methods) bool
    score_df      : pd.DataFrame           # (n_rows, n_methods) float
    consensus_mask: pd.Series              # majority vote
    consensus_score: pd.Series             # mean normalised score
    method_results: List[OutlierResult]
    summary       : Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierReport:
    """Final report merging everything."""
    dataset_shape    : Tuple[int, int]
    numeric_columns  : List[str]
    method_results   : List[OutlierResult]
    ensemble         : EnsembleResult
    smart_choice     : str
    smart_result     : OutlierResult
    recommendations  : List[str]
    stats_summary    : Dict[str, Any]

    @property
    def outlier_indices(self) -> pd.Index:
        """Returns indices of rows flagged by consensus."""
        return self.ensemble.consensus_mask[self.ensemble.consensus_mask].index

    @property
    def n_outliers(self) -> int:
        return int(self.ensemble.consensus_mask.sum())

    @property
    def outlier_rate(self) -> float:
        return float(self.ensemble.consensus_mask.mean())

    def to_markdown(self) -> str:
        """Generate a Markdown version of the report."""
        lines = [
            "### 🔍 Outlier Detection Report",
            f"**Dataset**: {self.dataset_shape[0]} rows × {self.dataset_shape[1]} cols",
            f"**Numeric Columns**: {len(self.numeric_columns)}",
            f"**Smart Choice**: `{self.smart_choice}`",
            "",
            "#### 📊 Ensemble Consensus",
            f"- **Outliers Detected**: {self.ensemble.summary.get('consensus_outliers')}",
            f"- **Outlier Rate**: {self.ensemble.summary.get('consensus_rate', 0):.2%}",
            "",
            "#### 💡 Recommendations",
        ]
        for rec in self.recommendations:
            lines.append(f"- {rec}")
        
        lines.append("\n#### 🧪 Method Breakdown")
        lines.append("| Method | Column | Count | Rate |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for r in self.method_results:
            lines.append(f"| {r.method} | {r.column or 'multivariate'} | {r.outlier_count} | {r.outlier_rate:.2%} |")
            
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns, drop all-NaN cols."""
    num = df.select_dtypes(include=[np.number])
    return num.dropna(axis=1, how="all")


def _series_from_column(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].dropna()


def _normalise_scores(scores: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; higher = more anomalous."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-12:
        return pd.Series(np.zeros(len(scores)), index=scores.index)
    return (scores - mn) / (mx - mn)


def _build_mask(scores: pd.Series, contamination: float) -> pd.Series:
    threshold = scores.quantile(1 - contamination)
    return scores >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL METHODS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class IQRDetector:
    """
    Interquartile Range method.
    Flags values below Q1 - k*IQR or above Q3 + k*IQR.
    Default k=1.5 (mild), k=3.0 (extreme).
    Works column by column.
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def detect(self, df: pd.DataFrame) -> List[OutlierResult]:
        results = []
        for col in _get_numeric(df).columns:
            s   = _series_from_column(df, col)
            q1  = s.quantile(0.25)
            q3  = s.quantile(0.75)
            iqr = q3 - q1
            lo  = q1 - self.k * iqr
            hi  = q3 + self.k * iqr

            mask  = (df[col] < lo) | (df[col] > hi)
            score = df[col].apply(
                lambda x: max(0, (x - hi) / (iqr + 1e-12)) if x > hi
                     else max(0, (lo - x) / (iqr + 1e-12)) if x < lo
                     else 0.0
            )
            results.append(OutlierResult(
                method        = f"IQR(k={self.k})",
                column        = col,
                outlier_mask  = mask,
                outlier_count = int(mask.sum()),
                outlier_rate  = float(mask.mean()),
                scores        = score,
                thresholds    = {"lower": lo, "upper": hi, "IQR": iqr},
                details       = {"Q1": q1, "Q3": q3},
            ))
        return results


class ZScoreDetector:
    """
    Standard Z-Score method.
    Flags |z| > threshold (default 3).
    Assumes approximate normality.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> List[OutlierResult]:
        results = []
        for col in _get_numeric(df).columns:
            s    = df[col].dropna()
            mu   = s.mean()
            sigma = s.std(ddof=1)
            if sigma < 1e-12:
                continue
            z    = ((df[col] - mu) / sigma).abs()
            mask = z > self.threshold
            results.append(OutlierResult(
                method        = f"Z-Score(t={self.threshold})",
                column        = col,
                outlier_mask  = mask.fillna(False),
                outlier_count = int(mask.sum()),
                outlier_rate  = float(mask.mean()),
                scores        = z,
                thresholds    = {"z_threshold": self.threshold},
                details       = {"mean": mu, "std": sigma},
            ))
        return results


class ModifiedZScoreDetector:
    """
    Modified Z-Score using Median Absolute Deviation (MAD).
    More robust than standard Z-Score.
    Recommended threshold: 3.5 (Iglewicz & Hoaglin).
    """

    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> List[OutlierResult]:
        results = []
        for col in _get_numeric(df).columns:
            s      = df[col].dropna()
            median = s.median()
            mad    = np.median(np.abs(s - median))
            if mad < 1e-12:
                mad = s.mean() * 0.6745 + 1e-12  # fallback

            mz   = (0.6745 * (df[col] - median) / mad).abs()
            mask = mz > self.threshold
            results.append(OutlierResult(
                method        = f"ModifiedZ(t={self.threshold})",
                column        = col,
                outlier_mask  = mask.fillna(False),
                outlier_count = int(mask.sum()),
                outlier_rate  = float(mask.mean()),
                scores        = mz,
                thresholds    = {"mz_threshold": self.threshold},
                details       = {"median": float(median), "MAD": float(mad)},
            ))
        return results


class GrubbsDetector:
    """
    Grubbs test for a single outlier in normally distributed data.
    Iterative version removes one outlier per pass (max_iter times).
    Best for small samples (n < 100) with known normality.
    """

    def __init__(self, alpha: float = 0.05, max_iter: int = 10):
        self.alpha    = alpha
        self.max_iter = max_iter

    def _critical_value(self, n: int) -> float:
        p   = self.alpha / (2 * n)
        t_c = stats.t.ppf(1 - p, df=n - 2)
        return ((n - 1) / np.sqrt(n)) * np.sqrt(t_c**2 / (n - 2 + t_c**2))

    def detect(self, df: pd.DataFrame) -> List[OutlierResult]:
        results = []
        for col in _get_numeric(df).columns:
            s    = df[col].dropna().copy()
            n0   = len(s)
            warn = []
            if n0 < 6:
                warn.append(f"Grubbs requires n≥6; {col} has {n0} rows.")
                results.append(OutlierResult(
                    method="Grubbs", column=col,
                    outlier_mask=pd.Series(False, index=df.index),
                    outlier_count=0, outlier_rate=0.0,
                    scores=None, thresholds={}, warnings=warn,
                ))
                continue

            removed_idx = []
            for _ in range(min(self.max_iter, n0 // 2)):
                if len(s) < 6:
                    break
                G    = (s - s.mean()).abs() / s.std(ddof=1)
                gc   = self._critical_value(len(s))
                gmax = G.max()
                if gmax > gc:
                    removed_idx.append(G.idxmax())
                    s = s.drop(G.idxmax())
                else:
                    break

            mask = pd.Series(False, index=df.index)
            mask.loc[removed_idx] = True

            results.append(OutlierResult(
                method        = f"Grubbs(α={self.alpha})",
                column        = col,
                outlier_mask  = mask,
                outlier_count = len(removed_idx),
                outlier_rate  = len(removed_idx) / n0,
                scores        = None,
                thresholds    = {"alpha": self.alpha},
                details       = {"removed_indices": removed_idx},
                warnings      = warn,
            ))
        return results


class DixonDetector:
    """
    Dixon Q-test — designed for very small samples (n=3..30).
    Tests the most extreme value (min or max).
    """

    # Critical Q values at α=0.05 for n=3..10
    _Q_CRITICAL = {
        3: 0.941, 4: 0.765, 5: 0.642, 6: 0.560,
        7: 0.507, 8: 0.468, 9: 0.437, 10: 0.412,
    }

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect(self, df: pd.DataFrame) -> List[OutlierResult]:
        results = []
        for col in _get_numeric(df).columns:
            s   = df[col].dropna().sort_values()
            n   = len(s)
            warn = []
            mask = pd.Series(False, index=df.index)

            if n < 3 or n > 30:
                warn.append(f"Dixon Q-test valid for n∈[3,30]; got {n}.")
                results.append(OutlierResult(
                    method="Dixon", column=col, outlier_mask=mask,
                    outlier_count=0, outlier_rate=0.0,
                    scores=None, thresholds={}, warnings=warn,
                ))
                continue

            rng    = float(s.iloc[-1] - s.iloc[0])
            q_lo   = (s.iloc[1] - s.iloc[0]) / rng if rng > 0 else 0
            q_hi   = (s.iloc[-1] - s.iloc[-2]) / rng if rng > 0 else 0
            q_crit = self._Q_CRITICAL.get(n, 0.412)

            outliers = []
            if q_lo > q_crit:
                outliers.append(s.index[0])
            if q_hi > q_crit:
                outliers.append(s.index[-1])

            mask.loc[outliers] = True
            results.append(OutlierResult(
                method        = f"Dixon(α={self.alpha})",
                column        = col,
                outlier_mask  = mask,
                outlier_count = len(outliers),
                outlier_rate  = len(outliers) / n,
                scores        = None,
                thresholds    = {"Q_critical": q_crit, "Q_low": q_lo, "Q_high": q_hi},
                warnings      = warn,
            ))
        return results


class MCDDetector:
    """
    Minimum Covariance Determinant — robust Mahalanobis distance.
    Multivariate method; requires sklearn.
    Chi-squared threshold with df = n_features, α = 0.975.
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 support_fraction: float = 0.75):
        self.contamination    = contamination
        self.support_fraction = support_fraction

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.covariance import MinCovDet

        num = _get_numeric(df).dropna()
        if len(num) < num.shape[1] * 2 + 1:
            return OutlierResult(
                method="MCD", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=["Not enough samples for MCD."],
            )

        mcd      = MinCovDet(support_fraction=self.support_fraction,
                             random_state=42)
        mcd.fit(num.values)
        dist     = mcd.mahalanobis(num.values)          # squared Mahal
        dist_s   = pd.Series(np.sqrt(dist), index=num.index)

        chi2_thr = np.sqrt(stats.chi2.ppf(0.975, df=num.shape[1]))
        mask     = dist_s > chi2_thr
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask[mask].index] = True

        return OutlierResult(
            method        = "MCD(RobustMahalanobis)",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = dist_s.reindex(df.index),
            thresholds    = {"chi2_threshold": chi2_thr,
                             "contamination":  self.contamination},
            details       = {"n_features": num.shape[1],
                             "support_fraction": self.support_fraction},
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ML-BASED METHODS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest — isolates anomalies by random partitioning.
    Works well in high dimensions, does not assume normality.
    Returns decision_function scores (more negative = more anomalous).
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.random_state  = random_state

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.ensemble import IsolationForest

        num = _get_numeric(df).dropna()
        if len(num) < MIN_SAMPLES_ML:
            return OutlierResult(
                method="IsolationForest", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=[f"Need ≥{MIN_SAMPLES_ML} samples; got {len(num)}."],
            )

        clf = IsolationForest(
            contamination = self.contamination,
            n_estimators  = self.n_estimators,
            random_state  = self.random_state,
        )
        clf.fit(num.values)
        preds  = clf.predict(num.values)          # -1=outlier, 1=inlier
        scores = -clf.decision_function(num.values)  # higher = more anomalous

        scores_s  = pd.Series(scores, index=num.index)
        mask_num  = pd.Series(preds == -1, index=num.index)
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask_num[mask_num].index] = True
        full_scores = scores_s.reindex(df.index)

        return OutlierResult(
            method        = "IsolationForest",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = full_scores,
            thresholds    = {"contamination": self.contamination},
            details       = {"n_estimators": self.n_estimators,
                             "n_features": num.shape[1]},
        )


class LOFDetector:
    """
    Local Outlier Factor — compares local density of a point to its neighbours.
    Excels at detecting local anomalies in clustered data.
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 n_neighbors: int = 20):
        self.contamination = contamination
        self.n_neighbors   = n_neighbors

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.neighbors import LocalOutlierFactor

        num = _get_numeric(df).dropna()
        n   = len(num)
        if n < MIN_SAMPLES_ML:
            return OutlierResult(
                method="LOF", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=[f"Need ≥{MIN_SAMPLES_ML} samples; got {n}."],
            )

        k = min(self.n_neighbors, n - 1)
        clf = LocalOutlierFactor(
            n_neighbors   = k,
            contamination = self.contamination,
        )
        preds  = clf.fit_predict(num.values)
        scores = -clf.negative_outlier_factor_  # higher = more anomalous

        scores_s  = pd.Series(scores, index=num.index)
        mask_num  = pd.Series(preds == -1, index=num.index)
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask_num[mask_num].index] = True

        return OutlierResult(
            method        = "LOF",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = scores_s.reindex(df.index),
            thresholds    = {"contamination": self.contamination,
                             "n_neighbors": k},
            details       = {"n_features": num.shape[1]},
        )


class OneClassSVMDetector:
    """
    One-Class SVM — learns a boundary around normal data.
    Sensitive to feature scaling — StandardScaler applied internally.
    Best for medium-sized datasets (scalability issues at n > 10k).
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 kernel: str = "rbf", nu: Optional[float] = None):
        self.contamination = contamination
        self.kernel        = kernel
        self.nu            = nu if nu else contamination

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler

        num = _get_numeric(df).dropna()
        n   = len(num)
        warn = []
        if n < MIN_SAMPLES_ML:
            return OutlierResult(
                method="OneClassSVM", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=[f"Need ≥{MIN_SAMPLES_ML} samples; got {n}."],
            )
        if n > 5_000:
            warn.append("OC-SVM slow on large datasets. Consider IsolationForest.")

        scaler = StandardScaler()
        X      = scaler.fit_transform(num.values)

        clf    = OneClassSVM(kernel=self.kernel, nu=self.nu)
        preds  = clf.fit_predict(X)
        scores = -clf.decision_function(X)  # higher = more anomalous

        scores_s  = pd.Series(scores, index=num.index)
        mask_num  = pd.Series(preds == -1, index=num.index)
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask_num[mask_num].index] = True

        return OutlierResult(
            method        = "OneClassSVM",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = scores_s.reindex(df.index),
            thresholds    = {"nu": self.nu, "kernel": self.kernel},
            warnings      = warn,
        )


class DBSCANDetector:
    """
    DBSCAN — density-based clustering. Points not belonging to any cluster
    are labelled noise (label=-1) → treated as outliers.
    eps is estimated via k-NN distance elbow if not provided.
    """

    def __init__(self, eps: Optional[float] = None,
                 min_samples: int = 5,
                 contamination: float = CONTAMINATION_DEFAULT):
        self.eps           = eps
        self.min_samples   = min_samples
        self.contamination = contamination

    def _estimate_eps(self, X: np.ndarray) -> float:
        from sklearn.neighbors import NearestNeighbors
        k   = min(self.min_samples, len(X) - 1)
        nn  = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        dists, _ = nn.kneighbors(X)
        kth      = np.sort(dists[:, -1])
        # elbow heuristic: 90th percentile
        return float(np.percentile(kth, 90))

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        num = _get_numeric(df).dropna()
        n   = len(num)
        if n < self.min_samples + 1:
            return OutlierResult(
                method="DBSCAN", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=["Not enough samples for DBSCAN."],
            )

        scaler = StandardScaler()
        X      = scaler.fit_transform(num.values)

        eps = self.eps if self.eps else self._estimate_eps(X)
        clf = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = clf.fit_predict(X)

        mask_num  = pd.Series(labels == -1, index=num.index)
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask_num[mask_num].index] = True

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        return OutlierResult(
            method        = "DBSCAN",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = None,
            thresholds    = {"eps": eps, "min_samples": self.min_samples},
            details       = {"n_clusters": n_clusters,
                             "noise_ratio": float(mask_num.mean())},
        )


class EllipticEnvelopeDetector:
    """
    Elliptic Envelope — fits a Gaussian to the data, uses Mahalanobis
    distance from the robust estimate. Assumes elliptic distribution.
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 support_fraction: float = 0.75):
        self.contamination    = contamination
        self.support_fraction = support_fraction

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        from sklearn.covariance import EllipticEnvelope

        num = _get_numeric(df).dropna()
        n, p = num.shape
        if n < p * 2:
            return OutlierResult(
                method="EllipticEnvelope", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
                warnings=["Need n ≥ 2p for EllipticEnvelope."],
            )

        clf = EllipticEnvelope(
            contamination    = self.contamination,
            support_fraction = self.support_fraction,
            random_state     = 42,
        )
        clf.fit(num.values)
        preds  = clf.predict(num.values)
        scores = -clf.decision_function(num.values)

        scores_s  = pd.Series(scores, index=num.index)
        mask_num  = pd.Series(preds == -1, index=num.index)
        full_mask = pd.Series(False, index=df.index)
        full_mask.loc[mask_num[mask_num].index] = True

        return OutlierResult(
            method        = "EllipticEnvelope",
            column        = None,
            outlier_mask  = full_mask,
            outlier_count = int(full_mask.sum()),
            outlier_rate  = float(full_mask.mean()),
            scores        = scores_s.reindex(df.index),
            thresholds    = {"contamination": self.contamination},
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleOutlierDetector:
    """
    Combines multiple methods via majority voting + score averaging.
    Only multivariate-capable results (column=None) participate in
    the ensemble. Univariate results are stored separately.

    Parameters
    ----------
    contamination : assumed outlier fraction
    voting_threshold : fraction of methods that must agree (0-1).
                       Default 0.5 = majority vote.
    methods : list of method names to include. None = all.
    """

    ALL_METHODS = [
        "iqr", "zscore", "modified_zscore",
        "isolation_forest", "lof", "one_class_svm",
        "dbscan", "elliptic_envelope", "mcd",
    ]

    def __init__(
        self,
        contamination     : float = CONTAMINATION_DEFAULT,
        voting_threshold  : float = 0.5,
        methods           : Optional[List[str]] = None,
        iqr_k             : float = 1.5,
        zscore_t          : float = 3.0,
        mz_t              : float = 3.5,
        if_n_estimators   : int   = 100,
        lof_neighbors     : int   = 20,
    ):
        self.contamination    = contamination
        self.voting_threshold = voting_threshold
        self.methods          = methods or self.ALL_METHODS
        self.iqr_k            = iqr_k
        self.zscore_t         = zscore_t
        self.mz_t             = mz_t
        self.if_n_est         = if_n_estimators
        self.lof_neighbors    = lof_neighbors

    # ── internal runners ─────────────────────────────────────────────────────

    def _run_statistical(self, df: pd.DataFrame) -> List[OutlierResult]:
        out = []
        if "iqr" in self.methods:
            out.extend(IQRDetector(k=self.iqr_k).detect(df))
        if "zscore" in self.methods:
            out.extend(ZScoreDetector(threshold=self.zscore_t).detect(df))
        if "modified_zscore" in self.methods:
            out.extend(ModifiedZScoreDetector(threshold=self.mz_t).detect(df))
        return out

    def _run_ml(self, df: pd.DataFrame) -> List[OutlierResult]:
        out = []
        if "isolation_forest" in self.methods:
            out.append(IsolationForestDetector(
                contamination=self.contamination,
                n_estimators=self.if_n_est).detect(df))
        if "lof" in self.methods:
            out.append(LOFDetector(
                contamination=self.contamination,
                n_neighbors=self.lof_neighbors).detect(df))
        if "one_class_svm" in self.methods:
            out.append(OneClassSVMDetector(
                contamination=self.contamination).detect(df))
        if "dbscan" in self.methods:
            out.append(DBSCANDetector(
                contamination=self.contamination).detect(df))
        if "elliptic_envelope" in self.methods:
            out.append(EllipticEnvelopeDetector(
                contamination=self.contamination).detect(df))
        if "mcd" in self.methods:
            out.append(MCDDetector(
                contamination=self.contamination).detect(df))
        return out

    # ── aggregate ─────────────────────────────────────────────────────────────

    def _aggregate_univariate(
        self, stat_results: List[OutlierResult], n_rows: int, index
    ) -> OutlierResult:
        """Merge per-column statistical results into one row-level mask."""
        combined = pd.Series(False, index=index)
        score    = pd.Series(0.0, index=index)
        count    = 0
        for r in stat_results:
            combined = combined | r.outlier_mask.reindex(index, fill_value=False)
            if r.scores is not None:
                norm = _normalise_scores(r.scores.reindex(index, fill_value=0))
                score += norm
                count += 1
        if count:
            score /= count
        return OutlierResult(
            method        = "StatisticalUnion",
            column        = None,
            outlier_mask  = combined,
            outlier_count = int(combined.sum()),
            outlier_rate  = float(combined.mean()),
            scores        = score,
            thresholds    = {},
        )

    def detect(self, df: pd.DataFrame) -> EnsembleResult:
        index = df.index
        stat_results = self._run_statistical(df)
        ml_results   = self._run_ml(df)

        # multivariate-capable results only
        mv_results   = [r for r in ml_results if r.column is None]
        # add merged statistical result
        stat_merged  = self._aggregate_univariate(stat_results, len(df), index)
        mv_results.append(stat_merged)

        # ── votes matrix ──────────────────────────────────────────────────────
        votes = {}
        score = {}
        for r in mv_results:
            votes[r.method] = r.outlier_mask.reindex(index, fill_value=False)
            if r.scores is not None:
                score[r.method] = _normalise_scores(
                    r.scores.reindex(index, fill_value=0))
            else:
                score[r.method] = votes[r.method].astype(float)

        votes_df = pd.DataFrame(votes, index=index)
        score_df = pd.DataFrame(score, index=index)

        vote_fraction   = votes_df.mean(axis=1)
        consensus_mask  = vote_fraction >= self.voting_threshold
        consensus_score = score_df.mean(axis=1)

        all_results = stat_results + ml_results

        summary = {
            "total_methods"     : len(mv_results),
            "voting_threshold"  : self.voting_threshold,
            "consensus_outliers": int(consensus_mask.sum()),
            "consensus_rate"    : float(consensus_mask.mean()),
            "method_agreement"  : {
                m: int(votes_df[m].sum()) for m in votes_df.columns
            },
        }

        return EnsembleResult(
            votes_df       = votes_df,
            score_df       = score_df,
            consensus_mask = consensus_mask,
            consensus_score= consensus_score,
            method_results = all_results,
            summary        = summary,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  SMART DETECTOR — auto-selects the best method
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class SmartDetector:
    """
    Analyses dataset characteristics and chooses the most suitable
    outlier detection strategy.

    Decision logic
    --------------
    | Condition                          | Recommended method         |
    |------------------------------------|----------------------------|
    | n < 30                             | Grubbs / Dixon             |
    | n < SMALL, p == 1, normal          | Z-Score                    |
    | n < SMALL, p == 1, skewed          | IQR or ModifiedZ           |
    | n < SMALL, p > 1                   | MCD / EllipticEnvelope     |
    | SMALL ≤ n < MEDIUM, p ≤ HIGH_DIM   | IsolationForest / LOF      |
    | n ≥ MEDIUM, p > HIGH_DIM           | IsolationForest            |
    | any with clusters visible          | DBSCAN                     |
    """

    def __init__(self, contamination: float = CONTAMINATION_DEFAULT,
                 verbose: bool = True):
        self.contamination = contamination
        self.verbose       = verbose
        self._reasoning: List[str] = []

    def _log(self, msg: str):
        self._reasoning.append(msg)
        if self.verbose:
            logger.info(f"[SmartDetector] {msg}")

    def _is_normal(self, s: pd.Series) -> bool:
        if len(s) < 8:
            return False
        _, p = stats.shapiro(s.sample(min(50, len(s)), random_state=42))
        return p > 0.05

    def _skewness(self, s: pd.Series) -> float:
        return float(s.skew())

    def choose(self, df: pd.DataFrame) -> str:
        """Return the name of the recommended method."""
        num = _get_numeric(df)
        n, p = num.shape
        self._reasoning.clear()
        self._log(f"Dataset: {n} rows × {p} numeric cols")

        if n < 3:
            self._log("Too small for any test.")
            return "iqr"

        if n < 30:
            self._log("Very small sample → Grubbs (single col) or IQR.")
            return "grubbs" if p == 1 else "iqr"

        if p == 1:
            col  = num.columns[0]
            norm = self._is_normal(num[col].dropna())
            skew = abs(self._skewness(num[col].dropna()))
            self._log(f"Univariate. Normal={norm}, |skew|={skew:.2f}")
            if norm and n < SMALL_DATASET_THRESHOLD:
                self._log("→ Z-Score (normal, small n)")
                return "zscore"
            elif skew > 1.0:
                self._log("→ ModifiedZ (skewed)")
                return "modified_zscore"
            else:
                self._log("→ IQR (robust fallback)")
                return "iqr"

        # multivariate
        if n < SMALL_DATASET_THRESHOLD:
            self._log("Small multivariate → MCD")
            return "mcd"

        if p > HIGH_DIM_THRESHOLD:
            self._log("High dimensional → IsolationForest")
            return "isolation_forest"

        if n < MEDIUM_DATASET_THRESHOLD:
            self._log("Medium dataset, low dims → LOF")
            return "lof"

        self._log("Large dataset → IsolationForest")
        return "isolation_forest"

    def detect(self, df: pd.DataFrame) -> OutlierResult:
        method_name = self.choose(df)
        self._log(f"Running: {method_name}")

        dispatch = {
            "iqr"              : lambda: IQRDetector().detect(df)[0]
                                 if _get_numeric(df).shape[1] > 0 else None,
            "zscore"           : lambda: ZScoreDetector().detect(df)[0]
                                 if _get_numeric(df).shape[1] > 0 else None,
            "modified_zscore"  : lambda: ModifiedZScoreDetector().detect(df)[0]
                                 if _get_numeric(df).shape[1] > 0 else None,
            "grubbs"           : lambda: GrubbsDetector().detect(df)[0]
                                 if _get_numeric(df).shape[1] > 0 else None,
            "mcd"              : lambda: MCDDetector(self.contamination).detect(df),
            "isolation_forest" : lambda: IsolationForestDetector(self.contamination).detect(df),
            "lof"              : lambda: LOFDetector(self.contamination).detect(df),
            "elliptic_envelope": lambda: EllipticEnvelopeDetector(self.contamination).detect(df),
        }

        fn = dispatch.get(method_name)
        if fn is None:
            fn = dispatch["iqr"]

        result = fn()
        if result is None:
            result = OutlierResult(
                method="SmartDetector(fallback)", column=None,
                outlier_mask=pd.Series(False, index=df.index),
                outlier_count=0, outlier_rate=0.0,
                scores=None, thresholds={},
            )
        result.details["smart_reasoning"] = self._reasoning.copy()
        result.details["chosen_method"]   = method_name
        return result

    @property
    def reasoning(self) -> List[str]:
        return self._reasoning.copy()


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  OUTLIER REPORTER — full analysis pipeline
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

class OutlierAnalyzer:
    """
    Main entry point for Nydra outlier analysis.
    Runs every available method, builds an ensemble, runs SmartDetector,
    and produces a complete OutlierReport.

    Usage
    -----
    >>> analyzer = OutlierAnalyzer(contamination=0.05)
    >>> report   = analyzer.analyze(df)
    >>> print(report.stats_summary)
    >>> clean_df = analyzer.get_clean_df(df, report)
    """

    def __init__(
        self,
        contamination    : float = CONTAMINATION_DEFAULT,
        voting_threshold : float = 0.5,
        run_slow_methods : bool  = True,   # OC-SVM can be slow
        verbose          : bool  = True,
    ):
        self.contamination    = contamination
        self.voting_threshold = voting_threshold
        self.run_slow_methods = run_slow_methods
        self.verbose          = verbose

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            logger.info(f"[OutlierAnalyzer] {msg}")

    def _recommendations(
        self, report: OutlierReport
    ) -> List[str]:
        recs = []
        rate = report.ensemble.summary.get("consensus_rate", 0)

        if rate == 0:
            recs.append("✅ No outliers detected by consensus. Data looks clean.")
            return recs

        if rate < 0.01:
            recs.append(
                f"🟡 Very few outliers ({rate:.1%}). "
                "Consider capping or Winsorization rather than removal."
            )
        elif rate < 0.05:
            recs.append(
                f"🟠 Moderate outlier rate ({rate:.1%}). "
                "Investigate each outlier — may be genuine extreme values."
            )
        else:
            recs.append(
                f"🔴 High outlier rate ({rate:.1%}). "
                "Possible data quality issues or wrong contamination assumption."
            )

        n, p = report.dataset_shape
        if p > HIGH_DIM_THRESHOLD:
            recs.append(
                "📐 High-dimensional data: IsolationForest or LOF are most reliable."
            )
        if n < SMALL_DATASET_THRESHOLD:
            recs.append(
                "📉 Small dataset: statistical tests (Grubbs, Modified Z) are preferred."
            )

        smart = report.smart_choice
        recs.append(f"🧠 SmartDetector chose '{smart}' based on data profile.")

        # method disagreement warning
        agree = report.ensemble.summary.get("method_agreement", {})
        counts = list(agree.values())
        if counts and (max(counts) - min(counts)) > 0.2 * n:
            recs.append(
                "⚠️  High disagreement between methods. "
                "Use ensemble consensus rather than any single method."
            )

        return recs

    def _build_stats_summary(
        self,
        df: pd.DataFrame,
        report_results: List[OutlierResult],
        ensemble: EnsembleResult,
        smart_result: OutlierResult,
    ) -> Dict[str, Any]:

        summary: Dict[str, Any] = {
            "dataset_rows"           : len(df),
            "dataset_cols"           : df.shape[1],
            "numeric_cols"           : len(_get_numeric(df).columns),
            "contamination_assumed"  : self.contamination,
            "consensus_outliers"     : int(ensemble.consensus_mask.sum()),
            "consensus_rate"         : float(ensemble.consensus_mask.mean()),
            "smart_method"           : smart_result.details.get("chosen_method"),
            "smart_outliers"         : smart_result.outlier_count,
            "method_breakdown"       : {},
        }

        for r in report_results:
            key = f"{r.method}|{r.column or 'multivariate'}"
            summary["method_breakdown"][key] = {
                "outlier_count": r.outlier_count,
                "outlier_rate" : round(r.outlier_rate, 4),
                "warnings"     : r.warnings,
            }

        return summary

    # ── main ─────────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> OutlierReport:
        """Run full outlier analysis. Returns an OutlierReport."""
        self._log(f"Starting analysis on {df.shape}")
        num_cols = list(_get_numeric(df).columns)

        if not num_cols:
            # Return a minimal empty report instead of crashing
            empty_series = pd.Series([False]*len(df), index=df.index)
            empty_scores = pd.Series([0.0]*len(df), index=df.index)
            empty_df = pd.DataFrame(index=df.index)

            return OutlierReport(
                dataset_shape=(len(df), df.shape[1]),
                numeric_columns=[],
                method_results=[],
                ensemble=EnsembleResult(
                    votes_df=empty_df,
                    score_df=empty_df,
                    consensus_mask=empty_series,
                    consensus_score=empty_scores,
                    method_results=[],
                    summary={"consensus_outliers": 0, "consensus_rate": 0.0}
                ),
                smart_choice="none",
                smart_result=OutlierResult(
                    method="none", 
                    column=None,
                    outlier_mask=empty_series,
                    outlier_count=0,
                    outlier_rate=0.0,
                    scores=empty_scores,
                    thresholds={},
                    warnings=["No numeric columns found for analysis."]
                ),
                recommendations=["No numeric columns found. Outlier analysis skipped."],
                stats_summary={
                    "dataset_rows": len(df),
                    "dataset_cols": df.shape[1],
                    "numeric_cols": 0,
                    "consensus_outliers": 0,
                    "consensus_rate": 0.0,
                },
            )

        # 1. Ensemble (runs everything internally)
        methods = EnsembleOutlierDetector.ALL_METHODS.copy()
        if not self.run_slow_methods:
            methods = [m for m in methods if m != "one_class_svm"]

        ensemble_detector = EnsembleOutlierDetector(
            contamination    = self.contamination,
            voting_threshold = self.voting_threshold,
            methods          = methods,
        )
        self._log("Running ensemble…")
        ensemble = ensemble_detector.detect(df)
        self._log(
            f"Ensemble done. Consensus outliers: {ensemble.summary['consensus_outliers']}"
        )

        # 2. SmartDetector
        self._log("Running SmartDetector…")
        smart_detector = SmartDetector(
            contamination=self.contamination, verbose=self.verbose)
        smart_result = smart_detector.detect(df)
        smart_choice = smart_result.details.get("chosen_method", "unknown")

        # 3. Stats summary
        stats_summary = self._build_stats_summary(
            df, ensemble.method_results, ensemble, smart_result
        )

        # 4. Build report
        report = OutlierReport(
            dataset_shape   = df.shape,
            numeric_columns = num_cols,
            method_results  = ensemble.method_results,
            ensemble        = ensemble,
            smart_choice    = smart_choice,
            smart_result    = smart_result,
            recommendations = [],
            stats_summary   = stats_summary,
        )

        report.recommendations = self._recommendations(report)
        self._log("Report ready.")
        return report

    # ── post-processing ───────────────────────────────────────────────────────

    def get_clean_df(
        self,
        df: pd.DataFrame,
        report: OutlierReport,
        strategy: str = "consensus",   # "consensus" | "smart" | "cap"
        cap_method: str = "iqr",
    ) -> pd.DataFrame:
        """
        Return a cleaned DataFrame using one of:
        - 'consensus' : remove rows flagged by ensemble majority vote
        - 'smart'     : remove rows flagged by SmartDetector
        - 'cap'       : Winsorize (cap) outliers using IQR bounds
        """
        if strategy == "consensus":
            mask = report.ensemble.consensus_mask
            return df[~mask].copy()

        elif strategy == "smart":
            mask = report.smart_result.outlier_mask
            return df[~mask.reindex(df.index, fill_value=False)].copy()

        elif strategy == "cap":
            df_clean = df.copy()
            for col in report.numeric_columns:
                if col not in df_clean.columns:
                    continue
                q1  = df_clean[col].quantile(0.25)
                q3  = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lo  = q1 - 1.5 * iqr
                hi  = q3 + 1.5 * iqr
                df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
            return df_clean

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use: consensus|smart|cap")

    def to_dataframe(self, report: OutlierReport) -> pd.DataFrame:
        """
        Convert the OutlierReport into a flat DataFrame (one row per method/column).
        Useful for display in Streamlit or CLI.
        """
        rows = []
        for r in report.method_results:
            rows.append({
                "method"       : r.method,
                "column"       : r.column or "multivariate",
                "outlier_count": r.outlier_count,
                "outlier_rate" : f"{r.outlier_rate:.2%}",
                "has_scores"   : r.scores is not None,
                "warnings"     : "; ".join(r.warnings) if r.warnings else "",
            })
        return pd.DataFrame(rows)

    def print_report(self, report: OutlierReport) -> None:
        """Print a text summary of the OutlierReport to stdout."""
        sep = "═" * 60
        print(f"\n{sep}")
        print("  Nydra — Advanced Outlier Report")
        print(sep)
        print(f"  Dataset     : {report.dataset_shape[0]} rows × {report.dataset_shape[1]} cols")
        print(f"  Numeric cols: {len(report.numeric_columns)}")
        print(f"  Smart method: {report.smart_choice}")
        print(f"  Consensus   : {report.ensemble.summary['consensus_outliers']} outliers "
              f"({report.ensemble.summary['consensus_rate']:.2%})")
        print(f"\n{'─'*60}")
        print("  Method breakdown:")
        for key, v in report.stats_summary["method_breakdown"].items():
            method, col = key.split("|")
            warn_str = f"  ⚠ {v['warnings']}" if v["warnings"] else ""
            print(f"    {method:<35} [{col}]  → {v['outlier_count']} ({v['outlier_rate']:.2%}){warn_str}")

        print(f"\n{'─'*60}")
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    {rec}")
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTION — one-liner API
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers(
    df                : pd.DataFrame,
    contamination     : float = CONTAMINATION_DEFAULT,
    strategy          : str   = "ensemble",
    method            : Optional[str] = None, # Alias for strategy
    voting_threshold  : float = 0.5,
    verbose           : bool  = False,
) -> OutlierReport:
    """
    One-liner outlier detection for nydra. Supports 'method' as alias for 'strategy'.
    """
    use_strategy = method if method else strategy
    analyzer = OutlierAnalyzer(
        contamination    = contamination,
        voting_threshold = voting_threshold,
        verbose          = verbose,
    )
    return analyzer.analyze(df)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPATIBILITY LAYER
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Alias for renamed classes
SmartOutlierDetector = SmartDetector
OutlierReporter      = OutlierAnalyzer
OutlierEvaluator     = OutlierReport  # OutlierReport acts as the container for evaluation results

def full_outlier_report(df: pd.DataFrame, contamination: float = CONTAMINATION_DEFAULT) -> OutlierReport:
    """Old name for detect_outliers."""
    return detect_outliers(df, contamination=contamination)

def benchmark_outliers(df: pd.DataFrame, contamination: float = CONTAMINATION_DEFAULT) -> pd.DataFrame:
    """Benchmark all methods and return a summary DataFrame."""
    analyzer = OutlierAnalyzer(contamination=contamination, verbose=False)
    report   = analyzer.analyze(df)
    return analyzer.to_dataframe(report)

# Add missing methods to OutlierAnalyzer for compatibility with src/core/agent.py
def _reporter_generate(self) -> Dict[str, Any]:
    """Generate a dictionary summary (used by agent.py)."""
    # Assuming self was initialized with (result, df) in the old version,
    # but here it's an analyzer. However, agent.py calls OutlierReporter(result, df).
    # We need a dedicated wrapper if we want to support that exact signature.
    return self.stats_summary if hasattr(self, "stats_summary") else {}

def _reporter_to_markdown(self) -> str:
    """Generate a markdown summary (used by agent.py)."""
    return "Outlier Report (Markdown version not fully implemented in compatibility layer)"

# If agent.py does: reporter = OutlierReporter(result, df), we need a wrapper
class OutlierReporterWrapper:
    def __init__(self, report: OutlierReport, df: pd.DataFrame):
        self.report = report
        self.df = df
    
    def generate(self) -> Dict[str, Any]:
        return self.report.stats_summary
    
    def to_markdown(self) -> str:
        # Simple markdown table from the results
        rows = []
        for r in self.report.method_results:
            rows.append(f"| {r.method} | {r.column or 'multivariate'} | {r.outlier_count} | {r.outlier_rate:.2%} |")
        
        table = "\n".join(rows)
        return f"""
### Outlier Analysis Report
| Method | Column | Count | Rate |
| :--- | :--- | :--- | :--- |
{table}

**Smart Choice:** {self.report.smart_choice}
**Consensus Outliers:** {self.report.ensemble.summary['consensus_outliers']}
"""

# Re-alias OutlierReporter to the wrapper
OutlierReporter = OutlierReporterWrapper

# Also fix detect_outliers to accept 'method' as an alias for 'strategy' if needed
# Actually, let's just wrap it.
def detect_outliers_compat(
    df                : pd.DataFrame,
    contamination     : float = CONTAMINATION_DEFAULT,
    strategy          : str   = "ensemble",
    method            : Optional[str] = None, # Alias for strategy
    voting_threshold  : float = 0.5,
    verbose           : bool  = False,
) -> OutlierReport:
    """Wrapper for detect_outliers that supports 'method' argument."""
    use_strategy = method if method else strategy
    analyzer = OutlierAnalyzer(
        contamination    = contamination,
        voting_threshold = voting_threshold,
        verbose          = verbose,
    )
    return analyzer.analyze(df)

# We can't easily overwrite the function name in the same file if we want to call it,
# but we can rename the original one.

# Let's just add the missing arguments to the original function if possible.
# I will do that in the next step or just use this wrapper logic.


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test (run with: python advanced_outlier.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    rng = np.random.default_rng(42)
    n   = 200

    df_test = pd.DataFrame({
        "age"     : rng.normal(35, 10, n).tolist() + [120, -5],
        "salary"  : rng.normal(50_000, 15_000, n).tolist() + [500_000, -1_000],
        "score"   : rng.uniform(0, 100, n + 2).tolist(),
    })

    print(f"Test DataFrame: {df_test.shape}")

    report  = detect_outliers(df_test, contamination=0.05, verbose=True)
    analyzer = OutlierAnalyzer(verbose=False)
    analyzer.print_report(report)

    clean_df = analyzer.get_clean_df(df_test, report, strategy="consensus")
    print(f"Clean DataFrame: {clean_df.shape}  (removed {len(df_test) - len(clean_df)} rows)")

    summary_df = analyzer.to_dataframe(report)
    print("\nMethod summary table:")
    print(summary_df.to_string(index=False))
