
"""
label_quality.py — Nydra v0.6.0
======================================
Advanced Label Quality Analysis & Noise Detection Pipeline.

The most comprehensive label quality analysis available in open-source Python.
Combines Confident Learning, MinHash LSH, statistical fairness tests,
calibration analysis, inter-annotator agreement, and PSI drift detection
into a single unified pipeline.

  Block 1  — ConfidentLearningDetector  : Confident Joint, noise matrix, per-sample issue types
  Block 2  — InconsistentLabelDetector  : MinHash LSH near-duplicate groups, label flip candidates
  Block 3  — AmbiguousSampleDetector    : margin score, entropy, CV boundary disagreement
  Block 4  — LabelDriftDetector         : KL divergence, PSI, chi-squared across time windows
  Block 5  — ImbalanceAnalyzer          : Gini, entropy, effective N, SMOTE factors, pairwise
  Block 6  — AnnotationConfidenceScorer : self-confidence, Platt calibration, CV agreement, Kappa
  Block 7  — OutlierLabelDetector       : isolation forest on label-conditioned features, LOF
  Block 8  — ClassSeparabilityAnalyzer  : Fisher's LDA ratio, Bhattacharyya, overlap coefficients
  Block 9  — LabelQualityScorer         : composite 0-100 score, per-class breakdown, verdict
  Block 10 — LabelQualityAnalyzer       : master orchestrator + convenience functions + CLI

Version : 0.6.0
Date    : April 2026
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import json
import logging
import math
import os
import time
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from scipy.stats import (
    chi2_contingency, entropy as scipy_entropy,
    ks_2samp, mannwhitneyu, kruskal, fisher_exact,
    permutation_test,
)
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    log_loss, brier_score_loss, cohen_kappa_score,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict,
)
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level  = logging.INFO,
    format = "[Nydra:label_quality] %(levelname)s — %(message)s",
)
logger = logging.getLogger("nydra.label_quality")

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    CLEANLAB_AVAILABLE = True
    logger.debug("cleanlab available — using native Confident Learning")
except ImportError:
    CLEANLAB_AVAILABLE = False
    logger.debug("cleanlab not installed — using built-in CL implementation")

try:
    from datasketch import MinHash, MinHashLSH
    MINHASH_AVAILABLE = True
except ImportError:
    MINHASH_AVAILABLE = False
    logger.debug("datasketch not installed — using cosine similarity fallback for duplicates")

try:
    import statsmodels.api as sm
    from statsmodels.stats.proportion import proportions_ztest
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_CV_FOLDS     = 5
DEFAULT_CLASSIFIER   = "auto"   # auto-selects based on dataset size
MIN_SAMPLES_PER_CLASS = 3
PSI_THRESHOLD_MINOR  = 0.10
PSI_THRESHOLD_MAJOR  = 0.25
KL_THRESHOLD_MINOR   = 0.05
KL_THRESHOLD_MAJOR   = 0.20
NOISE_RATE_CRITICAL  = 0.20    # >20% noise → critical
NOISE_RATE_HIGH      = 0.10
IMBALANCE_RATIO_MILD = 3.0
IMBALANCE_RATIO_HIGH = 10.0


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class LabelIssue:
    """Represents a single detected label issue."""
    sample_index:    int
    given_label:     Any
    predicted_label: Any
    issue_type:      str     # 'label_error'|'near_duplicate'|'ambiguous'|'outlier'|'low_confidence'
    confidence:      float   # model confidence in given label (0=wrong, 1=correct)
    severity:        str     # 'critical'|'high'|'medium'|'low'
    description:     str
    suggested_label: Any     = None
    duplicate_group: int     = None


@dataclass
class NoiseMatrix:
    """
    The Confident Joint / Noise Transition Matrix.
    Rows = given labels, Cols = estimated true labels.
    """
    matrix:          np.ndarray    # (n_classes, n_classes) counts
    noise_matrix:    np.ndarray    # (n_classes, n_classes) row-normalized rates
    inverse_matrix:  np.ndarray    # P(given=s|true=y)
    classes:         List[Any]
    noise_rate:      float         # overall noise rate
    per_class_noise: Dict[Any, float]


@dataclass
class LabelQualityReport:
    """Full output of LabelQualityAnalyzer.analyze()"""
    # Summary
    label_quality_score:   float        # 0-100
    verdict:               str
    noise_rate:            float
    n_label_errors:        int
    n_ambiguous:           int
    n_inconsistent:        int
    n_outliers:            int
    # Details
    issues:                List[LabelIssue]
    noise_matrix:          Optional[NoiseMatrix]
    per_class_quality:     Dict[Any, float]
    drift_report:          Dict
    imbalance_report:      Dict
    annotation_confidence: Dict
    separability_report:   Dict
    # Meta
    n_samples:             int
    n_classes:             int
    classes:               List[Any]
    processing_time_s:     float
    recommendations:       List[Dict]


# ==============================================================================
# BLOCK 1 — CONFIDENT LEARNING DETECTOR
# ==============================================================================

class ConfidentLearningDetector:
    """
    Full implementation of Confident Learning (Northcutt et al., 2021).

    The Confident Joint C[s,y] estimates the joint distribution of
    given (noisy) labels s and true (latent) labels y.

    Algorithm
    ---------
    1. Get out-of-sample predicted probabilities P(y|x) via cross-validation
    2. For each class s, compute per-class threshold t_s:
       t_s = average P(label=s | x_i) for all x_i where given_label=s
    3. Build Confident Joint: C[s,y] = count of samples where
       given_label=s  AND  argmax(P(y|x)) = y  AND  P(y|x) >= t_y
    4. Calibrate: C_calibrated = C * (n_s / sum(C[s,:])) for each s
    5. Normalize to get noise transition matrix T[s,y] = P(given=s|true=y)
    6. Estimate true class priors p_y from T and observed label frequencies
    7. Rank samples by label quality score = P(given_label|x) / max(P(y|x))

    Issue types detected
    --------------------
    label_error    : given label is wrong — model strongly disagrees
    near_duplicate : near-identical sample appears in multiple classes
    ambiguous      : sample is genuinely on decision boundary
    outlier        : low-confidence sample that doesn't fit any class well
    multi_label    : sample likely belongs to multiple classes simultaneously

    Fallback
    --------
    If cleanlab is installed: delegates to cleanlab.filter.find_label_issues()
    for maximum accuracy. Otherwise uses this full built-in implementation.
    """

    def __init__(
        self,
        classifier      = None,
        cv_folds:   int = DEFAULT_CV_FOLDS,
        n_jobs:     int = -1,
        seed:       int = 42,
        thresholds: Optional[np.ndarray] = None,
    ):
        self.classifier  = classifier
        self.cv_folds    = cv_folds
        self.n_jobs      = n_jobs
        self.seed        = seed
        self.thresholds  = thresholds  # custom per-class thresholds
        self._le         = LabelEncoder()

    # ── public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        pred_probs: Optional[np.ndarray] = None,
    ) -> Tuple[List[LabelIssue], NoiseMatrix, np.ndarray]:
        """
        Run Confident Learning detection.

        Parameters
        ----------
        X          : feature matrix (n_samples, n_features)
        y          : given labels (n_samples,)
        pred_probs : optional precomputed out-of-sample predicted probabilities
                     shape (n_samples, n_classes). If None, computed via CV.

        Returns
        -------
        issues      : list of LabelIssue objects
        noise_mx    : NoiseMatrix with confident joint + transition matrix
        qual_scores : per-sample quality scores (higher = cleaner label)
        """
        y_enc = self._le.fit_transform(y)
        classes = self._le.classes_.tolist()
        n_classes = len(classes)

        logger.info(f"Confident Learning: {len(X)} samples, {n_classes} classes")

        # Step 1: Get predicted probabilities
        if pred_probs is None:
            pred_probs = self._get_pred_probs(X, y_enc, n_classes)

        # Validate shape
        if pred_probs.shape[1] != n_classes:
            raise ValueError(
                f"pred_probs has {pred_probs.shape[1]} columns but "
                f"there are {n_classes} classes"
            )

        # Normalize rows (ensure valid probability distributions)
        row_sums = pred_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        pred_probs = pred_probs / row_sums

        # Use cleanlab if available (more accurate)
        if CLEANLAB_AVAILABLE:
            return self._cleanlab_detect(y_enc, pred_probs, classes)

        # Step 2: Per-class thresholds
        thresholds = self._compute_thresholds(y_enc, pred_probs, n_classes)

        # Step 3: Build Confident Joint
        confident_joint = self._build_confident_joint(y_enc, pred_probs, thresholds, n_classes)

        # Step 4: Calibrate
        confident_joint_calibrated = self._calibrate_joint(confident_joint, y_enc, n_classes)

        # Step 5: Noise transition matrix
        noise_matrix, inv_noise_matrix = self._compute_noise_matrices(
            confident_joint_calibrated, n_classes
        )

        # Step 6: True class priors
        py = self._estimate_true_priors(noise_matrix, y_enc, n_classes)

        # Step 7: Label quality scores
        quality_scores = self._compute_quality_scores(y_enc, pred_probs, thresholds)

        # Step 8: Build NoiseMatrix object
        per_class_noise = {
            classes[i]: float(1.0 - noise_matrix[i, i])
            for i in range(n_classes)
        }
        overall_noise = float(1.0 - np.trace(noise_matrix) / n_classes)
        # Better estimate: weighted by class frequency
        class_counts = np.bincount(y_enc, minlength=n_classes)
        overall_noise = float(
            sum(per_class_noise[classes[i]] * class_counts[i]
                for i in range(n_classes)) / len(y_enc)
        )

        noise_mx = NoiseMatrix(
            matrix         = confident_joint_calibrated,
            noise_matrix   = noise_matrix,
            inverse_matrix = inv_noise_matrix,
            classes        = classes,
            noise_rate     = overall_noise,
            per_class_noise= per_class_noise,
        )

        # Step 9: Classify issues
        issues = self._classify_issues(
            y_enc, pred_probs, quality_scores, thresholds,
            noise_mx, classes
        )

        logger.info(
            f"CL complete: noise_rate={overall_noise:.3f}, "
            f"{len(issues)} issues found"
        )
        return issues, noise_mx, quality_scores

    # ── cross-validated predicted probabilities ───────────────────────────────

    def _get_pred_probs(
        self, X: np.ndarray, y_enc: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """Get OOS predicted probabilities via stratified CV."""
        if self.cv_folds < 2:
            # Fallback for tiny datasets: In-Sample prediction
            # (Not ideal for CL, but necessary when n < 2)
            clf = self._build_classifier(X, y_enc)
            try:
                clf.fit(X, y_enc)
                return clf.predict_proba(X)
            except Exception:
                return np.ones((len(X), n_classes)) / n_classes

        clf = self._build_classifier(X, y_enc)
        
        # Check if StratifiedKFold is possible
        _, counts = np.unique(y_enc, return_counts=True)
        min_samples = np.min(counts)
        
        if min_samples >= self.cv_folds:
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.seed
            )
        else:
            from sklearn.model_selection import KFold
            cv = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.seed
            )

        pred_probs = np.zeros((len(X), n_classes), dtype=np.float64)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc) if min_samples >= self.cv_folds else cv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train        = y_enc[train_idx]

            # Skip fold if any class is missing in training split
            if len(np.unique(y_train)) < n_classes:
                # Expand with nearest available — use simple fallback
                clf_fold = clone(clf)
                try:
                    clf_fold.fit(X_train, y_train)
                    probs = clf_fold.predict_proba(X_val)
                    # Pad missing classes with zeros
                    trained_classes = clf_fold.classes_
                    full_probs = np.zeros((len(val_idx), n_classes))
                    for j, cls in enumerate(trained_classes):
                        full_probs[:, cls] = probs[:, j]
                    pred_probs[val_idx] = full_probs
                except Exception:
                    pred_probs[val_idx] = np.ones((len(val_idx), n_classes)) / n_classes
            else:
                clf_fold = clone(clf)
                clf_fold.fit(X_train, y_train)
                pred_probs[val_idx] = clf_fold.predict_proba(X_val)

            logger.debug(f"  Fold {fold_idx + 1}/{self.cv_folds} done")

        return pred_probs

    def _build_classifier(self, X: np.ndarray, y: np.ndarray):
        """Auto-select classifier based on dataset characteristics."""
        if self.classifier is not None and self.classifier != "auto":
            return self.classifier

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Rule-based selection
        if n_samples < 500:
            # Small: Logistic Regression (fast, calibrated)
            return LogisticRegression(
                max_iter=1000, solver="lbfgs",
                random_state=self.seed, C=1.0
            )
        elif n_samples < 5000:
            # Medium: Random Forest (robust, no scaling needed)
            return RandomForestClassifier(
                n_estimators=100, max_depth=None,
                n_jobs=self.n_jobs, random_state=self.seed,
                class_weight="balanced",
            )
        else:
            # Large: Gradient Boosting (best accuracy)
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=self.seed,
                subsample=0.8,
            )

    # ── confident joint construction ──────────────────────────────────────────

    def _compute_thresholds(
        self, y_enc: np.ndarray, pred_probs: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """
        Per-class confidence threshold t_s:
        Average predicted probability of class s among all samples
        where the given label IS s.
        This is the key insight of Confident Learning.
        """
        if self.thresholds is not None:
            return self.thresholds

        thresholds = np.zeros(n_classes, dtype=np.float64)
        for s in range(n_classes):
            mask = y_enc == s
            if mask.sum() == 0:
                thresholds[s] = 0.5
            else:
                thresholds[s] = float(pred_probs[mask, s].mean())
        return thresholds

    def _build_confident_joint(
        self,
        y_enc:      np.ndarray,
        pred_probs: np.ndarray,
        thresholds: np.ndarray,
        n_classes:  int,
    ) -> np.ndarray:
        """
        Build Confident Joint C[s,y]:
        For each sample i:
          - given label s = y_enc[i]
          - estimated true label y = argmax(pred_probs[i])
          - include in C[s,y] only if pred_probs[i, y] >= threshold[y]

        This filters out uncertain predictions to keep only confident ones.
        """
        C = np.zeros((n_classes, n_classes), dtype=np.int64)

        for i in range(len(y_enc)):
            s    = y_enc[i]                          # given label
            probs = pred_probs[i]
            y    = int(np.argmax(probs))             # estimated true label
            # Include only if prediction is confident
            if probs[y] >= thresholds[y]:
                C[s, y] += 1

        # Handle empty rows (classes with no confident assignments)
        # by adding a small smoothing count
        for s in range(n_classes):
            if C[s].sum() == 0:
                C[s, s] = 1  # assume self-consistent

        return C

    def _calibrate_joint(
        self,
        C:     np.ndarray,
        y_enc: np.ndarray,
        n_classes: int,
    ) -> np.ndarray:
        """
        Calibrate Confident Joint so that row sums match actual class counts.
        C_calibrated[s, y] = C[s,y] / sum(C[s,:]) * n_s
        where n_s = actual count of samples with given label s.
        """
        C_cal = C.astype(np.float64).copy()
        for s in range(n_classes):
            n_s     = int((y_enc == s).sum())
            row_sum = C_cal[s].sum()
            if row_sum > 0:
                C_cal[s] = C_cal[s] / row_sum * n_s
        return C_cal

    def _compute_noise_matrices(
        self, C: np.ndarray, n_classes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute noise transition matrix T and its inverse.

        T[s, y] = P(given=s | true=y) = C[s,y] / sum_s(C[s,y])
        T_inv[y, s] = P(true=y | given=s) = C[s,y] / sum_y(C[s,y])

        T is the noise matrix (columns sum to 1).
        T_inv is the inverse noise matrix (rows sum to 1).
        """
        # Column-normalized → T[s,y] = P(given=s | true=y)
        col_sums  = C.sum(axis=0)
        col_sums  = np.where(col_sums == 0, 1, col_sums)
        T         = C / col_sums[np.newaxis, :]

        # Row-normalized → T_inv[s,y] = P(true=y | given=s)
        row_sums  = C.sum(axis=1)
        row_sums  = np.where(row_sums == 0, 1, row_sums)
        T_inv     = C / row_sums[:, np.newaxis]

        return T, T_inv

    def _estimate_true_priors(
        self, noise_matrix: np.ndarray, y_enc: np.ndarray, n_classes: int
    ) -> np.ndarray:
        """
        Estimate true class priors p_y from noise matrix and observed frequencies.
        p_s = sum_y T[s,y] * p_y  →  p_y = T^{-1} p_s
        Solved via non-negative least squares.
        """
        from scipy.optimize import nnls
        p_s = np.bincount(y_enc, minlength=n_classes).astype(np.float64)
        p_s /= p_s.sum()
        try:
            p_y, _ = nnls(noise_matrix, p_s)
            if p_y.sum() > 0:
                p_y /= p_y.sum()
            else:
                p_y = p_s.copy()
        except Exception:
            p_y = p_s.copy()
        return p_y

    # ── quality scoring ───────────────────────────────────────────────────────

    def _compute_quality_scores(
        self,
        y_enc:      np.ndarray,
        pred_probs: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Per-sample label quality score ∈ [0, 1].
        Higher = cleaner label.

        Score = normalized self-confidence / normalized max-confidence
        = (P(given|x) / threshold_given) / (max_j P(j|x) / threshold_j)

        This makes scores comparable across classes with different difficulty.
        """
        n = len(y_enc)
        scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            s      = y_enc[i]
            probs  = pred_probs[i]
            # Self-confidence (normalized by class threshold)
            self_conf = probs[s] / max(thresholds[s], 1e-6)
            # Max-confidence across all classes (normalized)
            norm_probs = probs / np.maximum(thresholds, 1e-6)
            max_conf   = norm_probs.max()

            scores[i] = self_conf / max(max_conf, 1e-6)

        # Clip to [0, 1]
        return np.clip(scores, 0.0, 1.0)

    # ── issue classification ──────────────────────────────────────────────────

    def _classify_issues(
        self,
        y_enc:         np.ndarray,
        pred_probs:    np.ndarray,
        quality_scores:np.ndarray,
        thresholds:    np.ndarray,
        noise_mx:      NoiseMatrix,
        classes:       List,
    ) -> List[LabelIssue]:
        """
        Classify each suspected label issue into a specific type.

        Issue type decision tree
        ------------------------
        quality_score < 0.5 AND argmax != given → 'label_error'
        quality_score < 0.5 AND entropy high    → 'ambiguous'
        self_conf << threshold                  → 'low_confidence'
        P(top-2) both high                      → 'multi_label'
        """
        issues    = []
        n_classes = pred_probs.shape[1]

        for i in range(len(y_enc)):
            s      = y_enc[i]
            probs  = pred_probs[i]
            qs     = quality_scores[i]
            y_pred = int(np.argmax(probs))

            # Only flag as issue if quality_score < threshold
            if qs >= 0.5:
                continue

            # Compute entropy (0=certain, log(n)=uniform)
            ent = float(scipy_entropy(probs + 1e-10))
            max_ent = math.log(n_classes + 1e-10)
            rel_ent = ent / max_ent if max_ent > 0 else 0.0

            # Determine issue type
            if y_pred != s and probs[y_pred] > thresholds[y_pred] * 1.2:
                issue_type = "label_error"
                desc = (
                    f"Model predicts '{classes[y_pred]}' with "
                    f"P={probs[y_pred]:.3f} but label is '{classes[s]}'"
                )
                suggested = classes[y_pred]
            elif rel_ent > 0.75:
                issue_type = "ambiguous"
                desc = (
                    f"High prediction entropy ({ent:.3f}); sample sits on "
                    f"decision boundary across {n_classes} classes"
                )
                suggested = None
            elif n_classes >= 2:
                top2 = np.sort(probs)[-2:]
                if top2[0] > 0.30 and top2[1] > 0.30:
                    issue_type = "multi_label"
                    top2_idx   = np.argsort(probs)[-2:]
                    desc = (
                        f"Sample may belong to multiple classes: "
                        f"'{classes[top2_idx[1]]}' P={probs[top2_idx[1]]:.2f}, "
                        f"'{classes[top2_idx[0]]}' P={probs[top2_idx[0]]:.2f}"
                    )
                    suggested = classes[y_pred]
                else:
                    issue_type = "low_confidence"
                    desc = (
                        f"Low self-confidence P({classes[s]}|x)={probs[s]:.3f}; "
                        f"label quality score={qs:.3f}"
                    )
                    suggested = None
            else:
                issue_type = "low_confidence"
                desc = f"Quality score={qs:.3f}"
                suggested = None

            # Severity
            if qs < 0.1:
                severity = "critical"
            elif qs < 0.25:
                severity = "high"
            elif qs < 0.40:
                severity = "medium"
            else:
                severity = "low"

            issues.append(LabelIssue(
                sample_index    = i,
                given_label     = classes[s],
                predicted_label = classes[y_pred],
                issue_type      = issue_type,
                confidence      = float(probs[s]),
                severity        = severity,
                description     = desc,
                suggested_label = suggested,
            ))

        # Sort by severity then quality_score
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        issues.sort(key=lambda x: (severity_order[x.severity], x.confidence))
        return issues

    # ── cleanlab delegate ─────────────────────────────────────────────────────

    def _cleanlab_detect(
        self,
        y_enc:      np.ndarray,
        pred_probs: np.ndarray,
        classes:    List,
    ) -> Tuple[List[LabelIssue], NoiseMatrix, np.ndarray]:
        """Delegate to cleanlab for maximum accuracy."""
        logger.info("Using cleanlab native Confident Learning")
        issue_flags = find_label_issues(
            labels          = y_enc,
            pred_probs      = pred_probs,
            return_indices_ranked_by = "self_confidence",
        )
        quality_scores = get_label_quality_scores(y_enc, pred_probs)

        n_classes = len(classes)
        issues = []
        for idx in issue_flags:
            s      = y_enc[idx]
            probs  = pred_probs[idx]
            y_pred = int(np.argmax(probs))
            qs     = float(quality_scores[idx])

            severity = (
                "critical" if qs < 0.1 else
                "high"     if qs < 0.25 else
                "medium"   if qs < 0.40 else "low"
            )

            issues.append(LabelIssue(
                sample_index    = int(idx),
                given_label     = classes[s],
                predicted_label = classes[y_pred],
                issue_type      = "label_error",
                confidence      = float(probs[s]),
                severity        = severity,
                description     = (
                    f"[cleanlab] quality_score={qs:.3f}; "
                    f"predicted='{classes[y_pred]}'"
                ),
                suggested_label = classes[y_pred],
            ))

        # Build basic noise matrix from cleanlab
        thresholds = self._compute_thresholds(y_enc, pred_probs, n_classes)
        C = self._build_confident_joint(y_enc, pred_probs, thresholds, n_classes)
        C_cal = self._calibrate_joint(C, y_enc, n_classes)
        T, T_inv = self._compute_noise_matrices(C_cal, n_classes)
        col_sums = C_cal.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        per_class_noise = {classes[i]: float(1.0 - T[i, i]) for i in range(n_classes)}
        class_counts = np.bincount(y_enc, minlength=n_classes)
        overall_noise = float(
            sum(per_class_noise[classes[i]] * class_counts[i]
                for i in range(n_classes)) / len(y_enc)
        )

        noise_mx = NoiseMatrix(
            matrix         = C_cal,
            noise_matrix   = T,
            inverse_matrix = T_inv,
            classes        = classes,
            noise_rate     = overall_noise,
            per_class_noise= per_class_noise,
        )

        return issues, noise_mx, quality_scores


# ==============================================================================
# BLOCK 2 — INCONSISTENT LABEL DETECTOR
# ==============================================================================

class InconsistentLabelDetector:
    """
    Detect samples with near-identical features but different labels.

    Two strategies
    --------------
    A) MinHash LSH (datasketch)
       - Converts each sample's feature vector to a set of (feature, bin) tokens
       - Uses MinHash with 128 permutations → Jaccard similarity approximation
       - LSH index with threshold 0.8 → finds all pairs with similarity >= 0.8
       - Groups into duplicate clusters, checks label agreement within each cluster
       - Scales to millions of samples in O(n) time

    B) Cosine similarity fallback (scikit-learn)
       - Uses NearestNeighbors with cosine metric
       - Finds all pairs within distance threshold
       - Groups into inconsistency clusters

    Additional: Label Flip Candidates
    -----------------------------------
    For each sample, if the model predicts a class with probability > P(given_label)
    by a margin > 0.2, and that class != given_label → label flip candidate.
    Ranked by flip_score = P(predicted) - P(given).

    Output
    ------
    Per-sample columns added:
      is_near_duplicate     : bool
      duplicate_group_id    : int (-1 = not in group)
      group_label_agreement : float (0-1, 1 = all same label)
      inconsistency_score   : float (0-1, 1 = maximally inconsistent)
      is_label_flip_candidate: bool
      flip_score            : float
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        n_permutations:       int   = 128,
        n_neighbors:          int   = 10,
        flip_margin:          float = 0.20,
    ):
        self.similarity_threshold = similarity_threshold
        self.n_permutations       = n_permutations
        self.n_neighbors          = n_neighbors
        self.flip_margin          = flip_margin

    def detect(
        self,
        X:          np.ndarray,
        y:          np.ndarray,
        pred_probs: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, List[LabelIssue]]:
        """
        Returns
        -------
        result_df  : DataFrame with one row per sample, consistency columns
        issues     : LabelIssue list for inconsistent samples
        """
        logger.info(f"Inconsistency detection: {len(X)} samples")

        if MINHASH_AVAILABLE:
            groups = self._minhash_groups(X)
        else:
            groups = self._cosine_groups(X)

        # Build per-sample results
        n         = len(X)
        group_ids = np.full(n, -1, dtype=int)
        agreement = np.ones(n, dtype=float)
        incon_score = np.zeros(n, dtype=float)
        issues    : List[LabelIssue] = []

        for g_id, group in enumerate(groups):
            if len(group) < 2:
                continue
            group_labels = [y[i] for i in group]
            most_common  = Counter(group_labels).most_common(1)[0][0]
            agreement_rate = Counter(group_labels)[most_common] / len(group_labels)

            for idx in group:
                group_ids[idx]  = g_id
                agreement[idx]  = agreement_rate
                incon_score[idx]= 1.0 - agreement_rate

                if y[idx] != most_common and agreement_rate > 0.5:
                    issues.append(LabelIssue(
                        sample_index    = idx,
                        given_label     = y[idx],
                        predicted_label = most_common,
                        issue_type      = "near_duplicate",
                        confidence      = agreement_rate,
                        severity        = "high" if agreement_rate > 0.8 else "medium",
                        description     = (
                            f"Near-duplicate group {g_id} ({len(group)} samples): "
                            f"{int(agreement_rate * 100)}% labeled '{most_common}' "
                            f"but this sample is labeled '{y[idx]}'"
                        ),
                        suggested_label = most_common,
                        duplicate_group = g_id,
                    ))

        # Label flip candidates
        flip_scores = np.zeros(n, dtype=float)
        is_flip     = np.zeros(n, dtype=bool)

        if pred_probs is not None:
            le = LabelEncoder().fit(y)
            y_enc = le.transform(y)
            for i in range(n):
                s     = y_enc[i]
                probs = pred_probs[i]
                y_top = int(np.argmax(probs))
                if y_top != s:
                    margin = float(probs[y_top] - probs[s])
                    if margin >= self.flip_margin:
                        flip_scores[i] = margin
                        is_flip[i]     = True
                        issues.append(LabelIssue(
                            sample_index    = i,
                            given_label     = le.classes_[s],
                            predicted_label = le.classes_[y_top],
                            issue_type      = "label_error",
                            confidence      = float(probs[s]),
                            severity        = "critical" if margin > 0.5 else "high",
                            description     = (
                                f"Label flip candidate: given='{le.classes_[s]}' "
                                f"P={probs[s]:.3f}, predicted='{le.classes_[y_top]}' "
                                f"P={probs[y_top]:.3f}, margin={margin:.3f}"
                            ),
                            suggested_label = le.classes_[y_top],
                        ))

        result_df = pd.DataFrame({
            "sample_index":          np.arange(n),
            "given_label":           y,
            "is_near_duplicate":     group_ids >= 0,
            "duplicate_group_id":    group_ids,
            "group_label_agreement": agreement,
            "inconsistency_score":   incon_score,
            "is_label_flip_candidate": is_flip,
            "flip_score":            flip_scores,
        })

        n_incon = int((group_ids >= 0).sum())
        n_flip  = int(is_flip.sum())
        logger.info(f"  Near-duplicates: {n_incon}, flip candidates: {n_flip}")

        # Deduplicate issues (same sample may appear twice)
        seen = set()
        unique_issues = []
        for iss in issues:
            key = (iss.sample_index, iss.issue_type)
            if key not in seen:
                seen.add(key)
                unique_issues.append(iss)

        return result_df, unique_issues

    # ── MinHash LSH ───────────────────────────────────────────────────────────

    def _minhash_groups(self, X: np.ndarray) -> List[List[int]]:
        """
        Group near-identical samples using MinHash LSH.
        Converts continuous features to discrete tokens via percentile binning.
        """
        # Discretize features into bins for tokenization
        n_bins   = 20
        X_binned = self._discretize(X, n_bins)

        # Build MinHash signatures
        lsh = MinHashLSH(
            threshold   = self.similarity_threshold,
            num_perm    = self.n_permutations,
        )
        minhashes = []
        for i in range(len(X)):
            m = MinHash(num_perm=self.n_permutations)
            tokens = self._to_tokens(X_binned[i])
            for tok in tokens:
                m.update(tok.encode("utf-8"))
            minhashes.append(m)
            try:
                lsh.insert(str(i), m)
            except Exception:
                pass  # duplicate key

        # Query groups
        visited  = set()
        groups   = []
        for i in range(len(X)):
            if i in visited:
                continue
            try:
                result = lsh.query(minhashes[i])
                group  = [int(r) for r in result]
            except Exception:
                group = [i]
            if len(group) >= 2:
                groups.append(group)
                visited.update(group)

        return groups

    def _cosine_groups(self, X: np.ndarray) -> List[List[int]]:
        """Fallback: use NearestNeighbors with cosine distance."""
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        k      = min(self.n_neighbors, len(X) - 1)
        nn     = NearestNeighbors(
            n_neighbors = k + 1,
            metric      = "cosine",
            algorithm   = "brute",
            n_jobs      = -1,
        )
        nn.fit(X_sc)
        distances, indices = nn.kneighbors(X_sc)

        visited  = set()
        groups   = []
        dist_threshold = 1.0 - self.similarity_threshold

        for i in range(len(X)):
            if i in visited:
                continue
            group = [i]
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                if dist <= dist_threshold:
                    group.append(j)
            if len(group) >= 2:
                groups.append(group)
                visited.update(group)

        return groups

    def _discretize(self, X: np.ndarray, n_bins: int) -> np.ndarray:
        """Bin continuous features into integer buckets."""
        X_binned = np.zeros_like(X, dtype=int)
        for col in range(X.shape[1]):
            vals = X[:, col]
            if vals.std() < 1e-10:
                X_binned[:, col] = 0
            else:
                percentiles = np.linspace(0, 100, n_bins + 1)
                bins        = np.percentile(vals, percentiles)
                bins        = np.unique(bins)
                X_binned[:, col] = np.digitize(vals, bins[1:-1])
        return X_binned

    def _to_tokens(self, row: np.ndarray) -> List[str]:
        return [f"f{j}={v}" for j, v in enumerate(row)]


# ==============================================================================
# BLOCK 3 — AMBIGUOUS SAMPLE DETECTOR
# ==============================================================================

class AmbiguousSampleDetector:
    """
    Find samples that are genuinely hard to classify — on the decision boundary.

    Methods
    -------
    1. Margin Score
       margin_i = P(top-1 class | x_i) - P(top-2 class | x_i)
       Low margin → uncertain prediction → ambiguous sample.

    2. Prediction Entropy
       H(i) = -sum_j P(j|x_i) log P(j|x_i)
       Normalized: H_norm = H / log(n_classes)
       High entropy → model spread across many classes → ambiguous.

    3. Cross-Validation Boundary Disagreement
       For each sample, check how many CV folds predict a different class.
       High disagreement_rate = multiple folds disagree = boundary sample.

    4. k-NN Neighborhood Purity
       For each sample, compute label purity among its k nearest neighbors.
       Low purity → sample is in a mixed-label region → ambiguous.

    5. Loss-Based Difficulty
       Per-sample cross-entropy loss: -log P(given_label | x_i).
       High loss = model assigns low probability to given label → hard sample.

    Composite Ambiguity Score
    -------------------------
    ambiguity_score = 0.30 * (1 - margin_norm)
                    + 0.25 * entropy_norm
                    + 0.20 * disagreement_rate
                    + 0.15 * (1 - nn_purity)
                    + 0.10 * loss_norm

    Threshold: ambiguity_score > 0.65 → flagged as ambiguous
    """

    WEIGHTS = {
        "margin":       0.30,
        "entropy":      0.25,
        "disagreement": 0.20,
        "nn_purity":    0.15,
        "loss":         0.10,
    }

    def __init__(
        self,
        ambiguity_threshold: float = 0.65,
        n_neighbors:         int   = 15,
        cv_folds:            int   = DEFAULT_CV_FOLDS,
        seed:                int   = 42,
    ):
        self.ambiguity_threshold = ambiguity_threshold
        self.n_neighbors         = n_neighbors
        self.cv_folds            = cv_folds
        self.seed                = seed

    def detect(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        pred_probs:  np.ndarray,
        all_fold_preds: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, List[LabelIssue]]:
        """
        Parameters
        ----------
        X              : features
        y              : given labels (encoded)
        pred_probs     : OOS predicted probabilities (n, n_classes)
        all_fold_preds : optional (n, cv_folds) predicted class per fold

        Returns
        -------
        result_df : per-sample ambiguity scores
        issues    : LabelIssue list
        """
        n          = len(X)
        n_classes  = pred_probs.shape[1]
        le         = LabelEncoder().fit(y)
        y_enc      = le.transform(y)

        logger.info(f"Ambiguity detection: {n} samples, {n_classes} classes")

        # 1. Margin score
        sorted_probs  = np.sort(pred_probs, axis=1)
        margin_scores = sorted_probs[:, -1] - sorted_probs[:, -2]
        margin_norm   = np.clip(margin_scores, 0, 1)

        # 2. Entropy
        raw_entropy    = scipy_entropy(pred_probs.T + 1e-10)  # per-sample entropy
        max_entropy    = math.log(n_classes + 1e-10)
        entropy_norm   = raw_entropy / max_entropy
        entropy_norm   = np.clip(entropy_norm, 0, 1)

        # 3. CV disagreement
        if all_fold_preds is not None and all_fold_preds.shape == (n, self.cv_folds):
            # Count folds that predict a class different from the majority fold prediction
            majority_pred = stats.mode(all_fold_preds, axis=1, keepdims=True).mode.ravel()
            disagreements = (all_fold_preds != majority_pred[:, None]).sum(axis=1)
            disagree_rate = disagreements / self.cv_folds
        else:
            # Estimate from pred_probs: high entropy ≈ high disagreement
            disagree_rate = entropy_norm * 0.7

        # 4. k-NN neighborhood purity
        nn_purity = self._nn_purity(X, y_enc, n_classes)

        # 5. Per-sample log loss
        self_probs   = pred_probs[np.arange(n), y_enc]
        self_probs   = np.clip(self_probs, 1e-10, 1.0)
        per_loss     = -np.log(self_probs)
        max_loss     = -math.log(1.0 / n_classes + 1e-10)
        loss_norm    = np.clip(per_loss / max_loss, 0, 1)

        # Composite ambiguity score
        ambiguity = (
            self.WEIGHTS["margin"]       * (1.0 - margin_norm)
            + self.WEIGHTS["entropy"]    * entropy_norm
            + self.WEIGHTS["disagreement"] * disagree_rate
            + self.WEIGHTS["nn_purity"]  * (1.0 - nn_purity)
            + self.WEIGHTS["loss"]       * loss_norm
        )
        ambiguity = np.clip(ambiguity, 0, 1)

        is_ambiguous = ambiguity > self.ambiguity_threshold
        n_ambiguous  = int(is_ambiguous.sum())
        logger.info(f"  Ambiguous samples: {n_ambiguous} ({n_ambiguous/n*100:.1f}%)")

        result_df = pd.DataFrame({
            "sample_index":       np.arange(n),
            "given_label":        y,
            "ambiguity_score":    ambiguity,
            "is_ambiguous":       is_ambiguous,
            "margin_score":       margin_scores,
            "entropy_score":      entropy_norm,
            "disagreement_rate":  disagree_rate,
            "nn_purity":          nn_purity,
            "per_sample_loss":    per_loss,
        })

        issues = []
        for i in np.where(is_ambiguous)[0]:
            aq = float(ambiguity[i])
            issues.append(LabelIssue(
                sample_index    = int(i),
                given_label     = y[i],
                predicted_label = le.classes_[int(np.argmax(pred_probs[i]))],
                issue_type      = "ambiguous",
                confidence      = float(self_probs[i]),
                severity        = "high" if aq > 0.80 else "medium",
                description     = (
                    f"Ambiguity score={aq:.3f} | "
                    f"margin={margin_scores[i]:.3f} | "
                    f"entropy={entropy_norm[i]:.3f} | "
                    f"nn_purity={nn_purity[i]:.3f}"
                ),
                suggested_label = None,
            ))

        return result_df, issues

    def _nn_purity(self, X: np.ndarray, y_enc: np.ndarray,
                   n_classes: int) -> np.ndarray:
        """
        For each sample: fraction of k nearest neighbors sharing the same label.
        Purity = 1 → all neighbors have same label.
        Purity = 0 → all neighbors have different labels.
        """
        k   = min(self.n_neighbors, len(X) - 1)
        sc  = StandardScaler()
        X_s = sc.fit_transform(X)
        nn  = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn.fit(X_s)
        _, indices = nn.kneighbors(X_s)

        purity = np.zeros(len(X), dtype=float)
        for i in range(len(X)):
            neighbor_labels = y_enc[indices[i][1:]]  # exclude self
            purity[i] = float((neighbor_labels == y_enc[i]).mean())
        return purity


# ==============================================================================
# BLOCK 4 — LABEL DRIFT DETECTOR
# ==============================================================================

class LabelDriftDetector:
    """
    Detect shifts in label distribution across time windows or dataset splits.

    This catches the case where labeling policy changed mid-dataset,
    or where data was collected from different sources/time periods
    with different label distributions.

    Metrics
    -------
    1. Population Stability Index (PSI)
       PSI = sum_i (actual_i - expected_i) * log(actual_i / expected_i)
       PSI < 0.10 → stable
       PSI 0.10–0.25 → minor shift (investigate)
       PSI > 0.25 → major shift (action required)

    2. KL Divergence (Jensen-Shannon Divergence)
       JSD = (KL(P||M) + KL(Q||M)) / 2  where M = (P+Q)/2
       Symmetric, bounded [0, log(2)], JSD=0 → identical distributions.

    3. Chi-Squared Homogeneity Test
       Tests whether class distributions across windows are from the same
       underlying distribution. p < 0.05 → significant drift.

    4. Per-Class Drift Analysis
       For each class: tracks frequency across all windows.
       Classes with monotonically increasing/decreasing trends flagged
       as systematic drift.

    5. Window-to-Window KL
       Consecutive window KL divergences to pinpoint exactly when drift occurred.

    Output
    ------
    drift_score per class, overall_psi, jsd_matrix (n_windows × n_windows),
    chi2_p_value, drift_timeline (window → distribution), drifting_classes
    """

    def __init__(
        self,
        n_windows:   int   = 5,
        psi_minor:   float = PSI_THRESHOLD_MINOR,
        psi_major:   float = PSI_THRESHOLD_MAJOR,
        min_window_size: int = 30,
    ):
        self.n_windows       = n_windows
        self.psi_minor       = psi_minor
        self.psi_major       = psi_major
        self.min_window_size = min_window_size

    def detect(
        self,
        y:           np.ndarray,
        time_index:  Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        y          : label array (n_samples,) — assumed ordered by time/index
        time_index : optional time or index array for custom ordering

        Returns
        -------
        report dict with all drift metrics
        """
        n = len(y)
        logger.info(f"Label drift detection: {n} samples, {self.n_windows} windows")

        # Sort by time_index if provided
        if time_index is not None:
            sort_order = np.argsort(time_index)
            y = y[sort_order]

        classes    = sorted(set(y.tolist()))
        n_classes  = len(classes)
        class_idx  = {c: i for i, c in enumerate(classes)}

        # Split into windows
        window_size = max(n // self.n_windows, self.min_window_size)
        windows     = []
        for w in range(self.n_windows):
            start = w * window_size
            end   = min(start + window_size, n) if w < self.n_windows - 1 else n
            if end > start:
                windows.append(y[start:end])

        n_actual_windows = len(windows)

        # Per-window class distributions
        window_dists = np.zeros((n_actual_windows, n_classes), dtype=float)
        for w, win in enumerate(windows):
            counts = Counter(win.tolist())
            for c, cnt in counts.items():
                if c in class_idx:
                    window_dists[w, class_idx[c]] = cnt / len(win)

        # Reference distribution = overall
        overall_dist = np.zeros(n_classes, dtype=float)
        counts_all   = Counter(y.tolist())
        for c, cnt in counts_all.items():
            if c in class_idx:
                overall_dist[class_idx[c]] = cnt / n

        # 1. PSI per window vs reference
        psi_per_window = np.zeros(n_actual_windows, dtype=float)
        for w in range(n_actual_windows):
            psi_per_window[w] = self._psi(overall_dist, window_dists[w])

        overall_psi = float(psi_per_window.max())

        # 2. JSD matrix (pairwise between all windows)
        jsd_matrix = np.zeros((n_actual_windows, n_actual_windows), dtype=float)
        for i in range(n_actual_windows):
            for j in range(i + 1, n_actual_windows):
                jsd = self._jsd(window_dists[i], window_dists[j])
                jsd_matrix[i, j] = jsd
                jsd_matrix[j, i] = jsd

        # 3. Chi-squared homogeneity test
        # Build contingency table: rows=windows, cols=classes
        contingency = np.zeros((n_actual_windows, n_classes), dtype=int)
        for w, win in enumerate(windows):
            counts = Counter(win.tolist())
            for c, cnt in counts.items():
                if c in class_idx:
                    contingency[w, class_idx[c]] = cnt

        # Remove zero-sum columns
        col_sums = contingency.sum(axis=0)
        valid_cols = col_sums > 0
        contingency_valid = contingency[:, valid_cols]

        chi2_p = 1.0
        if contingency_valid.shape[1] > 1 and contingency_valid.shape[0] > 1:
            try:
                chi2, chi2_p, _, _ = chi2_contingency(contingency_valid)
            except Exception:
                chi2_p = 1.0

        # 4. Per-class PSI
        per_class_psi: Dict[Any, float] = {}
        for c, i in class_idx.items():
            actual   = window_dists[:, i]
            expected = np.full(n_actual_windows, overall_dist[i])
            per_class_psi[c] = float(self._psi(expected, actual))

        # 5. Trend detection per class
        drifting_classes = []
        class_trends     = {}
        for c, i in class_idx.items():
            freqs = window_dists[:, i]
            if len(freqs) >= 3:
                # Spearman correlation with window index
                rho, p_val = stats.spearmanr(np.arange(len(freqs)), freqs)
                class_trends[c] = {"rho": round(float(rho), 3), "p_value": round(float(p_val), 4)}
                if abs(rho) > 0.6 and p_val < 0.10:
                    direction = "increasing" if rho > 0 else "decreasing"
                    drifting_classes.append({
                        "class":     c,
                        "trend":     direction,
                        "rho":       round(float(rho), 3),
                        "psi":       round(per_class_psi[c], 4),
                    })

        # 6. Consecutive window KL divergences
        consec_kl = []
        for w in range(n_actual_windows - 1):
            kl = self._jsd(window_dists[w], window_dists[w + 1])
            consec_kl.append(round(float(kl), 4))

        # Overall verdict
        if overall_psi < self.psi_minor and chi2_p > 0.10:
            verdict = "Stable"
        elif overall_psi < self.psi_major:
            verdict = "Minor Drift"
        else:
            verdict = "Major Drift"

        report = {
            "verdict":              verdict,
            "overall_psi":          round(overall_psi, 4),
            "chi2_p_value":         round(chi2_p, 4),
            "significant_drift":    chi2_p < 0.05,
            "n_windows":            n_actual_windows,
            "window_distributions": window_dists.tolist(),
            "psi_per_window":       psi_per_window.tolist(),
            "jsd_matrix":           jsd_matrix.tolist(),
            "consecutive_jsd":      consec_kl,
            "per_class_psi":        {str(k): round(v, 4) for k, v in per_class_psi.items()},
            "class_trends":         {str(k): v for k, v in class_trends.items()},
            "drifting_classes":     drifting_classes,
            "classes":              [str(c) for c in classes],
        }

        logger.info(
            f"Drift: verdict={verdict}, PSI={overall_psi:.4f}, "
            f"chi2_p={chi2_p:.4f}, drifting_classes={len(drifting_classes)}"
        )
        return report

    def _psi(self, expected: np.ndarray, actual: np.ndarray,
              eps: float = 1e-6) -> float:
        """Population Stability Index."""
        e = np.clip(expected, eps, None)
        a = np.clip(actual,   eps, None)
        # Normalize
        e = e / e.sum()
        a = a / a.sum()
        return float(np.sum((a - e) * np.log(a / e)))

    def _jsd(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Jensen-Shannon Divergence (symmetric, bounded [0, log(2)])."""
        p  = np.clip(p, eps, None)
        q  = np.clip(q, eps, None)
        p /= p.sum()
        q /= q.sum()
        m  = 0.5 * (p + q)
        return float(0.5 * scipy_entropy(p, m) + 0.5 * scipy_entropy(q, m))


# ==============================================================================
# BLOCK 5 — IMBALANCE ANALYZER
# ==============================================================================

class ImbalanceAnalyzer:
    """
    Deep class imbalance analysis with actionable recommendations.

    Metrics computed
    ----------------
    1. Imbalance Ratio        : max_count / min_count
    2. Gini Coefficient       : measures inequality in class distribution
    3. Shannon Entropy        : H = -sum p_i log(p_i); max = log(n_classes)
    4. Effective N Classes    : exp(Shannon Entropy); 1=all same class, K=balanced
    5. Normalized Entropy     : H / H_max; 0=maximally imbalanced, 1=balanced
    6. Per-class statistics   : count, freq, minority/majority flag, IR_vs_next,
                                expected_recall_drop (empirical model), SMOTE_factor
    7. Pairwise IR matrix     : IR between every pair of classes
    8. Minority risk score    : estimated recall drop for minority classes
    9. Kurtosis of distribution: heavy tail = extreme imbalance pattern
    10. Imbalance type         : 'Binary-Skewed'|'Moderate'|'Long-tail'|'Extreme'

    SMOTE Factor
    ------------
    Recommended oversample factor per minority class:
    target = majority_count * 0.5 (default) or majority_count (full balance)
    smote_factor = (target - current_count) / current_count
    """

    def __init__(
        self,
        minority_threshold:   float = 0.10,  # freq < 10% → minority
        ir_mild:              float = IMBALANCE_RATIO_MILD,
        ir_high:              float = IMBALANCE_RATIO_HIGH,
        smote_target_ratio:   float = 0.5,   # target minority/majority ratio
    ):
        self.minority_threshold = minority_threshold
        self.ir_mild            = ir_mild
        self.ir_high            = ir_high
        self.smote_target_ratio = smote_target_ratio

    def analyze(self, y: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        y : label array

        Returns
        -------
        Full imbalance report dict
        """
        counts = Counter(y.tolist())
        classes = sorted(counts.keys(), key=lambda c: str(c))
        n_classes = len(classes)
        n_total   = len(y)

        counts_arr = np.array([counts[c] for c in classes], dtype=float)
        freqs      = counts_arr / n_total

        # 1. Imbalance Ratio
        max_cnt = float(counts_arr.max())
        min_cnt = float(counts_arr.min())
        ir      = max_cnt / max(min_cnt, 1)

        # 2. Gini Coefficient
        gini = self._gini(counts_arr)

        # 3. Shannon Entropy
        entropy  = float(scipy_entropy(freqs + 1e-10))
        max_ent  = math.log(n_classes + 1e-10)
        norm_ent = entropy / max_ent if max_ent > 0 else 1.0

        # 4. Effective N Classes
        eff_n = math.exp(entropy)

        # 5. Kurtosis of distribution
        kurt = float(stats.kurtosis(counts_arr))

        # 6. Per-class stats
        majority_class = classes[int(np.argmax(counts_arr))]
        majority_count = int(max_cnt)
        per_class = {}
        for cls, cnt, freq in zip(classes, counts_arr.astype(int), freqs):
            is_minority  = freq < self.minority_threshold
            smote_target = majority_count * self.smote_target_ratio
            smote_factor = max(0.0, (smote_target - cnt) / max(cnt, 1))
            recall_drop  = self._estimate_recall_drop(ir, freq)
            per_class[str(cls)] = {
                "count":        int(cnt),
                "frequency":    round(float(freq), 4),
                "is_minority":  is_minority,
                "is_majority":  cls == majority_class,
                "smote_factor": round(smote_factor, 2),
                "smote_samples_needed": max(0, int(smote_target - cnt)),
                "estimated_recall_drop": round(recall_drop, 3),
                "ir_vs_majority":        round(max_cnt / max(cnt, 1), 2),
            }

        # 7. Pairwise IR matrix
        pairwise_ir = {}
        for i, c1 in enumerate(classes):
            for j, c2 in enumerate(classes):
                if i < j:
                    ratio = counts_arr[i] / max(counts_arr[j], 1)
                    key   = f"{c1}_vs_{c2}"
                    pairwise_ir[key] = round(float(max(ratio, 1/ratio)), 2)

        # 8. Imbalance type
        imbalance_type = self._classify_imbalance(ir, gini, n_classes, kurt)

        # 9. Severity
        if ir < self.ir_mild:
            severity = "Balanced"
            recommendation = "No action needed"
        elif ir < self.ir_high:
            severity = "Moderate Imbalance"
            recommendation = "Consider class_weight='balanced' or mild SMOTE"
        else:
            severity = "Severe Imbalance"
            recommendation = "Use SMOTE/ADASYN oversampling + class_weight='balanced'"

        # 10. Overall imbalance score 0-100 (0=balanced, 100=maximally imbalanced)
        imbalance_score = round(float(
            0.40 * min(ir / 100.0, 1.0) * 100
            + 0.30 * gini * 100
            + 0.30 * (1.0 - norm_ent) * 100
        ), 1)

        report = {
            "verdict":           severity,
            "recommendation":    recommendation,
            "imbalance_score":   imbalance_score,
            "imbalance_type":    imbalance_type,
            "imbalance_ratio":   round(ir, 2),
            "gini_coefficient":  round(gini, 4),
            "shannon_entropy":   round(entropy, 4),
            "normalized_entropy":round(norm_ent, 4),
            "effective_n_classes":round(eff_n, 2),
            "kurtosis":          round(kurt, 4),
            "n_classes":         n_classes,
            "n_samples":         n_total,
            "majority_class":    str(majority_class),
            "majority_count":    majority_count,
            "minority_count":    int(min_cnt),
            "per_class":         per_class,
            "pairwise_ir":       pairwise_ir,
        }

        logger.info(
            f"Imbalance: IR={ir:.1f}, Gini={gini:.3f}, "
            f"EffN={eff_n:.1f}/{n_classes}, severity={severity}"
        )
        return report

    def _gini(self, x: np.ndarray) -> float:
        """Gini coefficient for array x. 0=equal, 1=maximally unequal."""
        x = np.sort(x.astype(float))
        n = len(x)
        if n == 0 or x.sum() == 0:
            return 0.0
        cumsum = np.cumsum(x)
        return float((2 * np.sum(cumsum) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]))

    def _estimate_recall_drop(self, ir: float, freq: float) -> float:
        """
        Empirical model: minority recall ≈ 1 - log(IR) * (1 - freq) * 0.3
        Based on meta-analysis of imbalanced learning literature.
        """
        if ir <= 1.0:
            return 0.0
        drop = min(math.log(ir) * (1.0 - freq) * 0.20, 0.60)
        return drop

    def _classify_imbalance(self, ir: float, gini: float,
                             n_classes: int, kurt: float) -> str:
        if ir < 2:
            return "Balanced"
        if n_classes == 2 and ir > 5:
            return "Binary-Skewed"
        if n_classes > 5 and gini > 0.5:
            return "Long-tail"
        if kurt > 3:
            return "Extreme-Outlier"
        return "Moderate"


# ==============================================================================
# BLOCK 6 — ANNOTATION CONFIDENCE SCORER
# ==============================================================================

class AnnotationConfidenceScorer:
    """
    Assign a confidence score 0-1 to each label in the dataset.

    Components
    ----------
    1. Raw Self-Confidence
       C_raw(i) = P(given_label_i | x_i)
       Direct model confidence in the assigned label.

    2. Calibrated Self-Confidence (Platt Scaling)
       Train a sigmoid calibration layer on held-out fold predictions.
       Corrects overconfident classifiers (e.g. SVMs, GBMs).
       C_cal(i) = sigmoid(a * logit(P(given|x)) + b)

    3. Cross-Validation Agreement
       agree(i) = fraction of CV folds that predict given_label_i
       Independent of model calibration — purely voting-based.

    4. Cohen's Kappa per class
       Measures agreement between model predictions and given labels
       adjusted for chance, per class.
       κ ∈ [-1, 1]: 0=chance, 1=perfect, <0=worse than chance

    5. Krippendorff's Alpha (if multiple annotators available)
       Measures inter-annotator reliability across all annotators.
       α ∈ [0, 1]: 0.80+ = good reliability

    6. Composite Annotation Confidence
       C_composite(i) = 0.35 * C_cal
                      + 0.35 * agree
                      + 0.20 * C_raw
                      + 0.10 * (kappa_class_i normalized)

    Global metrics
    --------------
    annotation_quality_score: mean C_composite
    global_kappa: Cohen's Kappa between model predictions and given labels
    calibration_error: ECE (Expected Calibration Error)
    brier_score: proper scoring rule for probability calibration
    """

    WEIGHTS = {
        "calibrated":  0.35,
        "agreement":   0.35,
        "raw":         0.20,
        "kappa_norm":  0.10,
    }

    def __init__(self, cv_folds: int = DEFAULT_CV_FOLDS, seed: int = 42):
        self.cv_folds = cv_folds
        self.seed     = seed

    def score(
        self,
        y:              np.ndarray,
        pred_probs:     np.ndarray,
        fold_preds:     Optional[np.ndarray] = None,
        annotators_df:  Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        y             : given labels (encoded integers)
        pred_probs    : OOS predicted probabilities (n, n_classes)
        fold_preds    : (n, cv_folds) predictions per fold (optional)
        annotators_df : DataFrame with columns = annotator IDs, rows = samples (optional)

        Returns
        -------
        report dict with per-sample and global metrics
        """
        n         = len(y)
        n_classes = pred_probs.shape[1]
        y_pred    = np.argmax(pred_probs, axis=1)

        logger.info(f"Annotation confidence scoring: {n} samples")

        # 1. Raw self-confidence
        self_probs  = pred_probs[np.arange(n), y]
        c_raw       = np.clip(self_probs, 0, 1)

        # 2. Platt-calibrated confidence
        c_cal = self._platt_calibrate(y, pred_probs)

        # 3. CV agreement
        if fold_preds is not None and fold_preds.shape == (n, self.cv_folds):
            agreement = (fold_preds == y[:, None]).mean(axis=1).astype(float)
        else:
            # Estimate from pred_probs
            agreement = c_raw.copy()

        # 4. Per-class Cohen's Kappa
        per_class_kappa = {}
        for cls in range(n_classes):
            y_bin    = (y     == cls).astype(int)
            yp_bin   = (y_pred == cls).astype(int)
            try:
                kappa = cohen_kappa_score(y_bin, yp_bin)
            except Exception:
                kappa = 0.0
            per_class_kappa[cls] = float(kappa)

        # Normalize kappa to [0, 1] for weighting
        kappa_arr = np.array([per_class_kappa[y[i]] for i in range(n)], dtype=float)
        kappa_norm = np.clip((kappa_arr + 1.0) / 2.0, 0, 1)

        # 5. Composite confidence
        c_composite = (
            self.WEIGHTS["calibrated"] * c_cal
            + self.WEIGHTS["agreement"] * agreement
            + self.WEIGHTS["raw"]       * c_raw
            + self.WEIGHTS["kappa_norm"]* kappa_norm
        )
        c_composite = np.clip(c_composite, 0, 1)

        # 6. Global metrics
        global_kappa = float(cohen_kappa_score(y, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
        ece          = self._expected_calibration_error(y, pred_probs)
        brier        = float(brier_score_loss(
            (y == np.argmax(pred_probs, axis=1)).astype(int),
            pred_probs.max(axis=1)
        ))

        # 7. Krippendorff's Alpha (if annotators provided)
        alpha = None
        if annotators_df is not None and not annotators_df.empty:
            alpha = self._krippendorff_alpha(annotators_df.values)

        # 8. Annotation quality score (0-100)
        quality_score = round(float(c_composite.mean()) * 100, 1)

        # 9. Low-confidence samples
        low_conf_mask  = c_composite < 0.40
        n_low_conf     = int(low_conf_mask.sum())

        report = {
            "annotation_quality_score": quality_score,
            "global_kappa":             round(global_kappa, 4),
            "kappa_interpretation":     self._kappa_interpret(global_kappa),
            "expected_calibration_error": round(ece, 4),
            "brier_score":              round(brier, 4),
            "krippendorff_alpha":       round(alpha, 4) if alpha is not None else None,
            "mean_raw_confidence":      round(float(c_raw.mean()), 4),
            "mean_calibrated_confidence": round(float(c_cal.mean()), 4),
            "mean_cv_agreement":        round(float(agreement.mean()), 4),
            "n_low_confidence":         n_low_conf,
            "pct_low_confidence":       round(n_low_conf / n * 100, 1),
            "per_class_kappa":          {str(k): round(v, 4) for k, v in per_class_kappa.items()},
            "per_sample_confidence":    c_composite.tolist(),
            "per_sample_raw":           c_raw.tolist(),
            "per_sample_calibrated":    c_cal.tolist(),
            "per_sample_agreement":     agreement.tolist(),
        }
        logger.info(
            f"Annotation quality: {quality_score}/100, "
            f"kappa={global_kappa:.3f}, ECE={ece:.4f}, "
            f"low_conf={n_low_conf}"
        )
        return report

    def _platt_calibrate(
        self, y: np.ndarray, pred_probs: np.ndarray
    ) -> np.ndarray:
        """
        Platt scaling: fit sigmoid on predicted probabilities.
        Uses the predicted probability of the given class as input.
        """
        self_probs = pred_probs[np.arange(len(y)), y]
        eps        = 1e-7
        logits     = np.log(
            np.clip(self_probs, eps, 1 - eps)
            / np.clip(1 - self_probs, eps, 1 - eps)
        ).reshape(-1, 1)

        try:
            lr = LogisticRegression(max_iter=500, solver="lbfgs")
            lr.fit(logits, y)
            # Use the probability of the given class label
            proba_all = lr.predict_proba(logits)
            calibrated = np.array([
                proba_all[i, j] for i, j in enumerate(y)
            ])
            return np.clip(calibrated, 0, 1)
        except Exception:
            return np.clip(self_probs, 0, 1)

    def _expected_calibration_error(
        self, y: np.ndarray, pred_probs: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE):
        Measures how well predicted probabilities match actual accuracy.
        Lower is better.
        """
        n          = len(y)
        y_pred     = np.argmax(pred_probs, axis=1)
        confidence = pred_probs.max(axis=1)
        correct    = (y_pred == y).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask   = (confidence > lo) & (confidence <= hi)
            if mask.sum() == 0:
                continue
            avg_conf  = confidence[mask].mean()
            avg_acc   = correct[mask].mean()
            ece      += (mask.sum() / n) * abs(avg_conf - avg_acc)

        return float(ece)

    def _krippendorff_alpha(self, data: np.ndarray,
                             level: str = "nominal") -> float:
        """
        Krippendorff's Alpha for inter-annotator reliability.
        data: (n_annotators, n_samples) with NaN for missing annotations.
        level: 'nominal'|'ordinal'|'interval'|'ratio'
        """
        try:
            # Transpose: (n_samples, n_annotators)
            data_T = data.T.astype(float)
            n_samp, n_ann = data_T.shape
            # Observed disagreement
            Do = 0.0
            n_pairs = 0
            for i in range(n_samp):
                row = data_T[i]
                valid = row[~np.isnan(row)]
                for j in range(len(valid)):
                    for k in range(j + 1, len(valid)):
                        Do += self._metric(valid[j], valid[k], level)
                        n_pairs += 1
            if n_pairs == 0:
                return 0.0
            Do /= n_pairs

            # Expected disagreement
            all_vals = data_T.ravel()
            all_vals = all_vals[~np.isnan(all_vals)]
            De = 0.0
            total = len(all_vals)
            for j in range(total):
                for k in range(j + 1, total):
                    De += self._metric(all_vals[j], all_vals[k], level)
            if total > 1:
                De /= (total * (total - 1) / 2)

            if De == 0:
                return 1.0 if Do == 0 else 0.0
            return float(1.0 - Do / De)
        except Exception:
            return 0.0

    def _metric(self, v1: float, v2: float, level: str) -> float:
        if level == "nominal":
            return 0.0 if v1 == v2 else 1.0
        elif level == "ordinal":
            return abs(v1 - v2)
        else:
            return (v1 - v2) ** 2

    def _kappa_interpret(self, kappa: float) -> str:
        if kappa < 0:
            return "Less than chance"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost perfect"


# ==============================================================================
# BLOCK 7 — OUTLIER LABEL DETECTOR
# ==============================================================================

class OutlierLabelDetector:
    """
    Detect samples that are statistical outliers WITHIN their own class.

    These are genuine data collection errors:
    samples labeled as class C but whose features look nothing like
    any other member of class C.

    Methods
    -------
    1. Isolation Forest per class
       Train IsolationForest on each class's samples separately.
       Anomaly score = average path length in random trees.
       Class-conditional: detects outliers relative to class distribution.

    2. Local Outlier Factor per class
       LOF measures local density relative to neighbors.
       LOF >> 1 → sample is in a low-density region compared to its neighbors.
       Class-conditional: only neighbors within same class are used.

    3. Mahalanobis Distance per class
       Robust Mahalanobis distance using MCD (Minimum Covariance Determinant).
       D_M(x) = sqrt((x-μ)^T Σ^{-1} (x-μ))
       Chi-squared test: D_M^2 ~ χ²(n_features) → p-value threshold.
       Handles correlated features correctly.

    4. Z-Score per class per feature
       For low-dimensional data: any feature with |z| > 3.5 → outlier.
       Per-class standardization.

    5. Ensemble outlier score
       Soft voting across all 4 methods.
       out_score(i) = 0.35*IF + 0.30*LOF + 0.25*Mahal + 0.10*Zscore

    Minimum class size
    ------------------
    If a class has < 10 samples, skip isolation forest + LOF (unstable).
    Fall back to Mahalanobis + Z-score only.
    """

    WEIGHTS = {
        "isolation_forest": 0.35,
        "lof":              0.30,
        "mahalanobis":      0.25,
        "zscore":           0.10,
    }
    MIN_SAMPLES_IF  = 10
    MIN_SAMPLES_LOF = 5
    MIN_SAMPLES_MAH = 5

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors:   int   = 20,
        threshold:     float = 0.70,
        seed:          int   = 42,
    ):
        self.contamination = contamination
        self.n_neighbors   = n_neighbors
        self.threshold     = threshold
        self.seed          = seed

    def detect(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[pd.DataFrame, List[LabelIssue]]:
        """
        Returns
        -------
        result_df : per-sample outlier scores
        issues    : LabelIssue list for class-conditional outliers
        """
        le      = LabelEncoder()
        y_enc   = le.fit_transform(y)
        classes = le.classes_.tolist()
        n       = len(X)

        sc    = StandardScaler()
        X_sc  = sc.fit_transform(X)

        if_scores    = np.zeros(n, dtype=float)
        lof_scores   = np.zeros(n, dtype=float)
        mahal_scores = np.zeros(n, dtype=float)
        z_scores_max = np.zeros(n, dtype=float)

        logger.info(f"Outlier label detection: {n} samples, {len(classes)} classes")

        for cls_idx, cls in enumerate(classes):
            mask = y_enc == cls_idx
            if mask.sum() < 2:
                continue
            X_cls = X_sc[mask]
            idx   = np.where(mask)[0]

            # 1. Isolation Forest
            if mask.sum() >= self.MIN_SAMPLES_IF:
                contamination = min(self.contamination, 0.5 - 1e-6)
                try:
                    iso = IsolationForest(
                        contamination = contamination,
                        random_state  = self.seed,
                        n_jobs        = -1,
                    )
                    iso.fit(X_cls)
                    raw_scores = iso.score_samples(X_cls)
                    # Convert: more negative = more anomalous → normalize to [0,1] outlier score
                    norm = (raw_scores - raw_scores.max()) / (raw_scores.min() - raw_scores.max() + 1e-10)
                    if_scores[idx] = np.clip(norm, 0, 1)
                except Exception:
                    pass

            # 2. LOF
            if mask.sum() >= self.MIN_SAMPLES_LOF:
                k = min(self.n_neighbors, mask.sum() - 1)
                try:
                    lof = LocalOutlierFactor(
                        n_neighbors   = k,
                        contamination = min(self.contamination, 0.5 - 1e-6),
                        n_jobs        = -1,
                    )
                    lof.fit(X_cls)
                    raw_lof = -lof.negative_outlier_factor_
                    # Normalize to [0,1]
                    lo, hi = raw_lof.min(), raw_lof.max()
                    if hi > lo:
                        lof_scores[idx] = (raw_lof - lo) / (hi - lo)
                except Exception:
                    pass

            # 3. Mahalanobis Distance
            if mask.sum() >= self.MIN_SAMPLES_MAH:
                m_scores = self._mahalanobis_scores(X_cls)
                mahal_scores[idx] = np.clip(m_scores, 0, 1)

            # 4. Z-score (per feature, per class)
            z = self._zscore_outlier(X_cls)
            z_scores_max[idx] = np.clip(z, 0, 1)

        # Ensemble
        ensemble_score = (
            self.WEIGHTS["isolation_forest"] * if_scores
            + self.WEIGHTS["lof"]            * lof_scores
            + self.WEIGHTS["mahalanobis"]    * mahal_scores
            + self.WEIGHTS["zscore"]         * z_scores_max
        )
        ensemble_score = np.clip(ensemble_score, 0, 1)

        is_outlier = ensemble_score > self.threshold

        result_df = pd.DataFrame({
            "sample_index":        np.arange(n),
            "given_label":         y,
            "outlier_score":       ensemble_score,
            "is_class_outlier":    is_outlier,
            "isolation_forest_score": if_scores,
            "lof_score":           lof_scores,
            "mahalanobis_score":   mahal_scores,
            "zscore_max":          z_scores_max,
        })

        issues = []
        for i in np.where(is_outlier)[0]:
            sc_val = float(ensemble_score[i])
            issues.append(LabelIssue(
                sample_index    = int(i),
                given_label     = y[i],
                predicted_label = y[i],
                issue_type      = "outlier",
                confidence      = 1.0 - sc_val,
                severity        = "critical" if sc_val > 0.90 else "high" if sc_val > 0.80 else "medium",
                description     = (
                    f"Class-conditional outlier in class '{y[i]}': "
                    f"ensemble_score={sc_val:.3f} "
                    f"(IF={if_scores[i]:.2f}, LOF={lof_scores[i]:.2f}, "
                    f"Mahal={mahal_scores[i]:.2f})"
                ),
                suggested_label = None,
            ))

        logger.info(f"  Class-conditional outliers: {len(issues)}")
        return result_df, issues

    def _mahalanobis_scores(self, X_cls: np.ndarray) -> np.ndarray:
        """Robust Mahalanobis distance using MCD, normalized to [0,1]."""
        try:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet(random_state=self.seed, support_fraction=0.75)
            mcd.fit(X_cls)
            distances = mcd.mahalanobis(X_cls)
        except Exception:
            # Fallback: standard Mahalanobis
            try:
                mu    = X_cls.mean(axis=0)
                cov   = np.cov(X_cls.T) + np.eye(X_cls.shape[1]) * 1e-6
                cov_inv = np.linalg.pinv(cov)
                diff  = X_cls - mu
                distances = np.array([
                    float(d @ cov_inv @ d) for d in diff
                ])
            except Exception:
                return np.zeros(len(X_cls))

        # Chi-squared p-value → outlier probability
        df    = X_cls.shape[1]
        pvals = 1.0 - stats.chi2.cdf(distances, df=df)
        # Low p-value → outlier → high outlier score
        return np.clip(1.0 - pvals, 0, 1)

    def _zscore_outlier(self, X_cls: np.ndarray,
                         threshold: float = 3.5) -> np.ndarray:
        """Per-sample max |z-score| across features, normalized."""
        if X_cls.shape[0] < 2:
            return np.zeros(len(X_cls))
        mu  = X_cls.mean(axis=0)
        std = X_cls.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        z   = np.abs((X_cls - mu) / std)
        max_z = z.max(axis=1)
        # Score: z > threshold → outlier; normalize to [0,1]
        return np.clip((max_z - threshold) / threshold, 0, 1)


# ==============================================================================
# BLOCK 8 — CLASS SEPARABILITY ANALYZER
# ==============================================================================

class ClassSeparabilityAnalyzer:
    """
    Measure how well-separated the classes are in feature space.
    Poor separability → labels may be noisy OR features are insufficient.

    Metrics
    -------
    1. Fisher's Linear Discriminant Ratio (F-ratio / ANOVA F-statistic)
       F = (between-class variance) / (within-class variance)
       High F → good separability.
       Computed per feature and aggregated (weighted by F-ratio).

    2. Bhattacharyya Distance (pairwise between classes)
       D_B = -ln(BC)  where BC = Bhattacharyya coefficient
       For Gaussian classes: D_B = (1/8)(μ1-μ2)^T Σ^{-1} (μ1-μ2) + (1/2)ln(...)
       D_B = 0 → identical distributions, D_B → ∞ → perfectly separable.

    3. Overlap Coefficient (pairwise)
       OVL = 1 - 0.5 * integral |p(x) - q(x)| dx
       Estimated via histogram intersection.
       OVL = 0 → no overlap (perfectly separable), OVL = 1 → identical.

    4. Average k-NN Purity (multi-class)
       For each sample: fraction of its k nearest neighbors with the same label.
       Global average purity measures class cluster cohesion.
       Purity = 1 → perfectly separated clusters.

    5. Silhouette Score
       silhouette(i) = (b_i - a_i) / max(a_i, b_i)
       a_i = mean intra-cluster distance
       b_i = mean nearest-cluster distance
       Score ∈ [-1, 1], higher = better separated.

    6. Davis-Bouldin Index
       DB = mean_i(max_{j≠i} (s_i + s_j) / d_ij)
       s_i = mean distance within cluster i
       d_ij = distance between cluster centroids i and j
       Lower = better separated.

    7. Class Overlap Score (0-100, lower = less overlap = better)
       Composite from the above metrics.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_pca_components: Optional[int] = None,
    ):
        self.n_neighbors       = n_neighbors
        self.n_pca_components  = n_pca_components

    def analyze(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict:
        """Returns full separability report."""
        le      = LabelEncoder()
        y_enc   = le.fit_transform(y)
        classes = le.classes_.tolist()
        n_cls   = len(classes)

        sc    = StandardScaler()
        X_sc  = sc.fit_transform(X)

        # Optional PCA for high-dimensional data
        if self.n_pca_components and X_sc.shape[1] > self.n_pca_components:
            from sklearn.decomposition import PCA
            pca   = PCA(n_components=self.n_pca_components, random_state=42)
            X_sc  = pca.fit_transform(X_sc)

        logger.info(f"Separability analysis: {n_cls} classes, {X_sc.shape[1]} features")

        # 1. Fisher F-ratio
        f_stats = self._fisher_f_ratio(X_sc, y_enc, n_cls)

        # 2. Bhattacharyya distances (pairwise)
        bhatt = self._bhattacharyya_pairwise(X_sc, y_enc, n_cls)

        # 3. Overlap coefficients (pairwise)
        overlap = self._overlap_pairwise(X_sc, y_enc, n_cls, classes)

        # 4. k-NN purity
        knn_purity = self._knn_purity(X_sc, y_enc)

        # 5. Silhouette score
        sil_score = self._silhouette(X_sc, y_enc)

        # 6. Davis-Bouldin index
        db_index = self._davis_bouldin(X_sc, y_enc, n_cls)

        # 7. Composite overlap score (0-100, lower = better)
        # Invert: 100 = no overlap (perfect), 0 = complete overlap
        norm_bhatt   = float(np.mean([v for v in bhatt.values()]))
        norm_overlap = float(np.mean([v for v in overlap.values()]))
        norm_sil     = (sil_score + 1) / 2  # [0,1]
        norm_db      = 1 / (1 + db_index)    # [0,1], higher = better
        norm_purity  = knn_purity

        separability_score = round(float(
            0.30 * (1 - norm_overlap)
            + 0.25 * norm_purity
            + 0.20 * norm_sil
            + 0.15 * norm_db
            + 0.10 * min(norm_bhatt / 2.0, 1.0)
        ) * 100, 1)

        # Verdict
        if separability_score > 70:
            verdict = "Well Separated"
        elif separability_score > 45:
            verdict = "Moderate Overlap"
        else:
            verdict = "High Overlap — Labels May Be Ambiguous"

        report = {
            "verdict":                verdict,
            "separability_score":     separability_score,
            "fisher_f_mean":          round(float(np.mean(f_stats)), 3),
            "fisher_f_per_feature":   [round(float(f), 3) for f in f_stats[:10]],
            "bhattacharyya_pairwise": {k: round(v, 4) for k, v in bhatt.items()},
            "mean_bhattacharyya":     round(float(np.mean(list(bhatt.values()))), 4),
            "overlap_pairwise":       {k: round(v, 4) for k, v in overlap.items()},
            "mean_overlap":           round(float(np.mean(list(overlap.values()))), 4),
            "knn_purity":             round(knn_purity, 4),
            "silhouette_score":       round(sil_score, 4),
            "davis_bouldin_index":    round(db_index, 4),
        }
        logger.info(
            f"Separability: score={separability_score}, "
            f"verdict={verdict}, sil={sil_score:.3f}"
        )
        return report

    def _fisher_f_ratio(
        self, X: np.ndarray, y_enc: np.ndarray, n_cls: int
    ) -> np.ndarray:
        """ANOVA F-statistic per feature."""
        groups = [X[y_enc == c] for c in range(n_cls) if (y_enc == c).sum() > 0]
        if len(groups) < 2:
            return np.zeros(X.shape[1])
        try:
            f_stats = np.array([
                float(stats.f_oneway(*[g[:, j] for g in groups]).statistic)
                for j in range(X.shape[1])
            ])
            return np.nan_to_num(f_stats, nan=0.0)
        except Exception:
            return np.zeros(X.shape[1])

    def _bhattacharyya_pairwise(
        self, X: np.ndarray, y_enc: np.ndarray, n_cls: int
    ) -> Dict[str, float]:
        """Pairwise Bhattacharyya distance between class Gaussians."""
        result = {}
        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                X_i = X[y_enc == i]
                X_j = X[y_enc == j]
                if len(X_i) < 2 or len(X_j) < 2:
                    continue
                try:
                    mu_i = X_i.mean(axis=0)
                    mu_j = X_j.mean(axis=0)
                    cov_i = np.cov(X_i.T) + np.eye(X_i.shape[1]) * 1e-6
                    cov_j = np.cov(X_j.T) + np.eye(X_j.shape[1]) * 1e-6
                    cov_m = (cov_i + cov_j) / 2

                    diff  = mu_i - mu_j
                    try:
                        cov_m_inv = np.linalg.pinv(cov_m)
                        term1 = 0.125 * float(diff @ cov_m_inv @ diff)
                    except Exception:
                        term1 = 0.0

                    try:
                        sign_m, logdet_m = np.linalg.slogdet(cov_m)
                        sign_i, logdet_i = np.linalg.slogdet(cov_i)
                        sign_j, logdet_j = np.linalg.slogdet(cov_j)
                        term2 = 0.5 * (logdet_m - 0.5 * (logdet_i + logdet_j))
                    except Exception:
                        term2 = 0.0

                    d_b = term1 + term2
                    result[f"class_{i}_vs_{j}"] = float(max(d_b, 0))
                except Exception:
                    result[f"class_{i}_vs_{j}"] = 0.0
        return result

    def _overlap_pairwise(
        self, X: np.ndarray, y_enc: np.ndarray, n_cls: int, classes: List
    ) -> Dict[str, float]:
        """Pairwise overlap via histogram intersection on PCA-projected data."""
        result = {}
        # Use first PC for overlap estimation
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1, random_state=42)
            x1d = pca.fit_transform(X).ravel()
        except Exception:
            x1d = X[:, 0]

        for i in range(n_cls):
            for j in range(i + 1, n_cls):
                Xi = x1d[y_enc == i]
                Xj = x1d[y_enc == j]
                if len(Xi) < 2 or len(Xj) < 2:
                    continue
                # Histogram intersection
                lo  = min(Xi.min(), Xj.min())
                hi  = max(Xi.max(), Xj.max())
                if hi <= lo:
                    result[f"{classes[i]}_vs_{classes[j]}"] = 1.0
                    continue
                bins = np.linspace(lo, hi, 50)
                hi_i, _ = np.histogram(Xi, bins=bins, density=True)
                hi_j, _ = np.histogram(Xj, bins=bins, density=True)
                width    = bins[1] - bins[0]
                ovl      = float(np.minimum(hi_i, hi_j).sum() * width)
                result[f"{classes[i]}_vs_{classes[j]}"] = round(min(ovl, 1.0), 4)
        return result

    def _knn_purity(self, X: np.ndarray, y_enc: np.ndarray) -> float:
        k  = min(self.n_neighbors, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn.fit(X)
        _, idx = nn.kneighbors(X)
        purity = float((y_enc[idx[:, 1:]] == y_enc[:, None]).mean())
        return purity

    def _silhouette(self, X: np.ndarray, y_enc: np.ndarray) -> float:
        from sklearn.metrics import silhouette_score
        if len(np.unique(y_enc)) < 2:
            return 0.0
        try:
            n_sample = min(5000, len(X))
            idx = np.random.choice(len(X), n_sample, replace=False)
            return float(silhouette_score(X[idx], y_enc[idx], metric="euclidean"))
        except Exception:
            return 0.0

    def _davis_bouldin(self, X: np.ndarray, y_enc: np.ndarray, n_cls: int) -> float:
        from sklearn.metrics import davies_bouldin_score
        if len(np.unique(y_enc)) < 2:
            return 0.0
        try:
            return float(davies_bouldin_score(X, y_enc))
        except Exception:
            return 0.0


# ==============================================================================
# BLOCK 9 — LABEL QUALITY SCORER
# ==============================================================================

class LabelQualityScorer:
    """
    Compute a unified Label Quality Score 0-100 from all analysis blocks.

    Weights
    -------
    Noise Rate          : 30%  (most critical — directly impacts model accuracy)
    Inconsistency Rate  : 20%  (same features → different labels = confusion)
    Ambiguity Rate      : 15%  (borderline samples → irreducible error)
    Drift Severity      : 15%  (policy changes → distribution mismatch)
    Imbalance Severity  : 10%  (minority class disadvantage)
    Annotation Confidence: 10% (overall labeler reliability)

    Per-class quality breakdown
    ---------------------------
    Each class gets its own quality score from:
    - per-class noise rate
    - per-class annotation confidence
    - per-class kappa
    - class sample count (small classes get uncertainty penalty)

    Verdicts
    --------
    90–100 : Excellent ✅ — production-ready dataset
    75–89  : Good      🟢 — minor cleanup recommended
    60–74  : Needs Work ⚠️ — clean before training
    40–59  : Poor      🔴 — significant label quality issues
    0–39   : Critical  ❌ — dataset not suitable for training
    """

    WEIGHTS = {
        "noise":       0.30,
        "inconsistency": 0.20,
        "ambiguity":   0.15,
        "drift":       0.15,
        "imbalance":   0.10,
        "confidence":  0.10,
    }

    VERDICT_THRESHOLDS = [
        (90, "Excellent ✅"),
        (75, "Good 🟢"),
        (60, "Needs Work ⚠️"),
        (40, "Poor 🔴"),
        (0,  "Critical ❌"),
    ]

    def compute(
        self,
        n_samples:       int,
        noise_rate:      float,
        n_issues:        int,
        n_inconsistent:  int,
        n_ambiguous:     int,
        drift_report:    Dict,
        imbalance_report:Dict,
        annotation_conf: Dict,
        noise_matrix:    Optional[NoiseMatrix],
    ) -> Dict:
        """
        Returns
        -------
        dict with overall_score, verdict, per-component scores, recommendations
        """
        # Component scores (0-100, higher = better quality)

        # Noise: score = 100 * (1 - noise_rate)
        noise_score = max(0, 100 * (1 - min(noise_rate, 1.0)))

        # Inconsistency
        inconsistency_rate = n_inconsistent / max(n_samples, 1)
        inconsistency_score = max(0, 100 * (1 - min(inconsistency_rate * 5, 1.0)))

        # Ambiguity
        ambiguity_rate  = n_ambiguous / max(n_samples, 1)
        ambiguity_score = max(0, 100 * (1 - min(ambiguity_rate * 3, 1.0)))

        # Drift
        psi     = drift_report.get("overall_psi", 0.0)
        if psi < 0.10:
            drift_score = 100.0
        elif psi < 0.25:
            drift_score = 100.0 - (psi - 0.10) / 0.15 * 50
        else:
            drift_score = max(0, 50.0 - (psi - 0.25) * 100)

        # Imbalance
        imbalance_sc = imbalance_report.get("imbalance_score", 0)
        imbalance_score = max(0, 100 - imbalance_sc)

        # Annotation confidence
        conf_score = annotation_conf.get("annotation_quality_score", 75)

        # Composite
        overall_score = (
            self.WEIGHTS["noise"]         * noise_score
            + self.WEIGHTS["inconsistency"]* inconsistency_score
            + self.WEIGHTS["ambiguity"]   * ambiguity_score
            + self.WEIGHTS["drift"]       * drift_score
            + self.WEIGHTS["imbalance"]   * imbalance_score
            + self.WEIGHTS["confidence"]  * conf_score
        )
        overall_score = round(float(np.clip(overall_score, 0, 100)), 1)

        # Verdict
        verdict = self.VERDICT_THRESHOLDS[-1][1]
        for threshold, label in self.VERDICT_THRESHOLDS:
            if overall_score >= threshold:
                verdict = label
                break

        # Per-class quality
        per_class_quality: Dict[Any, float] = {}
        if noise_matrix is not None:
            for cls, cls_noise in noise_matrix.per_class_noise.items():
                cls_conf = annotation_conf.get(
                    "per_class_kappa", {}
                ).get(str(cls), 0.5)
                cls_score = round(float(
                    0.60 * max(0, 100 * (1 - cls_noise))
                    + 0.40 * max(0, (cls_conf + 1) / 2 * 100)
                ), 1)
                per_class_quality[cls] = cls_score

        # Priority recommendations
        recommendations = self._build_recommendations(
            noise_rate, noise_score, inconsistency_rate, inconsistency_score,
            ambiguity_rate, ambiguity_score, psi, drift_score,
            imbalance_report, conf_score, noise_matrix
        )

        return {
            "overall_score":        overall_score,
            "verdict":              verdict,
            "component_scores": {
                "noise":          round(noise_score, 1),
                "inconsistency":  round(inconsistency_score, 1),
                "ambiguity":      round(ambiguity_score, 1),
                "drift":          round(drift_score, 1),
                "imbalance":      round(imbalance_score, 1),
                "confidence":     round(conf_score, 1),
            },
            "component_rates": {
                "noise_rate":         round(noise_rate, 4),
                "inconsistency_rate": round(inconsistency_rate, 4),
                "ambiguity_rate":     round(ambiguity_rate, 4),
                "psi":                round(psi, 4),
                "imbalance_ratio":    imbalance_report.get("imbalance_ratio", 1.0),
                "annotation_confidence": round(conf_score, 1),
            },
            "per_class_quality": {str(k): v for k, v in per_class_quality.items()},
            "recommendations":   recommendations,
        }

    def _build_recommendations(
        self,
        noise_rate, noise_score,
        inconsistency_rate, inconsistency_score,
        ambiguity_rate, ambiguity_score,
        psi, drift_score,
        imbalance_report, conf_score,
        noise_matrix,
    ) -> List[Dict]:
        recs = []

        if noise_rate > NOISE_RATE_CRITICAL:
            recs.append({
                "priority": 1, "area": "Label Noise",
                "severity": "critical",
                "issue": f"Noise rate {noise_rate:.1%} exceeds critical threshold {NOISE_RATE_CRITICAL:.0%}",
                "action": "Re-label all samples flagged as label_errors. Use Confident Learning "
                          "to prioritize: fix critical → high → medium severity issues first.",
                "impact": "High noise will cause model to learn wrong decision boundaries. "
                          "Expected accuracy drop: ~{:.0%}".format(min(noise_rate * 2, 0.30)),
            })
        elif noise_rate > NOISE_RATE_HIGH:
            recs.append({
                "priority": 2, "area": "Label Noise",
                "severity": "high",
                "issue": f"Noise rate {noise_rate:.1%} is elevated",
                "action": "Review and correct flagged label errors. Consider noise-robust "
                          "loss functions (symmetric cross-entropy, GCE loss) as interim measure.",
                "impact": "Moderate impact on model performance. Training with cleanlab's "
                          "LearningWithNoisyLabels can partially mitigate.",
            })

        if inconsistency_rate > 0.05:
            recs.append({
                "priority": 2, "area": "Label Inconsistency",
                "severity": "high",
                "issue": f"{inconsistency_rate:.1%} of samples are near-duplicates with conflicting labels",
                "action": "Review duplicate groups; establish consistent labeling guidelines. "
                          "Consider majority-vote resolution for duplicate groups.",
                "impact": "Inconsistent labels confuse the model about true class boundaries.",
            })

        if psi > PSI_THRESHOLD_MAJOR:
            recs.append({
                "priority": 3, "area": "Label Drift",
                "severity": "high",
                "issue": f"Major label distribution shift detected (PSI={psi:.3f})",
                "action": "Stratify train/val/test splits by time window. Consider separate "
                          "models per data period or drift-aware training.",
                "impact": "Model trained on early data may not generalize to later data.",
            })

        ir = imbalance_report.get("imbalance_ratio", 1.0)
        if ir > IMBALANCE_RATIO_HIGH:
            recs.append({
                "priority": 3, "area": "Class Imbalance",
                "severity": "high",
                "issue": f"Severe imbalance: IR={ir:.1f}",
                "action": "Apply SMOTE oversampling to minority classes. Use class_weight='balanced'. "
                          "Consider stratified sampling and balanced batch construction.",
                "impact": f"Minority class recall will drop by ~{min(math.log(ir) * 0.2, 0.6):.0%} without mitigation.",
            })

        if ambiguity_rate > 0.15:
            recs.append({
                "priority": 4, "area": "Ambiguous Samples",
                "severity": "medium",
                "issue": f"{ambiguity_rate:.1%} of samples are near the decision boundary",
                "action": "Consider collecting more data near decision boundaries. "
                          "Ambiguous samples may benefit from expert re-review or multi-label treatment.",
                "impact": "High ambiguity suggests irreducible label noise or genuine multi-label structure.",
            })

        if conf_score < 60:
            recs.append({
                "priority": 4, "area": "Annotation Confidence",
                "severity": "medium",
                "issue": f"Annotation quality score {conf_score:.0f}/100 is low",
                "action": "Improve annotator training, establish clear labeling guidelines, "
                          "consider inter-annotator agreement metrics in future annotation rounds.",
                "impact": "Low annotator confidence propagates to model uncertainty.",
            })

        # Sort by priority
        recs.sort(key=lambda r: r["priority"])
        return recs


# ==============================================================================
# BLOCK 10 — MASTER ORCHESTRATOR + CONVENIENCE FUNCTIONS + CLI
# ==============================================================================

class LabelQualityAnalyzer:
    """
    🔬 Nydra Label Quality Analyzer — Master Class

    The most comprehensive label quality analysis in open-source Python.

    Runs 8 specialized analysis blocks in a single pipeline call:
      1. Confident Learning — noise detection + noise transition matrix
      2. Inconsistency      — near-duplicate groups + label flip candidates
      3. Ambiguity          — decision boundary samples
      4. Drift              — label distribution shift across time
      5. Imbalance          — Gini, entropy, SMOTE factors
      6. Annotation Conf.   — Platt calibration, Kappa, Krippendorff's Alpha
      7. Outlier Labels     — class-conditional outliers (IF + LOF + Mahal)
      8. Separability       — Fisher F, Bhattacharyya, Silhouette, DB-Index
      9. Quality Scorer     — composite 0-100 score + recommendations

    Usage
    -----
    >>> from label_quality import LabelQualityAnalyzer
    >>> analyzer = LabelQualityAnalyzer()
    >>> report = analyzer.analyze(df, label_col="label", feature_cols=["f1","f2","f3"])
    >>> print(report.label_quality_score)   # e.g. 73.4
    >>> print(report.verdict)               # e.g. "Needs Work ⚠️"
    >>> print(report.n_label_errors)        # e.g. 234

    Convenience functions
    ---------------------
    >>> from label_quality import quick_noise_check, find_label_errors, get_quality_score
    """

    def __init__(
        self,
        classifier      = None,     # custom classifier; None = auto-select
        cv_folds:   int = DEFAULT_CV_FOLDS,
        n_jobs:     int = -1,
        seed:       int = 42,
        run_all:    bool = True,    # run all blocks; set False to skip slow ones
        run_separability: bool = True,
        run_outliers:     bool = True,
        run_drift:        bool = True,
        verbose:    bool = True,
    ):
        self.cv_folds         = cv_folds
        self.n_jobs           = n_jobs
        self.seed             = seed
        self.run_all          = run_all
        self.run_separability = run_all and run_separability
        self.run_outliers     = run_all and run_outliers
        self.run_drift        = run_all and run_drift
        self.verbose          = verbose

        # Initialize all blocks
        self._cl       = ConfidentLearningDetector(
            classifier=classifier, cv_folds=cv_folds, n_jobs=n_jobs, seed=seed
        )
        self._incon    = InconsistentLabelDetector()
        self._ambig    = AmbiguousSampleDetector(cv_folds=cv_folds, seed=seed)
        self._drift    = LabelDriftDetector()
        self._imbal    = ImbalanceAnalyzer()
        self._conf     = AnnotationConfidenceScorer(cv_folds=cv_folds, seed=seed)
        self._outlier  = OutlierLabelDetector(seed=seed)
        self._sep      = ClassSeparabilityAnalyzer()
        self._scorer   = LabelQualityScorer()

    # ── primary API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        df:             pd.DataFrame,
        label_col:      str,
        feature_cols:   Optional[List[str]] = None,
        pred_probs:     Optional[np.ndarray] = None,
        time_col:       Optional[str]        = None,
        annotators_df:  Optional[pd.DataFrame] = None,
    ) -> LabelQualityReport:
        """
        Run the full label quality analysis pipeline.

        Parameters
        ----------
        df           : input DataFrame
        label_col    : name of the label/target column
        feature_cols : list of feature column names (None = all non-label cols)
        pred_probs   : precomputed OOS predicted probabilities (n, n_classes)
                       If None, computed internally via cross-validation.
        time_col     : column for drift analysis ordering (None = row order)
        annotators_df: DataFrame with annotator columns for Krippendorff's Alpha

        Returns
        -------
        LabelQualityReport
        """
        t0 = time.time()
        if self.verbose:
            logger.info("=" * 65)
            logger.info("🔬 Nydra — Label Quality Analyzer v0.6.0")
            logger.info(f"   Samples : {len(df)}")
            logger.info(f"   Label   : {label_col}")
            logger.info("=" * 65)

        # ── prepare data ──────────────────────────────────────────────────────
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]

        y_raw  = df[label_col].values
        le     = LabelEncoder()
        y_enc  = le.fit_transform(y_raw)
        classes = le.classes_.tolist()
        n_classes = len(classes)

        # Feature matrix
        X = self._prepare_features(df[feature_cols])

        # ── Adjust CV folds for small datasets ──────────────────────────────
        n_samples = len(X)
        _, counts = np.unique(y_enc, return_counts=True)
        min_samples = np.min(counts) if len(counts) > 0 else 0

        if n_samples < self.cv_folds or min_samples < self.cv_folds:
            old_folds = self.cv_folds
            # Stratified CV needs at least 2 samples per class.
            # Regular CV needs at least 2 samples total.
            if min_samples >= 2:
                self.cv_folds = min(old_folds, min_samples)
            elif n_samples >= 2:
                self.cv_folds = min(old_folds, n_samples)
            else:
                self.cv_folds = 0
            
            if self.verbose and old_folds != self.cv_folds:
                logger.warning(
                    f"Dataset too small ({n_samples} samples, min_class={min_samples}). "
                    f"Reducing CV folds from {old_folds} to {self.cv_folds}."
                )
            
            # Synchronize with sub-blocks
            self._cl.cv_folds    = self.cv_folds
            self._ambig.cv_folds = self.cv_folds
            self._conf.cv_folds  = self.cv_folds

        # ── Block 1: Confident Learning ───────────────────────────────────────
        if self.verbose:
            logger.info("Block 1: Confident Learning...")
        cl_issues, noise_mx, quality_scores = self._cl.detect(X, y_enc, pred_probs)

        # Get pred_probs for downstream blocks
        if pred_probs is None:
            pred_probs = self._cl._get_pred_probs(X, y_enc, n_classes)

        # ── Block 2: Inconsistency ────────────────────────────────────────────
        if self.verbose:
            logger.info("Block 2: Inconsistency detection...")
        incon_df, incon_issues = self._incon.detect(X, y_raw, pred_probs)

        # ── Block 3: Ambiguity ────────────────────────────────────────────────
        if self.verbose:
            logger.info("Block 3: Ambiguity detection...")
        ambig_df, ambig_issues = self._ambig.detect(X, y_enc, pred_probs)

        # ── Block 4: Drift ────────────────────────────────────────────────────
        drift_report = {}
        if self.run_drift:
            if self.verbose:
                logger.info("Block 4: Label drift detection...")
            time_idx = df[time_col].values if time_col else None
            drift_report = self._drift.detect(y_raw, time_idx)

        # ── Block 5: Imbalance ────────────────────────────────────────────────
        if self.verbose:
            logger.info("Block 5: Imbalance analysis...")
        imbalance_report = self._imbal.analyze(y_raw)

        # ── Block 6: Annotation Confidence ───────────────────────────────────
        if self.verbose:
            logger.info("Block 6: Annotation confidence scoring...")
        annotation_conf = self._conf.score(y_enc, pred_probs,
                                            annotators_df=annotators_df)

        # ── Block 7: Outlier Labels ───────────────────────────────────────────
        outlier_issues: List[LabelIssue] = []
        if self.run_outliers:
            if self.verbose:
                logger.info("Block 7: Class-conditional outlier detection...")
            _, outlier_issues = self._outlier.detect(X, y_raw)

        # ── Block 8: Separability ─────────────────────────────────────────────
        sep_report = {}
        if self.run_separability:
            if self.verbose:
                logger.info("Block 8: Class separability analysis...")
            sep_report = self._sep.analyze(X, y_enc)

        # ── Block 9: Quality Score ────────────────────────────────────────────
        if self.verbose:
            logger.info("Block 9: Computing quality score...")

        all_issues = cl_issues + incon_issues + ambig_issues + outlier_issues

        n_ambiguous    = int(ambig_df["is_ambiguous"].sum())
        n_inconsistent = int(incon_df["is_near_duplicate"].sum())
        n_outliers     = len(outlier_issues)

        score_report = self._scorer.compute(
            n_samples        = len(df),
            noise_rate       = noise_mx.noise_rate,
            n_issues         = len(cl_issues),
            n_inconsistent   = n_inconsistent,
            n_ambiguous      = n_ambiguous,
            drift_report     = drift_report,
            imbalance_report = imbalance_report,
            annotation_conf  = annotation_conf,
            noise_matrix     = noise_mx,
        )

        elapsed = time.time() - t0

        report = LabelQualityReport(
            label_quality_score   = score_report["overall_score"],
            verdict               = score_report["verdict"],
            noise_rate            = noise_mx.noise_rate,
            n_label_errors        = len(cl_issues),
            n_ambiguous           = n_ambiguous,
            n_inconsistent        = n_inconsistent,
            n_outliers            = n_outliers,
            issues                = all_issues,
            noise_matrix          = noise_mx,
            per_class_quality     = score_report["per_class_quality"],
            drift_report          = drift_report,
            imbalance_report      = imbalance_report,
            annotation_confidence = annotation_conf,
            separability_report   = sep_report,
            n_samples             = len(df),
            n_classes             = n_classes,
            classes               = classes,
            processing_time_s     = round(elapsed, 2),
            recommendations       = score_report["recommendations"],
        )

        if self.verbose:
            self._print_summary(report)

        return report

    def to_dataframe(self, report: LabelQualityReport) -> pd.DataFrame:
        """Convert issues list to a DataFrame for easy inspection."""
        if not report.issues:
            return pd.DataFrame()
        rows = []
        for iss in report.issues:
            rows.append({
                "sample_index":    iss.sample_index,
                "given_label":     iss.given_label,
                "predicted_label": iss.predicted_label,
                "issue_type":      iss.issue_type,
                "confidence":      round(iss.confidence, 4),
                "severity":        iss.severity,
                "suggested_label": iss.suggested_label,
                "duplicate_group": iss.duplicate_group,
                "description":     iss.description,
            })
        df = pd.DataFrame(rows)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        df["severity_rank"] = df["severity"].map(severity_order)
        return df.sort_values(["severity_rank", "confidence"]).drop(
            columns=["severity_rank"]
        ).reset_index(drop=True)

    def save_report(self, report: LabelQualityReport, output_dir: str) -> Dict:
        """Save full report to JSON + CSV."""
        os.makedirs(output_dir, exist_ok=True)

        # JSON summary
        summary = {
            "label_quality_score": report.label_quality_score,
            "verdict":             report.verdict,
            "noise_rate":          report.noise_rate,
            "n_label_errors":      report.n_label_errors,
            "n_ambiguous":         report.n_ambiguous,
            "n_inconsistent":      report.n_inconsistent,
            "n_outliers":          report.n_outliers,
            "n_samples":           report.n_samples,
            "n_classes":           report.n_classes,
            "classes":             [str(c) for c in report.classes],
            "processing_time_s":   report.processing_time_s,
            "per_class_quality":   report.per_class_quality,
            "recommendations":     report.recommendations,
            "drift":               report.drift_report,
            "imbalance":           report.imbalance_report,
            "annotation_confidence": {
                k: v for k, v in report.annotation_confidence.items()
                if not isinstance(v, list)  # skip per-sample lists
            },
            "separability":        report.separability_report,
        }
        if report.noise_matrix:
            summary["noise_matrix"] = {
                "classes":        [str(c) for c in report.noise_matrix.classes],
                "noise_rate":     report.noise_matrix.noise_rate,
                "per_class_noise": {
                    str(k): v for k, v in report.noise_matrix.per_class_noise.items()
                },
            }

        json_path = os.path.join(output_dir, "label_quality_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

        # CSV of issues
        issues_df  = self.to_dataframe(report)
        csv_path   = os.path.join(output_dir, "label_issues.csv")
        issues_df.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(f"Report saved → {output_dir}/")
        return {"json": json_path, "csv": csv_path}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _prepare_features(self, df_features: pd.DataFrame) -> np.ndarray:
        """Convert feature DataFrame to numeric numpy array."""
        # One-hot encode categoricals
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns
        num_cols = df_features.select_dtypes(include=[np.number]).columns

        parts = []
        if len(num_cols) > 0:
            X_num = df_features[num_cols].values.astype(np.float64)
            # Fill NaN with column median
            col_medians = np.nanmedian(X_num, axis=0)
            for j in range(X_num.shape[1]):
                mask = np.isnan(X_num[:, j])
                X_num[mask, j] = col_medians[j]
            parts.append(X_num)

        if len(cat_cols) > 0:
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            X_cat = ohe.fit_transform(df_features[cat_cols].fillna("__missing__"))
            parts.append(X_cat)

        if not parts:
            raise ValueError("No usable feature columns found")

        X = np.hstack(parts)
        # Standardize
        sc = StandardScaler()
        return sc.fit_transform(X)

    def _print_summary(self, report: LabelQualityReport) -> None:
        sep = "─" * 55
        print(f"\n{sep}")
        print(f"  🔬 Label Quality Report")
        print(sep)
        print(f"  Overall Score : {report.label_quality_score:.1f}/100")
        print(f"  Verdict       : {report.verdict}")
        print(f"  Noise Rate    : {report.noise_rate:.2%}")
        print(f"  Label Errors  : {report.n_label_errors}")
        print(f"  Ambiguous     : {report.n_ambiguous}")
        print(f"  Inconsistent  : {report.n_inconsistent}")
        print(f"  Outliers      : {report.n_outliers}")
        print(f"  Time          : {report.processing_time_s}s")
        print(sep)
        if report.recommendations:
            print("  Top Recommendations:")
            for rec in report.recommendations[:3]:
                print(f"  [{rec['severity'].upper()}] {rec['area']}: {rec['issue']}")
        print(sep + "\n")


# ── CONVENIENCE FUNCTIONS ─────────────────────────────────────────────────────

def quick_noise_check(
    df:          pd.DataFrame,
    label_col:   str,
    feature_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Fast noise check — runs Confident Learning only.
    Returns dict with noise_rate, n_errors, error_indices.
    """
    analyzer = LabelQualityAnalyzer(
        run_separability=False,
        run_outliers=False,
        run_drift=False,
        verbose=False,
    )
    report = analyzer.analyze(df, label_col, feature_cols)
    return {
        "noise_rate":    report.noise_rate,
        "n_errors":      report.n_label_errors,
        "error_indices": [iss.sample_index for iss in report.issues
                          if iss.issue_type == "label_error"],
        "verdict":       report.verdict,
    }


def find_label_errors(
    df:           pd.DataFrame,
    label_col:    str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Find label errors and return them as a DataFrame.
    Sorted by severity (critical first).
    """
    analyzer = LabelQualityAnalyzer(
        run_separability=False, run_drift=False, verbose=False
    )
    report = analyzer.analyze(df, label_col, feature_cols)
    return analyzer.to_dataframe(report)


def get_quality_score(
    df:           pd.DataFrame,
    label_col:    str,
    feature_cols: Optional[List[str]] = None,
) -> float:
    """Quick single-number quality score 0-100."""
    analyzer = LabelQualityAnalyzer(verbose=False)
    report   = analyzer.analyze(df, label_col, feature_cols)
    return report.label_quality_score


def get_priority_fixes(
    df:           pd.DataFrame,
    label_col:    str,
    feature_cols: Optional[List[str]] = None,
    top_n:        int = 50,
) -> pd.DataFrame:
    """
    Return the top-N samples most urgently needing re-labeling.
    Ordered by severity + confidence.
    """
    issues_df = find_label_errors(df, label_col, feature_cols)
    return issues_df.head(top_n)


def full_analysis(
    df:           pd.DataFrame,
    label_col:    str,
    feature_cols: Optional[List[str]] = None,
    output_dir:   Optional[str]       = None,
) -> LabelQualityReport:
    """
    Run complete label quality analysis and optionally save report.

    Returns LabelQualityReport.
    """
    analyzer = LabelQualityAnalyzer()
    report   = analyzer.analyze(df, label_col, feature_cols)
    if output_dir:
        analyzer.save_report(report, output_dir)
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        prog        = "label_quality",
        description = "🔬 Nydra — Label Quality Analyzer v0.6.0",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv",          help="Input CSV file")
    parser.add_argument("--label",      required=True, metavar="COL",
                        help="Label/target column name")
    parser.add_argument("--features",   default=None, metavar="COL1,COL2",
                        help="Comma-separated feature columns (default: all non-label)")
    parser.add_argument("--output",     default="label_quality_report", metavar="DIR",
                        help="Output directory for report")
    parser.add_argument("--folds",      type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no-sep",     action="store_true", help="Skip separability analysis")
    parser.add_argument("--no-outlier", action="store_true", help="Skip outlier detection")
    parser.add_argument("--no-drift",   action="store_true", help="Skip drift detection")
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    feature_cols = args.features.split(",") if args.features else None

    analyzer = LabelQualityAnalyzer(
        cv_folds         = args.folds,
        seed             = args.seed,
        run_separability = not args.no_sep,
        run_outliers     = not args.no_outlier,
        run_drift        = not args.no_drift,
        verbose          = not args.quiet,
    )
    report = analyzer.analyze(df, args.label, feature_cols)
    paths  = analyzer.save_report(report, args.output)

    print(f"\n✅ Label Quality Analysis Complete")
    print(f"   Score   : {report.label_quality_score:.1f}/100 — {report.verdict}")
    print(f"   Errors  : {report.n_label_errors}")
    print(f"   Report  : {paths['json']}")
    print(f"   Issues  : {paths['csv']}")



