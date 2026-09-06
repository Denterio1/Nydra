"""
image_analyzer.py — Dataset-Level Image Analysis Engine
========================================================
Part of Nydra v0.6.0 — Images & Text Module

Role:
    While image_loader.py reads individual images and image_quality.py
    inspects each image alone — image_analyzer.py looks at the ENTIRE
    dataset and discovers problems that only appear when comparing images.

Blocks:
    1. Color Distribution Analysis      (Shortcut Learning / Color Bias)
    2. Class Balance Deep Analysis      (Gini, Overlap, Imbalance)
    3. Size & Dimension Analysis        (Size Bias, Aspect Ratio)
    4. Dataset Diversity & Redundancy   (Vendi Score, pHash, dHash)
    5. Distribution Shift Detection     (KS-Test, optional MMD)
    6. Outlier Image Detection          (Isolation Forest, KNN, OOD)
    7. ML Readiness Final Score         (Weighted composite, Verdict)

Dependencies:
    numpy, pandas, PIL, scipy, sklearn, imagehash (optional), collections

Usage:
    from src.data.image_analyzer import ImageAnalyzer

    analyzer = ImageAnalyzer(
        loader_df=image_loader_df,
        quality_df=image_quality_df,
        image_dir="path/to/images",
        label_column="label",
        split_column="split"          # optional — for shift detection
    )
    results = analyzer.analyze_all()
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import warnings
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

# scipy
from scipy import stats
from scipy.spatial.distance import cdist, hamming
from scipy.stats import (
    ks_2samp, chi2_contingency, entropy as scipy_entropy,
    kurtosis, skew
)
from scipy.cluster.hierarchy import linkage, fcluster

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.covariance import EllipticEnvelope

# Optional
try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    from scipy.stats import wasserstein_distance
    HAS_WASSERSTEIN = True
except ImportError:
    HAS_WASSERSTEIN = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("nydra.image_analyzer")


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES — Structured Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ColorBiasResult:
    """Results for Block 1: Color Distribution Analysis."""
    class_mean_rgb: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    class_std_rgb: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    color_bias_detected: bool = False
    biased_classes: List[str] = field(default_factory=list)
    bias_severity: str = "none"             # none / low / medium / high
    channel_variance_ratio: Dict[str, float] = field(default_factory=dict)
    shortcut_risk_score: float = 0.0        # 0–100
    shortcut_risk_level: str = "low"        # low / medium / high / critical
    histogram_divergence: Dict[str, float] = field(default_factory=dict)  # per class
    dominant_colors: Dict[str, List[Tuple[int,int,int]]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassBalanceResult:
    """Results for Block 2: Class Balance Deep Analysis."""
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_percentages: Dict[str, float] = field(default_factory=dict)
    total_images: int = 0
    num_classes: int = 0
    imbalance_ratio: float = 1.0            # max_count / min_count
    gini_coefficient: float = 0.0           # 0=perfect balance, 1=total imbalance
    balance_score: float = 100.0            # 0–100 (100 = perfectly balanced)
    is_severely_imbalanced: bool = False
    majority_class: str = ""
    minority_class: str = ""
    suggested_augmentation: Dict[str, int] = field(default_factory=dict)
    class_overlap_matrix: Optional[np.ndarray] = None
    overlapping_class_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SizeDimensionResult:
    """Results for Block 3: Size & Dimension Analysis."""
    width_stats: Dict[str, float] = field(default_factory=dict)
    height_stats: Dict[str, float] = field(default_factory=dict)
    aspect_ratio_stats: Dict[str, float] = field(default_factory=dict)
    class_size_stats: Dict[str, Dict] = field(default_factory=dict)
    size_bias_detected: bool = False
    biased_size_classes: List[str] = field(default_factory=list)
    recommended_target_size: Tuple[int, int] = (224, 224)
    resize_quality_impact: Dict[str, str] = field(default_factory=dict)  # img -> impact
    aspect_ratio_distribution: Dict[str, int] = field(default_factory=dict)
    dominant_aspect_ratio: str = "unknown"
    size_outlier_images: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiversityRedundancyResult:
    """Results for Block 4: Dataset Diversity & Redundancy."""
    vendi_score: float = 0.0                # Higher = more diverse
    diversity_grade: str = "F"              # A–F
    effective_dataset_size: int = 0         # After removing redundancy
    redundancy_rate: float = 0.0            # 0–1 (0 = no redundancy)
    near_duplicate_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    near_duplicate_clusters: List[List[str]] = field(default_factory=list)
    augmentation_copies_detected: int = 0
    cross_class_similar_pairs: List[Tuple[str, str, str, str, float]] = field(default_factory=list)
    coverage_score: float = 0.0             # 0–100
    similarity_matrix_sample: Optional[np.ndarray] = None
    hash_method_used: str = "pixel_stats"
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionShiftResult:
    """Results for Block 5: Distribution Shift Detection."""
    shift_detected: bool = False
    shift_severity: str = "none"            # none / low / medium / high / critical
    covariate_shift_score: float = 0.0      # 0–100
    ks_test_results: Dict[str, Dict] = field(default_factory=dict)  # per channel
    domain_gap_score: float = 0.0
    splits_compared: Tuple[str, str] = ("train", "test")
    pixel_mean_shift: Dict[str, float] = field(default_factory=dict)
    pixel_std_shift: Dict[str, float] = field(default_factory=dict)
    wasserstein_distances: Dict[str, float] = field(default_factory=dict)
    mmd_score: Optional[float] = None       # None if not computed
    hidden_stratification_risk: float = 0.0
    context_bias_risk: str = "low"
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierImageResult:
    """Results for Block 6: Outlier Image Detection."""
    outlier_images: List[str] = field(default_factory=list)
    outlier_scores: Dict[str, float] = field(default_factory=dict)
    outlier_rate: float = 0.0
    outlier_method: str = "ensemble"
    isolation_forest_outliers: List[str] = field(default_factory=list)
    lof_outliers: List[str] = field(default_factory=list)
    knn_outliers: List[str] = field(default_factory=list)
    ood_images: List[str] = field(default_factory=list)         # Out-of-Distribution
    suggestions: Dict[str, str] = field(default_factory=dict)  # img -> delete/keep/review
    outlier_cluster_labels: Optional[np.ndarray] = None
    recommendations: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLReadinessResult:
    """Results for Block 7: ML Readiness Final Score."""
    # Sub-scores (0–100)
    diversity_score: float = 0.0
    balance_score: float = 0.0
    shift_score: float = 0.0
    quality_score: float = 0.0
    coverage_score: float = 0.0
    size_score: float = 0.0
    outlier_score: float = 0.0

    # Weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "diversity":  0.20,
        "balance":    0.25,
        "shift":      0.20,
        "quality":    0.20,
        "coverage":   0.15,
    })

    # Final
    final_score: float = 0.0
    grade: str = "F"                        # A / B / C / D / F
    verdict: str = "Not Ready"              # Ready ✅ / Needs Work ⚠️ / Not Ready ❌
    priority_issues: List[str] = field(default_factory=list)
    recommendations_ranked: List[Dict] = field(default_factory=list)
    estimated_accuracy_impact: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report — all blocks combined."""
    dataset_name: str = "Unknown"
    analysis_timestamp: str = ""
    total_images: int = 0
    total_classes: int = 0
    image_dir: str = ""

    color_bias: Optional[ColorBiasResult] = None
    class_balance: Optional[ClassBalanceResult] = None
    size_dimension: Optional[SizeDimensionResult] = None
    diversity_redundancy: Optional[DiversityRedundancyResult] = None
    distribution_shift: Optional[DistributionShiftResult] = None
    outlier_detection: Optional[OutlierImageResult] = None
    ml_readiness: Optional[MLReadinessResult] = None

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    analysis_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for JSON/Streamlit."""
        return {
            "dataset_name": self.dataset_name,
            "analysis_timestamp": self.analysis_timestamp,
            "total_images": self.total_images,
            "total_classes": self.total_classes,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.analysis_duration_seconds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _load_image_safe(path: str, max_size: int = 512) -> Optional[np.ndarray]:
    """Load image as RGB numpy array. Returns None on failure."""
    try:
        img = Image.open(path).convert("RGB")
        # Resize large images for speed (analysis quality unaffected)
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return np.array(img, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Cannot load image {path}: {e}")
        return None


def _rgb_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute per-channel statistics for an RGB image array."""
    result = {}
    channel_names = ["R", "G", "B"]
    for i, ch in enumerate(channel_names):
        channel = arr[:, :, i].flatten()
        result[f"{ch}_mean"]   = float(np.mean(channel))
        result[f"{ch}_std"]    = float(np.std(channel))
        result[f"{ch}_median"] = float(np.median(channel))
        result[f"{ch}_skew"]   = float(skew(channel))
        result[f"{ch}_kurt"]   = float(kurtosis(channel))
    result["brightness"]  = float(np.mean(arr))
    result["contrast"]    = float(np.std(arr))
    result["saturation"]  = _compute_saturation(arr)
    return result


def _compute_saturation(arr: np.ndarray) -> float:
    """Compute mean saturation from RGB array (0–255 scale)."""
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 0, (max_c - min_c) / (max_c + 1e-8), 0)
    return float(np.mean(sat) * 255)


def _compute_histogram(arr: np.ndarray, bins: int = 32) -> np.ndarray:
    """Compute normalized RGB histogram (flattened, bins per channel)."""
    histograms = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
        h = h / (h.sum() + 1e-8)
        histograms.append(h)
    return np.concatenate(histograms)


def _histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Histogram intersection similarity (0=no overlap, 1=identical)."""
    return float(np.sum(np.minimum(h1, h2)))


def _histogram_bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    """Bhattacharyya distance between two histograms."""
    bc = np.sum(np.sqrt(h1 * h2 + 1e-10))
    return float(-np.log(bc + 1e-10))


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient. 0 = perfect equality, 1 = total inequality."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumvals = np.cumsum(values)
    return float((2 * np.sum((np.arange(1, n+1)) * values) - (n+1) * values.sum())
                 / (n * values.sum()))


def _vendi_score(similarity_matrix: np.ndarray) -> float:
    """
    Vendi Score — diversity metric from Princeton (Friedman & Dieng, 2022).
    VS = exp(H(eigenvalues of K/n))
    where K is the similarity matrix and H is Shannon entropy.
    Higher = more diverse.
    """
    n = similarity_matrix.shape[0]
    if n == 0:
        return 0.0
    K = similarity_matrix / n
    # eigenvalues (only real part needed for PSD matrices)
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]   # filter numerical noise
    eigenvalues = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-10))
    return float(np.exp(entropy))


def _score_to_grade(score: float) -> str:
    """Convert 0–100 score to letter grade."""
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def _aspect_ratio_label(w: float, h: float) -> str:
    """Return human-readable aspect ratio label."""
    ratio = w / (h + 1e-8)
    if abs(ratio - 1.0) < 0.05:   return "1:1 (Square)"
    if abs(ratio - 4/3) < 0.08:   return "4:3"
    if abs(ratio - 16/9) < 0.08:  return "16:9"
    if abs(ratio - 3/2) < 0.08:   return "3:2"
    if ratio > 1.5:                return f"Wide ({ratio:.2f}:1)"
    return f"Tall (1:{1/ratio:.2f})"


def _perceptual_hash_numpy(arr: np.ndarray, hash_size: int = 16) -> int:
    """
    dHash implementation (difference hash) using numpy only.
    No external library needed. Returns int hash.
    """
    # Resize to (hash_size+1) x hash_size
    img_pil = Image.fromarray(arr.astype(np.uint8)).convert("L")
    img_small = img_pil.resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img_small, dtype=np.float32)
    # Row differences
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Flatten and convert to int
    hash_bits = diff.flatten()
    hash_int = int(sum(bit << i for i, bit in enumerate(hash_bits)))
    return hash_int


def _hamming_distance_int(h1: int, h2: int) -> int:
    """Hamming distance between two integer hashes."""
    xor = h1 ^ h2
    return bin(xor).count('1')


def _compute_mmd(X: np.ndarray, Y: np.ndarray,
                  kernel: str = "rbf", sigma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy (MMD) between distributions X and Y.
    Uses RBF kernel. O(n²) — use on samples only.
    """
    def rbf(A, B):
        dist = cdist(A, B, "sqeuclidean")
        return np.exp(-dist / (2 * sigma ** 2))

    n, m = len(X), len(Y)
    if n == 0 or m == 0:
        return 0.0
    # Sub-sample for speed
    max_n = 200
    if n > max_n: X = X[np.random.choice(n, max_n, replace=False)]
    if m > max_n: Y = Y[np.random.choice(m, max_n, replace=False)]

    K_xx = rbf(X, X)
    K_yy = rbf(Y, Y)
    K_xy = rbf(X, Y)
    mmd = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
    return float(max(0.0, mmd))


def _extract_feature_vector(arr: np.ndarray) -> np.ndarray:
    """
    Extract compact feature vector from image for ML-based analysis.
    Uses pixel statistics + histogram — no deep learning needed.
    Dimensions: 3*5 (channel stats) + 3*32 (histograms) = 111
    """
    features = []
    channel_names = [0, 1, 2]
    for ch in channel_names:
        c = arr[:, :, ch].flatten()
        features.extend([
            np.mean(c), np.std(c), np.median(c),
            skew(c), kurtosis(c)
        ])
    # Histogram features
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256))
        h = h / (h.sum() + 1e-8)
        features.extend(h.tolist())

    # Spatial statistics (quadrant means)
    h_half = arr.shape[0] // 2
    w_half = arr.shape[1] // 2
    quadrants = [
        arr[:h_half, :w_half],
        arr[:h_half, w_half:],
        arr[h_half:, :w_half],
        arr[h_half:, w_half:],
    ]
    for q in quadrants:
        features.append(np.mean(q))

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1: COLOR DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class ColorDistributionAnalyzer:
    """
    Analyzes color distributions across classes to detect:
    - Color Bias: specific class always has same color scheme
    - Shortcut Learning Risk: model may learn COLOR instead of CONTENT
    - Histogram Divergence: how different class color distributions are
    - Dominant color clusters per class (K-means style)
    """

    def __init__(self, max_images_per_class: int = 200,
                 shortcut_threshold: float = 25.0):
        self.max_images_per_class = max_images_per_class
        self.shortcut_threshold = shortcut_threshold  # mean RGB diff to flag

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                image_dir: str = "") -> ColorBiasResult:
        """
        Main entry point.
        image_paths: list of relative or absolute paths
        labels: corresponding class labels
        image_dir: base directory (if paths are relative)
        """
        result = ColorBiasResult()
        result.recommendations = []

        # Group by class
        class_images: Dict[str, List[str]] = defaultdict(list)
        for path, label in zip(image_paths, labels):
            full = os.path.join(image_dir, path) if image_dir else path
            class_images[str(label)].append(full)

        if len(class_images) < 2:
            result.recommendations.append(
                "Need at least 2 classes for color bias detection."
            )
            return result

        # Per-class stats
        class_stats: Dict[str, Dict] = {}
        class_histograms: Dict[str, np.ndarray] = {}
        class_dominant: Dict[str, List] = {}

        for label, paths in class_images.items():
            paths = paths[:self.max_images_per_class]
            all_stats = []
            all_histograms = []
            dominant_accumulator = []

            for p in paths:
                arr = _load_image_safe(p, max_size=256)
                if arr is None:
                    continue
                stats_dict = _rgb_stats(arr)
                all_stats.append(stats_dict)
                all_histograms.append(_compute_histogram(arr, bins=32))
                # Sample 3x3 grid dominant colors
                h, w = arr.shape[:2]
                for row in [h//4, h//2, 3*h//4]:
                    for col in [w//4, w//2, 3*w//4]:
                        pixel = arr[row, col, :].astype(int)
                        dominant_accumulator.append(tuple(pixel))

            if not all_stats:
                continue

            # Aggregate
            keys = all_stats[0].keys()
            agg = {k: float(np.mean([s[k] for s in all_stats])) for k in keys}
            agg_std = {k: float(np.std([s[k] for s in all_stats])) for k in keys}
            class_stats[label] = {"mean": agg, "std": agg_std}

            # Mean histogram
            class_histograms[label] = np.mean(all_histograms, axis=0)

            # Dominant colors: quantize to 8 levels and find top-5
            quantized = [(r//32*32, g//32*32, b//32*32)
                         for r, g, b in dominant_accumulator]
            top5 = Counter(quantized).most_common(5)
            class_dominant[label] = [color for color, _ in top5]

        # ── Channel variance across classes ───────────────────────────────────
        channel_var: Dict[str, float] = {}
        classes = list(class_stats.keys())
        for ch in ["R", "G", "B"]:
            means = [class_stats[c]["mean"][f"{ch}_mean"] for c in classes]
            channel_var[ch] = float(np.std(means))

        result.class_mean_rgb = {
            c: (
                class_stats[c]["mean"]["R_mean"],
                class_stats[c]["mean"]["G_mean"],
                class_stats[c]["mean"]["B_mean"],
            )
            for c in classes
        }
        result.class_std_rgb = {
            c: (
                class_stats[c]["std"]["R_mean"],
                class_stats[c]["std"]["G_mean"],
                class_stats[c]["std"]["B_mean"],
            )
            for c in classes
        }
        result.channel_variance_ratio = channel_var
        result.dominant_colors = class_dominant

        # ── Histogram Divergence (Bhattacharyya per class pair) ───────────────
        divergences: Dict[str, float] = {}
        class_list = list(class_histograms.keys())
        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                c1, c2 = class_list[i], class_list[j]
                div = _histogram_bhattacharyya(
                    class_histograms[c1], class_histograms[c2]
                )
                divergences[f"{c1}_vs_{c2}"] = div
        result.histogram_divergence = divergences

        # ── Shortcut Risk Score ───────────────────────────────────────────────
        max_channel_var = max(channel_var.values()) if channel_var else 0
        mean_divergence = np.mean(list(divergences.values())) if divergences else 0

        # Weighted shortcut risk: high channel variance + high divergence = risk
        shortcut_risk = min(100.0, (max_channel_var / 255 * 60) +
                                    (mean_divergence * 40))
        result.shortcut_risk_score = round(float(shortcut_risk), 2)

        if shortcut_risk < 20:
            result.shortcut_risk_level = "low"
        elif shortcut_risk < 40:
            result.shortcut_risk_level = "medium"
        elif shortcut_risk < 65:
            result.shortcut_risk_level = "high"
        else:
            result.shortcut_risk_level = "critical"

        # ── Biased classes ────────────────────────────────────────────────────
        overall_mean_rgb = np.mean(
            [list(v) for v in result.class_mean_rgb.values()], axis=0
        )
        biased = []
        for cls, (r, g, b) in result.class_mean_rgb.items():
            diff = float(np.linalg.norm(
                np.array([r, g, b]) - overall_mean_rgb
            ))
            if diff > self.shortcut_threshold:
                biased.append(cls)

        result.biased_classes = biased
        result.color_bias_detected = len(biased) > 0

        if not biased:
            result.bias_severity = "none"
        elif len(biased) == 1:
            result.bias_severity = "low"
        elif len(biased) <= len(classes) // 2:
            result.bias_severity = "medium"
        else:
            result.bias_severity = "high"

        # ── Recommendations ───────────────────────────────────────────────────
        if result.shortcut_risk_level in ("high", "critical"):
            result.recommendations.append(
                f"🚨 High Shortcut Learning Risk ({shortcut_risk:.1f}/100): "
                f"Model may learn color instead of content. "
                f"Apply color jitter augmentation."
            )
        if result.color_bias_detected:
            result.recommendations.append(
                f"⚠️ Color bias in classes: {', '.join(biased)}. "
                f"These classes have significantly different color profiles. "
                f"Consider: (1) Color normalization, (2) Grayscale training, "
                f"(3) Random color augmentation."
            )
        if max_channel_var > 40:
            result.recommendations.append(
                f"📊 High inter-class color variance (max channel std={max_channel_var:.1f}). "
                f"Use ColorJitter with brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1."
            )
        if not result.recommendations:
            result.recommendations.append(
                "✅ Color distribution is consistent across classes. Low shortcut learning risk."
            )

        result.details = {
            "class_stats": {c: class_stats[c]["mean"] for c in class_stats},
            "max_channel_variance": max_channel_var,
            "mean_histogram_divergence": float(mean_divergence),
            "num_classes_analyzed": len(classes),
        }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2: CLASS BALANCE DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalanceAnalyzer:
    """
    Deep analysis of class balance:
    - Gini Coefficient of distribution (ecology-inspired)
    - Imbalance Ratio (majority / minority)
    - Suggested augmentation targets
    - Class Overlap: are similar-looking images in different classes?
    """

    SEVERE_IMBALANCE_RATIO = 5.0
    MODERATE_IMBALANCE_RATIO = 3.0

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                feature_vectors: Optional[Dict[str, np.ndarray]] = None,
                image_dir: str = "") -> ClassBalanceResult:
        result = ClassBalanceResult()

        label_array = np.array(labels)
        unique_classes, counts = np.unique(label_array, return_counts=True)
        count_dict = dict(zip(unique_classes.tolist(), counts.tolist()))

        result.class_counts = count_dict
        result.total_images = int(len(labels))
        result.num_classes = int(len(unique_classes))
        result.class_percentages = {
            cls: round(cnt / result.total_images * 100, 2)
            for cls, cnt in count_dict.items()
        }

        if result.num_classes < 2:
            result.recommendations.append("Need at least 2 classes for balance analysis.")
            return result

        # ── Imbalance Ratio ───────────────────────────────────────────────────
        max_count = max(counts)
        min_count = min(counts)
        result.imbalance_ratio = round(max_count / (min_count + 1e-8), 2)
        result.majority_class = unique_classes[np.argmax(counts)].tolist()
        result.minority_class = unique_classes[np.argmin(counts)].tolist()
        result.is_severely_imbalanced = result.imbalance_ratio > self.SEVERE_IMBALANCE_RATIO

        # ── Gini Coefficient ──────────────────────────────────────────────────
        result.gini_coefficient = round(_gini_coefficient(counts.astype(float)), 4)

        # ── Balance Score (0–100) ─────────────────────────────────────────────
        # Perfect balance = 100, total imbalance = 0
        # Based on normalized entropy
        proportions = counts / counts.sum()
        max_entropy = np.log(result.num_classes)
        actual_entropy = scipy_entropy(proportions)
        if max_entropy > 0:
            balance_score = (actual_entropy / max_entropy) * 100
        else:
            balance_score = 100.0
        result.balance_score = round(float(balance_score), 2)

        # ── Suggested Augmentation ────────────────────────────────────────────
        # Target = mean count of top-50% classes (robust target)
        sorted_counts = np.sort(counts)[::-1]
        target_count = int(np.mean(sorted_counts[:max(1, len(sorted_counts)//2)]))
        result.suggested_augmentation = {}
        for cls, cnt in count_dict.items():
            if cnt < target_count:
                need = target_count - cnt
                result.suggested_augmentation[cls] = need

        # ── Class Overlap Detection ───────────────────────────────────────────
        if feature_vectors and len(feature_vectors) >= 2:
            result.overlapping_class_pairs = self._detect_class_overlap(
                feature_vectors, count_dict
            )

        # ── Recommendations ───────────────────────────────────────────────────
        if result.imbalance_ratio > self.SEVERE_IMBALANCE_RATIO:
            result.recommendations.append(
                f"🚨 Severe class imbalance (ratio={result.imbalance_ratio:.1f}x). "
                f"'{result.majority_class}' has {max_count} images vs "
                f"'{result.minority_class}' with {min_count}. "
                f"Recommended: SMOTE/ADASYN or class weights."
            )
        elif result.imbalance_ratio > self.MODERATE_IMBALANCE_RATIO:
            result.recommendations.append(
                f"⚠️ Moderate class imbalance (ratio={result.imbalance_ratio:.1f}x). "
                f"Consider augmenting minority classes."
            )

        if result.gini_coefficient > 0.4:
            result.recommendations.append(
                f"📊 High Gini coefficient ({result.gini_coefficient:.3f}). "
                f"Dataset is unevenly distributed. Target balanced collection."
            )

        if result.suggested_augmentation:
            total_needed = sum(result.suggested_augmentation.values())
            result.recommendations.append(
                f"💡 To balance dataset: add ~{total_needed} augmented images "
                f"across {len(result.suggested_augmentation)} minority classes."
            )

        if result.overlapping_class_pairs:
            pairs_str = ", ".join(
                f"{a}↔{b}" for a, b, _ in result.overlapping_class_pairs[:3]
            )
            result.recommendations.append(
                f"🔍 Class overlap detected: {pairs_str}. "
                f"These class pairs may confuse the model."
            )

        if not result.recommendations:
            result.recommendations.append(
                f"✅ Classes are well-balanced (Gini={result.gini_coefficient:.3f}, "
                f"ratio={result.imbalance_ratio:.1f}x)."
            )

        result.details = {
            "entropy": float(actual_entropy),
            "max_entropy": float(max_entropy),
            "sorted_counts": sorted_counts.tolist(),
            "target_count_for_balance": target_count,
        }

        return result

    def _detect_class_overlap(self,
                               feature_vectors: Dict[str, np.ndarray],
                               count_dict: Dict[str, int],
                               overlap_threshold: float = 0.3
                               ) -> List[Tuple[str, str, float]]:
        """
        Detect overlapping classes using centroid distance in feature space.
        Returns list of (class1, class2, overlap_score) sorted by overlap.
        """
        overlapping = []
        classes = list(feature_vectors.keys())
        centroids = {cls: np.mean(vecs, axis=0)
                     for cls, vecs in feature_vectors.items()
                     if len(vecs) > 0}

        if len(centroids) < 2:
            return []

        # Compute within-class spread (std)
        within_spreads = {}
        for cls, vecs in feature_vectors.items():
            if len(vecs) > 1:
                within_spreads[cls] = float(np.mean(np.std(vecs, axis=0)))
            else:
                within_spreads[cls] = 0.0

        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                c1, c2 = classes[i], classes[j]
                if c1 not in centroids or c2 not in centroids:
                    continue
                dist = float(np.linalg.norm(centroids[c1] - centroids[c2]))
                spread = within_spreads.get(c1, 0) + within_spreads.get(c2, 0)
                # Overlap if distance < combined spread
                if spread > 0:
                    overlap_ratio = max(0.0, 1.0 - dist / (spread + 1e-8))
                    if overlap_ratio > overlap_threshold:
                        overlapping.append((c1, c2, round(overlap_ratio, 3)))

        return sorted(overlapping, key=lambda x: -x[2])


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3: SIZE & DIMENSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class SizeDimensionAnalyzer:
    """
    Analyzes image sizes and dimensions across the dataset:
    - Width/Height/Aspect Ratio distributions
    - Size Bias: specific class consistently larger/smaller
    - Optimal target resize recommendation
    - Resize quality impact per image
    """

    # Common ML target sizes sorted by area
    COMMON_TARGET_SIZES = [
        (32, 32), (64, 64), (96, 96), (128, 128),
        (160, 160), (192, 192), (224, 224), (256, 256),
        (299, 299), (384, 384), (448, 448), (512, 512),
    ]
    # Standard ML target (ImageNet default)
    DEFAULT_TARGET = (224, 224)

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                image_dir: str = "") -> SizeDimensionResult:
        result = SizeDimensionResult()
        result.recommendations = []

        widths, heights, aspect_ratios = [], [], []
        size_outliers = []
        class_sizes: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        path_to_size: Dict[str, Tuple[int, int]] = {}

        for path, label in zip(image_paths, labels):
            full = os.path.join(image_dir, path) if image_dir else path
            try:
                with Image.open(full) as img:
                    w, h = img.size
                widths.append(w)
                heights.append(h)
                ar = w / (h + 1e-8)
                aspect_ratios.append(ar)
                class_sizes[str(label)].append((w, h))
                path_to_size[path] = (w, h)
            except Exception:
                continue

        if not widths:
            result.recommendations.append("No valid images found for size analysis.")
            return result

        widths_arr = np.array(widths)
        heights_arr = np.array(heights)
        ar_arr = np.array(aspect_ratios)

        # ── Global Stats ──────────────────────────────────────────────────────
        def _stats_dict(arr: np.ndarray) -> Dict[str, float]:
            return {
                "mean":   round(float(np.mean(arr)), 2),
                "std":    round(float(np.std(arr)), 2),
                "min":    round(float(np.min(arr)), 2),
                "max":    round(float(np.max(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "p25":    round(float(np.percentile(arr, 25)), 2),
                "p75":    round(float(np.percentile(arr, 75)), 2),
            }

        result.width_stats  = _stats_dict(widths_arr)
        result.height_stats = _stats_dict(heights_arr)
        result.aspect_ratio_stats = _stats_dict(ar_arr)

        # ── Aspect Ratio Distribution ─────────────────────────────────────────
        ar_labels = [_aspect_ratio_label(w, h)
                     for w, h in zip(widths, heights)]
        ar_counts = Counter(ar_labels)
        result.aspect_ratio_distribution = dict(ar_counts)
        result.dominant_aspect_ratio = ar_counts.most_common(1)[0][0]

        # ── Per-Class Size Analysis + Size Bias Detection ─────────────────────
        class_mean_areas = {}
        for cls, sizes in class_sizes.items():
            ws = [s[0] for s in sizes]
            hs = [s[1] for s in sizes]
            areas = [w * h for w, h in sizes]
            result.class_size_stats[cls] = {
                "mean_width":  round(np.mean(ws), 1),
                "mean_height": round(np.mean(hs), 1),
                "mean_area":   round(np.mean(areas), 1),
                "std_width":   round(np.std(ws), 1),
                "std_height":  round(np.std(hs), 1),
                "count":       len(sizes),
            }
            class_mean_areas[cls] = float(np.mean(areas))

        # Size bias: class area differs by >50% from overall mean
        overall_mean_area = float(np.mean(widths_arr * heights_arr))
        biased_size_classes = []
        for cls, mean_area in class_mean_areas.items():
            ratio = mean_area / (overall_mean_area + 1e-8)
            if ratio > 1.5 or ratio < 0.67:
                biased_size_classes.append(cls)

        result.size_bias_detected = len(biased_size_classes) > 0
        result.biased_size_classes = biased_size_classes

        # ── Recommended Target Size ───────────────────────────────────────────
        result.recommended_target_size = self._recommend_target_size(
            widths_arr, heights_arr
        )

        # ── Resize Quality Impact ─────────────────────────────────────────────
        target_w, target_h = result.recommended_target_size
        impact_dict: Dict[str, str] = {}
        for path, (w, h) in path_to_size.items():
            scale_w = target_w / (w + 1e-8)
            scale_h = target_h / (h + 1e-8)
            scale = min(scale_w, scale_h)
            if scale >= 0.8:
                impact_dict[path] = "minimal"
            elif scale >= 0.4:
                impact_dict[path] = "moderate"
            else:
                impact_dict[path] = "significant"
        result.resize_quality_impact = impact_dict

        # Size outliers (IQR method on area)
        areas = widths_arr * heights_arr
        q1, q3 = np.percentile(areas, 25), np.percentile(areas, 75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        for path, (w, h) in path_to_size.items():
            area = w * h
            if area < lower or area > upper:
                size_outliers.append(path)
        result.size_outlier_images = size_outliers

        # ── Recommendations ───────────────────────────────────────────────────
        std_w = result.width_stats["std"]
        std_h = result.height_stats["std"]
        mean_w = result.width_stats["mean"]
        mean_h = result.height_stats["mean"]

        if std_w / (mean_w + 1e-8) > 0.3 or std_h / (mean_h + 1e-8) > 0.3:
            result.recommendations.append(
                f"⚠️ High size variability "
                f"(W: {mean_w:.0f}±{std_w:.0f}, H: {mean_h:.0f}±{std_h:.0f}). "
                f"Resize to {result.recommended_target_size[0]}×"
                f"{result.recommended_target_size[1]} before training."
            )

        if result.size_bias_detected:
            result.recommendations.append(
                f"🔍 Size bias detected in classes: "
                f"{', '.join(result.biased_size_classes)}. "
                f"These classes have systematically different image sizes — "
                f"model may use size as a shortcut."
            )

        sig_impact = sum(1 for v in impact_dict.values() if v == "significant")
        if sig_impact > 0:
            result.recommendations.append(
                f"📉 {sig_impact} images will lose significant detail when resized "
                f"to {result.recommended_target_size}. Consider higher target resolution."
            )

        if size_outliers:
            result.recommendations.append(
                f"🔎 {len(size_outliers)} images have extreme sizes (outliers). "
                f"Review before training."
            )

        if len(ar_counts) > 4:
            result.recommendations.append(
                f"📐 {len(ar_counts)} different aspect ratios detected. "
                f"Use center-crop or padding instead of stretch resizing."
            )

        if not result.recommendations:
            result.recommendations.append(
                f"✅ Image sizes are consistent. "
                f"Recommended resize: {result.recommended_target_size}."
            )

        result.details = {
            "total_images_analyzed": len(widths),
            "overall_mean_area": round(overall_mean_area, 1),
            "dominant_aspect_ratio": result.dominant_aspect_ratio,
            "size_outlier_count": len(size_outliers),
        }

        return result

    def _recommend_target_size(self,
                                widths: np.ndarray,
                                heights: np.ndarray) -> Tuple[int, int]:
        """
        Recommend optimal target size for resizing.
        Strategy: pick the standard size closest to the 25th percentile
        of image areas — preserves most information without upscaling small images.
        """
        # Use P25 to avoid upscaling too many images
        p25_area = float(np.percentile(widths * heights, 25))
        p25_side = int(math.sqrt(p25_area))

        # Find closest standard size
        best_size = self.DEFAULT_TARGET
        best_diff = float("inf")
        for size in self.COMMON_TARGET_SIZES:
            diff = abs(size[0] - p25_side)
            if diff < best_diff:
                best_diff = diff
                best_size = size

        return best_size


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4: DATASET DIVERSITY & REDUNDANCY
# ─────────────────────────────────────────────────────────────────────────────

class DiversityRedundancyAnalyzer:
    """
    Measures true diversity and detects redundancy:
    - Vendi Score: scientific diversity metric (Princeton, 2022)
    - pHash / dHash: near-duplicate detection without heavy dependencies
    - Augmentation copies detection
    - Cross-class similarity (dangerous: different labels, same appearance)
    - Effective Dataset Size after redundancy removal
    """

    NEAR_DUPLICATE_THRESHOLD = 10   # Hamming distance
    SIMILAR_THRESHOLD = 20          # Less strict: similar but not duplicate
    MAX_SAMPLE_FOR_VENDI = 500      # Vendi is O(n²)
    MAX_SAMPLE_FOR_HASH = 5000      # Hash comparison is fast

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                image_dir: str = "") -> DiversityRedundancyResult:
        result = DiversityRedundancyResult()
        result.recommendations = []

        full_paths = [
            os.path.join(image_dir, p) if image_dir else p
            for p in image_paths
        ]

        # ── Step 1: Compute perceptual hashes ─────────────────────────────────
        logger.info("Computing perceptual hashes...")
        hashes: List[Optional[int]] = []
        valid_indices: List[int] = []

        for i, p in enumerate(full_paths[:self.MAX_SAMPLE_FOR_HASH]):
            arr = _load_image_safe(p, max_size=128)
            if arr is not None:
                h = _perceptual_hash_numpy(arr, hash_size=16)
                hashes.append(h)
                valid_indices.append(i)
            else:
                hashes.append(None)

        valid_hashes = [(i, hashes[j], labels[i])
                        for j, i in enumerate(valid_indices)
                        if hashes[j] is not None]

        # ── Step 2: Near-Duplicate Detection ──────────────────────────────────
        near_dup_pairs: List[Tuple[str, str, float]] = []
        cross_class_pairs: List[Tuple[str, str, str, str, float]] = []
        cluster_membership: Dict[int, int] = {}  # index -> cluster_id
        cluster_id = 0

        for i in range(len(valid_hashes)):
            idx_i, hash_i, label_i = valid_hashes[i]
            for j in range(i + 1, min(len(valid_hashes), i + 500)):
                idx_j, hash_j, label_j = valid_hashes[j]
                dist = _hamming_distance_int(hash_i, hash_j)

                if dist <= self.NEAR_DUPLICATE_THRESHOLD:
                    similarity = 1.0 - dist / 256.0
                    near_dup_pairs.append((
                        image_paths[idx_i],
                        image_paths[idx_j],
                        round(similarity, 3)
                    ))
                    # Cluster tracking
                    ci = cluster_membership.get(idx_i, cluster_id)
                    if idx_i not in cluster_membership:
                        cluster_membership[idx_i] = cluster_id
                        cluster_id += 1
                    cluster_membership[idx_j] = cluster_membership[idx_i]

                    # Cross-class similarity (different labels, same look)
                    if str(label_i) != str(label_j):
                        cross_class_pairs.append((
                            image_paths[idx_i], image_paths[idx_j],
                            str(label_i), str(label_j),
                            round(similarity, 3)
                        ))

        result.near_duplicate_pairs = near_dup_pairs[:200]  # cap for display
        result.cross_class_similar_pairs = cross_class_pairs[:100]

        # ── Near-Duplicate Clusters ───────────────────────────────────────────
        clusters: Dict[int, List[str]] = defaultdict(list)
        for idx, cid in cluster_membership.items():
            clusters[cid].append(image_paths[idx])
        result.near_duplicate_clusters = [
            v for v in clusters.values() if len(v) > 1
        ]

        # ── Augmentation copies ───────────────────────────────────────────────
        # Augmented copies tend to appear in same class, very high similarity
        aug_count = sum(
            1 for p1, p2, sim in near_dup_pairs
            if sim > 0.95
        )
        result.augmentation_copies_detected = aug_count

        # ── Redundancy Rate ───────────────────────────────────────────────────
        n_total = len(valid_hashes)
        n_duplicated = len(set(
            idx for pair in result.near_duplicate_clusters
            for idx in range(len(pair))
        ))
        result.redundancy_rate = round(
            len(near_dup_pairs) / max(1, n_total * (n_total - 1) / 2), 4
        )
        result.effective_dataset_size = max(0, len(image_paths) - len(near_dup_pairs))

        # ── Vendi Score (diversity) ───────────────────────────────────────────
        logger.info("Computing Vendi Score...")
        feature_vecs = []
        sample_paths = full_paths[:self.MAX_SAMPLE_FOR_VENDI]
        for p in sample_paths:
            arr = _load_image_safe(p, max_size=128)
            if arr is not None:
                fv = _extract_feature_vector(arr)
                feature_vecs.append(fv)

        if len(feature_vecs) >= 10:
            fv_matrix = np.array(feature_vecs)
            # Normalize
            fv_matrix = (fv_matrix - fv_matrix.mean(axis=0)) / (fv_matrix.std(axis=0) + 1e-8)
            # Compute RBF similarity matrix
            sigma = float(np.median(cdist(fv_matrix[:100], fv_matrix[:100])))
            if sigma < 1e-6:
                sigma = 1.0
            # Sample for Vendi if too large
            if len(fv_matrix) > 300:
                idx = np.random.choice(len(fv_matrix), 300, replace=False)
                fv_sample = fv_matrix[idx]
            else:
                fv_sample = fv_matrix

            dist_matrix = cdist(fv_sample, fv_sample, "sqeuclidean")
            sim_matrix = np.exp(-dist_matrix / (2 * sigma ** 2))
            result.vendi_score = round(_vendi_score(sim_matrix), 3)
            # Store sample (max 50x50)
            result.similarity_matrix_sample = sim_matrix[:50, :50]
        else:
            result.vendi_score = 0.0

        # ── Diversity Grade ───────────────────────────────────────────────────
        # Vendi Score: 1.0 = no diversity, grows with diversity
        # Normalize: assume max useful vendi ~ num_classes
        n_classes = len(set(labels))
        normalized_vendi = min(100.0, (result.vendi_score / max(1, n_classes)) * 100)

        # Combine with redundancy penalty
        redundancy_penalty = min(50.0, result.redundancy_rate * 5000)
        coverage = max(0.0, normalized_vendi - redundancy_penalty * 0.3)
        result.coverage_score = round(float(coverage), 2)
        result.diversity_grade = _score_to_grade(result.coverage_score)
        result.hash_method_used = "dHash (numpy)" if not HAS_IMAGEHASH else "pHash (imagehash)"

        # ── Recommendations ───────────────────────────────────────────────────
        if result.redundancy_rate > 0.05:
            result.recommendations.append(
                f"🔁 High redundancy detected ({result.redundancy_rate*100:.1f}% of pairs). "
                f"{len(near_dup_pairs)} near-duplicate pairs found. "
                f"Effective dataset size: {result.effective_dataset_size} "
                f"(was {len(image_paths)})."
            )

        if result.near_duplicate_clusters:
            result.recommendations.append(
                f"🗂️ {len(result.near_duplicate_clusters)} near-duplicate clusters found. "
                f"Remove redundant images to improve training efficiency."
            )

        if result.cross_class_similar_pairs:
            result.recommendations.append(
                f"⚠️ {len(result.cross_class_similar_pairs)} cross-class near-similar pairs "
                f"(same appearance, different labels). "
                f"This will confuse the classifier — review labels."
            )

        if result.augmentation_copies_detected > 10:
            result.recommendations.append(
                f"🔄 {result.augmentation_copies_detected} possible augmentation copies "
                f"detected (>95% similar). "
                f"Ensure augmented copies are not leaking into test set."
            )

        if result.vendi_score < 2.0 and len(feature_vecs) > 0:
            result.recommendations.append(
                f"📉 Low Vendi diversity score ({result.vendi_score:.2f}). "
                f"Dataset lacks visual variety. Add more diverse samples."
            )

        if result.coverage_score >= 80:
            result.recommendations.append(
                f"✅ Good dataset diversity (coverage={result.coverage_score:.1f}/100, "
                f"Vendi={result.vendi_score:.2f})."
            )

        result.details = {
            "total_hashed": len(valid_hashes),
            "near_dup_pairs": len(near_dup_pairs),
            "cross_class_pairs": len(cross_class_pairs),
            "vendi_score_raw": result.vendi_score,
            "redundancy_rate": result.redundancy_rate,
        }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5: DISTRIBUTION SHIFT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class DistributionShiftDetector:
    """
    Detects distribution shift between dataset splits (train vs test):
    - Covariate Shift: input distributions differ
    - Pixel mean/std shift per channel
    - KS-Test (Kolmogorov-Smirnov) per channel — fast and robust
    - Wasserstein Distance (Earth Mover's Distance) per channel
    - Optional MMD (Maximum Mean Discrepancy) on feature vectors
    - Hidden Stratification risk estimation
    - Context Bias detection
    """

    MMD_SAMPLE_SIZE = 150           # MMD is O(n²), keep small

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                splits: Optional[List[str]] = None,
                image_dir: str = "",
                compute_mmd: bool = False) -> DistributionShiftResult:
        result = DistributionShiftResult()
        result.recommendations = []

        if splits is None:
            result.recommendations.append(
                "No split column provided. Skipping distribution shift analysis."
            )
            return result

        unique_splits = list(set(splits))
        if len(unique_splits) < 2:
            result.recommendations.append(
                f"Only one split found ({unique_splits[0]}). Need train+test."
            )
            return result

        # Use first two splits (usually train/test or train/val)
        split_a = "train" if "train" in unique_splits else unique_splits[0]
        split_b = "test" if "test" in unique_splits else unique_splits[1]
        result.splits_compared = (split_a, split_b)

        # Group paths by split
        split_paths: Dict[str, List[str]] = defaultdict(list)
        for path, split in zip(image_paths, splits):
            full = os.path.join(image_dir, path) if image_dir else path
            split_paths[split].append(full)

        paths_a = split_paths.get(split_a, [])
        paths_b = split_paths.get(split_b, [])

        if not paths_a or not paths_b:
            result.recommendations.append("Empty split(s) found.")
            return result

        # ── Extract pixel statistics for each split ───────────────────────────
        logger.info(f"Extracting pixel stats for {split_a} / {split_b}...")
        stats_a = self._extract_split_stats(paths_a)
        stats_b = self._extract_split_stats(paths_b)

        if not stats_a or not stats_b:
            result.recommendations.append("Could not load images from splits.")
            return result

        # ── Per-channel pixel mean/std shift ─────────────────────────────────
        channel_names = ["R", "G", "B", "brightness"]
        for ch in channel_names:
            mean_a = np.mean([s[f"{ch}_mean"] for s in stats_a if f"{ch}_mean" in s])
            mean_b = np.mean([s[f"{ch}_mean"] for s in stats_b if f"{ch}_mean" in s])
            std_a  = np.mean([s[f"{ch}_std"]  for s in stats_a if f"{ch}_std"  in s])
            std_b  = np.mean([s[f"{ch}_std"]  for s in stats_b if f"{ch}_std"  in s])
            result.pixel_mean_shift[ch] = round(float(abs(mean_a - mean_b)), 3)
            result.pixel_std_shift[ch]  = round(float(abs(std_a  - std_b)),  3)

        # ── KS-Test per channel ───────────────────────────────────────────────
        channels = ["R_mean", "G_mean", "B_mean", "brightness"]
        total_ks_pvalue = 0.0
        significant_channels = 0

        for ch in channels:
            vals_a = [s[ch] for s in stats_a if ch in s]
            vals_b = [s[ch] for s in stats_b if ch in s]
            if len(vals_a) < 5 or len(vals_b) < 5:
                continue
            stat, pvalue = ks_2samp(vals_a, vals_b)
            result.ks_test_results[ch] = {
                "statistic": round(float(stat), 4),
                "pvalue":    round(float(pvalue), 4),
                "shift_detected": pvalue < 0.05,
            }
            total_ks_pvalue += pvalue
            if pvalue < 0.05:
                significant_channels += 1

        # ── Wasserstein Distance ──────────────────────────────────────────────
        if HAS_WASSERSTEIN:
            for ch in ["R_mean", "G_mean", "B_mean"]:
                vals_a = [s[ch] for s in stats_a if ch in s]
                vals_b = [s[ch] for s in stats_b if ch in s]
                if vals_a and vals_b:
                    wd = wasserstein_distance(vals_a, vals_b)
                    result.wasserstein_distances[ch] = round(float(wd), 3)

        # ── MMD (optional, expensive) ─────────────────────────────────────────
        if compute_mmd:
            logger.info("Computing MMD (this may take a moment)...")
            fv_a = self._extract_feature_vectors(
                paths_a[:self.MMD_SAMPLE_SIZE]
            )
            fv_b = self._extract_feature_vectors(
                paths_b[:self.MMD_SAMPLE_SIZE]
            )
            if len(fv_a) > 5 and len(fv_b) > 5:
                sigma = float(np.median(
                    cdist(fv_a[:50], fv_a[:50])
                )) + 1e-6
                result.mmd_score = round(_compute_mmd(fv_a, fv_b, sigma=sigma), 4)

        # ── Covariate Shift Score (0–100) ─────────────────────────────────────
        # Based on significant channels + mean pixel shift magnitude
        max_pixel_shift = max(result.pixel_mean_shift.values()) if result.pixel_mean_shift else 0
        channel_shift_score = (significant_channels / max(1, len(channels))) * 60
        pixel_shift_score = min(40.0, (max_pixel_shift / 255) * 40 * 3)
        result.covariate_shift_score = round(
            float(channel_shift_score + pixel_shift_score), 2
        )
        result.shift_detected = result.covariate_shift_score > 20

        # ── Shift Severity ────────────────────────────────────────────────────
        score = result.covariate_shift_score
        if score < 10:
            result.shift_severity = "none"
        elif score < 25:
            result.shift_severity = "low"
        elif score < 50:
            result.shift_severity = "medium"
        elif score < 75:
            result.shift_severity = "high"
        else:
            result.shift_severity = "critical"

        # ── Domain Gap Score ──────────────────────────────────────────────────
        result.domain_gap_score = round(
            float(np.mean(list(result.pixel_mean_shift.values()))), 2
        )

        # ── Hidden Stratification Risk ────────────────────────────────────────
        # Proxy: high within-class variance in split A vs B
        unique_labels = list(set(labels))
        label_splits = list(zip(labels, splits))
        within_class_shift = []
        for lbl in unique_labels[:10]:  # sample classes
            mask_a = [i for i, (l, s) in enumerate(label_splits)
                      if l == lbl and s == split_a]
            mask_b = [i for i, (l, s) in enumerate(label_splits)
                      if l == lbl and s == split_b]
            if len(mask_a) > 3 and len(mask_b) > 3:
                # Compare count ratios as proxy for stratification
                ratio = len(mask_a) / (len(mask_b) + 1e-8)
                within_class_shift.append(abs(ratio - 1.0))

        result.hidden_stratification_risk = round(
            float(np.mean(within_class_shift)) * 100
            if within_class_shift else 0.0, 2
        )

        # ── Context Bias ──────────────────────────────────────────────────────
        # Proxy: high saturation difference → different backgrounds
        sat_shift = abs(
            np.mean([s.get("saturation", 0) for s in stats_a]) -
            np.mean([s.get("saturation", 0) for s in stats_b])
        )
        if sat_shift < 5:
            result.context_bias_risk = "low"
        elif sat_shift < 15:
            result.context_bias_risk = "medium"
        else:
            result.context_bias_risk = "high"

        # ── Recommendations ───────────────────────────────────────────────────
        if result.shift_severity in ("high", "critical"):
            result.recommendations.append(
                f"🚨 Critical distribution shift detected "
                f"({split_a} vs {split_b}, score={score:.1f}/100). "
                f"Your model will likely underperform on {split_b}. "
                f"Rebalance your data split or apply domain adaptation."
            )
        elif result.shift_severity == "medium":
            result.recommendations.append(
                f"⚠️ Moderate distribution shift between {split_a} and {split_b} "
                f"(score={score:.1f}/100). "
                f"Consider stratified sampling."
            )

        if significant_channels > 2:
            shifted = [ch for ch, v in result.ks_test_results.items() if v["shift_detected"]]
            result.recommendations.append(
                f"📊 KS-Test failed on channels: {', '.join(shifted)}. "
                f"These channels have different distributions across splits."
            )

        if result.hidden_stratification_risk > 30:
            result.recommendations.append(
                f"🔍 High hidden stratification risk ({result.hidden_stratification_risk:.1f}). "
                f"Some classes may be under-represented in test set. "
                f"Use stratified train/test split."
            )

        if result.context_bias_risk != "low":
            result.recommendations.append(
                f"🖼️ Context/background bias risk: {result.context_bias_risk}. "
                f"Background color patterns differ between splits."
            )

        if not result.shift_detected:
            result.recommendations.append(
                f"✅ No significant distribution shift detected between "
                f"{split_a} and {split_b}."
            )

        result.details = {
            "paths_a_count": len(paths_a),
            "paths_b_count": len(paths_b),
            "significant_channels": significant_channels,
            "max_pixel_shift": max_pixel_shift,
        }

        return result

    def _extract_split_stats(self, paths: List[str],
                              max_images: int = 300) -> List[Dict]:
        """Extract RGB stats list from a split's image paths."""
        stats_list = []
        sample = paths[:max_images]
        for p in sample:
            arr = _load_image_safe(p, max_size=128)
            if arr is not None:
                stats_list.append(_rgb_stats(arr))
        return stats_list

    def _extract_feature_vectors(self, paths: List[str]) -> np.ndarray:
        """Extract feature vectors from a list of paths."""
        vecs = []
        for p in paths:
            arr = _load_image_safe(p, max_size=128)
            if arr is not None:
                vecs.append(_extract_feature_vector(arr))
        return np.array(vecs) if vecs else np.array([]).reshape(0, 115)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6: OUTLIER IMAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class OutlierImageDetector:
    """
    Detects outlier/anomalous images in the dataset:
    - Isolation Forest: global outliers in feature space
    - Local Outlier Factor: local density-based anomalies
    - KNN Distance: images far from any cluster
    - Ensemble: combined vote for robust detection
    - OOD (Out-of-Distribution): truly alien images
    - Per-image suggestion: delete / keep / review
    """

    MAX_IMAGES = 2000   # Feature extraction limit
    CONTAMINATION = 0.05  # Expected outlier rate

    def analyze(self,
                image_paths: List[str],
                labels: List[str],
                quality_scores: Optional[Dict[str, float]] = None,
                image_dir: str = "") -> OutlierImageResult:
        result = OutlierImageResult()
        result.recommendations = []

        # ── Extract features ──────────────────────────────────────────────────
        logger.info("Extracting features for outlier detection...")
        features: List[np.ndarray] = []
        valid_paths: List[str] = []
        valid_labels: List[str] = []

        sample_pairs = list(zip(image_paths, labels))[:self.MAX_IMAGES]
        for path, label in sample_pairs:
            full = os.path.join(image_dir, path) if image_dir else path
            arr = _load_image_safe(full, max_size=128)
            if arr is not None:
                fv = _extract_feature_vector(arr)
                features.append(fv)
                valid_paths.append(path)
                valid_labels.append(str(label))

        if len(features) < 20:
            result.recommendations.append(
                "Not enough valid images for outlier detection (need ≥20)."
            )
            return result

        # ── Preprocess: Normalize + PCA ───────────────────────────────────────
        X = np.array(features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA reduction for efficiency
        n_components = min(50, X_scaled.shape[1], X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        # ── Method 1: Isolation Forest ────────────────────────────────────────
        logger.info("Running Isolation Forest...")
        iso = IsolationForest(
            n_estimators=200,
            contamination=self.CONTAMINATION,
            random_state=42,
            n_jobs=-1
        )
        iso_labels = iso.fit_predict(X_pca)
        iso_scores = iso.decision_function(X_pca)
        iso_outlier_paths = [
            valid_paths[i] for i, lbl in enumerate(iso_labels) if lbl == -1
        ]
        result.isolation_forest_outliers = iso_outlier_paths

        # ── Method 2: Local Outlier Factor ────────────────────────────────────
        logger.info("Running LOF...")
        n_neighbors = min(20, len(X_pca) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.CONTAMINATION,
            n_jobs=-1
        )
        lof_labels = lof.fit_predict(X_pca)
        lof_scores = lof.negative_outlier_factor_
        lof_outlier_paths = [
            valid_paths[i] for i, lbl in enumerate(lof_labels) if lbl == -1
        ]
        result.lof_outliers = lof_outlier_paths

        # ── Method 3: KNN Distance ────────────────────────────────────────────
        logger.info("Running KNN distance...")
        n_nn = min(5, len(X_pca) - 1)
        knn = NearestNeighbors(n_neighbors=n_nn, n_jobs=-1)
        knn.fit(X_pca)
        distances, _ = knn.kneighbors(X_pca)
        mean_dist = distances[:, 1:].mean(axis=1)  # exclude self
        knn_threshold = np.percentile(mean_dist, 100 * (1 - self.CONTAMINATION))
        knn_outlier_paths = [
            valid_paths[i] for i, d in enumerate(mean_dist) if d > knn_threshold
        ]
        result.knn_outliers = knn_outlier_paths

        # ── Ensemble: vote across all 3 methods ───────────────────────────────
        iso_set  = set(iso_outlier_paths)
        lof_set  = set(lof_outlier_paths)
        knn_set  = set(knn_outlier_paths)

        # Vote: flag as outlier if ≥2 methods agree
        all_paths_set = set(valid_paths)
        ensemble_outliers = []
        ensemble_scores: Dict[str, float] = {}

        for i, path in enumerate(valid_paths):
            votes = (
                (1 if path in iso_set else 0) +
                (1 if path in lof_set else 0) +
                (1 if path in knn_set else 0)
            )
            # Normalized anomaly score (0–100, higher = more anomalous)
            iso_s = float(np.clip(-iso_scores[i] * 50 + 50, 0, 100))
            lof_s = float(np.clip(-lof_scores[i] * 20, 0, 100))
            knn_s = float(np.clip(mean_dist[i] / (knn_threshold + 1e-8) * 50, 0, 100))
            combined_score = round((iso_s + lof_s + knn_s) / 3, 2)
            ensemble_scores[path] = combined_score

            if votes >= 2:
                ensemble_outliers.append(path)

        result.outlier_images = ensemble_outliers
        result.outlier_scores = ensemble_scores
        result.outlier_rate = round(len(ensemble_outliers) / max(1, len(valid_paths)), 4)
        result.outlier_method = "Ensemble (IsolationForest + LOF + KNN)"

        # ── OOD Detection: Elliptic Envelope ─────────────────────────────────
        try:
            ee = EllipticEnvelope(
                contamination=self.CONTAMINATION,
                random_state=42
            )
            # Use only first 10 PCA components for robustness
            X_ood = X_pca[:, :min(10, X_pca.shape[1])]
            ee.fit(X_ood)
            ood_pred = ee.predict(X_ood)
            result.ood_images = [
                valid_paths[i] for i, p in enumerate(ood_pred) if p == -1
            ]
        except Exception as e:
            logger.warning(f"OOD detection failed: {e}")
            result.ood_images = []

        # ── Per-image Suggestions ─────────────────────────────────────────────
        suggestions: Dict[str, str] = {}
        for path in ensemble_outliers:
            score = ensemble_scores.get(path, 50)
            # Check quality score if provided
            quality = (quality_scores or {}).get(path, None)
            if score > 80 or (quality is not None and quality < 30):
                suggestions[path] = "delete"
            elif score > 60:
                suggestions[path] = "review"
            else:
                suggestions[path] = "keep"
        result.suggestions = suggestions

        # ── Recommendations ───────────────────────────────────────────────────
        pct = result.outlier_rate * 100
        if pct > 10:
            result.recommendations.append(
                f"🚨 High outlier rate ({pct:.1f}%): {len(ensemble_outliers)} images "
                f"detected as anomalous by ensemble. Dataset needs cleaning."
            )
        elif pct > 5:
            result.recommendations.append(
                f"⚠️ Moderate outlier rate ({pct:.1f}%): {len(ensemble_outliers)} images. "
                f"Review suggested images before training."
            )
        else:
            result.recommendations.append(
                f"✅ Low outlier rate ({pct:.1f}%). Dataset is clean."
            )

        if result.ood_images:
            result.recommendations.append(
                f"🔍 {len(result.ood_images)} Out-of-Distribution (OOD) images detected. "
                f"These don't fit any class distribution."
            )

        delete_count = sum(1 for v in suggestions.values() if v == "delete")
        if delete_count:
            result.recommendations.append(
                f"🗑️ {delete_count} images recommended for deletion "
                f"(anomaly score >80 or quality <30)."
            )

        # Methods agreement insight
        all_agree = iso_set & lof_set & knn_set
        if all_agree:
            result.recommendations.append(
                f"🎯 {len(all_agree)} images flagged by ALL 3 methods — "
                f"very high confidence these are true outliers."
            )

        result.details = {
            "total_analyzed": len(valid_paths),
            "iso_outliers": len(iso_outlier_paths),
            "lof_outliers": len(lof_outlier_paths),
            "knn_outliers": len(knn_outlier_paths),
            "ood_count": len(result.ood_images),
            "all_3_agree": len(all_agree),
            "pca_variance_explained": round(
                float(pca.explained_variance_ratio_.sum()), 3
            ),
        }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7: ML READINESS FINAL SCORE
# ─────────────────────────────────────────────────────────────────────────────

class MLReadinessScorer:
    """
    Aggregates all analysis blocks into a final ML Readiness score:
    - Weighted composite of 5 dimensions
    - Letter grade A–F
    - Verdict: Ready / Needs Work / Not Ready
    - Priority-ranked recommendations
    - Estimated accuracy impact per issue
    """

    # Impact estimates (rough, based on literature)
    IMPACT_TABLE = {
        "severe_imbalance":       "~10–25% accuracy drop on minority classes",
        "high_redundancy":        "~5–15% wasted training capacity",
        "color_bias":             "~10–30% accuracy drop on unseen domains",
        "distribution_shift":     "~15–40% accuracy drop on test set",
        "high_outlier_rate":      "~5–10% degraded model generalization",
        "low_diversity":          "~10–20% poor generalization to new data",
        "size_inconsistency":     "~2–5% minor accuracy degradation",
    }

    def compute(self,
                color_result:    Optional[ColorBiasResult],
                balance_result:  Optional[ClassBalanceResult],
                size_result:     Optional[SizeDimensionResult],
                diversity_result: Optional[DiversityRedundancyResult],
                shift_result:    Optional[DistributionShiftResult],
                outlier_result:  Optional[OutlierImageResult],
                quality_score_from_quality_module: float = 75.0
                ) -> MLReadinessResult:

        result = MLReadinessResult()
        issues = []
        recs = []

        # ── Diversity Score ───────────────────────────────────────────────────
        if diversity_result:
            raw = diversity_result.coverage_score
            # Penalty for high redundancy
            redundancy_penalty = min(30, diversity_result.redundancy_rate * 3000)
            result.diversity_score = round(max(0.0, raw - redundancy_penalty), 2)
            if result.diversity_score < 50:
                issues.append(("low_diversity", result.diversity_score))
                recs.append({
                    "issue": "Low Diversity",
                    "score": result.diversity_score,
                    "impact": self.IMPACT_TABLE["low_diversity"],
                    "action": "Add more visually diverse training samples.",
                    "priority": 2,
                })
        else:
            result.diversity_score = 50.0

        # ── Balance Score ─────────────────────────────────────────────────────
        if balance_result:
            result.balance_score = balance_result.balance_score
            if balance_result.is_severely_imbalanced:
                issues.append(("severe_imbalance", result.balance_score))
                recs.append({
                    "issue": "Severe Class Imbalance",
                    "score": result.balance_score,
                    "impact": self.IMPACT_TABLE["severe_imbalance"],
                    "action": (
                        f"Augment '{balance_result.minority_class}' or "
                        f"use class weights / SMOTE."
                    ),
                    "priority": 1,
                })
        else:
            result.balance_score = 50.0

        # ── Shift Score (inverted: 0 shift = 100 score) ───────────────────────
        if shift_result:
            result.shift_score = round(
                max(0.0, 100.0 - shift_result.covariate_shift_score), 2
            )
            if shift_result.shift_detected:
                issues.append(("distribution_shift", result.shift_score))
                recs.append({
                    "issue": "Distribution Shift",
                    "score": result.shift_score,
                    "impact": self.IMPACT_TABLE["distribution_shift"],
                    "action": (
                        "Rebalance train/test split with stratified sampling. "
                        "Ensure same data source for train and test."
                    ),
                    "priority": 1,
                })
        else:
            result.shift_score = 75.0

        # ── Quality Score (from image_quality.py) ────────────────────────────
        result.quality_score = round(float(quality_score_from_quality_module), 2)
        if result.quality_score < 60:
            recs.append({
                "issue": "Low Image Quality",
                "score": result.quality_score,
                "impact": "~5–15% degraded feature learning",
                "action": "Filter blurry, corrupted, or very dark images.",
                "priority": 2,
            })

        # ── Coverage Score ────────────────────────────────────────────────────
        if diversity_result:
            result.coverage_score = diversity_result.coverage_score
        else:
            result.coverage_score = 50.0

        # ── Size Score ────────────────────────────────────────────────────────
        if size_result:
            # Penalize if too many sizes / size bias
            size_variety = len(set(size_result.aspect_ratio_distribution.keys()))
            size_bias_penalty = 20 if size_result.size_bias_detected else 0
            variety_penalty = min(30, (size_variety - 1) * 5)
            result.size_score = round(
                max(0.0, 100 - size_bias_penalty - variety_penalty), 2
            )
            if size_result.size_bias_detected:
                issues.append(("size_inconsistency", result.size_score))
                recs.append({
                    "issue": "Size Bias",
                    "score": result.size_score,
                    "impact": self.IMPACT_TABLE["size_inconsistency"],
                    "action": (
                        f"Resize all images to "
                        f"{size_result.recommended_target_size} consistently."
                    ),
                    "priority": 3,
                })
        else:
            result.size_score = 75.0

        # ── Outlier Score ─────────────────────────────────────────────────────
        if outlier_result:
            outlier_penalty = min(50, outlier_result.outlier_rate * 1000)
            result.outlier_score = round(max(0.0, 100 - outlier_penalty), 2)
            if outlier_result.outlier_rate > 0.05:
                issues.append(("high_outlier_rate", result.outlier_score))
                recs.append({
                    "issue": "High Outlier Rate",
                    "score": result.outlier_score,
                    "impact": self.IMPACT_TABLE["high_outlier_rate"],
                    "action": (
                        f"Remove {len(outlier_result.outlier_images)} flagged images. "
                        f"Start with the 'delete' suggestions."
                    ),
                    "priority": 2,
                })
        else:
            result.outlier_score = 75.0

        # ── Color Bias ────────────────────────────────────────────────────────
        if color_result and color_result.color_bias_detected:
            recs.append({
                "issue": "Color Bias / Shortcut Learning Risk",
                "score": round(100 - color_result.shortcut_risk_score, 2),
                "impact": self.IMPACT_TABLE["color_bias"],
                "action": (
                    "Apply ColorJitter augmentation. "
                    "Consider training on grayscale images to test if "
                    "model relies on color."
                ),
                "priority": 2,
            })

        # ── Weighted Final Score ──────────────────────────────────────────────
        weights = result.weights
        final = (
            result.diversity_score  * weights["diversity"] +
            result.balance_score    * weights["balance"]   +
            result.shift_score      * weights["shift"]     +
            result.quality_score    * weights["quality"]   +
            result.coverage_score   * weights["coverage"]
        )
        result.final_score = round(float(final), 2)
        result.grade = _score_to_grade(result.final_score)

        # ── Verdict ───────────────────────────────────────────────────────────
        critical_issues = [k for k, _ in issues
                           if k in ("distribution_shift", "severe_imbalance")]
        if result.final_score >= 75 and not critical_issues:
            result.verdict = "Ready ✅"
        elif result.final_score >= 50 or not critical_issues:
            result.verdict = "Needs Work ⚠️"
        else:
            result.verdict = "Not Ready ❌"

        # ── Priority Issues ───────────────────────────────────────────────────
        result.priority_issues = [k for k, _ in sorted(issues, key=lambda x: x[1])]

        # ── Ranked Recommendations ────────────────────────────────────────────
        result.recommendations_ranked = sorted(recs, key=lambda r: r["priority"])

        # ── Estimated Accuracy Impact ─────────────────────────────────────────
        result.estimated_accuracy_impact = {
            r["issue"]: r["impact"]
            for r in result.recommendations_ranked
        }

        result.details = {
            "sub_scores": {
                "diversity":  result.diversity_score,
                "balance":    result.balance_score,
                "shift":      result.shift_score,
                "quality":    result.quality_score,
                "coverage":   result.coverage_score,
                "size":       result.size_score,
                "outlier":    result.outlier_score,
            },
            "final_score": result.final_score,
            "grade": result.grade,
            "verdict": result.verdict,
            "issue_count": len(issues),
        }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS: ImageAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ImageAnalyzer:
    """
    Main entry point for dataset-level image analysis.

    Orchestrates all 7 blocks and returns a unified AnalysisReport.

    Example:
        analyzer = ImageAnalyzer(
            loader_df=loader_df,
            quality_df=quality_df,
            image_dir="data/images/",
            label_column="label",
            split_column="split",
        )
        report = analyzer.analyze_all()
        print(report.ml_readiness.verdict)
    """

    def __init__(self,
                 loader_df: pd.DataFrame,
                 quality_df: Optional[pd.DataFrame] = None,
                 image_dir: str = "",
                 label_column: str = "label",
                 path_column: str = "path",
                 split_column: Optional[str] = None,
                 dataset_name: str = "Dataset",
                 max_images_per_class: int = 200,
                 compute_mmd: bool = False):
        """
        Args:
            loader_df:    DataFrame from image_loader.py (must have path_column)
            quality_df:   DataFrame from image_quality.py (optional)
            image_dir:    Base directory for images
            label_column: Column name for class labels
            path_column:  Column name for image paths
            split_column: Column name for train/test/val splits (optional)
            dataset_name: Human-readable name
            max_images_per_class: Limit per class for heavy analyses
            compute_mmd:  Enable expensive MMD computation in shift detection
        """
        self.df = loader_df.copy()
        self.quality_df = quality_df
        self.image_dir = image_dir
        self.label_col = label_column
        self.path_col = path_column
        self.split_col = split_column
        self.dataset_name = dataset_name
        self.max_images_per_class = max_images_per_class
        self.compute_mmd = compute_mmd

        # Validate
        self._validate_inputs()

        # Extract core arrays
        self.image_paths = self.df[self.path_col].tolist()
        self.labels = self.df[self.label_col].astype(str).tolist() \
            if self.label_col in self.df.columns else ["unknown"] * len(self.df)
        self.splits = self.df[self.split_col].astype(str).tolist() \
            if self.split_col and self.split_col in self.df.columns else None

        # Quality scores dict
        self.quality_scores: Optional[Dict[str, float]] = None
        if quality_df is not None and self.path_col in quality_df.columns:
            score_col = next(
                (c for c in quality_df.columns
                 if "score" in c.lower() or "quality" in c.lower()),
                None
            )
            if score_col:
                self.quality_scores = dict(
                    zip(quality_df[self.path_col].tolist(),
                        quality_df[score_col].tolist())
                )

        # Mean quality score
        self.mean_quality_score = float(
            np.mean(list(self.quality_scores.values()))
            if self.quality_scores else 75.0
        )

        logger.info(
            f"ImageAnalyzer initialized: {len(self.image_paths)} images, "
            f"{len(set(self.labels))} classes, dir={self.image_dir}"
        )

    def _validate_inputs(self):
        """Validate inputs and raise clear errors."""
        if self.df.empty:
            raise ValueError("loader_df is empty.")
        if self.path_col not in self.df.columns:
            raise ValueError(
                f"path_column='{self.path_col}' not found in loader_df. "
                f"Available: {list(self.df.columns)}"
            )

    def analyze_all(self,
                    blocks: Optional[List[int]] = None,
                    verbose: bool = True) -> AnalysisReport:
        """
        Run all analysis blocks and return unified AnalysisReport.

        Args:
            blocks: List of block numbers to run (1–7). None = all.
            verbose: Print progress.

        Returns:
            AnalysisReport with all results populated.
        """
        start_time = time.time()
        blocks = blocks or [1, 2, 3, 4, 5, 6, 7]

        report = AnalysisReport(
            dataset_name=self.dataset_name,
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            total_images=len(self.image_paths),
            total_classes=len(set(self.labels)),
            image_dir=self.image_dir,
        )

        def _log(msg: str):
            if verbose:
                logger.info(msg)
                print(f"[Nydra] {msg}")

        # ── Block 1: Color Distribution ───────────────────────────────────────
        if 1 in blocks:
            _log("Block 1/7: Color Distribution Analysis...")
            try:
                analyzer = ColorDistributionAnalyzer(
                    max_images_per_class=self.max_images_per_class
                )
                report.color_bias = analyzer.analyze(
                    self.image_paths, self.labels, self.image_dir
                )
            except Exception as e:
                report.errors.append(f"Block 1 error: {e}")
                logger.error(f"Block 1 failed: {e}")

        # ── Block 2: Class Balance ────────────────────────────────────────────
        if 2 in blocks:
            _log("Block 2/7: Class Balance Deep Analysis...")
            try:
                analyzer = ClassBalanceAnalyzer()
                report.class_balance = analyzer.analyze(
                    self.image_paths, self.labels,
                    image_dir=self.image_dir
                )
            except Exception as e:
                report.errors.append(f"Block 2 error: {e}")
                logger.error(f"Block 2 failed: {e}")

        # ── Block 3: Size & Dimension ─────────────────────────────────────────
        if 3 in blocks:
            _log("Block 3/7: Size & Dimension Analysis...")
            try:
                analyzer = SizeDimensionAnalyzer()
                report.size_dimension = analyzer.analyze(
                    self.image_paths, self.labels, self.image_dir
                )
            except Exception as e:
                report.errors.append(f"Block 3 error: {e}")
                logger.error(f"Block 3 failed: {e}")

        # ── Block 4: Diversity & Redundancy ───────────────────────────────────
        if 4 in blocks:
            _log("Block 4/7: Diversity & Redundancy Analysis...")
            try:
                analyzer = DiversityRedundancyAnalyzer()
                report.diversity_redundancy = analyzer.analyze(
                    self.image_paths, self.labels, self.image_dir
                )
            except Exception as e:
                report.errors.append(f"Block 4 error: {e}")
                logger.error(f"Block 4 failed: {e}")

        # ── Block 5: Distribution Shift ───────────────────────────────────────
        if 5 in blocks:
            _log("Block 5/7: Distribution Shift Detection...")
            try:
                detector = DistributionShiftDetector()
                report.distribution_shift = detector.analyze(
                    self.image_paths, self.labels,
                    splits=self.splits,
                    image_dir=self.image_dir,
                    compute_mmd=self.compute_mmd
                )
            except Exception as e:
                report.errors.append(f"Block 5 error: {e}")
                logger.error(f"Block 5 failed: {e}")

        # ── Block 6: Outlier Detection ────────────────────────────────────────
        if 6 in blocks:
            _log("Block 6/7: Outlier Image Detection...")
            try:
                detector = OutlierImageDetector()
                report.outlier_detection = detector.analyze(
                    self.image_paths, self.labels,
                    quality_scores=self.quality_scores,
                    image_dir=self.image_dir
                )
            except Exception as e:
                report.errors.append(f"Block 6 error: {e}")
                logger.error(f"Block 6 failed: {e}")

        # ── Block 7: ML Readiness Final Score ─────────────────────────────────
        if 7 in blocks:
            _log("Block 7/7: Computing ML Readiness Score...")
            try:
                scorer = MLReadinessScorer()
                report.ml_readiness = scorer.compute(
                    color_result    = report.color_bias,
                    balance_result  = report.class_balance,
                    size_result     = report.size_dimension,
                    diversity_result= report.diversity_redundancy,
                    shift_result    = report.distribution_shift,
                    outlier_result  = report.outlier_detection,
                    quality_score_from_quality_module=self.mean_quality_score,
                )
            except Exception as e:
                report.errors.append(f"Block 7 error: {e}")
                logger.error(f"Block 7 failed: {e}")

        report.analysis_duration_seconds = round(time.time() - start_time, 2)

        if verbose:
            self._print_summary(report)

        return report

    def analyze_block(self, block_num: int, **kwargs) -> Any:
        """Run a single block and return its result."""
        return self.analyze_all(blocks=[block_num], **kwargs)

    def _print_summary(self, report: AnalysisReport):
        """Print a concise summary of the analysis."""
        print("\n" + "="*60)
        print(f"🩺 Nydra — Image Dataset Analysis Report")
        print(f"   Dataset: {report.dataset_name}")
        print(f"   Images:  {report.total_images} | Classes: {report.total_classes}")
        print(f"   Time:    {report.analysis_duration_seconds}s")
        print("="*60)

        if report.ml_readiness:
            ml = report.ml_readiness
            print(f"\n   🎯 ML Readiness: {ml.final_score:.1f}/100 "
                  f"[Grade: {ml.grade}] — {ml.verdict}")
            print(f"\n   Sub-scores:")
            print(f"     Diversity:  {ml.diversity_score:.1f}")
            print(f"     Balance:    {ml.balance_score:.1f}")
            print(f"     Shift:      {ml.shift_score:.1f}")
            print(f"     Quality:    {ml.quality_score:.1f}")
            print(f"     Coverage:   {ml.coverage_score:.1f}")

            if ml.recommendations_ranked:
                print(f"\n   ⚠️ Top Issues:")
                for rec in ml.recommendations_ranked[:3]:
                    print(f"     [{rec['priority']}] {rec['issue']} "
                          f"→ {rec['impact']}")

        if report.errors:
            print(f"\n   ❌ Errors: {len(report.errors)}")
            for err in report.errors:
                print(f"     {err}")

        print("="*60 + "\n")

    # ── Convenience Methods ────────────────────────────────────────────────────

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Run full analysis and return a single-row summary DataFrame.
        Useful for comparing multiple datasets.
        """
        report = self.analyze_all(verbose=False)
        row = {
            "dataset": report.dataset_name,
            "total_images": report.total_images,
            "total_classes": report.total_classes,
            "analysis_time_s": report.analysis_duration_seconds,
        }
        if report.ml_readiness:
            ml = report.ml_readiness
            row.update({
                "ml_readiness_score": ml.final_score,
                "grade": ml.grade,
                "verdict": ml.verdict,
                "diversity_score": ml.diversity_score,
                "balance_score": ml.balance_score,
                "shift_score": ml.shift_score,
                "quality_score": ml.quality_score,
                "coverage_score": ml.coverage_score,
            })
        if report.color_bias:
            row["shortcut_risk"] = report.color_bias.shortcut_risk_score
        if report.class_balance:
            row["gini_coefficient"] = report.class_balance.gini_coefficient
            row["imbalance_ratio"] = report.class_balance.imbalance_ratio
        if report.diversity_redundancy:
            row["vendi_score"] = report.diversity_redundancy.vendi_score
            row["redundancy_rate"] = report.diversity_redundancy.redundancy_rate
        if report.outlier_detection:
            row["outlier_rate"] = report.outlier_detection.outlier_rate
        return pd.DataFrame([row])

    def compare_datasets(self, other: "ImageAnalyzer") -> pd.DataFrame:
        """
        Compare this dataset with another.
        Returns a comparison DataFrame.
        """
        df1 = self.get_summary_dataframe()
        df2 = other.get_summary_dataframe()
        return pd.concat([df1, df2], ignore_index=True)

    def get_top_outliers(self, n: int = 10) -> pd.DataFrame:
        """Return top-n outlier images sorted by anomaly score."""
        report = self.analyze_all(blocks=[6], verbose=False)
        if not report.outlier_detection:
            return pd.DataFrame()
        scores = report.outlier_detection.outlier_scores
        suggestions = report.outlier_detection.suggestions
        rows = [
            {
                "path": path,
                "anomaly_score": score,
                "suggestion": suggestions.get(path, "review"),
                "is_ood": path in (report.outlier_detection.ood_images or []),
            }
            for path, score in scores.items()
        ]
        df = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False)
        return df.head(n).reset_index(drop=True)

    def get_near_duplicate_report(self) -> pd.DataFrame:
        """Return near-duplicate pairs as DataFrame."""
        report = self.analyze_all(blocks=[4], verbose=False)
        if not report.diversity_redundancy:
            return pd.DataFrame()
        pairs = report.diversity_redundancy.near_duplicate_pairs
        return pd.DataFrame(
            pairs, columns=["image_1", "image_2", "similarity"]
        ).sort_values("similarity", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE (python image_analyzer.py <image_dir> <csv_file>)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python image_analyzer.py <image_dir> <metadata_csv>")
        print("       CSV must have 'path' and 'label' columns.")
        print("       Optional: 'split' column for shift detection.")
        sys.exit(1)

    image_dir  = sys.argv[1]
    csv_path   = sys.argv[2]
    output     = sys.argv[3] if len(sys.argv) > 3 else "analysis_report.json"

    print(f"\n🩺 Nydra — Image Analyzer")
    print(f"   Image Dir: {image_dir}")
    print(f"   Metadata:  {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   Found {len(df)} rows in metadata CSV.")

    split_col = "split" if "split" in df.columns else None

    analyzer = ImageAnalyzer(
        loader_df=df,
        image_dir=image_dir,
        label_column="label",
        path_column="path",
        split_column=split_col,
        dataset_name=Path(csv_path).stem,
        max_images_per_class=150,
        compute_mmd=False,
    )

    report = analyzer.analyze_all(verbose=True)

    # Save JSON report
    report_dict = report.to_dict()
    if report.ml_readiness:
        report_dict["ml_readiness"] = report.ml_readiness.details
    if report.class_balance:
        report_dict["class_balance"] = {
            "imbalance_ratio": report.class_balance.imbalance_ratio,
            "gini": report.class_balance.gini_coefficient,
            "balance_score": report.class_balance.balance_score,
        }
    if report.diversity_redundancy:
        report_dict["diversity"] = {
            "vendi_score": report.diversity_redundancy.vendi_score,
            "redundancy_rate": report.diversity_redundancy.redundancy_rate,
            "near_dup_pairs": len(report.diversity_redundancy.near_duplicate_pairs),
        }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved to: {output}")
