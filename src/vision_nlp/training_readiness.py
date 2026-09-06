"""
training_readiness.py — Nydra v0.6.0
==========================================
Evaluates whether an image dataset is ready for model training.
Provides a score (0-100), grade (A→F), blocking issues, warnings,
model recommendations, augmentation plan, hardware requirements,
and training time estimates.

Author  : Nydra Team
Project : Nydra — https://github.com/Denterio1/Nydra
Version : 0.6.0
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import math
import time
import json
import logging
import hashlib
import warnings
import platform
import collections
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Set, NamedTuple
)
from dataclasses import dataclass, field, asdict
from enum import Enum, auto

import numpy as np

# Optional heavy deps — graceful fallback
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("nydra.training_readiness")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "0.6.0"

# Minimum samples per class thresholds
MIN_SAMPLES_CRITICAL   = 10      # below this → blocking
MIN_SAMPLES_POOR       = 50      # very risky
MIN_SAMPLES_FAIR       = 100     # workable with augmentation
MIN_SAMPLES_GOOD       = 500     # good
MIN_SAMPLES_EXCELLENT  = 1000    # ideal

# Imbalance ratio thresholds
IMBALANCE_CRITICAL     = 10.0    # majority:minority > 10 → blocking
IMBALANCE_SEVERE       = 5.0     # > 5 → warning
IMBALANCE_MODERATE     = 2.0     # > 2 → note

# Total dataset thresholds
TOTAL_CRITICAL         = 50
TOTAL_POOR             = 200
TOTAL_FAIR             = 1000
TOTAL_GOOD             = 5000
TOTAL_EXCELLENT        = 20000

# Split ratios
DEFAULT_TRAIN_RATIO    = 0.70
DEFAULT_VAL_RATIO      = 0.15
DEFAULT_TEST_RATIO     = 0.15

# Grade boundaries (same as quality_score.py)
GRADE_A = 90
GRADE_B = 75
GRADE_C = 60
GRADE_D = 45

# Hardware estimation constants
GPU_THROUGHPUT = {
    "cpu":         50,    # images/second
    "gpu_t4":    1200,
    "gpu_v100":  3000,
    "gpu_a100":  6000,
    "gpu_rtx3090": 2500,
    "gpu_rtx4090": 4500,
}

MODEL_PARAMS = {
    "mobilenet_v2":    3_400_000,
    "efficientnet_b0": 5_300_000,
    "efficientnet_b4": 19_000_000,
    "resnet50":        25_600_000,
    "resnet101":       44_500_000,
    "vit_base":        86_000_000,
    "vit_large":      307_000_000,
    "distilbert":      66_000_000,
    "bert_base":      110_000_000,
    "roberta_base":   125_000_000,
    "clip_vit_b32":   151_000_000,
    "yolov8n":         3_200_000,
    "yolov8m":        25_900_000,
    "faster_rcnn":    41_000_000,
}

# Augmentation safety per task
SAFE_AUGMENTATIONS = {
    "classification": [
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation(±15°)",
        "ColorJitter(brightness, contrast)",
        "RandomGrayscale(p=0.1)",
        "GaussianBlur(kernel=3)",
        "RandomCrop",
        "Normalize(ImageNet mean/std)",
    ],
    "detection": [
        "RandomHorizontalFlip",
        "RandomBrightness",
        "RandomContrast",
        "Mosaic (YOLOv5+)",
        "MixUp",
        "RandomScale",
    ],
    "segmentation": [
        "RandomHorizontalFlip",
        "RandomRotation(±10°)",
        "RandomCrop (with mask)",
        "ColorJitter",
        "ElasticTransform",
    ],
    "text_classification": [
        "Synonym Replacement",
        "Random Insertion",
        "Random Swap",
        "Random Deletion",
        "Back Translation",
        "EDA (Easy Data Augmentation)",
    ],
}

RISKY_AUGMENTATIONS = {
    "classification": [
        "RandomRotation(90°) — only if rotation-invariant",
        "CutMix — needs n > 500 per class",
        "MixUp — needs balanced classes",
        "AutoAugment — computationally expensive",
    ],
    "detection": [
        "RandomRotation — breaks bbox alignment",
        "CutOut — may hide small objects",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TaskType(Enum):
    IMAGE_CLASSIFICATION  = "image_classification"
    OBJECT_DETECTION      = "object_detection"
    IMAGE_SEGMENTATION    = "image_segmentation"
    IMAGE_REGRESSION      = "image_regression"
    TEXT_CLASSIFICATION   = "text_classification"
    MULTI_MODAL           = "multi_modal"
    UNKNOWN               = "unknown"


class HardwareType(Enum):
    CPU         = "cpu"
    GPU_T4      = "gpu_t4"
    GPU_V100    = "gpu_v100"
    GPU_A100    = "gpu_a100"
    GPU_RTX3090 = "gpu_rtx3090"
    GPU_RTX4090 = "gpu_rtx4090"


class ReadinessGrade(Enum):
    A = "A"   # 90-100  Excellent
    B = "B"   # 75-89   Good
    C = "C"   # 60-74   Fair
    D = "D"   # 45-59   Poor
    F = "F"   # 0-44    Critical


@dataclass
class ClassInfo:
    """Information about a single class in the dataset."""
    name: str
    count: int
    percentage: float
    is_minority: bool = False
    is_majority: bool = False
    has_enough_samples: bool = True
    split_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Score for a single readiness dimension."""
    name: str
    score: float          # 0-100
    weight: float         # contribution weight
    weighted_score: float # score * weight
    status: str           # excellent / good / fair / poor / critical
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ModelRecommendation:
    """A recommended model for training."""
    name: str
    architecture: str
    task: str
    reason: str
    params: int
    pretrained: bool
    pretrained_on: str
    min_samples_needed: int
    expected_accuracy_range: Tuple[float, float]
    training_difficulty: str    # easy / medium / hard
    library: str                # torchvision / huggingface / ultralytics
    code_snippet: str


@dataclass
class AugmentationPlan:
    """Complete augmentation plan for the dataset."""
    task: str
    safe_transforms: List[str]
    risky_transforms: List[str]
    not_recommended: List[str]
    expected_multiplier: float   # how much data augmentation adds
    library_code: str            # albumentations / torchvision code
    notes: str


@dataclass
class HardwareRequirements:
    """Hardware requirements for training."""
    min_ram_gb: float
    recommended_ram_gb: float
    min_vram_gb: float
    recommended_vram_gb: float
    cpu_cores_recommended: int
    gpu_required: bool
    gpu_recommended: str
    estimated_disk_gb: float
    notes: List[str]


@dataclass
class TrainingTimeEstimate:
    """Training time estimates per hardware type."""
    n_samples: int
    n_epochs: int
    batch_size: int
    model_name: str
    estimates: Dict[str, str]   # hardware_type → "~2.5 hours"
    bottleneck: str
    tips: List[str]


@dataclass
class SplitRecommendation:
    """Recommended train/val/test split."""
    strategy: str               # standard / stratified / time_series / k_fold
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train_count: int
    val_count: int
    test_count: int
    warnings: List[str]
    per_class_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class ReadinessReport:
    """
    Complete training readiness report — the main output of this module.
    """
    # Metadata
    timestamp: str
    dataset_path: str
    task_type: str
    n_samples: int
    n_classes: int
    modality: str               # image / text / multi-modal

    # Overall result
    overall_score: float
    grade: str
    ready_to_train: bool
    confidence: float           # how confident is this assessment

    # Issues
    blocking_issues: List[str]
    warnings: List[str]
    info_notes: List[str]

    # Detailed dimensions
    dimensions: Dict[str, Any]

    # Actionable output
    recommendations: List[str]
    model_recommendations: List[Any]
    augmentation_plan: Any
    hardware_requirements: Any
    training_time_estimate: Any
    split_recommendation: Any
    transfer_learning_advice: Dict[str, Any]

    # Summary for UI
    summary: str
    next_steps: List[str]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _score_to_status(score: float) -> str:
    """Convert numeric score to status string."""
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "good"
    elif score >= 60:
        return "fair"
    elif score >= 45:
        return "poor"
    else:
        return "critical"


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= GRADE_A:
        return "A"
    elif score >= GRADE_B:
        return "B"
    elif score >= GRADE_C:
        return "C"
    elif score >= GRADE_D:
        return "D"
    else:
        return "F"


def _grade_emoji(grade: str) -> str:
    """Emoji for grade."""
    return {
        "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⛔"
    }.get(grade, "❓")


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"~{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"~{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"~{days:.1f} days"


def _format_params(n: int) -> str:
    """Format parameter count."""
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division."""
    return a / b if b != 0 else default


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _gini_impurity(counts: List[int]) -> float:
    """
    Gini impurity — measures class imbalance.
    0 = perfectly imbalanced, 1 = perfectly balanced (normalized).
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts]
    return 1.0 - sum(p ** 2 for p in probs)


def _entropy(counts: List[int]) -> float:
    """Shannon entropy of class distribution."""
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _imbalance_ratio(counts: List[int]) -> float:
    """Ratio of majority to minority class."""
    if not counts or min(counts) == 0:
        return float("inf")
    return max(counts) / min(counts)


def _detect_task_from_structure(
    dataset_info: Dict[str, Any]
) -> TaskType:
    """
    Infer task type from dataset structure.
    Uses presence of bounding boxes, masks, labels.
    """
    has_boxes   = dataset_info.get("has_bounding_boxes", False)
    has_masks   = dataset_info.get("has_masks", False)
    has_labels  = dataset_info.get("has_labels", True)
    has_text    = dataset_info.get("has_text", False)
    n_classes   = dataset_info.get("n_classes", 0)
    modality    = dataset_info.get("modality", "image")

    if has_text and modality == "image":
        return TaskType.MULTI_MODAL
    if has_masks:
        return TaskType.IMAGE_SEGMENTATION
    if has_boxes:
        return TaskType.OBJECT_DETECTION
    if modality == "text":
        return TaskType.TEXT_CLASSIFICATION
    if has_labels and n_classes >= 2:
        return TaskType.IMAGE_CLASSIFICATION
    if has_labels and n_classes == 0:
        return TaskType.IMAGE_REGRESSION

    return TaskType.IMAGE_CLASSIFICATION  # default safe assumption


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1 — DatasetSizeAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class DatasetSizeAnalyzer:
    """
    Analyzes whether the dataset has enough samples for training.

    Checks:
    - Total sample count vs thresholds
    - Per-class sample count
    - Samples available after train/val/test split
    - Minimum effective samples with augmentation
    """

    def __init__(
        self,
        task_type: TaskType = TaskType.IMAGE_CLASSIFICATION,
        augmentation_factor: float = 3.0
    ):
        self.task_type = task_type
        self.augmentation_factor = augmentation_factor

    def analyze(
        self,
        n_total: int,
        class_counts: Dict[str, int],
        n_channels: int = 3,
        image_size: Tuple[int, int] = (224, 224),
    ) -> DimensionScore:
        """
        Analyze dataset size and return a DimensionScore.

        Parameters
        ----------
        n_total       : total number of images
        class_counts  : dict of {class_name: count}
        n_channels    : image channels (1=gray, 3=RGB, 4=RGBA)
        image_size    : (width, height) of images
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}

        n_classes  = max(len(class_counts), 1)
        counts     = list(class_counts.values()) if class_counts else [n_total]
        min_count  = min(counts) if counts else 0
        max_count  = max(counts) if counts else 0
        avg_count  = _safe_divide(sum(counts), len(counts))

        details["total_samples"]       = n_total
        details["n_classes"]           = n_classes
        details["min_per_class"]       = min_count
        details["max_per_class"]       = max_count
        details["avg_per_class"]       = round(avg_count, 1)
        details["image_size"]          = f"{image_size[0]}×{image_size[1]}"
        details["n_channels"]          = n_channels

        # Estimated disk size
        bytes_per_image = image_size[0] * image_size[1] * n_channels
        est_disk_mb = (n_total * bytes_per_image) / (1024 ** 2)
        details["estimated_disk_mb"]   = round(est_disk_mb, 1)

        # Samples after split (70/15/15)
        train_n = int(n_total * DEFAULT_TRAIN_RATIO)
        val_n   = int(n_total * DEFAULT_VAL_RATIO)
        test_n  = n_total - train_n - val_n
        details["after_split"] = {
            "train": train_n,
            "val":   val_n,
            "test":  test_n,
        }

        # With augmentation
        effective_train = int(train_n * self.augmentation_factor)
        details["effective_train_with_augmentation"] = effective_train

        # --- Scoring ---
        score = 0.0

        # Total size score (50 points)
        if n_total >= TOTAL_EXCELLENT:
            score += 50.0
        elif n_total >= TOTAL_GOOD:
            score += 40.0 + 10.0 * (n_total - TOTAL_GOOD) / (TOTAL_EXCELLENT - TOTAL_GOOD)
        elif n_total >= TOTAL_FAIR:
            score += 25.0 + 15.0 * (n_total - TOTAL_FAIR) / (TOTAL_GOOD - TOTAL_FAIR)
        elif n_total >= TOTAL_POOR:
            score += 10.0 + 15.0 * (n_total - TOTAL_POOR) / (TOTAL_FAIR - TOTAL_POOR)
        elif n_total >= TOTAL_CRITICAL:
            score += 5.0
        else:
            score += 0.0
            issues.append(
                f"BLOCKING: Only {n_total} total samples — absolute minimum is {TOTAL_CRITICAL}."
            )

        # Per-class minimum score (50 points)
        classes_below_critical = sum(1 for c in counts if c < MIN_SAMPLES_CRITICAL)
        classes_below_fair     = sum(1 for c in counts if c < MIN_SAMPLES_FAIR)

        if classes_below_critical > 0:
            score += 0.0
            issues.append(
                f"BLOCKING: {classes_below_critical} class(es) have fewer than "
                f"{MIN_SAMPLES_CRITICAL} samples — model cannot learn these."
            )
        elif min_count >= MIN_SAMPLES_EXCELLENT:
            score += 50.0
        elif min_count >= MIN_SAMPLES_GOOD:
            score += 40.0
        elif min_count >= MIN_SAMPLES_FAIR:
            score += 25.0
            suggestions.append(
                f"Min class has {min_count} samples. "
                f"Apply augmentation to reach {MIN_SAMPLES_GOOD}+."
            )
        elif min_count >= MIN_SAMPLES_POOR:
            score += 10.0
            suggestions.append(
                f"Min class has only {min_count} samples. "
                f"Use heavy augmentation + transfer learning."
            )
        else:
            score += 0.0

        if classes_below_fair > 0:
            suggestions.append(
                f"{classes_below_fair} class(es) have fewer than {MIN_SAMPLES_FAIR} samples. "
                f"Consider collecting more data or merging similar classes."
            )

        # Task-specific checks
        if self.task_type == TaskType.OBJECT_DETECTION and n_total < 500:
            issues.append(
                "Object detection requires at least 500 images. "
                "Strongly recommend 1000+ with bounding boxes."
            )
        elif self.task_type == TaskType.IMAGE_SEGMENTATION and n_total < 200:
            issues.append(
                "Segmentation requires pixel-level labels. "
                "200 is absolute minimum — target 500+."
            )

        score = _clamp(score)
        details["score_breakdown"] = {
            "total_size_score": min(score, 50),
            "per_class_score":  max(score - 50, 0) if score > 50 else 0,
        }

        return DimensionScore(
            name="Dataset Size",
            score=round(score, 2),
            weight=0.20,
            weighted_score=round(score * 0.20, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def get_size_category(self, n_total: int) -> str:
        """Return a human-readable size category."""
        if n_total >= TOTAL_EXCELLENT:
            return "Large dataset ✅"
        elif n_total >= TOTAL_GOOD:
            return "Medium dataset 🟡"
        elif n_total >= TOTAL_FAIR:
            return "Small dataset 🟠"
        elif n_total >= TOTAL_POOR:
            return "Tiny dataset 🔴"
        else:
            return "Insufficient dataset ⛔"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2 — ClassBalanceAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalanceAnalyzer:
    """
    Analyzes class balance / imbalance in the dataset.

    Checks:
    - Imbalance ratio (majority:minority)
    - Gini impurity / Shannon entropy
    - Minority class percentage
    - Recommended rebalancing strategies
    """

    def __init__(self):
        pass

    def analyze(
        self,
        class_counts: Dict[str, int],
        task_type: TaskType = TaskType.IMAGE_CLASSIFICATION,
    ) -> DimensionScore:
        """
        Analyze class balance.

        Parameters
        ----------
        class_counts : {class_name: count}
        task_type    : affects tolerance (detection more tolerant)
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}

        if not class_counts:
            return DimensionScore(
                name="Class Balance",
                score=0.0,
                weight=0.18,
                weighted_score=0.0,
                status="critical",
                details={"error": "No class information provided"},
                issues=["BLOCKING: No class labels found."],
                suggestions=["Provide labels for your dataset."],
            )

        counts    = list(class_counts.values())
        total     = sum(counts)
        n_classes = len(counts)
        ratio     = _imbalance_ratio(counts)
        gini      = _gini_impurity(counts)
        entropy   = _entropy(counts)
        max_entropy = math.log2(n_classes) if n_classes > 1 else 1.0

        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        minority_class = sorted_classes[0]
        majority_class = sorted_classes[-1]

        minority_pct = _safe_divide(minority_class[1], total) * 100
        majority_pct = _safe_divide(majority_class[1], total) * 100

        details["n_classes"]           = n_classes
        details["total_samples"]       = total
        details["imbalance_ratio"]     = round(ratio, 2)
        details["gini_impurity"]       = round(gini, 4)
        details["shannon_entropy"]     = round(entropy, 4)
        details["max_entropy"]         = round(max_entropy, 4)
        details["entropy_normalized"]  = round(
            _safe_divide(entropy, max_entropy), 4
        )
        details["minority_class"]      = {
            "name": minority_class[0],
            "count": minority_class[1],
            "percentage": round(minority_pct, 2),
        }
        details["majority_class"]      = {
            "name": majority_class[0],
            "count": majority_class[1],
            "percentage": round(majority_pct, 2),
        }
        details["class_distribution"]  = {
            k: {"count": v, "percentage": round(_safe_divide(v, total) * 100, 2)}
            for k, v in sorted(class_counts.items(), key=lambda x: -x[1])
        }

        # Rebalancing strategies
        strategies = self._recommend_strategies(
            ratio=ratio,
            n_minority=minority_class[1],
            n_classes=n_classes,
            task_type=task_type,
        )
        details["recommended_strategies"] = strategies

        # --- Scoring ---
        score = 0.0

        if ratio == float("inf") or ratio >= IMBALANCE_CRITICAL:
            score = 0.0
            issues.append(
                f"BLOCKING: Imbalance ratio is {ratio:.1f}:1 "
                f"(majority={majority_class[0]}: {majority_class[1]}, "
                f"minority={minority_class[0]}: {minority_class[1]}). "
                f"Model will ignore minority classes."
            )
        elif ratio >= IMBALANCE_SEVERE:
            score = 25.0
            issues.append(
                f"Severe imbalance ratio {ratio:.1f}:1. "
                f"Apply oversampling or class weights."
            )
        elif ratio >= IMBALANCE_MODERATE:
            score = 55.0
            suggestions.append(
                f"Moderate imbalance ratio {ratio:.1f}:1. "
                f"Consider stratified sampling."
            )
        elif ratio >= 1.5:
            score = 80.0
            suggestions.append(
                f"Mild imbalance ({ratio:.1f}:1). "
                f"Use stratified splits."
            )
        else:
            score = 100.0

        # Bonus for more classes being balanced
        balanced_classes = sum(
            1 for c in counts
            if abs(c - (total / n_classes)) / (total / n_classes) < 0.2
        )
        balance_pct = _safe_divide(balanced_classes, n_classes)
        if ratio < IMBALANCE_SEVERE:
            score = score * (0.7 + 0.3 * balance_pct)

        if minority_pct < 5.0 and n_classes > 2:
            suggestions.append(
                f"Minority class '{minority_class[0]}' is only {minority_pct:.1f}% "
                f"of total. Very likely to be underfit."
            )

        score = _clamp(score)

        return DimensionScore(
            name="Class Balance",
            score=round(score, 2),
            weight=0.18,
            weighted_score=round(score * 0.18, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def _recommend_strategies(
        self,
        ratio: float,
        n_minority: int,
        n_classes: int,
        task_type: TaskType,
    ) -> List[Dict[str, str]]:
        """Recommend rebalancing strategies based on imbalance severity."""
        strategies = []

        if ratio < 2.0:
            strategies.append({
                "method": "Stratified Split",
                "reason": "Mild imbalance — just ensure stratified train/val/test split.",
                "priority": "low",
            })
            return strategies

        # Always recommend class weights for moderate+
        strategies.append({
            "method": "Class Weights",
            "reason": "Easy to implement — weight=total/(n_classes×class_count).",
            "priority": "high",
            "code_hint": "class_weight='balanced' in sklearn / weight= in PyTorch CrossEntropyLoss",
        })

        if ratio < IMBALANCE_SEVERE and n_minority >= 50:
            strategies.append({
                "method": "SMOTE (oversampling)",
                "reason": "Synthesize minority samples in feature space.",
                "priority": "medium",
                "code_hint": "from imblearn.over_sampling import SMOTE",
            })

        if n_minority >= 20:
            strategies.append({
                "method": "Data Augmentation on minority classes",
                "reason": "Apply heavier augmentation only to minority class images.",
                "priority": "high",
                "code_hint": "Apply RandomFlip, ColorJitter, RandomRotation to minority classes only.",
            })

        if ratio >= IMBALANCE_SEVERE:
            strategies.append({
                "method": "Undersampling majority class",
                "reason": "Reduce majority class to balance the dataset.",
                "priority": "medium",
                "code_hint": "from imblearn.under_sampling import RandomUnderSampler",
            })

        if ratio >= IMBALANCE_CRITICAL:
            strategies.append({
                "method": "Focal Loss",
                "reason": "Loss function that down-weights easy examples, focus on hard minority.",
                "priority": "high",
                "code_hint": "from torchvision.ops import sigmoid_focal_loss",
            })
            strategies.append({
                "method": "Collect more data for minority classes",
                "reason": "Most reliable solution — ratio is critical.",
                "priority": "critical",
            })

        return strategies

    def get_balance_summary(self, class_counts: Dict[str, int]) -> str:
        """One-line balance summary."""
        if not class_counts:
            return "No data"
        counts = list(class_counts.values())
        ratio  = _imbalance_ratio(counts)
        if ratio < 1.5:
            return f"Well balanced ({ratio:.1f}:1) ✅"
        elif ratio < IMBALANCE_MODERATE:
            return f"Mildly imbalanced ({ratio:.1f}:1) 🟡"
        elif ratio < IMBALANCE_SEVERE:
            return f"Moderately imbalanced ({ratio:.1f}:1) 🟠"
        elif ratio < IMBALANCE_CRITICAL:
            return f"Severely imbalanced ({ratio:.1f}:1) 🔴"
        else:
            return f"Critically imbalanced ({ratio:.1f}:1) ⛔"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3 — SplitAdvisor
# ─────────────────────────────────────────────────────────────────────────────

class SplitAdvisor:
    """
    Recommends the best train/val/test split strategy.

    Considers:
    - Total dataset size
    - Class balance
    - Temporal data (sequential images)
    - Cross-validation when dataset is small
    """

    def __init__(self):
        pass

    def recommend(
        self,
        n_total: int,
        class_counts: Dict[str, int],
        is_temporal: bool = False,
        min_val_samples_per_class: int = 10,
        min_test_samples_per_class: int = 20,
    ) -> SplitRecommendation:
        """
        Recommend train/val/test split.

        Parameters
        ----------
        n_total                     : total samples
        class_counts                : {class: count}
        is_temporal                 : sequential images (dashcam, medical scans...)
        min_val_samples_per_class   : minimum val samples per class
        min_test_samples_per_class  : minimum test samples per class
        """
        warnings_list: List[str] = []
        n_classes = len(class_counts) if class_counts else 1
        counts    = list(class_counts.values()) if class_counts else [n_total]
        min_count = min(counts) if counts else n_total

        # Choose strategy
        if is_temporal:
            strategy     = "time_series_split"
            train_ratio  = 0.70
            val_ratio    = 0.15
            test_ratio   = 0.15
            warnings_list.append(
                "Temporal data detected — do NOT shuffle. "
                "Keep chronological order: train[first] → val[mid] → test[last]."
            )
        elif n_total < 500 or min_count < 50:
            strategy     = "stratified_k_fold"
            train_ratio  = 0.80
            val_ratio    = 0.10
            test_ratio   = 0.10
            warnings_list.append(
                f"Small dataset ({n_total} samples). "
                "Use Stratified K-Fold (k=5) instead of a fixed split."
            )
        elif _imbalance_ratio(counts) > IMBALANCE_MODERATE:
            strategy     = "stratified_split"
            train_ratio  = 0.70
            val_ratio    = 0.15
            test_ratio   = 0.15
            warnings_list.append(
                "Imbalanced classes detected. "
                "Stratified split ensures all classes appear in all splits."
            )
        elif n_total >= TOTAL_EXCELLENT:
            strategy     = "standard_split"
            train_ratio  = 0.80
            val_ratio    = 0.10
            test_ratio   = 0.10
        else:
            strategy     = "stratified_split"
            train_ratio  = 0.70
            val_ratio    = 0.15
            test_ratio   = 0.15

        train_n = int(n_total * train_ratio)
        val_n   = int(n_total * val_ratio)
        test_n  = n_total - train_n - val_n

        # Compute per-class counts
        per_class: Dict[str, Dict[str, int]] = {}
        for cls, cnt in class_counts.items():
            per_class[cls] = {
                "train": int(cnt * train_ratio),
                "val":   int(cnt * val_ratio),
                "test":  cnt - int(cnt * train_ratio) - int(cnt * val_ratio),
            }

        # Validate minimums
        if class_counts:
            for cls, splits in per_class.items():
                if splits["val"] < min_val_samples_per_class:
                    warnings_list.append(
                        f"Class '{cls}' will have only {splits['val']} val samples "
                        f"(minimum recommended: {min_val_samples_per_class})."
                    )
                if splits["test"] < min_test_samples_per_class:
                    warnings_list.append(
                        f"Class '{cls}' will have only {splits['test']} test samples "
                        f"(minimum recommended: {min_test_samples_per_class})."
                    )

        if val_n < 50:
            warnings_list.append(
                f"Val set has only {val_n} samples. "
                "Consider using cross-validation for more reliable evaluation."
            )

        if test_n < 100:
            warnings_list.append(
                f"Test set has only {test_n} samples. "
                "Results may have high variance — interpret carefully."
            )

        return SplitRecommendation(
            strategy=strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            train_count=train_n,
            val_count=val_n,
            test_count=test_n,
            warnings=warnings_list,
            per_class_counts=per_class,
        )

    def get_strategy_description(self, strategy: str) -> str:
        """Human-readable strategy description."""
        descriptions = {
            "standard_split": (
                "Random 80/10/10 split. Best for large balanced datasets."
            ),
            "stratified_split": (
                "Stratified 70/15/15 split. Ensures each class is proportionally "
                "represented in all splits."
            ),
            "stratified_k_fold": (
                "Stratified K-Fold (k=5). Best for small datasets — "
                "uses all data for training and evaluation."
            ),
            "time_series_split": (
                "Chronological split. First 70% for training, "
                "next 15% for validation, last 15% for testing."
            ),
        }
        return descriptions.get(strategy, "Unknown strategy.")

    def generate_sklearn_code(self, strategy: str, test_ratio: float) -> str:
        """Generate sklearn code for the recommended split."""
        if strategy == "stratified_split":
            return (
                "from sklearn.model_selection import train_test_split\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                f"    X, y, test_size={test_ratio}, stratify=y, random_state=42\n"
                ")"
            )
        elif strategy == "stratified_k_fold":
            return (
                "from sklearn.model_selection import StratifiedKFold\n\n"
                "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
                "for train_idx, val_idx in skf.split(X, y):\n"
                "    X_train, X_val = X[train_idx], X[val_idx]\n"
                "    y_train, y_val = y[train_idx], y[val_idx]"
            )
        elif strategy == "time_series_split":
            return (
                "# Chronological split — do NOT shuffle\n"
                "n = len(X)\n"
                "train_end = int(n * 0.70)\n"
                "val_end   = int(n * 0.85)\n"
                "X_train, y_train = X[:train_end], y[:train_end]\n"
                "X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]\n"
                "X_test,  y_test  = X[val_end:], y[val_end:]"
            )
        else:
            return (
                "from sklearn.model_selection import train_test_split\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                f"    X, y, test_size={test_ratio}, random_state=42\n"
                ")"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4 — AugmentationAdvisor
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationAdvisor:
    """
    Recommends augmentation strategies for the dataset.

    Considers:
    - Task type (classification vs detection vs segmentation)
    - Dataset size (more augmentation for small datasets)
    - Domain-specific rules (medical, satellite, documents)
    - Expected data multiplication factor
    """

    # Domain-specific restrictions
    DOMAIN_RESTRICTIONS = {
        "medical": [
            "Avoid aggressive color changes — preserve clinical colors",
            "Avoid heavy rotation — anatomical orientation matters",
            "Blur only slightly — don't lose diagnostic details",
        ],
        "satellite": [
            "RandomVerticalFlip is safe — no top/bottom orientation",
            "RandomRotation 90° is safe — rotation-invariant",
            "Heavy zoom may remove context",
        ],
        "documents": [
            "Avoid rotation — text must be readable",
            "Avoid heavy color changes — OCR depends on contrast",
            "Perspective transform can help with scanned docs",
        ],
        "faces": [
            "Avoid RandomVerticalFlip — faces are not upside down",
            "Heavy occlusion (CutOut) may remove key features",
            "LandmarkJitter is useful for face alignment tasks",
        ],
    }

    def __init__(self):
        pass

    def recommend(
        self,
        n_total: int,
        task_type: TaskType = TaskType.IMAGE_CLASSIFICATION,
        image_domain: str = "general",
        class_counts: Optional[Dict[str, int]] = None,
        is_color: bool = True,
    ) -> AugmentationPlan:
        """
        Recommend augmentation plan.

        Parameters
        ----------
        n_total       : total dataset size
        task_type     : classification / detection / segmentation
        image_domain  : general / medical / satellite / documents / faces
        class_counts  : for imbalance-specific advice
        is_color      : False if grayscale images
        """
        task_key = "classification"
        if task_type == TaskType.OBJECT_DETECTION:
            task_key = "detection"
        elif task_type == TaskType.IMAGE_SEGMENTATION:
            task_key = "segmentation"
        elif task_type == TaskType.TEXT_CLASSIFICATION:
            task_key = "text_classification"

        safe_transforms  = list(SAFE_AUGMENTATIONS.get(task_key, []))
        risky_transforms = list(RISKY_AUGMENTATIONS.get(task_key, []))
        not_recommended  : List[str] = []

        # Remove color-based transforms for grayscale
        if not is_color:
            safe_transforms = [
                t for t in safe_transforms
                if "Color" not in t and "Grayscale" not in t
                and "Jitter" not in t
            ]
            not_recommended.append(
                "ColorJitter — grayscale images have no color channels"
            )

        # Apply domain restrictions
        domain_notes = self.DOMAIN_RESTRICTIONS.get(image_domain, [])
        if domain_notes:
            risky_transforms.extend(domain_notes)

        # Size-based augmentation intensity
        if n_total < TOTAL_POOR:
            multiplier = 10.0
            notes = (
                f"Very small dataset ({n_total} samples). "
                "Apply aggressive augmentation — target 5-10x multiplication."
            )
        elif n_total < TOTAL_FAIR:
            multiplier = 5.0
            notes = (
                f"Small dataset ({n_total} samples). "
                "Apply moderate augmentation — target 3-5x multiplication."
            )
        elif n_total < TOTAL_GOOD:
            multiplier = 3.0
            notes = (
                f"Medium dataset ({n_total} samples). "
                "Standard augmentation — target 2-3x multiplication."
            )
        else:
            multiplier = 1.5
            notes = (
                f"Large dataset ({n_total} samples). "
                "Light augmentation — just prevent overfitting."
            )

        # Add advanced augmentation for small datasets
        if n_total < TOTAL_FAIR and task_key == "classification":
            safe_transforms.extend([
                "RandomErasing(p=0.3)",
                "TrivialAugmentWide (torchvision)",
            ])
            risky_transforms.append(
                "CutMix — requires n > 500 per class, handle carefully"
            )

        # Class imbalance augmentation
        if class_counts:
            ratio = _imbalance_ratio(list(class_counts.values()))
            if ratio > IMBALANCE_MODERATE:
                safe_transforms.append(
                    "Apply heavier augmentation to minority classes "
                    f"(imbalance ratio: {ratio:.1f}:1)"
                )

        library_code = self._generate_albumentations_code(
            task_key, n_total, is_color
        )

        return AugmentationPlan(
            task=task_key,
            safe_transforms=safe_transforms,
            risky_transforms=risky_transforms,
            not_recommended=not_recommended,
            expected_multiplier=multiplier,
            library_code=library_code,
            notes=notes,
        )

    def _generate_albumentations_code(
        self, task: str, n_total: int, is_color: bool
    ) -> str:
        """Generate albumentations transform pipeline code."""
        code_lines = [
            "import albumentations as A",
            "from albumentations.pytorch import ToTensorV2",
            "",
            "train_transform = A.Compose([",
            "    A.RandomHorizontalFlip(p=0.5),",
        ]

        if is_color:
            code_lines.append(
                "    A.ColorJitter(brightness=0.2, contrast=0.2, "
                "saturation=0.2, hue=0.1, p=0.4),"
            )

        if n_total < TOTAL_FAIR:
            code_lines.extend([
                "    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,",
                "                      rotate_limit=15, p=0.5),",
                "    A.GaussianBlur(blur_limit=(3, 5), p=0.2),",
                "    A.RandomBrightnessContrast(p=0.3),",
                "    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),",
            ])
        else:
            code_lines.extend([
                "    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05,",
                "                      rotate_limit=10, p=0.3),",
            ])

        code_lines.extend([
            "    A.Normalize(mean=[0.485, 0.456, 0.406],",
            "                std=[0.229, 0.224, 0.225]),",
            "    ToTensorV2(),",
            "])",
            "",
            "val_transform = A.Compose([",
            "    A.Normalize(mean=[0.485, 0.456, 0.406],",
            "                std=[0.229, 0.224, 0.225]),",
            "    ToTensorV2(),",
            "])",
        ])

        return "\n".join(code_lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5 — HardwareEstimator
# ─────────────────────────────────────────────────────────────────────────────

class HardwareEstimator:
    """
    Estimates hardware requirements for training.

    Considers:
    - Model size (parameters)
    - Dataset size
    - Batch size
    - Precision (fp32 vs fp16)
    """

    def __init__(self):
        self._has_psutil = HAS_PSUTIL
        self._has_torch  = HAS_TORCH

    def estimate(
        self,
        n_total: int,
        model_name: str = "resnet50",
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        precision: str = "fp32",
        n_channels: int = 3,
    ) -> HardwareRequirements:
        """
        Estimate hardware requirements.

        Parameters
        ----------
        n_total     : total images
        model_name  : model architecture name
        image_size  : (width, height)
        batch_size  : training batch size
        precision   : 'fp32' or 'fp16' (mixed precision)
        n_channels  : image channels
        """
        notes : List[str] = []

        params         = MODEL_PARAMS.get(model_name, 25_000_000)
        bytes_per_param = 4 if precision == "fp32" else 2
        n_workers       = min(4, os.cpu_count() or 2)

        # RAM estimation
        image_bytes       = image_size[0] * image_size[1] * n_channels * 4
        batch_image_bytes = image_bytes * batch_size
        # Model in RAM: params × 4 bytes (fp32)
        model_ram_mb = (params * 4) / (1024 ** 2)
        # DataLoader cache
        loader_ram_mb = (batch_image_bytes * n_workers * 2) / (1024 ** 2)
        # Misc overhead
        overhead_mb = 512

        total_ram_mb          = model_ram_mb + loader_ram_mb + overhead_mb
        recommended_ram_gb    = max(8.0, math.ceil(total_ram_mb / 1024) * 2)
        min_ram_gb            = max(4.0, math.ceil(total_ram_mb / 1024))

        # VRAM estimation
        activation_factor  = 2.5   # rough multiplier for activations
        model_vram_mb      = (params * bytes_per_param) / (1024 ** 2)
        batch_vram_mb      = (batch_image_bytes * activation_factor) / (1024 ** 2)
        grad_vram_mb       = model_vram_mb * 1.2   # gradients similar size to model
        optimizer_vram_mb  = model_vram_mb * 2.0   # Adam optimizer states

        total_vram_mb        = model_vram_mb + batch_vram_mb + grad_vram_mb + optimizer_vram_mb
        min_vram_gb          = max(2.0, math.ceil(total_vram_mb / 1024))
        recommended_vram_gb  = min_vram_gb * 2

        # GPU decision
        gpu_required    = n_total > 5000 or params > 50_000_000
        gpu_recommended = "NVIDIA T4 (16GB) or better"

        if params > 100_000_000:
            gpu_recommended = "NVIDIA A100 (40GB) for large models"
        elif params > 50_000_000:
            gpu_recommended = "NVIDIA V100 (32GB) or RTX 3090 (24GB)"
        elif params < 10_000_000:
            gpu_recommended = "Any GPU with 4GB+ VRAM"

        # Disk estimation
        image_bytes_raw  = image_size[0] * image_size[1] * n_channels
        total_disk_bytes = n_total * image_bytes_raw
        checkpoint_mb    = (model_vram_mb * 3)  # keep 3 checkpoints
        total_disk_gb    = (total_disk_bytes / (1024 ** 3)) + (checkpoint_mb / 1024)

        # Notes
        if precision == "fp16":
            notes.append(
                "Mixed precision (fp16) halves VRAM usage. "
                "Use torch.cuda.amp.autocast() in your training loop."
            )
        if n_total > 100_000:
            notes.append(
                "Large dataset — consider using multiple data loader workers (num_workers=4+) "
                "and prefetching (pin_memory=True)."
            )
        if params > 100_000_000:
            notes.append(
                "Very large model — consider gradient checkpointing to reduce VRAM: "
                "torch.utils.checkpoint.checkpoint_sequential()"
            )
        if not gpu_required:
            notes.append(
                "Small model + small dataset — CPU training is feasible "
                "but will be slow. A GPU will be 10-100× faster."
            )

        # Detect current system hardware
        system_info = self._detect_system()
        if system_info:
            details = system_info
            if details.get("gpu_available"):
                notes.append(
                    f"Detected GPU: {details.get('gpu_name', 'Unknown')} "
                    f"({details.get('gpu_vram_gb', '?')}GB VRAM)"
                )
            else:
                notes.append("No GPU detected on current system.")

        return HardwareRequirements(
            min_ram_gb=min_ram_gb,
            recommended_ram_gb=recommended_ram_gb,
            min_vram_gb=min_vram_gb,
            recommended_vram_gb=recommended_vram_gb,
            cpu_cores_recommended=n_workers + 2,
            gpu_required=gpu_required,
            gpu_recommended=gpu_recommended,
            estimated_disk_gb=round(total_disk_gb, 2),
            notes=notes,
        )

    def _detect_system(self) -> Dict[str, Any]:
        """Detect current system hardware."""
        info: Dict[str, Any] = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }

        if self._has_psutil:
            try:
                mem = psutil.virtual_memory()
                info["system_ram_gb"]    = round(mem.total / (1024 ** 3), 1)
                info["available_ram_gb"] = round(mem.available / (1024 ** 3), 1)
                info["cpu_count"]        = psutil.cpu_count(logical=False)
                info["cpu_count_logical"] = psutil.cpu_count(logical=True)
            except Exception:
                pass

        if self._has_torch:
            try:
                info["gpu_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    info["gpu_name"]     = torch.cuda.get_device_name(0)
                    info["gpu_vram_gb"]  = round(
                        torch.cuda.get_device_properties(0).total_memory
                        / (1024 ** 3), 1
                    )
                    info["gpu_count"]    = torch.cuda.device_count()
                else:
                    info["gpu_available"] = False
            except Exception:
                info["gpu_available"] = False
        else:
            info["gpu_available"] = False

        return info


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6 — TrainingTimeEstimator
# ─────────────────────────────────────────────────────────────────────────────

class TrainingTimeEstimator:
    """
    Estimates training time on different hardware configurations.

    Formula:
        time = (n_samples × n_epochs) / (throughput × batch_size) × overhead
    """

    # Overhead multipliers per model type
    OVERHEAD = {
        "classification": 1.0,
        "detection":       2.5,    # bbox regression + classification heads
        "segmentation":    3.0,    # pixel-level output is expensive
        "multi_modal":     2.0,
    }

    def __init__(self):
        pass

    def estimate(
        self,
        n_train: int,
        n_epochs: int,
        batch_size: int,
        model_name: str = "resnet50",
        task_type: TaskType = TaskType.IMAGE_CLASSIFICATION,
        image_size: Tuple[int, int] = (224, 224),
    ) -> TrainingTimeEstimate:
        """
        Estimate training time.

        Parameters
        ----------
        n_train     : number of training samples
        n_epochs    : number of training epochs
        batch_size  : training batch size
        model_name  : model architecture
        task_type   : affects overhead
        image_size  : affects throughput (larger = slower)
        """
        tips : List[str] = []

        # Determine task overhead
        task_key = "classification"
        if task_type == TaskType.OBJECT_DETECTION:
            task_key = "detection"
        elif task_type == TaskType.IMAGE_SEGMENTATION:
            task_key = "segmentation"
        elif task_type == TaskType.MULTI_MODAL:
            task_key = "multi_modal"

        overhead = self.OVERHEAD.get(task_key, 1.0)

        # Image size penalty (relative to 224×224)
        base_pixels   = 224 * 224
        curr_pixels   = image_size[0] * image_size[1]
        size_penalty  = curr_pixels / base_pixels

        # Batches per epoch
        batches_per_epoch = math.ceil(n_train / batch_size)
        total_batches     = batches_per_epoch * n_epochs

        estimates: Dict[str, str] = {}

        for hw_name, hw_throughput in GPU_THROUGHPUT.items():
            # Effective throughput considering model size
            params = MODEL_PARAMS.get(model_name, 25_000_000)
            # Large models are slower proportionally
            model_penalty = 1.0
            if params > 100_000_000:
                model_penalty = 2.0
            elif params > 50_000_000:
                model_penalty = 1.5
            elif params < 5_000_000:
                model_penalty = 0.8

            effective_throughput = (
                hw_throughput
                / size_penalty
                / model_penalty
                * overhead ** -1
            )
            effective_throughput = max(effective_throughput, 1.0)

            total_images_processed = n_train * n_epochs
            seconds = total_images_processed / effective_throughput
            estimates[hw_name] = _format_duration(seconds)

        # Determine bottleneck
        if image_size[0] > 512:
            bottleneck = "Image resolution — very high resolution slows down I/O and GPU"
        elif task_type == TaskType.IMAGE_SEGMENTATION:
            bottleneck = "Per-pixel prediction — segmentation is compute-heavy"
        elif n_train > 100_000:
            bottleneck = "Data loading — use num_workers=4+ and pin_memory=True"
        else:
            bottleneck = "Model forward/backward pass"

        # Tips
        if n_epochs > 100:
            tips.append(
                "Early stopping (patience=10) recommended — "
                "prevents overfitting and saves time."
            )
        if batch_size < 16:
            tips.append(
                f"Batch size {batch_size} is small. "
                "Try gradient accumulation to simulate larger batches."
            )
        if batch_size > 128:
            tips.append(
                f"Large batch size ({batch_size}) — "
                "may need learning rate warmup."
            )
        if image_size[0] > 384:
            tips.append(
                f"Large image size ({image_size[0]}×{image_size[1]}). "
                "Consider resizing to 224 or 256 for faster iteration."
            )

        tips.append(
            "Use a learning rate scheduler (CosineAnnealingLR or ReduceLROnPlateau)."
        )

        return TrainingTimeEstimate(
            n_samples=n_train,
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=model_name,
            estimates=estimates,
            bottleneck=bottleneck,
            tips=tips,
        )

    def recommend_epochs(
        self,
        n_train: int,
        task_type: TaskType,
        use_transfer_learning: bool = True,
    ) -> int:
        """Recommend number of epochs."""
        if use_transfer_learning:
            if n_train < 500:
                return 30
            elif n_train < 2000:
                return 50
            elif n_train < 10000:
                return 30
            else:
                return 20
        else:
            if n_train < 500:
                return 100
            elif n_train < 5000:
                return 150
            else:
                return 100

    def recommend_batch_size(
        self,
        n_train: int,
        vram_gb: float = 8.0,
        image_size: Tuple[int, int] = (224, 224),
    ) -> int:
        """Recommend batch size based on VRAM."""
        pixels_mb = (image_size[0] * image_size[1] * 3 * 4) / (1024 ** 2)
        max_batch = int((vram_gb * 1024 * 0.5) / (pixels_mb * 2.5))
        max_batch = max(1, min(max_batch, 256))

        # Round to power of 2
        batch = 1
        while batch * 2 <= max_batch:
            batch *= 2

        return max(8, batch)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7 — ModelRecommender
# ─────────────────────────────────────────────────────────────────────────────

class ModelRecommender:
    """
    Recommends the most suitable model architectures.

    Decision tree based on:
    - Task type
    - Dataset size
    - Number of classes
    - Hardware constraints
    - Transfer learning availability
    """

    def recommend(
        self,
        task_type: TaskType,
        n_total: int,
        n_classes: int,
        use_transfer_learning: bool = True,
        max_vram_gb: float = 8.0,
        prefer_lightweight: bool = False,
    ) -> List[ModelRecommendation]:
        """
        Recommend models for the given dataset configuration.

        Returns a ranked list of ModelRecommendation (best first).
        """
        if task_type in (
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.IMAGE_REGRESSION,
        ):
            return self._recommend_classification(
                n_total, n_classes, use_transfer_learning,
                max_vram_gb, prefer_lightweight
            )
        elif task_type == TaskType.OBJECT_DETECTION:
            return self._recommend_detection(n_total, max_vram_gb)
        elif task_type == TaskType.IMAGE_SEGMENTATION:
            return self._recommend_segmentation(n_total, max_vram_gb)
        elif task_type == TaskType.TEXT_CLASSIFICATION:
            return self._recommend_text(n_total, n_classes)
        elif task_type == TaskType.MULTI_MODAL:
            return self._recommend_multimodal(n_total)
        else:
            return self._recommend_classification(
                n_total, n_classes, use_transfer_learning,
                max_vram_gb, prefer_lightweight
            )

    def _recommend_classification(
        self,
        n_total: int,
        n_classes: int,
        use_transfer_learning: bool,
        max_vram_gb: float,
        prefer_lightweight: bool,
    ) -> List[ModelRecommendation]:
        recs: List[ModelRecommendation] = []

        if n_total < 1000 or prefer_lightweight:
            recs.append(ModelRecommendation(
                name="MobileNetV2",
                architecture="MobileNetV2",
                task="image_classification",
                reason=(
                    f"Small dataset ({n_total} samples). "
                    "MobileNetV2 is lightweight and works well with transfer learning "
                    "even on tiny datasets."
                ),
                params=MODEL_PARAMS["mobilenet_v2"],
                pretrained=True,
                pretrained_on="ImageNet-1K",
                min_samples_needed=50,
                expected_accuracy_range=(0.70, 0.92),
                training_difficulty="easy",
                library="torchvision",
                code_snippet=(
                    "import torchvision.models as models\n"
                    "model = models.mobilenet_v2(pretrained=True)\n"
                    f"model.classifier[1] = nn.Linear(1280, {n_classes})"
                ),
            ))

        if n_total < 5000:
            recs.append(ModelRecommendation(
                name="EfficientNet-B0",
                architecture="EfficientNet-B0",
                task="image_classification",
                reason=(
                    "Best accuracy/efficiency trade-off for small-medium datasets. "
                    "State-of-the-art performance with few parameters."
                ),
                params=MODEL_PARAMS["efficientnet_b0"],
                pretrained=True,
                pretrained_on="ImageNet-1K",
                min_samples_needed=100,
                expected_accuracy_range=(0.75, 0.95),
                training_difficulty="easy",
                library="torchvision",
                code_snippet=(
                    "from torchvision.models import efficientnet_b0\n"
                    "model = efficientnet_b0(pretrained=True)\n"
                    f"model.classifier[1] = nn.Linear(1280, {n_classes})"
                ),
            ))

        if n_total >= 1000:
            recs.append(ModelRecommendation(
                name="ResNet50",
                architecture="ResNet-50",
                task="image_classification",
                reason=(
                    "Classic workhorse — excellent performance, "
                    "well understood, extensive community support."
                ),
                params=MODEL_PARAMS["resnet50"],
                pretrained=True,
                pretrained_on="ImageNet-1K",
                min_samples_needed=200,
                expected_accuracy_range=(0.78, 0.96),
                training_difficulty="medium",
                library="torchvision",
                code_snippet=(
                    "from torchvision.models import resnet50\n"
                    "model = resnet50(pretrained=True)\n"
                    f"model.fc = nn.Linear(2048, {n_classes})"
                ),
            ))

        if n_total >= 5000 and max_vram_gb >= 8:
            recs.append(ModelRecommendation(
                name="ViT-Base/16",
                architecture="Vision Transformer",
                task="image_classification",
                reason=(
                    "Transformer-based — best accuracy on large datasets. "
                    "Requires more data than CNNs but achieves state-of-the-art results."
                ),
                params=MODEL_PARAMS["vit_base"],
                pretrained=True,
                pretrained_on="ImageNet-21K",
                min_samples_needed=1000,
                expected_accuracy_range=(0.85, 0.98),
                training_difficulty="hard",
                library="huggingface/timm",
                code_snippet=(
                    "import timm\n"
                    f"model = timm.create_model('vit_base_patch16_224', "
                    f"pretrained=True, num_classes={n_classes})"
                ),
            ))

        return recs[:3]  # Return top 3

    def _recommend_detection(
        self, n_total: int, max_vram_gb: float
    ) -> List[ModelRecommendation]:
        recs: List[ModelRecommendation] = []

        recs.append(ModelRecommendation(
            name="YOLOv8n (nano)",
            architecture="YOLOv8",
            task="object_detection",
            reason=(
                "Fastest inference, smallest size. "
                "Best for real-time applications and limited hardware."
            ),
            params=MODEL_PARAMS["yolov8n"],
            pretrained=True,
            pretrained_on="COCO",
            min_samples_needed=300,
            expected_accuracy_range=(0.50, 0.80),
            training_difficulty="easy",
            library="ultralytics",
            code_snippet=(
                "from ultralytics import YOLO\n"
                "model = YOLO('yolov8n.pt')\n"
                "model.train(data='dataset.yaml', epochs=100)"
            ),
        ))

        if n_total >= 1000 and max_vram_gb >= 8:
            recs.append(ModelRecommendation(
                name="YOLOv8m (medium)",
                architecture="YOLOv8",
                task="object_detection",
                reason="Best balance of speed and accuracy for medium datasets.",
                params=MODEL_PARAMS["yolov8m"],
                pretrained=True,
                pretrained_on="COCO",
                min_samples_needed=500,
                expected_accuracy_range=(0.60, 0.88),
                training_difficulty="medium",
                library="ultralytics",
                code_snippet=(
                    "from ultralytics import YOLO\n"
                    "model = YOLO('yolov8m.pt')\n"
                    "model.train(data='dataset.yaml', epochs=100)"
                ),
            ))

        return recs

    def _recommend_segmentation(
        self, n_total: int, max_vram_gb: float
    ) -> List[ModelRecommendation]:
        recs: List[ModelRecommendation] = []

        recs.append(ModelRecommendation(
            name="U-Net",
            architecture="U-Net (encoder-decoder)",
            task="image_segmentation",
            reason=(
                "Classic segmentation architecture. "
                "Works well with small medical/scientific image datasets."
            ),
            params=31_000_000,
            pretrained=True,
            pretrained_on="ImageNet (encoder only)",
            min_samples_needed=100,
            expected_accuracy_range=(0.70, 0.92),
            training_difficulty="medium",
            library="segmentation-models-pytorch",
            code_snippet=(
                "import segmentation_models_pytorch as smp\n"
                "model = smp.Unet(\n"
                "    encoder_name='resnet34',\n"
                "    encoder_weights='imagenet',\n"
                f"    classes={1},  # per class or 1 for binary\n"
                ")"
            ),
        ))

        if n_total >= 500:
            recs.append(ModelRecommendation(
                name="DeepLabV3+",
                architecture="DeepLabV3+ with ResNet",
                task="image_segmentation",
                reason="State-of-the-art semantic segmentation for larger datasets.",
                params=41_000_000,
                pretrained=True,
                pretrained_on="ImageNet + COCO",
                min_samples_needed=300,
                expected_accuracy_range=(0.80, 0.95),
                training_difficulty="hard",
                library="torchvision / segmentation-models-pytorch",
                code_snippet=(
                    "import segmentation_models_pytorch as smp\n"
                    "model = smp.DeepLabV3Plus(\n"
                    "    encoder_name='resnet50',\n"
                    "    encoder_weights='imagenet',\n"
                    "    classes=num_classes,\n"
                    ")"
                ),
            ))

        return recs

    def _recommend_text(
        self, n_total: int, n_classes: int
    ) -> List[ModelRecommendation]:
        recs: List[ModelRecommendation] = []

        if n_total < 1000:
            recs.append(ModelRecommendation(
                name="TF-IDF + LogisticRegression",
                architecture="Traditional ML",
                task="text_classification",
                reason=(
                    f"Very small dataset ({n_total} samples). "
                    "Deep learning needs more data — start with classical ML."
                ),
                params=0,
                pretrained=False,
                pretrained_on="N/A",
                min_samples_needed=50,
                expected_accuracy_range=(0.65, 0.88),
                training_difficulty="easy",
                library="scikit-learn",
                code_snippet=(
                    "from sklearn.pipeline import Pipeline\n"
                    "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                    "from sklearn.linear_model import LogisticRegression\n\n"
                    "pipeline = Pipeline([\n"
                    "    ('tfidf', TfidfVectorizer(max_features=10000)),\n"
                    "    ('clf', LogisticRegression(max_iter=1000)),\n"
                    "])"
                ),
            ))

        recs.append(ModelRecommendation(
            name="DistilBERT",
            architecture="Transformer (DistilBERT)",
            task="text_classification",
            reason=(
                "40% smaller than BERT, 60% faster, retains 97% performance. "
                "Best balance for text classification."
            ),
            params=MODEL_PARAMS["distilbert"],
            pretrained=True,
            pretrained_on="BookCorpus + Wikipedia",
            min_samples_needed=500,
            expected_accuracy_range=(0.82, 0.95),
            training_difficulty="medium",
            library="huggingface/transformers",
            code_snippet=(
                "from transformers import DistilBertForSequenceClassification\n"
                f"model = DistilBertForSequenceClassification.from_pretrained(\n"
                f"    'distilbert-base-uncased', num_labels={n_classes}\n"
                ")"
            ),
        ))

        return recs

    def _recommend_multimodal(self, n_total: int) -> List[ModelRecommendation]:
        return [
            ModelRecommendation(
                name="CLIP (ViT-B/32)",
                architecture="CLIP — Contrastive Language-Image Pretraining",
                task="multi_modal",
                reason=(
                    "Zero-shot and few-shot capability. "
                    "Jointly understands images and text. "
                    "Excellent for image-text matching tasks."
                ),
                params=MODEL_PARAMS["clip_vit_b32"],
                pretrained=True,
                pretrained_on="400M image-text pairs (OpenAI)",
                min_samples_needed=100,
                expected_accuracy_range=(0.75, 0.95),
                training_difficulty="medium",
                library="openai/clip",
                code_snippet=(
                    "import clip\n"
                    "model, preprocess = clip.load('ViT-B/32', device='cuda')"
                ),
            )
        ]


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 8 — TransferLearningAdvisor
# ─────────────────────────────────────────────────────────────────────────────

class TransferLearningAdvisor:
    """
    Advises on transfer learning strategy.

    Decides:
    - Whether to use transfer learning (almost always yes)
    - How many layers to freeze
    - Learning rate strategy (feature extraction vs fine-tuning)
    - Which pretrained model to start from
    """

    def advise(
        self,
        n_total: int,
        task_type: TaskType,
        domain_similarity: str = "moderate",
        n_classes: int = 10,
    ) -> Dict[str, Any]:
        """
        Produce transfer learning advice.

        Parameters
        ----------
        n_total           : total samples
        task_type         : image/text task
        domain_similarity : 'high' (natural photos), 'moderate', 'low' (medical/satellite)
        n_classes         : number of output classes
        """
        advice: Dict[str, Any] = {}

        # Should we use TL at all?
        use_tl = n_total < 50_000 or domain_similarity in ("high", "moderate")
        advice["use_transfer_learning"]     = use_tl
        advice["domain_similarity"]         = domain_similarity

        if not use_tl:
            advice["strategy"]              = "train_from_scratch"
            advice["reason"]                = (
                "Large dataset with high domain specificity — "
                "training from scratch may outperform transfer learning."
            )
            advice["recommended_lr"]        = 0.01
            advice["freeze_layers"]         = "none"
            return advice

        # Determine strategy based on size + similarity
        if n_total < 500 and domain_similarity == "high":
            strategy        = "feature_extraction"
            freeze          = "all_except_last"
            lr              = 0.001
            epochs_phase1   = 10
            epochs_phase2   = None
            reason = (
                "Very small dataset + similar domain. "
                "Freeze all pretrained layers, only train the classifier head."
            )
        elif n_total < 500 and domain_similarity in ("moderate", "low"):
            strategy        = "fine_tune_top_layers"
            freeze          = "first_50_percent"
            lr              = 0.0001
            epochs_phase1   = 10
            epochs_phase2   = 20
            reason = (
                "Small dataset + different domain. "
                "Freeze early layers (general features), "
                "fine-tune top layers (domain-specific features)."
            )
        elif n_total < 5000:
            strategy        = "progressive_unfreezing"
            freeze          = "gradual"
            lr              = 0.0001
            epochs_phase1   = 5
            epochs_phase2   = 20
            reason = (
                "Medium dataset. Use progressive unfreezing: "
                "start with head only, then gradually unfreeze deeper layers."
            )
        else:
            strategy        = "full_fine_tune"
            freeze          = "none"
            lr              = 0.00001
            epochs_phase1   = 0
            epochs_phase2   = 30
            reason = (
                "Sufficient data for full fine-tuning. "
                "Use a small learning rate to avoid catastrophic forgetting."
            )

        advice["strategy"]          = strategy
        advice["reason"]            = reason
        advice["freeze_layers"]     = freeze
        advice["recommended_lr"]    = lr
        advice["epochs_phase1"]     = epochs_phase1
        advice["epochs_phase2"]     = epochs_phase2
        advice["lr_schedule"]       = "CosineAnnealingLR or ReduceLROnPlateau"

        # Warmup
        advice["use_warmup"]        = lr < 0.0005
        advice["warmup_epochs"]     = 3 if lr < 0.0005 else 0

        # Code hint
        advice["code_hint"] = self._generate_code_hint(
            strategy=strategy,
            lr=lr,
            n_classes=n_classes,
        )

        # Pretrained source recommendation
        if domain_similarity == "low":
            if task_type == TaskType.IMAGE_CLASSIFICATION:
                advice["pretrained_source"] = (
                    "Consider domain-specific pretrained weights:\n"
                    "- Medical: torchxrayvision, pathology models\n"
                    "- Satellite: torchgeo\n"
                    "- Documents: DocFormer, LayoutLM"
                )
        else:
            advice["pretrained_source"] = "ImageNet-1K or ImageNet-21K (standard)"

        return advice

    def _generate_code_hint(
        self, strategy: str, lr: float, n_classes: int
    ) -> str:
        """Generate PyTorch code hint for the strategy."""
        if strategy == "feature_extraction":
            return (
                "# Freeze all layers\n"
                "for param in model.parameters():\n"
                "    param.requires_grad = False\n\n"
                "# Unfreeze classifier head only\n"
                "for param in model.fc.parameters():  # adjust layer name\n"
                "    param.requires_grad = True\n\n"
                f"optimizer = torch.optim.Adam(model.fc.parameters(), lr={lr})"
            )
        elif strategy == "progressive_unfreezing":
            return (
                "# Phase 1: Train head only\n"
                "for param in model.parameters():\n"
                "    param.requires_grad = False\n"
                "for param in model.fc.parameters():\n"
                "    param.requires_grad = True\n\n"
                "# Train for 5 epochs, then unfreeze all\n"
                "for param in model.parameters():\n"
                "    param.requires_grad = True\n"
                f"optimizer = torch.optim.Adam(model.parameters(), lr={lr})"
            )
        else:
            return (
                "# Full fine-tune\n"
                "for param in model.parameters():\n"
                "    param.requires_grad = True\n\n"
                f"optimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay=1e-4)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 9 — LeakageRiskAssessor
# ─────────────────────────────────────────────────────────────────────────────

class LeakageRiskAssessor:
    """
    Assesses the risk of data leakage between train/val/test splits.

    Checks:
    - Duplicate images across splits
    - Near-duplicate images (perceptual hash similarity)
    - Label leakage (test labels visible during training)
    - Temporal leakage (future data in training set)
    - Patient/subject leakage (same subject in train and test)
    """

    def __init__(self):
        self._has_pil = HAS_PIL

    def assess(
        self,
        image_hashes: Optional[Dict[str, List[str]]] = None,
        has_subjects: bool = False,
        subject_ids: Optional[Dict[str, List[str]]] = None,
        is_temporal: bool = False,
        timestamps: Optional[List] = None,
    ) -> DimensionScore:
        """
        Assess data leakage risk.

        Parameters
        ----------
        image_hashes  : {split_name: [hash1, hash2, ...]}
        has_subjects  : dataset has multiple images per subject (face, medical)
        subject_ids   : {split_name: [subject_id1, ...]}
        is_temporal   : dataset is time-series
        timestamps    : list of timestamps (for temporal check)
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}
        score       = 100.0

        leakage_found = False

        # Check 1: Exact duplicate images across splits
        if image_hashes and len(image_hashes) >= 2:
            splits      = list(image_hashes.keys())
            train_hashes = set(image_hashes.get("train", []))

            for split_name, hashes in image_hashes.items():
                if split_name == "train":
                    continue
                overlap = train_hashes.intersection(set(hashes))
                if overlap:
                    leakage_found = True
                    pct = _safe_divide(len(overlap), len(hashes)) * 100
                    issues.append(
                        f"BLOCKING: {len(overlap)} exact duplicate images "
                        f"found between train and {split_name} "
                        f"({pct:.1f}% of {split_name} set). "
                        f"Remove duplicates before training."
                    )
                    score -= 40.0
                    details[f"train_{split_name}_duplicates"] = len(overlap)

        # Check 2: Subject leakage
        if has_subjects and subject_ids:
            train_subjects = set(subject_ids.get("train", []))
            for split_name, sids in subject_ids.items():
                if split_name == "train":
                    continue
                overlap = train_subjects.intersection(set(sids))
                if overlap:
                    leakage_found = True
                    issues.append(
                        f"BLOCKING: {len(overlap)} subjects appear in both "
                        f"train and {split_name} sets. "
                        "This causes inflated test metrics."
                    )
                    score -= 30.0

        # Check 3: Temporal leakage
        if is_temporal and timestamps:
            # Check if data was shuffled (timestamps not monotonic in splits)
            ts = [t for t in timestamps if t is not None]
            if ts:
                diffs = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
                n_backward = sum(1 for d in diffs if d < 0)
                backward_pct = _safe_divide(n_backward, len(diffs)) * 100
                if backward_pct > 5:
                    issues.append(
                        f"Temporal leakage risk: {backward_pct:.1f}% of samples "
                        "appear to be shuffled in a time-series dataset. "
                        "Use chronological split — never shuffle time-series."
                    )
                    score -= 25.0

        if not leakage_found:
            details["status"] = "No leakage detected"
            suggestions.append(
                "Run perceptual hash check to catch near-duplicate images "
                "that differ only in brightness/compression."
            )
        else:
            suggestions.append(
                "Use imagededup or PIL.ImageChops to find and remove "
                "duplicate/near-duplicate images across splits."
            )

        details["has_subject_info"]   = has_subjects
        details["is_temporal"]        = is_temporal
        details["leakage_detected"]   = leakage_found

        # General best practices
        details["best_practices"] = [
            "Always split BEFORE any augmentation.",
            "Use subject/patient-level split if dataset has multiple images per subject.",
            "For temporal data: never shuffle, use chronological split.",
            "Check image hashes across splits after splitting.",
        ]

        score = _clamp(score)

        return DimensionScore(
            name="Split Validity",
            score=round(score, 2),
            weight=0.10,
            weighted_score=round(score * 0.10, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def compute_image_hash(self, image_path: Union[str, Path]) -> Optional[str]:
        """Compute SHA256 hash of image content."""
        try:
            with open(image_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def compute_perceptual_hash(
        self, image_path: Union[str, Path], hash_size: int = 16
    ) -> Optional[str]:
        """
        Compute perceptual hash (pHash) for near-duplicate detection.
        Two images with Hamming distance < 10 are likely duplicates.
        """
        if not self._has_pil:
            return None
        try:
            img    = Image.open(image_path).convert("L").resize(
                (hash_size, hash_size), Image.LANCZOS
            )
            pixels = np.array(img).flatten()
            avg    = pixels.mean()
            bits   = "".join("1" if p > avg else "0" for p in pixels)
            return hex(int(bits, 2))[2:].zfill(hash_size ** 2 // 4)
        except Exception:
            return None

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Hamming distance between two hex hashes."""
        if len(hash1) != len(hash2):
            return float("inf")
        b1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        b2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        return sum(c1 != c2 for c1, c2 in zip(b1, b2))


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 10 — ReadinessScoreCalculator
# ─────────────────────────────────────────────────────────────────────────────

class ReadinessScoreCalculator:
    """
    Aggregates all dimension scores into a final readiness score.

    8 Dimensions with weights:
        Dataset Size        20%
        Class Balance       18%
        Data Quality        17%
        Label Quality       15%
        Split Validity      10%
        Augmentation        8%
        Hardware            7%
        Model Fit           5%

    Grade scale (matches quality_score.py):
        A: 90-100  Excellent 🟢
        B: 75-89   Good 🟡
        C: 60-74   Fair 🟠
        D: 45-59   Poor 🔴
        F:  0-44   Critical ⛔
    """

    DIMENSION_WEIGHTS = {
        "dataset_size":    0.20,
        "class_balance":   0.18,
        "data_quality":    0.17,
        "label_quality":   0.15,
        "split_validity":  0.10,
        "augmentation":    0.08,
        "hardware":        0.07,
        "model_fit":       0.05,
    }

    def __init__(self):
        pass

    def calculate(
        self,
        dimension_scores: Dict[str, DimensionScore],
        blocking_issues: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate overall readiness score.

        Parameters
        ----------
        dimension_scores : mapping of dimension_key → DimensionScore
        blocking_issues  : list of blocking issue strings
        """
        total_weighted = 0.0
        total_weight   = 0.0
        dim_summary    : Dict[str, Any] = {}

        for key, dim_score in dimension_scores.items():
            w = self.DIMENSION_WEIGHTS.get(key, dim_score.weight)
            weighted = dim_score.score * w
            total_weighted += weighted
            total_weight   += w

            dim_summary[key] = {
                "score":          round(dim_score.score, 2),
                "grade":          _score_to_grade(dim_score.score),
                "status":         dim_score.status,
                "weight":         round(w * 100, 1),
                "weighted_score": round(weighted, 2),
                "issues":         dim_score.issues,
                "suggestions":    dim_score.suggestions,
                "details":        dim_score.details,
            }

        overall = _safe_divide(total_weighted, total_weight)
        overall = _clamp(overall)

        # Blocking issues force max score down
        if blocking_issues:
            # Each blocking issue caps the score at 50
            overall = min(overall, 50.0)

        grade          = _score_to_grade(overall)
        ready_to_train = not blocking_issues and overall >= GRADE_C

        # Confidence: how complete is our information?
        coverage = _safe_divide(len(dimension_scores), len(self.DIMENSION_WEIGHTS))
        confidence = coverage * 100

        return {
            "overall_score":   round(overall, 2),
            "grade":           grade,
            "grade_emoji":     _grade_emoji(grade),
            "ready_to_train":  ready_to_train,
            "confidence":      round(confidence, 1),
            "dimensions":      dim_summary,
            "blocking_issues": blocking_issues,
        }

    def generate_score_table(
        self, dimension_scores: Dict[str, DimensionScore]
    ) -> str:
        """Generate ASCII table of dimension scores."""
        lines = [
            "┌─────────────────────────┬───────┬────────┬──────────┐",
            "│ Dimension               │ Score │ Grade  │ Weight   │",
            "├─────────────────────────┼───────┼────────┼──────────┤",
        ]
        for key, dim in dimension_scores.items():
            w     = self.DIMENSION_WEIGHTS.get(key, dim.weight)
            grade = _score_to_grade(dim.score)
            emoji = _grade_emoji(grade)
            lines.append(
                f"│ {dim.name:<23} │ {dim.score:>5.1f} │ {emoji} {grade:<5} │ {w*100:>6.1f}%  │"
            )
        lines.append(
            "└─────────────────────────┴───────┴────────┴──────────┘"
        )
        return "\n".join(lines)

    def get_priority_fixes(
        self, dimension_scores: Dict[str, DimensionScore]
    ) -> List[Dict[str, str]]:
        """
        Return prioritized list of fixes — worst dimension first.
        """
        fixes = []
        for key, dim in sorted(
            dimension_scores.items(),
            key=lambda x: x[1].score
        ):
            if dim.score < 75:
                fixes.append({
                    "dimension": dim.name,
                    "score":     str(round(dim.score, 1)),
                    "priority":  "critical" if dim.score < 45 else
                                 "high"     if dim.score < 60 else "medium",
                    "fixes":     dim.suggestions[:2],
                })
        return fixes


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 11 — MultiModalReadiness
# ─────────────────────────────────────────────────────────────────────────────

class MultiModalReadiness:
    """
    Analyzes readiness for multi-modal datasets (images + text).

    Checks:
    - Alignment between images and their text descriptions/labels
    - Completeness of each modality
    - Consistency of multi-modal pairs
    - Recommended multi-modal architectures
    """

    def analyze(
        self,
        n_image_samples: int,
        n_text_samples: int,
        n_aligned_pairs: int,
        image_quality_score: float = 80.0,
        text_quality_score: float  = 80.0,
        task_type: TaskType = TaskType.MULTI_MODAL,
    ) -> DimensionScore:
        """
        Analyze multi-modal alignment and readiness.

        Parameters
        ----------
        n_image_samples  : number of images
        n_text_samples   : number of text items
        n_aligned_pairs  : number of (image, text) pairs
        image_quality_score : 0-100 score from image_quality.py
        text_quality_score  : 0-100 score from text_quality.py
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}
        score       = 100.0

        total_possible = max(n_image_samples, n_text_samples)
        alignment_pct  = _safe_divide(n_aligned_pairs, total_possible) * 100

        details["n_image_samples"]      = n_image_samples
        details["n_text_samples"]       = n_text_samples
        details["n_aligned_pairs"]      = n_aligned_pairs
        details["alignment_percentage"] = round(alignment_pct, 2)
        details["image_quality_score"]  = image_quality_score
        details["text_quality_score"]   = text_quality_score

        # Alignment check
        if alignment_pct < 50:
            score -= 40.0
            issues.append(
                f"BLOCKING: Only {alignment_pct:.1f}% of samples have both "
                "image and text. Multi-modal models need aligned pairs."
            )
        elif alignment_pct < 80:
            score -= 20.0
            suggestions.append(
                f"Only {alignment_pct:.1f}% alignment. "
                "Fill missing text with captions or descriptions."
            )
        elif alignment_pct < 95:
            score -= 5.0
            suggestions.append(
                f"{100 - alignment_pct:.1f}% of samples missing one modality. "
                "Consider removing unaligned samples."
            )

        # Modality size mismatch
        size_diff_pct = abs(n_image_samples - n_text_samples) / max(total_possible, 1) * 100
        if size_diff_pct > 20:
            suggestions.append(
                f"Large size mismatch: {n_image_samples} images vs {n_text_samples} texts. "
                "Align or drop unmatched samples."
            )

        # Quality penalty
        avg_quality = (image_quality_score + text_quality_score) / 2
        if avg_quality < 60:
            score -= 20.0
            issues.append(
                f"Combined modality quality too low ({avg_quality:.1f}/100). "
                "Fix quality issues before training."
            )
        elif avg_quality < 75:
            score -= 10.0
            suggestions.append(
                f"Modality quality ({avg_quality:.1f}/100) could be improved."
            )

        # Size check for multi-modal
        if n_aligned_pairs < 100:
            score -= 30.0
            issues.append(
                f"Only {n_aligned_pairs} aligned pairs. "
                "Multi-modal models need at least 500-1000 pairs."
            )
        elif n_aligned_pairs < 500:
            score -= 10.0
            suggestions.append(
                f"Small aligned dataset ({n_aligned_pairs} pairs). "
                "Use CLIP zero-shot — avoid fine-tuning with so few pairs."
            )

        details["alignment_status"] = (
            "well-aligned"  if alignment_pct >= 95 else
            "mostly-aligned" if alignment_pct >= 80 else
            "partially-aligned" if alignment_pct >= 50 else
            "poorly-aligned"
        )

        details["recommended_approach"] = (
            "Full fine-tuning (CLIP, BLIP)" if n_aligned_pairs >= 1000
            else "Zero-shot CLIP inference" if n_aligned_pairs >= 100
            else "Collect more aligned pairs first"
        )

        score = _clamp(score)

        return DimensionScore(
            name="Multi-Modal Alignment",
            score=round(score, 2),
            weight=0.10,
            weighted_score=round(score * 0.10, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 12 — DataQualityBridge
# ─────────────────────────────────────────────────────────────────────────────

class DataQualityBridge:
    """
    Bridges image_quality.py and text_quality.py results
    into a training-readiness dimension score.

    Converts quality module outputs into a DimensionScore
    suitable for the readiness score aggregation.
    """

    def from_image_quality(
        self,
        quality_report: Dict[str, Any],
    ) -> DimensionScore:
        """
        Convert image_quality.py output to DimensionScore.

        Parameters
        ----------
        quality_report : dict from image_quality.py analyze() output
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}

        # Extract key metrics
        overall_quality  = quality_report.get("overall_quality_score", 75.0)
        corrupt_pct      = quality_report.get("corrupt_percentage", 0.0)
        blur_pct         = quality_report.get("blur_percentage", 0.0)
        dark_pct         = quality_report.get("dark_percentage", 0.0)
        duplicate_pct    = quality_report.get("duplicate_percentage", 0.0)
        low_res_pct      = quality_report.get("low_resolution_percentage", 0.0)

        details.update({
            "overall_quality_score": overall_quality,
            "corrupt_images_pct":    corrupt_pct,
            "blurry_images_pct":     blur_pct,
            "dark_images_pct":       dark_pct,
            "duplicate_images_pct":  duplicate_pct,
            "low_resolution_pct":    low_res_pct,
        })

        score = overall_quality

        if corrupt_pct > 5:
            issues.append(
                f"BLOCKING: {corrupt_pct:.1f}% of images are corrupt — remove them."
            )
            score = min(score, 40.0)
        elif corrupt_pct > 1:
            suggestions.append(
                f"{corrupt_pct:.1f}% corrupt images found. Clean before training."
            )

        if blur_pct > 30:
            issues.append(
                f"High blur rate: {blur_pct:.1f}% of images are blurry. "
                "Remove or threshold blur before training."
            )
            score = min(score, 60.0)
        elif blur_pct > 10:
            suggestions.append(
                f"{blur_pct:.1f}% blurry images. Consider blur threshold filtering."
            )

        if duplicate_pct > 10:
            issues.append(
                f"BLOCKING: {duplicate_pct:.1f}% duplicate images. "
                "Remove duplicates — they inflate metrics artificially."
            )
            score = min(score, 50.0)
        elif duplicate_pct > 3:
            suggestions.append(
                f"{duplicate_pct:.1f}% duplicate images found. "
                "Run image_cleaner.py deduplication."
            )

        if low_res_pct > 20:
            suggestions.append(
                f"{low_res_pct:.1f}% of images are low resolution. "
                "May affect model learning quality."
            )

        score = _clamp(score)

        return DimensionScore(
            name="Data Quality",
            score=round(score, 2),
            weight=0.17,
            weighted_score=round(score * 0.17, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def from_label_quality(
        self,
        label_report: Dict[str, Any],
    ) -> DimensionScore:
        """
        Convert label_quality.py output to DimensionScore.

        Parameters
        ----------
        label_report : dict from label_quality.py analyze() output
        """
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}

        missing_pct   = label_report.get("missing_labels_pct", 0.0)
        noise_rate    = label_report.get("noise_rate", 0.0)
        consistency   = label_report.get("consistency_score", 100.0)
        agreement     = label_report.get("inter_annotator_agreement", 1.0)
        n_labeled     = label_report.get("n_labeled", 0)
        n_total       = label_report.get("n_total", 1)

        details.update({
            "missing_labels_pct":        missing_pct,
            "noise_rate":                noise_rate,
            "consistency_score":         consistency,
            "inter_annotator_agreement": agreement,
            "labeled_pct":               round(_safe_divide(n_labeled, n_total) * 100, 2),
        })

        score = 100.0

        if missing_pct > 20:
            score -= 40.0
            issues.append(
                f"BLOCKING: {missing_pct:.1f}% of images have no labels. "
                "Cannot train supervised model."
            )
        elif missing_pct > 5:
            score -= 20.0
            suggestions.append(
                f"{missing_pct:.1f}% missing labels. "
                "Fill or remove unlabeled samples."
            )

        if noise_rate > 30:
            score -= 30.0
            issues.append(
                f"High label noise: {noise_rate:.1f}% estimated mislabeled. "
                "Clean labels — noise > 30% severely degrades performance."
            )
        elif noise_rate > 10:
            score -= 15.0
            suggestions.append(
                f"Label noise: {noise_rate:.1f}%. "
                "Use label smoothing or noisy-label training techniques."
            )

        if consistency < 70:
            score -= 20.0
            suggestions.append(
                f"Low label consistency: {consistency:.1f}%. "
                "Review labeling guidelines and re-annotate inconsistent samples."
            )

        if agreement < 0.7 and agreement > 0:
            score -= 15.0
            suggestions.append(
                f"Inter-annotator agreement: {agreement:.2f} (Cohen's κ). "
                "Target κ > 0.8 for reliable labels."
            )

        score = _clamp(score)

        return DimensionScore(
            name="Label Quality",
            score=round(score, 2),
            weight=0.15,
            weighted_score=round(score * 0.15, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def estimate_quality_without_reports(
        self,
        n_total: int,
        has_quality_check: bool = False,
        has_label_check: bool   = False,
    ) -> Tuple[DimensionScore, DimensionScore]:
        """
        Estimate quality scores when quality reports are not available.
        Returns conservative placeholder scores.
        """
        data_q_score  = 60.0 if not has_quality_check else 75.0
        label_q_score = 60.0 if not has_label_check   else 75.0

        note = "Quality not verified — run image_quality.py and label_quality.py for accurate score."

        data_quality = DimensionScore(
            name="Data Quality",
            score=data_q_score,
            weight=0.17,
            weighted_score=data_q_score * 0.17,
            status=_score_to_status(data_q_score),
            details={"note": note, "verified": False},
            issues=[],
            suggestions=["Run image_quality.py to get accurate quality score."],
        )

        label_quality = DimensionScore(
            name="Label Quality",
            score=label_q_score,
            weight=0.15,
            weighted_score=label_q_score * 0.15,
            status=_score_to_status(label_q_score),
            details={"note": note, "verified": False},
            issues=[],
            suggestions=["Run label_quality.py to assess label accuracy and noise."],
        )

        return data_quality, label_quality


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 13 — AugmentationScorer
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationScorer:
    """
    Scores the augmentation potential of the dataset.

    Considers:
    - Current dataset size (small = high augmentation potential)
    - Task type constraints
    - Whether augmentation has been applied
    - Effective dataset size after augmentation
    """

    def score(
        self,
        n_total: int,
        task_type: TaskType = TaskType.IMAGE_CLASSIFICATION,
        augmentation_applied: bool = False,
        augmentation_factor: float = 1.0,
        image_domain: str = "general",
    ) -> DimensionScore:
        """Score augmentation dimension."""
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}

        effective_n = int(n_total * augmentation_factor)
        potential_multiplier = self._estimate_max_multiplier(task_type, image_domain)

        details["current_n"]             = n_total
        details["effective_n"]           = effective_n
        details["augmentation_applied"]  = augmentation_applied
        details["augmentation_factor"]   = augmentation_factor
        details["max_potential_multiplier"] = potential_multiplier

        score = 0.0

        if augmentation_applied and augmentation_factor >= 3.0:
            score = 90.0
        elif augmentation_applied and augmentation_factor >= 1.5:
            score = 70.0
        elif not augmentation_applied and n_total >= TOTAL_EXCELLENT:
            score = 80.0
            suggestions.append(
                "Large dataset — light augmentation is sufficient. "
                "Apply RandomFlip + ColorJitter at minimum."
            )
        elif not augmentation_applied and n_total >= TOTAL_GOOD:
            score = 60.0
            suggestions.append(
                f"Medium dataset ({n_total}). "
                "Apply standard augmentation (3-5× multiplication recommended)."
            )
        elif not augmentation_applied and n_total >= TOTAL_FAIR:
            score = 40.0
            suggestions.append(
                f"Small dataset ({n_total}). "
                f"Heavy augmentation up to {potential_multiplier}× is strongly recommended."
            )
        else:
            score = 20.0
            issues.append(
                f"Very small dataset ({n_total} samples) with no augmentation. "
                f"Apply aggressive augmentation — you can reach {potential_multiplier}× the data."
            )

        details["recommended_action"] = (
            "Continue with current augmentation" if score >= 80
            else "Apply standard augmentation pipeline"  if score >= 60
            else "Apply aggressive augmentation + collect more data"
        )

        score = _clamp(score)

        return DimensionScore(
            name="Augmentation Potential",
            score=round(score, 2),
            weight=0.08,
            weighted_score=round(score * 0.08, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def _estimate_max_multiplier(
        self, task_type: TaskType, image_domain: str
    ) -> float:
        """Estimate maximum safe augmentation multiplier."""
        if image_domain == "medical":
            return 5.0
        elif image_domain == "documents":
            return 3.0
        elif task_type == TaskType.OBJECT_DETECTION:
            return 5.0
        elif task_type == TaskType.IMAGE_SEGMENTATION:
            return 4.0
        else:
            return 10.0


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 14 — HardwareScorer
# ─────────────────────────────────────────────────────────────────────────────

class HardwareScorer:
    """
    Scores hardware feasibility for training.

    A dataset that requires a GPU the user doesn't have
    still scores lower — the score reflects practical readiness.
    """

    def score(
        self,
        hw_requirements: HardwareRequirements,
        available_ram_gb: Optional[float] = None,
        available_vram_gb: Optional[float] = None,
        has_gpu: bool = False,
    ) -> DimensionScore:
        """Score hardware dimension."""
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}
        score       = 100.0

        details["required_ram_gb"]       = hw_requirements.min_ram_gb
        details["recommended_ram_gb"]    = hw_requirements.recommended_ram_gb
        details["required_vram_gb"]      = hw_requirements.min_vram_gb
        details["gpu_required"]          = hw_requirements.gpu_required
        details["gpu_recommended"]       = hw_requirements.gpu_recommended

        # RAM check
        if available_ram_gb is not None:
            details["available_ram_gb"] = available_ram_gb
            if available_ram_gb < hw_requirements.min_ram_gb:
                score -= 30.0
                issues.append(
                    f"Insufficient RAM: {available_ram_gb}GB available, "
                    f"{hw_requirements.min_ram_gb}GB required."
                )
            elif available_ram_gb < hw_requirements.recommended_ram_gb:
                score -= 10.0
                suggestions.append(
                    f"RAM is sufficient ({available_ram_gb}GB) but more "
                    f"({hw_requirements.recommended_ram_gb}GB) is recommended."
                )

        # GPU check
        details["has_gpu"] = has_gpu
        if hw_requirements.gpu_required and not has_gpu:
            score -= 35.0
            issues.append(
                "GPU required for this dataset/model size but none detected. "
                "Training on CPU will take extremely long."
            )
            suggestions.append(
                "Use Google Colab (free T4 GPU), Kaggle (free P100), "
                "or AWS/GCP spot instances."
            )
        elif not hw_requirements.gpu_required and not has_gpu:
            score -= 10.0
            suggestions.append(
                "CPU training is feasible but slow. "
                "Consider Google Colab for free GPU access."
            )

        # VRAM check
        if available_vram_gb is not None and has_gpu:
            details["available_vram_gb"] = available_vram_gb
            if available_vram_gb < hw_requirements.min_vram_gb:
                score -= 20.0
                suggestions.append(
                    f"VRAM: {available_vram_gb}GB available, "
                    f"{hw_requirements.min_vram_gb}GB required. "
                    "Reduce batch size or use mixed precision (fp16)."
                )

        details["cloud_alternatives"] = [
            "Google Colab (free T4 16GB, limited hours)",
            "Kaggle Notebooks (free P100 30h/week)",
            "Lightning.AI (free T4 access)",
            "Vast.ai (cheap GPU rental)",
        ]

        score = _clamp(score)

        return DimensionScore(
            name="Hardware Feasibility",
            score=round(score, 2),
            weight=0.07,
            weighted_score=round(score * 0.07, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 15 — ModelFitScorer
# ─────────────────────────────────────────────────────────────────────────────

class ModelFitScorer:
    """
    Scores how well-fitted the available models are for the dataset.

    Checks:
    - Whether suitable pretrained models exist for the task
    - Whether transfer learning is applicable
    - Estimated maximum achievable accuracy
    """

    def score(
        self,
        task_type: TaskType,
        n_total: int,
        n_classes: int,
        domain_similarity: str = "moderate",
    ) -> DimensionScore:
        """Score model fit dimension."""
        issues      : List[str] = []
        suggestions : List[str] = []
        details     : Dict[str, Any] = {}
        score       = 100.0

        # All common tasks have pretrained models
        has_pretrained = task_type in (
            TaskType.IMAGE_CLASSIFICATION,
            TaskType.OBJECT_DETECTION,
            TaskType.IMAGE_SEGMENTATION,
            TaskType.TEXT_CLASSIFICATION,
            TaskType.MULTI_MODAL,
        )

        details["task_type"]       = task_type.value
        details["has_pretrained"]  = has_pretrained
        details["n_classes"]       = n_classes
        details["domain_similarity"] = domain_similarity

        if not has_pretrained:
            score -= 40.0
            suggestions.append(
                "No standard pretrained model for this task. "
                "You'll need to design a custom architecture."
            )

        # Too many classes
        if n_classes > 1000:
            score -= 20.0
            suggestions.append(
                f"Very large number of classes ({n_classes}). "
                "Consider hierarchical classification or class grouping."
            )
        elif n_classes > 100:
            score -= 5.0
            suggestions.append(
                f"Large number of classes ({n_classes}). "
                "Ensure sufficient samples per class (200+ each)."
            )

        # Domain mismatch
        if domain_similarity == "low":
            score -= 15.0
            suggestions.append(
                "Low domain similarity with ImageNet. "
                "Look for domain-specific pretrained models "
                "(medical, satellite, documents)."
            )

        # Extreme class count with small dataset
        if n_classes > 50 and n_total < 5000:
            score -= 20.0
            issues.append(
                f"Too many classes ({n_classes}) for dataset size ({n_total}). "
                f"Each class has ~{n_total//n_classes} samples on average — too few."
            )

        details["estimated_max_accuracy"] = self._estimate_accuracy(
            n_total, n_classes, domain_similarity, has_pretrained
        )

        score = _clamp(score)

        return DimensionScore(
            name="Model Fit",
            score=round(score, 2),
            weight=0.05,
            weighted_score=round(score * 0.05, 2),
            status=_score_to_status(score),
            details=details,
            issues=issues,
            suggestions=suggestions,
        )

    def _estimate_accuracy(
        self,
        n_total: int,
        n_classes: int,
        domain_similarity: str,
        has_pretrained: bool,
    ) -> str:
        """Rough estimate of achievable accuracy range."""
        if not has_pretrained:
            return "Unknown — custom task"

        base = 0.85
        if n_total < 500:
            base -= 0.15
        elif n_total < 2000:
            base -= 0.05

        if n_classes > 100:
            base -= 0.10
        elif n_classes > 50:
            base -= 0.05

        if domain_similarity == "low":
            base -= 0.10
        elif domain_similarity == "moderate":
            base -= 0.03

        base = max(0.40, min(base, 0.98))
        return f"{base-0.05:.0%} – {base+0.05:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS — TrainingReadinessAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TrainingReadinessAnalyzer:
    """
    Main entry point for training readiness analysis.

    Orchestrates all sub-analyzers and produces a complete
    ReadinessReport with score, grade, recommendations,
    model suggestions, augmentation plan, hardware requirements,
    and training time estimates.

    Usage
    -----
    ::

        from src.data.training_readiness import TrainingReadinessAnalyzer

        analyzer = TrainingReadinessAnalyzer()
        report   = analyzer.analyze(
            dataset_info={
                "n_total":       1200,
                "class_counts":  {"cats": 600, "dogs": 400, "birds": 200},
                "image_size":    (224, 224),
                "n_channels":    3,
                "task_type":     "image_classification",
                "domain":        "general",
            }
        )

        print(f"Score: {report.overall_score}/100  Grade: {report.grade}")
        print(f"Ready: {report.ready_to_train}")
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        # Instantiate all sub-analyzers
        self._size_analyzer        = DatasetSizeAnalyzer()
        self._balance_analyzer     = ClassBalanceAnalyzer()
        self._split_advisor        = SplitAdvisor()
        self._augmentation_advisor = AugmentationAdvisor()
        self._hw_estimator         = HardwareEstimator()
        self._time_estimator       = TrainingTimeEstimator()
        self._model_recommender    = ModelRecommender()
        self._tl_advisor           = TransferLearningAdvisor()
        self._leakage_assessor     = LeakageRiskAssessor()
        self._score_calculator     = ReadinessScoreCalculator()
        self._multimodal           = MultiModalReadiness()
        self._quality_bridge       = DataQualityBridge()
        self._aug_scorer           = AugmentationScorer()
        self._hw_scorer            = HardwareScorer()
        self._model_fit_scorer     = ModelFitScorer()

        logger.info("TrainingReadinessAnalyzer initialized — v%s", VERSION)

    def analyze(
        self,
        dataset_info: Dict[str, Any],
        image_quality_report:  Optional[Dict[str, Any]] = None,
        label_quality_report:  Optional[Dict[str, Any]] = None,
        image_hashes:          Optional[Dict[str, List[str]]] = None,
        subject_ids:           Optional[Dict[str, List[str]]] = None,
        available_ram_gb:      Optional[float] = None,
        available_vram_gb:     Optional[float] = None,
        has_gpu:               bool = False,
    ) -> ReadinessReport:
        """
        Full training readiness analysis.

        Parameters
        ----------
        dataset_info : dict with keys:
            n_total         : int — total samples
            class_counts    : Dict[str,int] — {class: count}
            image_size      : Tuple[int,int] — (w,h)
            n_channels      : int — 1/3/4
            task_type       : str — 'image_classification' / 'object_detection' / etc.
            domain          : str — 'general'/'medical'/'satellite'/'documents'/'faces'
            is_temporal     : bool — time-series images
            is_multimodal   : bool
            n_text_samples  : int (multi-modal)
            n_aligned_pairs : int (multi-modal)
            dataset_path    : str
            modality        : str — 'image'/'text'/'multi_modal'

        image_quality_report : output from image_quality.py (optional)
        label_quality_report : output from label_quality.py (optional)
        image_hashes         : {split_name: [hashes]} for leakage check
        subject_ids          : {split_name: [ids]} for subject leakage check
        available_ram_gb     : system RAM available
        available_vram_gb    : GPU VRAM available
        has_gpu              : whether a GPU is available
        """
        start_time = time.time()
        if self.verbose:
            logger.info("Starting training readiness analysis...")

        # ── Extract dataset_info fields ──────────────────────────────────
        n_total         = int(dataset_info.get("n_total", 0))
        class_counts    = dict(dataset_info.get("class_counts", {}))
        image_size      = tuple(dataset_info.get("image_size", (224, 224)))
        n_channels      = int(dataset_info.get("n_channels", 3))
        domain          = str(dataset_info.get("domain", "general"))
        is_temporal     = bool(dataset_info.get("is_temporal", False))
        is_multimodal   = bool(dataset_info.get("is_multimodal", False))
        n_text_samples  = int(dataset_info.get("n_text_samples", 0))
        n_aligned_pairs = int(dataset_info.get("n_aligned_pairs", 0))
        dataset_path    = str(dataset_info.get("dataset_path", ""))
        modality        = str(dataset_info.get("modality", "image"))
        n_classes       = len(class_counts) if class_counts else 1
        is_color        = n_channels >= 3

        # Infer task type
        raw_task = dataset_info.get("task_type", "")
        task_type = self._parse_task_type(raw_task, dataset_info)

        if self.verbose:
            logger.info(
                "Dataset: n=%d  classes=%d  task=%s",
                n_total, n_classes, task_type.value
            )

        # ── Sub-analysis ─────────────────────────────────────────────────

        # 1. Dataset Size
        size_score = self._size_analyzer.analyze(
            n_total=n_total,
            class_counts=class_counts,
            n_channels=n_channels,
            image_size=image_size,
        )

        # 2. Class Balance
        balance_score = self._balance_analyzer.analyze(
            class_counts=class_counts,
            task_type=task_type,
        )

        # 3. Data Quality (from report or estimated)
        if image_quality_report:
            data_quality_score = self._quality_bridge.from_image_quality(
                image_quality_report
            )
        else:
            data_quality_score, _ = self._quality_bridge.estimate_quality_without_reports(
                n_total=n_total,
                has_quality_check=False,
            )

        # 4. Label Quality (from report or estimated)
        if label_quality_report:
            label_quality_score = self._quality_bridge.from_label_quality(
                label_quality_report
            )
        else:
            _, label_quality_score = self._quality_bridge.estimate_quality_without_reports(
                n_total=n_total,
                has_label_check=False,
            )

        # 5. Split Validity / Leakage
        split_score = self._leakage_assessor.assess(
            image_hashes=image_hashes,
            has_subjects=bool(subject_ids),
            subject_ids=subject_ids,
            is_temporal=is_temporal,
        )

        # 6. Augmentation
        aug_score = self._aug_scorer.score(
            n_total=n_total,
            task_type=task_type,
            augmentation_applied=dataset_info.get("augmentation_applied", False),
            augmentation_factor=float(dataset_info.get("augmentation_factor", 1.0)),
            image_domain=domain,
        )

        # 7. Hardware
        model_name     = dataset_info.get("preferred_model", "resnet50")
        hw_reqs        = self._hw_estimator.estimate(
            n_total=n_total,
            model_name=model_name,
            image_size=image_size,
            n_channels=n_channels,
        )
        hw_score       = self._hw_scorer.score(
            hw_requirements=hw_reqs,
            available_ram_gb=available_ram_gb,
            available_vram_gb=available_vram_gb,
            has_gpu=has_gpu,
        )

        # 8. Model Fit
        domain_similarity = dataset_info.get("domain_similarity", "moderate")
        model_fit_score   = self._model_fit_scorer.score(
            task_type=task_type,
            n_total=n_total,
            n_classes=n_classes,
            domain_similarity=domain_similarity,
        )

        # Multi-modal bonus dimension (not in main score but reported)
        multimodal_score = None
        if is_multimodal and n_text_samples > 0:
            multimodal_score = self._multimodal.analyze(
                n_image_samples=n_total,
                n_text_samples=n_text_samples,
                n_aligned_pairs=n_aligned_pairs,
                image_quality_score=data_quality_score.score,
                text_quality_score=label_quality_score.score,
                task_type=task_type,
            )

        # ── Aggregate Scores ─────────────────────────────────────────────
        dimension_scores = {
            "dataset_size":   size_score,
            "class_balance":  balance_score,
            "data_quality":   data_quality_score,
            "label_quality":  label_quality_score,
            "split_validity": split_score,
            "augmentation":   aug_score,
            "hardware":       hw_score,
            "model_fit":      model_fit_score,
        }

        # Collect all blocking issues
        all_blocking = []
        for dim in dimension_scores.values():
            all_blocking.extend(
                [i for i in dim.issues if i.startswith("BLOCKING")]
            )

        aggregated = self._score_calculator.calculate(
            dimension_scores=dimension_scores,
            blocking_issues=all_blocking,
        )

        # ── Recommendations ───────────────────────────────────────────────
        model_recs = self._model_recommender.recommend(
            task_type=task_type,
            n_total=n_total,
            n_classes=n_classes,
            max_vram_gb=available_vram_gb or 8.0,
        )

        aug_plan = self._augmentation_advisor.recommend(
            n_total=n_total,
            task_type=task_type,
            image_domain=domain,
            class_counts=class_counts,
            is_color=is_color,
        )

        split_rec = self._split_advisor.recommend(
            n_total=n_total,
            class_counts=class_counts,
            is_temporal=is_temporal,
        )

        tl_advice = self._tl_advisor.advise(
            n_total=n_total,
            task_type=task_type,
            domain_similarity=domain_similarity,
            n_classes=n_classes,
        )

        # Training time — use recommended model
        best_model   = model_recs[0].name if model_recs else "resnet50"
        best_model_key = best_model.lower().replace("-", "_").replace("/", "_").split("(")[0].strip()
        n_epochs     = self._time_estimator.recommend_epochs(
            n_train=split_rec.train_count,
            task_type=task_type,
            use_transfer_learning=tl_advice.get("use_transfer_learning", True),
        )
        batch_size   = self._time_estimator.recommend_batch_size(
            n_train=split_rec.train_count,
            vram_gb=available_vram_gb or 8.0,
            image_size=image_size,
        )
        time_est     = self._time_estimator.estimate(
            n_train=split_rec.train_count,
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=best_model_key,
            task_type=task_type,
            image_size=image_size,
        )

        # ── Collect warnings & info ───────────────────────────────────────
        all_warnings = []
        all_info     = []
        for dim in dimension_scores.values():
            non_blocking = [i for i in dim.issues if not i.startswith("BLOCKING")]
            all_warnings.extend(non_blocking)
            all_info.extend(dim.suggestions[:1])  # top suggestion per dimension

        all_warnings.extend(split_rec.warnings)

        # ── Build recommendations list ────────────────────────────────────
        priority_fixes = self._score_calculator.get_priority_fixes(dimension_scores)
        recommendations = []
        for fix in priority_fixes:
            for fix_text in fix.get("fixes", []):
                recommendations.append(f"[{fix['dimension']}] {fix_text}")

        # ── Summary text ──────────────────────────────────────────────────
        grade      = aggregated["grade"]
        overall    = aggregated["overall_score"]
        grade_emoji = _grade_emoji(grade)
        summary    = self._build_summary(
            overall=overall,
            grade=grade,
            grade_emoji=grade_emoji,
            ready=aggregated["ready_to_train"],
            n_blocking=len(all_blocking),
            n_warnings=len(all_warnings),
            task=task_type.value,
            n_total=n_total,
            n_classes=n_classes,
        )

        # ── Next steps ────────────────────────────────────────────────────
        next_steps = self._build_next_steps(
            ready=aggregated["ready_to_train"],
            blocking_issues=all_blocking,
            grade=grade,
            tl_advice=tl_advice,
            best_model=best_model,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

        elapsed = round(time.time() - start_time, 3)
        if self.verbose:
            logger.info(
                "Analysis complete in %.3fs — Score: %.1f/100  Grade: %s  Ready: %s",
                elapsed, overall, grade, aggregated["ready_to_train"]
            )

        # Add multimodal to dimensions if present
        dim_output = aggregated["dimensions"]
        if multimodal_score:
            dim_output["multi_modal_alignment"] = {
                "score":          multimodal_score.score,
                "grade":          _score_to_grade(multimodal_score.score),
                "status":         multimodal_score.status,
                "details":        multimodal_score.details,
                "issues":         multimodal_score.issues,
                "suggestions":    multimodal_score.suggestions,
            }

        return ReadinessReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            dataset_path=dataset_path,
            task_type=task_type.value,
            n_samples=n_total,
            n_classes=n_classes,
            modality=modality,
            overall_score=overall,
            grade=grade,
            ready_to_train=aggregated["ready_to_train"],
            confidence=aggregated["confidence"],
            blocking_issues=all_blocking,
            warnings=all_warnings[:10],        # cap at 10
            info_notes=all_info[:10],
            dimensions=dim_output,
            recommendations=recommendations[:15],
            model_recommendations=[asdict(m) for m in model_recs],
            augmentation_plan=asdict(aug_plan),
            hardware_requirements=asdict(hw_reqs),
            training_time_estimate=asdict(time_est),
            split_recommendation=asdict(split_rec),
            transfer_learning_advice=tl_advice,
            summary=summary,
            next_steps=next_steps,
        )

    # ── Private Helpers ────────────────────────────────────────────────────

    def _parse_task_type(
        self, raw: str, dataset_info: Dict[str, Any]
    ) -> TaskType:
        """Parse task type string to TaskType enum."""
        mapping = {
            "image_classification":  TaskType.IMAGE_CLASSIFICATION,
            "classification":        TaskType.IMAGE_CLASSIFICATION,
            "object_detection":      TaskType.OBJECT_DETECTION,
            "detection":             TaskType.OBJECT_DETECTION,
            "image_segmentation":    TaskType.IMAGE_SEGMENTATION,
            "segmentation":          TaskType.IMAGE_SEGMENTATION,
            "image_regression":      TaskType.IMAGE_REGRESSION,
            "regression":            TaskType.IMAGE_REGRESSION,
            "text_classification":   TaskType.TEXT_CLASSIFICATION,
            "multi_modal":           TaskType.MULTI_MODAL,
            "multimodal":            TaskType.MULTI_MODAL,
        }
        if raw.lower() in mapping:
            return mapping[raw.lower()]
        return _detect_task_from_structure(dataset_info)

    def _build_summary(
        self,
        overall: float,
        grade: str,
        grade_emoji: str,
        ready: bool,
        n_blocking: int,
        n_warnings: int,
        task: str,
        n_total: int,
        n_classes: int,
    ) -> str:
        """Build human-readable summary paragraph."""
        status = "ready for training" if ready else "NOT ready for training"
        blocking_note = (
            f" {n_blocking} blocking issue(s) must be resolved before training."
            if n_blocking > 0
            else ""
        )
        warning_note = (
            f" {n_warnings} warning(s) to address."
            if n_warnings > 0
            else ""
        )
        return (
            f"Your {task} dataset ({n_total:,} images, {n_classes} classes) "
            f"scored {overall:.1f}/100 — Grade {grade} {grade_emoji}. "
            f"The dataset is {status}.{blocking_note}{warning_note}"
        )

    def _build_next_steps(
        self,
        ready: bool,
        blocking_issues: List[str],
        grade: str,
        tl_advice: Dict[str, Any],
        best_model: str,
        n_epochs: int,
        batch_size: int,
    ) -> List[str]:
        """Build ordered list of next steps."""
        steps: List[str] = []

        if blocking_issues:
            steps.append("1. ⛔ Fix blocking issues first (see Issues section).")

        if grade in ("D", "F"):
            steps.append(
                "2. 🔴 Address critical dimension scores before training."
            )

        if ready:
            strategy = tl_advice.get("strategy", "fine_tune")
            steps.append(
                f"3. ✅ Apply augmentation pipeline (albumentations recommended)."
            )
            steps.append(
                f"4. 🚀 Start training with {best_model} "
                f"using {strategy} strategy."
            )
            steps.append(
                f"5. ⚙️  Use batch_size={batch_size}, "
                f"n_epochs={n_epochs}, "
                f"lr={tl_advice.get('recommended_lr', 0.0001)}."
            )
            steps.append(
                "6. 📊 Monitor val loss — apply early stopping (patience=10)."
            )
            steps.append(
                "7. 🔍 After training: run bias_detector.py to check for bias."
            )
        else:
            steps.append(
                "3. 📦 Collect more data or apply augmentation to meet minimums."
            )
            steps.append(
                "4. 🏷️  Review and clean labels if label quality is low."
            )
            steps.append(
                "5. 🔁 Re-run training_readiness.py after fixes."
            )

        return steps

    # ── Convenience Methods ────────────────────────────────────────────────

    def quick_check(
        self,
        n_total: int,
        n_classes: int,
        class_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Fast check without full analysis.
        Returns a summary dict in milliseconds.
        """
        if class_counts is None:
            class_counts = {
                f"class_{i}": n_total // n_classes
                for i in range(n_classes)
            }

        size_ok      = n_total >= TOTAL_POOR
        balance_ok   = _imbalance_ratio(list(class_counts.values())) < IMBALANCE_CRITICAL
        min_per_class = min(class_counts.values()) if class_counts else 0

        return {
            "n_total":        n_total,
            "n_classes":      n_classes,
            "size_ok":        size_ok,
            "balance_ok":     balance_ok,
            "min_per_class":  min_per_class,
            "quick_verdict":  (
                "⚠️  Needs work" if not size_ok or not balance_ok
                else "✅ Looks promising — run full analysis for details"
            ),
        }

    def compare_datasets(
        self,
        dataset_a: Dict[str, Any],
        dataset_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare two datasets and return which is more ready for training.
        """
        report_a = self.analyze(dataset_a)
        report_b = self.analyze(dataset_b)

        winner = "A" if report_a.overall_score >= report_b.overall_score else "B"

        return {
            "dataset_a_score": report_a.overall_score,
            "dataset_a_grade": report_a.grade,
            "dataset_b_score": report_b.overall_score,
            "dataset_b_grade": report_b.grade,
            "winner":          winner,
            "score_diff":      abs(report_a.overall_score - report_b.overall_score),
            "recommendation":  (
                f"Dataset {winner} is more ready for training. "
                f"Score difference: {abs(report_a.overall_score - report_b.overall_score):.1f} points."
            ),
        }

    def to_dict(self, report: ReadinessReport) -> Dict[str, Any]:
        """Convert ReadinessReport to plain dict (JSON-serializable)."""
        return asdict(report)

    def to_json(self, report: ReadinessReport, indent: int = 2) -> str:
        """Convert ReadinessReport to JSON string."""
        return json.dumps(self.to_dict(report), indent=indent, default=str)

    def print_summary(self, report: ReadinessReport) -> None:
        """Print a formatted summary to console."""
        grade_emoji = _grade_emoji(report.grade)
        print("\n" + "═" * 60)
        print(f"  🩺 Nydra — Training Readiness Report  v{VERSION}")
        print("═" * 60)
        print(f"  Dataset : {report.dataset_path or 'N/A'}")
        print(f"  Task    : {report.task_type}")
        print(f"  Samples : {report.n_samples:,}  |  Classes: {report.n_classes}")
        print(f"  Score   : {report.overall_score:.1f}/100")
        print(f"  Grade   : {report.grade} {grade_emoji}")
        print(f"  Ready   : {'✅ Yes' if report.ready_to_train else '❌ No'}")
        print("─" * 60)

        if report.blocking_issues:
            print(f"  🚫 Blocking Issues ({len(report.blocking_issues)}):")
            for issue in report.blocking_issues:
                print(f"     • {issue}")
            print()

        if report.warnings:
            print(f"  ⚠️  Warnings ({len(report.warnings)}):")
            for w in report.warnings[:5]:
                print(f"     • {w}")
            print()

        print(f"  📊 Dimension Scores:")
        print(self._score_calculator.generate_score_table(
            {
                k: DimensionScore(
                    name=v["name"] if "name" in v else k.replace("_", " ").title(),
                    score=v["score"],
                    weight=v["weight"] / 100,
                    weighted_score=v["weighted_score"],
                    status=v["status"],
                )
                for k, v in report.dimensions.items()
            }
        ))

        print(f"\n  📝 Summary:")
        print(f"     {report.summary}")

        print(f"\n  🔜 Next Steps:")
        for step in report.next_steps:
            print(f"     {step}")

        if report.model_recommendations:
            print(f"\n  🤖 Top Model: {report.model_recommendations[0].get('name', 'N/A')}")
            print(f"     Reason: {report.model_recommendations[0].get('reason', '')[:80]}...")

        print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT INTEGRATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_streamlit_report(report: ReadinessReport) -> None:
    """
    Render the training readiness report in a Streamlit tab.

    Call this from app.py tab 🚂 Training Readiness.
    Requires: import streamlit as st
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        print("Streamlit/Plotly not available — use print_summary() instead.")
        return

    grade_emoji = _grade_emoji(report.grade)
    score       = report.overall_score

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown(f"## 🚂 Training Readiness Score")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Score",  f"{score:.1f}/100")
    col2.metric("Grade",          f"{report.grade} {grade_emoji}")
    col3.metric("Ready to Train", "✅ Yes" if report.ready_to_train else "❌ No")
    col4.metric("Confidence",     f"{report.confidence:.0f}%")

    # ── Gauge chart ───────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Readiness Score"},
        gauge={
            "axis":       {"range": [0, 100]},
            "bar":        {"color": "darkblue"},
            "steps": [
                {"range": [0, 44],   "color": "#ff4444"},
                {"range": [44, 59],  "color": "#ff8800"},
                {"range": [59, 74],  "color": "#ffcc00"},
                {"range": [74, 89],  "color": "#88cc00"},
                {"range": [89, 100], "color": "#00bb44"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Blocking issues ───────────────────────────────────────────────────
    if report.blocking_issues:
        st.error(f"🚫 **{len(report.blocking_issues)} Blocking Issue(s) — Fix Before Training**")
        for issue in report.blocking_issues:
            st.error(f"• {issue}")

    # ── Warnings ──────────────────────────────────────────────────────────
    if report.warnings:
        with st.expander(f"⚠️ Warnings ({len(report.warnings)})", expanded=False):
            for w in report.warnings:
                st.warning(f"• {w}")

    # ── Dimension breakdown ───────────────────────────────────────────────
    st.markdown("### 📊 Dimension Scores")
    dims      = report.dimensions
    dim_names = [v.get("name", k.replace("_", " ").title())
                 if isinstance(v, dict) else k
                 for k, v in dims.items()]
    dim_scores = [
        v.get("score", 0) if isinstance(v, dict) else 0
        for v in dims.values()
    ]

    fig_bar = go.Figure(go.Bar(
        x=dim_scores,
        y=dim_names,
        orientation="h",
        marker_color=[
            "#00bb44" if s >= 90 else
            "#88cc00" if s >= 75 else
            "#ffcc00" if s >= 60 else
            "#ff8800" if s >= 45 else
            "#ff4444"
            for s in dim_scores
        ],
        text=[f"{s:.1f}" for s in dim_scores],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis_range=[0, 110],
        height=max(300, len(dim_names) * 45),
        margin=dict(l=200),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tabs for details ──────────────────────────────────────────────────
    tabs = st.tabs([
        "🤖 Models", "🔄 Augmentation", "💻 Hardware",
        "⏱ Time", "🔀 Split", "🔁 Transfer Learning"
    ])

    with tabs[0]:
        st.markdown("#### Recommended Models")
        for rec in report.model_recommendations:
            with st.expander(
                f"**{rec.get('name', '')}** — "
                f"{_format_params(rec.get('params', 0))} params"
            ):
                st.markdown(f"**Reason:** {rec.get('reason', '')}")
                st.markdown(f"**Pretrained on:** {rec.get('pretrained_on', 'N/A')}")
                st.markdown(f"**Difficulty:** {rec.get('training_difficulty', 'N/A')}")
                st.markdown(
                    f"**Expected accuracy:** "
                    f"{rec.get('expected_accuracy_range', (0,0))[0]:.0%} – "
                    f"{rec.get('expected_accuracy_range', (0,0))[1]:.0%}"
                )
                st.code(rec.get("code_snippet", ""), language="python")

    with tabs[1]:
        aug = report.augmentation_plan
        st.markdown(f"**Expected multiplier:** {aug.get('expected_multiplier', 1)}×")
        st.markdown(f"**Notes:** {aug.get('notes', '')}")
        st.markdown("**Safe transforms:**")
        for t in aug.get("safe_transforms", []):
            st.markdown(f"  ✅ {t}")
        st.markdown("**Use with caution:**")
        for t in aug.get("risky_transforms", []):
            st.markdown(f"  ⚠️ {t}")
        st.code(aug.get("library_code", ""), language="python")

    with tabs[2]:
        hw = report.hardware_requirements
        c1, c2 = st.columns(2)
        c1.metric("Min RAM",       f"{hw.get('min_ram_gb', '?')} GB")
        c1.metric("Min VRAM",      f"{hw.get('min_vram_gb', '?')} GB")
        c2.metric("Estimated Disk", f"{hw.get('estimated_disk_gb', '?')} GB")
        c2.metric("GPU Required",  "Yes" if hw.get("gpu_required") else "No")
        st.info(f"**Recommended GPU:** {hw.get('gpu_recommended', 'N/A')}")
        for note in hw.get("notes", []):
            st.caption(f"💡 {note}")

    with tabs[3]:
        te = report.training_time_estimate
        st.markdown(
            f"**Setup:** {te.get('n_samples',0):,} samples × "
            f"{te.get('n_epochs',0)} epochs | "
            f"batch={te.get('batch_size',0)} | "
            f"model={te.get('model_name','?')}"
        )
        estimates = te.get("estimates", {})
        hw_names  = {
            "cpu":         "💻 CPU",
            "gpu_t4":      "🖥️ T4 (Colab)",
            "gpu_v100":    "🖥️ V100",
            "gpu_a100":    "🖥️ A100",
            "gpu_rtx3090": "🖥️ RTX 3090",
            "gpu_rtx4090": "🖥️ RTX 4090",
        }
        for hw_key, hw_label in hw_names.items():
            if hw_key in estimates:
                st.metric(hw_label, estimates[hw_key])
        st.warning(f"⚠️ Bottleneck: {te.get('bottleneck', 'N/A')}")
        for tip in te.get("tips", []):
            st.caption(f"💡 {tip}")

    with tabs[4]:
        split = report.split_recommendation
        strategy_desc = SplitAdvisor().get_strategy_description(
            split.get("strategy", "")
        )
        st.info(f"**Strategy:** {split.get('strategy', 'N/A')} — {strategy_desc}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train",      f"{split.get('train_count',0):,} ({split.get('train_ratio',0):.0%})")
        c2.metric("Validation", f"{split.get('val_count',0):,} ({split.get('val_ratio',0):.0%})")
        c3.metric("Test",       f"{split.get('test_count',0):,} ({split.get('test_ratio',0):.0%})")
        for w in split.get("warnings", []):
            st.warning(f"• {w}")
        code = SplitAdvisor().generate_sklearn_code(
            split.get("strategy", "standard_split"),
            split.get("test_ratio", 0.15),
        )
        st.code(code, language="python")

    with tabs[5]:
        tl = report.transfer_learning_advice
        st.markdown(f"**Use Transfer Learning:** {'✅ Yes' if tl.get('use_transfer_learning') else '❌ No'}")
        st.markdown(f"**Strategy:** `{tl.get('strategy', 'N/A')}`")
        st.markdown(f"**Reason:** {tl.get('reason', '')}")
        st.markdown(f"**Learning Rate:** `{tl.get('recommended_lr', 'N/A')}`")
        st.markdown(f"**Freeze Layers:** `{tl.get('freeze_layers', 'N/A')}`")
        if tl.get("pretrained_source"):
            st.info(tl["pretrained_source"])
        if tl.get("code_hint"):
            st.code(tl["code_hint"], language="python")

    # ── Next steps ────────────────────────────────────────────────────────
    st.markdown("### 🔜 Next Steps")
    for step in report.next_steps:
        st.markdown(step)

    st.caption(
        f"Analysis timestamp: {report.timestamp} | "
        f"Nydra v{VERSION}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _cli_main() -> None:
    """Command-line entry: python training_readiness.py <dataset_path>"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Nydra — Training Readiness Analyzer v" + VERSION
    )
    parser.add_argument("dataset_path",   help="Path to dataset directory")
    parser.add_argument("--task",         default="image_classification",
                        help="Task type (image_classification/object_detection/...)")
    parser.add_argument("--n-total",      type=int,   default=None)
    parser.add_argument("--n-classes",    type=int,   default=None)
    parser.add_argument("--domain",       default="general")
    parser.add_argument("--gpu",          action="store_true")
    parser.add_argument("--json",         action="store_true",
                        help="Output JSON instead of formatted report")
    args = parser.parse_args()

    # Build dataset_info from args / directory scan
    dataset_info: Dict[str, Any] = {
        "dataset_path": args.dataset_path,
        "task_type":    args.task,
        "domain":       args.domain,
        "modality":     "image",
    }

    # If path exists, scan for images
    p = Path(args.dataset_path)
    if p.exists():
        image_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
        all_images  = [f for f in p.rglob("*") if f.suffix.lower() in image_exts]

        if not args.n_total:
            dataset_info["n_total"] = len(all_images)

        # Try to infer classes from subdirectories
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        if subdirs and not args.n_classes:
            class_counts: Dict[str, int] = {}
            for subdir in subdirs:
                imgs = [
                    f for f in subdir.iterdir()
                    if f.suffix.lower() in image_exts
                ]
                if imgs:
                    class_counts[subdir.name] = len(imgs)
            if class_counts:
                dataset_info["class_counts"] = class_counts
                dataset_info["n_total"]      = sum(class_counts.values())

    if args.n_total:
        dataset_info["n_total"]  = args.n_total
    if args.n_classes and "class_counts" not in dataset_info:
        n = dataset_info.get("n_total", 1000)
        dataset_info["class_counts"] = {
            f"class_{i}": n // args.n_classes
            for i in range(args.n_classes)
        }

    # Defaults
    dataset_info.setdefault("n_total",      1000)
    dataset_info.setdefault("class_counts", {"class_0": 500, "class_1": 500})
    dataset_info.setdefault("image_size",   (224, 224))
    dataset_info.setdefault("n_channels",   3)

    analyzer = TrainingReadinessAnalyzer(verbose=True)
    report   = analyzer.analyze(
        dataset_info=dataset_info,
        has_gpu=args.gpu,
    )

    if args.json:
        print(analyzer.to_json(report))
    else:
        analyzer.print_summary(report)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """Quick demo with a sample dataset."""
    print("\n" + "="*60)
    print("  Nydra — Training Readiness Demo")
    print("="*60)

    analyzer = TrainingReadinessAnalyzer(verbose=True)

    # Demo 1: Good dataset
    print("\n📦 Demo 1: Good balanced dataset")
    report = analyzer.analyze({
        "n_total":       5000,
        "class_counts":  {
            "cats":  1700,
            "dogs":  1600,
            "birds": 1700,
        },
        "image_size":    (224, 224),
        "n_channels":    3,
        "task_type":     "image_classification",
        "domain":        "general",
        "dataset_path":  "/data/animals",
    })
    analyzer.print_summary(report)

    # Demo 2: Problematic dataset
    print("\n📦 Demo 2: Small imbalanced dataset")
    report2 = analyzer.analyze({
        "n_total":       300,
        "class_counts":  {
            "normal":   270,
            "disease":   30,
        },
        "image_size":    (512, 512),
        "n_channels":    1,   # grayscale medical
        "task_type":     "image_classification",
        "domain":        "medical",
        "dataset_path":  "/data/xrays",
    })
    analyzer.print_summary(report2)

    # Quick check
    print("\n⚡ Quick Check:")
    qc = analyzer.quick_check(n_total=150, n_classes=5)
    print(f"  Verdict: {qc['quick_verdict']}")
    print(f"  Min per class: {qc['min_per_class']}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("--demo", "-d"):
        _cli_main()
    else:
        _demo()
