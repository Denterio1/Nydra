"""
image_quality.py — Nydra v0.6.0
======================================
Advanced Image Quality Assessment for ML/AI Datasets

Analyzes every quality dimension of an image or full dataset:
- Basic metrics: brightness, contrast, sharpness, blur, noise, exposure
- Advanced perceptual quality: BRISQUE, NIQE, blocking/ringing artifacts
- ML-specific problems: resolution, edge density, information content
- Per-image quality score (0-100) with verdict and fix suggestion
- Dataset-level quality statistics and class-level quality bias detection
- Smart recovery suggestions linked to image_cleaner.py

Version: 0.6.0
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import os
import sys
import json
import math
import time
import logging
import warnings
import traceback
from io import BytesIO
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Pillow
try:
    from PIL import Image, ImageFilter, ImageStat
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("Pillow not installed. Run: pip install Pillow")

# OpenCV — core quality analysis
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not installed. Run: pip install opencv-python")

# NumPy & SciPy — statistical analysis
try:
    from scipy import stats, signal, ndimage
    from scipy.fft import fft2, fftshift
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# scikit-image — advanced image metrics
try:
    from skimage import exposure, filters, measure, feature, restoration
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# tqdm — progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("nydra.image_quality")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Brightness thresholds (0-255 mean pixel value)
BRIGHTNESS_TOO_DARK    = 40
BRIGHTNESS_DARK        = 70
BRIGHTNESS_BRIGHT      = 200
BRIGHTNESS_TOO_BRIGHT  = 230

# Blur thresholds (Laplacian variance)
BLUR_SEVERE   = 20
BLUR_HIGH     = 50
BLUR_MEDIUM   = 100
BLUR_LOW      = 200

# Noise thresholds (estimated sigma)
NOISE_SEVERE  = 25
NOISE_HIGH    = 15
NOISE_MEDIUM  = 8
NOISE_LOW     = 3

# Resolution thresholds
MIN_DIMENSION_CRITICAL = 32    # pixels
MIN_DIMENSION_WARNING  = 64
MIN_DIMENSION_ML       = 224   # recommended minimum for most models

# BRISQUE thresholds (lower = better quality)
BRISQUE_EXCELLENT = 20
BRISQUE_GOOD      = 40
BRISQUE_FAIR      = 60
BRISQUE_POOR      = 80

# Contrast thresholds (RMS contrast 0-1)
CONTRAST_TOO_LOW  = 0.05
CONTRAST_LOW      = 0.15
CONTRAST_HIGH     = 0.85

# Entropy thresholds (information content)
ENTROPY_TOO_LOW   = 3.0
ENTROPY_LOW       = 5.0

# Artifact thresholds
BLOCKING_THRESHOLD = 0.15
RINGING_THRESHOLD  = 0.10

# Score weights for overall quality
SCORE_WEIGHTS = {
    "sharpness":  0.25,
    "exposure":   0.25,
    "noise":      0.20,
    "contrast":   0.15,
    "artifacts":  0.15,
}

# Verdict thresholds
VERDICT_HIGH    = 80
VERDICT_MEDIUM  = 55
VERDICT_LOW     = 35


# ─────────────────────────────────────────────
# ENUMS & DATACLASSES
# ─────────────────────────────────────────────

class QualityVerdict(Enum):
    HIGH    = "High ✅"
    MEDIUM  = "Medium ⚠️"
    LOW     = "Low ❌"
    REJECT  = "Reject 🚫"


class BlurType(Enum):
    NONE    = "none"
    DEFOCUS = "defocus"
    MOTION  = "motion"
    GAUSSIAN = "gaussian"


class NoiseType(Enum):
    NONE      = "none"
    GAUSSIAN  = "gaussian"
    SALT_PEPPER = "salt_pepper"
    POISSON   = "poisson"
    MIXED     = "mixed"


class ExposureType(Enum):
    CORRECT       = "correct"
    UNDEREXPOSED  = "underexposed"
    OVEREXPOSED   = "overexposed"
    HIGH_DYNAMIC  = "high_dynamic_range"


@dataclass
class QualityMetrics:
    """All quality measurements for a single image."""
    # ── File info ──────────────────────────────────────────────────────────
    file_path:          str   = ""
    file_name:          str   = ""
    width:              int   = 0
    height:             int   = 0

    # ── Basic metrics ──────────────────────────────────────────────────────
    brightness_mean:    float = 0.0
    brightness_std:     float = 0.0
    brightness_score:   float = 0.0

    contrast_rms:       float = 0.0
    contrast_michelson: float = 0.0
    contrast_score:     float = 0.0

    sharpness_laplacian: float = 0.0
    sharpness_tenengrad: float = 0.0
    sharpness_brenner:   float = 0.0
    sharpness_score:     float = 0.0

    blur_laplacian:     float = 0.0
    blur_fft_ratio:     float = 0.0
    blur_score:         float = 0.0
    blur_type:          str   = BlurType.NONE.value
    is_blurry:          bool  = False

    noise_sigma:        float = 0.0
    noise_snr_db:       float = 0.0
    noise_score:        float = 0.0
    noise_type:         str   = NoiseType.NONE.value
    is_noisy:           bool  = False

    # ── Exposure ───────────────────────────────────────────────────────────
    exposure_type:      str   = ExposureType.CORRECT.value
    overexposed_ratio:  float = 0.0   # fraction of pixels at/near 255
    underexposed_ratio: float = 0.0   # fraction of pixels at/near 0
    histogram_spread:   float = 0.0
    exposure_score:     float = 0.0
    is_overexposed:     bool  = False
    is_underexposed:    bool  = False

    # ── Advanced perceptual ────────────────────────────────────────────────
    brisque_score:      float = -1.0   # -1 = not computed
    niqe_score:         float = -1.0
    entropy:            float = 0.0
    edge_density:       float = 0.0
    frequency_energy:   float = 0.0

    # ── Artifacts ──────────────────────────────────────────────────────────
    blocking_score:     float = 0.0
    ringing_score:      float = 0.0
    has_blocking:       bool  = False
    has_ringing:        bool  = False
    jpeg_quality_est:   int   = -1    # estimated JPEG quality factor
    artifact_score:     float = 100.0

    # ── ML-specific ────────────────────────────────────────────────────────
    is_low_resolution:  bool  = False
    resolution_verdict: str   = ""
    info_content:       float = 0.0   # information content score
    color_cast:         str   = ""    # dominant color cast if any
    dynamic_range:      float = 0.0

    # ── Overall ────────────────────────────────────────────────────────────
    overall_score:      float = 0.0
    verdict:            str   = QualityVerdict.MEDIUM.value
    reject_reason:      str   = ""
    suggested_fix:      str   = ""
    quality_issues:     List  = field(default_factory=list)

    # ── Timing ────────────────────────────────────────────────────────────
    processing_time_ms: float = 0.0


@dataclass
class DatasetQualityReport:
    """Dataset-level quality statistics."""
    total_images:       int   = 0
    high_quality:       int   = 0
    medium_quality:     int   = 0
    low_quality:        int   = 0
    rejected:           int   = 0

    mean_overall_score: float = 0.0
    std_overall_score:  float = 0.0
    min_score:          float = 0.0
    max_score:          float = 0.0

    mean_sharpness:     float = 0.0
    mean_brightness:    float = 0.0
    mean_contrast:      float = 0.0
    mean_noise:         float = 0.0

    blurry_count:       int   = 0
    dark_count:         int   = 0
    overexposed_count:  int   = 0
    noisy_count:        int   = 0
    artifact_count:     int   = 0
    low_res_count:      int   = 0

    # Class-level quality bias
    class_quality_bias: Dict  = field(default_factory=dict)
    biased_classes:     List  = field(default_factory=list)

    # Split-level quality check
    split_quality:      Dict  = field(default_factory=dict)
    split_quality_gap:  float = 0.0

    # Recommendations
    recommendations:    List  = field(default_factory=list)
    fix_priority:       List  = field(default_factory=list)

    generated_at:       str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — BRIGHTNESS ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class BrightnessAnalyzer:
    """
    Analyzes image brightness with multiple methods.
    Detects: too dark, too bright, correct exposure.
    Computes: mean, std, histogram spread, zone distribution.
    """

    def analyze(self, gray: np.ndarray) -> Dict:
        """
        Args: gray — single-channel uint8 numpy array
        Returns: dict with brightness metrics and score
        """
        h, w = gray.shape
        total_pixels = h * w

        mean_val  = float(np.mean(gray))
        std_val   = float(np.std(gray))
        median_val = float(np.median(gray))

        # Histogram
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 255))
        hist_norm  = hist / total_pixels

        # Zone distribution (Ansel Adams zones 0-9)
        zone_size = 256 // 10
        zones = [float(hist_norm[i*zone_size:(i+1)*zone_size].sum()) for i in range(10)]

        # Overexposed / underexposed ratio
        overexposed_ratio  = float(np.sum(gray >= 250) / total_pixels)
        underexposed_ratio = float(np.sum(gray <= 5)   / total_pixels)
        clipped_ratio      = overexposed_ratio + underexposed_ratio

        # Histogram spread (interquartile range normalized)
        p25, p75 = np.percentile(gray, [25, 75])
        hist_spread = float((p75 - p25) / 255.0)

        # Score calculation
        score = self._compute_score(mean_val, std_val, overexposed_ratio, underexposed_ratio)

        # Exposure type
        if overexposed_ratio > 0.15:
            exp_type = ExposureType.OVEREXPOSED.value
        elif underexposed_ratio > 0.15:
            exp_type = ExposureType.UNDEREXPOSED.value
        elif mean_val < BRIGHTNESS_DARK:
            exp_type = ExposureType.UNDEREXPOSED.value
        elif mean_val > BRIGHTNESS_BRIGHT:
            exp_type = ExposureType.OVEREXPOSED.value
        else:
            exp_type = ExposureType.CORRECT.value

        # Dynamic range
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float((p99 - p1) / 255.0)

        return {
            "brightness_mean":      round(mean_val, 3),
            "brightness_std":       round(std_val, 3),
            "brightness_median":    round(median_val, 3),
            "brightness_score":     round(score, 1),
            "overexposed_ratio":    round(overexposed_ratio, 4),
            "underexposed_ratio":   round(underexposed_ratio, 4),
            "clipped_ratio":        round(clipped_ratio, 4),
            "histogram_spread":     round(hist_spread, 4),
            "dynamic_range":        round(dynamic_range, 4),
            "exposure_type":        exp_type,
            "is_overexposed":       overexposed_ratio > 0.15 or mean_val > BRIGHTNESS_BRIGHT,
            "is_underexposed":      underexposed_ratio > 0.15 or mean_val < BRIGHTNESS_DARK,
            "zone_distribution":    zones,
        }

    def _compute_score(
        self,
        mean: float,
        std: float,
        over: float,
        under: float
    ) -> float:
        score = 100.0

        # Penalize extreme brightness
        if mean < BRIGHTNESS_TOO_DARK:
            score -= 60
        elif mean < BRIGHTNESS_DARK:
            score -= 30 * (1 - (mean - BRIGHTNESS_TOO_DARK) / (BRIGHTNESS_DARK - BRIGHTNESS_TOO_DARK))
        elif mean > BRIGHTNESS_TOO_BRIGHT:
            score -= 60
        elif mean > BRIGHTNESS_BRIGHT:
            score -= 30 * ((mean - BRIGHTNESS_BRIGHT) / (BRIGHTNESS_TOO_BRIGHT - BRIGHTNESS_BRIGHT))

        # Penalize clipping
        score -= min(40, over * 200)
        score -= min(40, under * 200)

        # Reward good standard deviation (texture variety)
        if std < 10:
            score -= 20  # flat / no texture

        return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — CONTRAST ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class ContrastAnalyzer:
    """
    Measures image contrast using multiple methods:
    - RMS Contrast (global)
    - Michelson Contrast (local max-min)
    - Weber Contrast
    - Local contrast map
    - CLAHE potential improvement estimate
    """

    def analyze(self, gray: np.ndarray) -> Dict:
        h, w = gray.shape
        img_float = gray.astype(np.float64) / 255.0

        # RMS Contrast — global measure
        rms = float(np.std(img_float))

        # Michelson Contrast — (max-min)/(max+min)
        i_max = float(img_float.max())
        i_min = float(img_float.min())
        michelson = (i_max - i_min) / (i_max + i_min + 1e-8)

        # Weber Contrast — relative to background (estimated as mean)
        bg = float(img_float.mean())
        weber = abs(i_max - bg) / (bg + 1e-8)

        # Local contrast map using sliding window (8x8 blocks)
        local_contrasts = self._local_contrast_map(gray)
        local_mean  = float(np.mean(local_contrasts))
        local_std   = float(np.std(local_contrasts))
        low_contrast_regions = float(np.mean(local_contrasts < 0.05))

        # Histogram-based contrast
        hist, _ = np.histogram(gray, bins=256, range=(0, 255))
        hist_norm = hist / hist.sum()
        cumhist = np.cumsum(hist_norm)
        p5  = float(np.searchsorted(cumhist, 0.05))
        p95 = float(np.searchsorted(cumhist, 0.95))
        contrast_range = (p95 - p5) / 255.0

        # Score
        score = self._compute_score(rms, michelson, low_contrast_regions)

        return {
            "contrast_rms":              round(rms, 4),
            "contrast_michelson":        round(michelson, 4),
            "contrast_weber":            round(weber, 4),
            "contrast_local_mean":       round(local_mean, 4),
            "contrast_local_std":        round(local_std, 4),
            "contrast_range":            round(contrast_range, 4),
            "low_contrast_region_ratio": round(low_contrast_regions, 4),
            "contrast_score":            round(score, 1),
        }

    def _local_contrast_map(self, gray: np.ndarray, block_size: int = 8) -> np.ndarray:
        """Compute local RMS contrast per block."""
        h, w = gray.shape
        img_f = gray.astype(np.float64) / 255.0
        contrasts = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img_f[y:y+block_size, x:x+block_size]
                contrasts.append(np.std(block))
        return np.array(contrasts) if contrasts else np.array([0.0])

    def _compute_score(
        self,
        rms: float,
        michelson: float,
        low_regions: float
    ) -> float:
        score = 100.0

        if rms < CONTRAST_TOO_LOW:
            score -= 70
        elif rms < CONTRAST_LOW:
            score -= 35 * (1 - (rms - CONTRAST_TOO_LOW) / (CONTRAST_LOW - CONTRAST_TOO_LOW))

        if michelson < 0.1:
            score -= 20

        # Penalize too many low-contrast regions
        score -= min(30, low_regions * 60)

        return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — SHARPNESS & BLUR ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class SharpnessBlurAnalyzer:
    """
    Multi-method sharpness and blur detection:
    - Laplacian variance (fast, reliable)
    - Tenengrad (gradient-based)
    - Brenner gradient
    - FFT-based blur detection (frequency domain)
    - Motion blur vs defocus blur classification
    - Local sharpness map (finds regions of blur)
    """

    def analyze(self, gray: np.ndarray) -> Dict:
        h, w = gray.shape

        # ── Laplacian variance ──────────────────────────────────────────────
        if CV2_AVAILABLE:
            laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
            lap_var    = float(laplacian.var())
            lap_mean   = float(np.abs(laplacian).mean())
        else:
            # Fallback: numpy-based Laplacian
            kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
            from scipy.signal import convolve2d
            laplacian = convolve2d(gray.astype(np.float64), kernel, mode="same")
            lap_var   = float(laplacian.var())
            lap_mean  = float(np.abs(laplacian).mean())

        # ── Tenengrad (Sobel-based) ─────────────────────────────────────────
        if CV2_AVAILABLE:
            sobel_x   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = float(np.mean(sobel_x**2 + sobel_y**2))
        else:
            tenengrad = lap_var  # fallback

        # ── Brenner gradient ────────────────────────────────────────────────
        gray_f  = gray.astype(np.float64)
        brenner = float(np.mean((gray_f[2:, :] - gray_f[:-2, :]) ** 2))

        # ── FFT-based blur detection ─────────────────────────────────────────
        fft_blur_ratio = self._fft_blur_ratio(gray)

        # ── Local sharpness map ──────────────────────────────────────────────
        local_sharpness = self._local_sharpness_map(gray)
        local_sharp_mean = float(np.mean(local_sharpness))
        blurry_region_ratio = float(np.mean(local_sharpness < BLUR_SEVERE))

        # ── Blur type detection ──────────────────────────────────────────────
        blur_type  = self._detect_blur_type(gray, lap_var, fft_blur_ratio)
        is_blurry  = lap_var < BLUR_HIGH

        # ── Sharpness score ──────────────────────────────────────────────────
        sharpness_score = self._sharpness_score(lap_var, tenengrad, brenner)
        blur_score      = self._blur_score(lap_var, fft_blur_ratio, blurry_region_ratio)

        return {
            "sharpness_laplacian":    round(lap_var, 3),
            "sharpness_laplacian_mean": round(lap_mean, 3),
            "sharpness_tenengrad":    round(tenengrad, 3),
            "sharpness_brenner":      round(brenner, 3),
            "sharpness_score":        round(sharpness_score, 1),
            "blur_laplacian":         round(lap_var, 3),
            "blur_fft_ratio":         round(fft_blur_ratio, 4),
            "blur_score":             round(blur_score, 1),
            "blur_type":              blur_type,
            "is_blurry":              is_blurry,
            "blurry_region_ratio":    round(blurry_region_ratio, 4),
            "local_sharpness_mean":   round(local_sharp_mean, 3),
        }

    def _fft_blur_ratio(self, gray: np.ndarray) -> float:
        """
        Frequency domain blur detection.
        Blurry images have less high-frequency content.
        Returns ratio of high-freq energy to total energy.
        """
        try:
            f      = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            mag    = np.abs(fshift)

            h, w   = gray.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 6   # inner 1/6 = low freq

            y, x = np.ogrid[:h, :w]
            mask_low  = (y - cy)**2 + (x - cx)**2 <= radius**2
            mask_high = ~mask_low

            low_energy  = float(mag[mask_low].sum())
            high_energy = float(mag[mask_high].sum())
            total       = low_energy + high_energy + 1e-8

            return high_energy / total
        except Exception:
            return 0.5  # neutral fallback

    def _local_sharpness_map(self, gray: np.ndarray, block_size: int = 32) -> np.ndarray:
        """Compute Laplacian variance for each block."""
        h, w = gray.shape
        values = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size]
                if CV2_AVAILABLE:
                    lap = cv2.Laplacian(block, cv2.CV_64F)
                    values.append(float(lap.var()))
                else:
                    values.append(float(np.std(block.astype(float))))
        return np.array(values) if values else np.array([0.0])

    def _detect_blur_type(
        self,
        gray: np.ndarray,
        lap_var: float,
        fft_ratio: float
    ) -> str:
        """Distinguish motion blur from defocus/Gaussian blur."""
        if lap_var >= BLUR_HIGH:
            return BlurType.NONE.value

        if not CV2_AVAILABLE:
            return BlurType.DEFOCUS.value if lap_var < BLUR_MEDIUM else BlurType.NONE.value

        # Motion blur: directional gradients are asymmetric
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        energy_x = float(np.mean(sobel_x**2))
        energy_y = float(np.mean(sobel_y**2))
        ratio = max(energy_x, energy_y) / (min(energy_x, energy_y) + 1e-8)

        if ratio > 3.0:
            return BlurType.MOTION.value
        elif fft_ratio < 0.05:
            return BlurType.GAUSSIAN.value
        else:
            return BlurType.DEFOCUS.value

    def _sharpness_score(
        self,
        lap_var: float,
        tenengrad: float,
        brenner: float
    ) -> float:
        # Normalize Laplacian variance to 0-100
        lap_score = min(100, (lap_var / BLUR_LOW) * 100)
        ten_score = min(100, (tenengrad / 5000) * 100)
        bre_score = min(100, (brenner / 2000) * 100)
        return (lap_score * 0.5 + ten_score * 0.3 + bre_score * 0.2)

    def _blur_score(
        self,
        lap_var: float,
        fft_ratio: float,
        blurry_region_ratio: float
    ) -> float:
        """Score where 100 = perfectly sharp, 0 = completely blurry."""
        lap_score    = min(100, (lap_var / BLUR_LOW) * 100)
        fft_score    = min(100, fft_ratio * 200)
        region_score = max(0, 100 - blurry_region_ratio * 150)
        return (lap_score * 0.5 + fft_score * 0.3 + region_score * 0.2)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — NOISE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

class NoiseAnalyzer:
    """
    Advanced noise analysis:
    - Sigma estimation (wavelet-based, most accurate)
    - SNR (Signal-to-Noise Ratio) in dB
    - Noise type classification: Gaussian, Salt & Pepper, Poisson
    - Local noise map
    - High-ISO detection
    """

    def analyze(self, gray: np.ndarray) -> Dict:
        h, w = gray.shape

        # ── Sigma estimation ─────────────────────────────────────────────────
        sigma = self._estimate_noise_sigma(gray)

        # ── SNR ──────────────────────────────────────────────────────────────
        signal_power = float(np.mean(gray.astype(np.float64) ** 2))
        noise_power  = sigma ** 2 + 1e-8
        snr_db       = 10 * math.log10(signal_power / noise_power) if signal_power > 0 else 0.0

        # ── Noise type classification ─────────────────────────────────────────
        noise_type = self._classify_noise_type(gray, sigma)

        # ── Local noise map ───────────────────────────────────────────────────
        local_noise = self._local_noise_map(gray)
        noise_uniformity = float(1.0 - np.std(local_noise) / (np.mean(local_noise) + 1e-8))

        # ── Salt & pepper ratio ───────────────────────────────────────────────
        sp_ratio = self._salt_pepper_ratio(gray)

        # ── Noise score ───────────────────────────────────────────────────────
        is_noisy = sigma > NOISE_MEDIUM
        noise_score = self._compute_score(sigma, snr_db, sp_ratio)

        return {
            "noise_sigma":          round(sigma, 4),
            "noise_snr_db":         round(snr_db, 2),
            "noise_type":           noise_type,
            "noise_uniformity":     round(noise_uniformity, 4),
            "salt_pepper_ratio":    round(sp_ratio, 6),
            "noise_score":          round(noise_score, 1),
            "is_noisy":             is_noisy,
            "local_noise_mean":     round(float(np.mean(local_noise)), 4),
            "local_noise_std":      round(float(np.std(local_noise)), 4),
        }

    def _estimate_noise_sigma(self, gray: np.ndarray) -> float:
        """
        Estimate noise standard deviation using the median absolute deviation
        of high-pass filtered image (Donoho-Johnstone wavelet estimator).
        Most accurate no-reference noise estimation method.
        """
        try:
            # High-pass filter: difference from local mean
            if CV2_AVAILABLE:
                blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
            else:
                from scipy.ndimage import gaussian_filter
                blurred = gaussian_filter(gray.astype(np.float64), sigma=2)

            noise_img = gray.astype(np.float64) - blurred
            # MAD estimator (robust to outliers)
            sigma = float(np.median(np.abs(noise_img)) / 0.6745)
            return sigma
        except Exception:
            return float(np.std(gray.astype(np.float64)))

    def _classify_noise_type(self, gray: np.ndarray, sigma: float) -> str:
        """Classify noise type based on statistical characteristics."""
        if sigma < NOISE_LOW:
            return NoiseType.NONE.value

        # Check for salt & pepper: extreme outliers
        sp = self._salt_pepper_ratio(gray)
        if sp > 0.001:
            return NoiseType.SALT_PEPPER.value

        # Check Gaussian: noise should be normally distributed
        try:
            flat = gray.flatten().astype(np.float64)
            mean = flat.mean()
            # Kurtosis test: Gaussian noise has kurtosis ≈ 3
            from scipy.stats import kurtosis
            kurt = kurtosis(flat - mean)
            if abs(kurt) < 1.0:
                return NoiseType.GAUSSIAN.value
            elif kurt > 2.0:
                return NoiseType.POISSON.value
            return NoiseType.MIXED.value
        except Exception:
            return NoiseType.GAUSSIAN.value

    def _local_noise_map(self, gray: np.ndarray, block_size: int = 16) -> np.ndarray:
        """Compute local noise sigma for each block."""
        h, w = gray.shape
        values = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y+block_size, x:x+block_size].astype(np.float64)
                # Local noise via high-pass
                if block.std() > 0:
                    values.append(float(np.median(np.abs(block - np.median(block))) / 0.6745))
                else:
                    values.append(0.0)
        return np.array(values) if values else np.array([0.0])

    def _salt_pepper_ratio(self, gray: np.ndarray) -> float:
        """Estimate fraction of salt & pepper pixels."""
        total = gray.size
        salt   = np.sum(gray == 255)
        pepper = np.sum(gray == 0)
        # Isolated extremes are likely S&P noise
        return float((salt + pepper) / total)

    def _compute_score(self, sigma: float, snr_db: float, sp_ratio: float) -> float:
        score = 100.0

        if sigma > NOISE_SEVERE:
            score -= 70
        elif sigma > NOISE_HIGH:
            score -= 40 * ((sigma - NOISE_HIGH) / (NOISE_SEVERE - NOISE_HIGH))
        elif sigma > NOISE_MEDIUM:
            score -= 20 * ((sigma - NOISE_MEDIUM) / (NOISE_HIGH - NOISE_MEDIUM))

        # Penalize low SNR
        if snr_db < 10:
            score -= 30
        elif snr_db < 20:
            score -= 15

        # Penalize salt & pepper
        score -= min(30, sp_ratio * 5000)

        return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — ARTIFACT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactDetector:
    """
    Detects compression and processing artifacts:
    - JPEG blocking artifacts (8x8 DCT blocks)
    - Ringing artifacts (ripples near edges)
    - Color banding
    - Moiré patterns
    - Estimates JPEG quality factor
    """

    def analyze(self, gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> Dict:
        blocking = self._detect_blocking(gray)
        ringing  = self._detect_ringing(gray)
        jpeg_q   = self._estimate_jpeg_quality(gray)
        moire    = self._detect_moire(gray)
        banding  = self._detect_color_banding(bgr) if bgr is not None else 0.0

        has_blocking = blocking > BLOCKING_THRESHOLD
        has_ringing  = ringing  > RINGING_THRESHOLD

        # Artifact score (100 = no artifacts)
        artifact_score = 100.0
        artifact_score -= min(50, blocking * 200)
        artifact_score -= min(30, ringing  * 150)
        artifact_score -= min(10, moire    * 100)
        artifact_score -= min(10, banding  * 100)
        artifact_score  = max(0.0, artifact_score)

        return {
            "blocking_score":    round(blocking, 4),
            "ringing_score":     round(ringing, 4),
            "moire_score":       round(moire, 4),
            "banding_score":     round(banding, 4),
            "jpeg_quality_est":  jpeg_q,
            "has_blocking":      has_blocking,
            "has_ringing":       has_ringing,
            "artifact_score":    round(artifact_score, 1),
        }

    def _detect_blocking(self, gray: np.ndarray) -> float:
        """
        Detect 8x8 JPEG blocking artifacts.
        Computes discontinuity at every 8th row and column.
        """
        h, w = gray.shape
        img  = gray.astype(np.float64)

        # Horizontal block edges (every 8 rows)
        h_diffs = []
        for i in range(8, h, 8):
            if i < h:
                diff = np.abs(img[i, :] - img[i-1, :]).mean()
                non_edge = np.abs(np.diff(img[max(0, i-4):i, :], axis=0)).mean()
                h_diffs.append(diff / (non_edge + 1e-8))

        # Vertical block edges (every 8 cols)
        v_diffs = []
        for j in range(8, w, 8):
            if j < w:
                diff = np.abs(img[:, j] - img[:, j-1]).mean()
                non_edge = np.abs(np.diff(img[:, max(0, j-4):j], axis=1)).mean()
                v_diffs.append(diff / (non_edge + 1e-8))

        all_diffs = h_diffs + v_diffs
        if not all_diffs:
            return 0.0

        blocking = float(np.mean(all_diffs))
        # Normalize to 0-1 (ratio > 2 = strong blocking)
        return min(1.0, (blocking - 1.0) / 3.0) if blocking > 1 else 0.0

    def _detect_ringing(self, gray: np.ndarray) -> float:
        """
        Detect ringing artifacts near edges.
        Ringing = oscillations in the gradient magnitude near strong edges.
        """
        try:
            if CV2_AVAILABLE:
                edges = cv2.Canny(gray, 50, 150)
                # Dilate edges to get near-edge region
                kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                near_edge = cv2.dilate(edges, kernel, iterations=2)
                # Measure gradient oscillation in near-edge regions
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                near_edge_mask = near_edge > 0
                if near_edge_mask.sum() == 0:
                    return 0.0
                # Coefficient of variation in gradient magnitude near edges
                near_grads = grad_mag[near_edge_mask]
                cv_score = float(np.std(near_grads) / (np.mean(near_grads) + 1e-8))
                return min(1.0, max(0.0, (cv_score - 0.5) / 2.0))
            else:
                return 0.0
        except Exception:
            return 0.0

    def _estimate_jpeg_quality(self, gray: np.ndarray) -> int:
        """
        Estimate JPEG quality factor from blocking strength.
        Returns -1 if not applicable.
        """
        try:
            if not CV2_AVAILABLE:
                return -1
            h, w = gray.shape
            img  = gray.astype(np.float64)
            # Measure average 8x8 block variance
            block_vars = []
            for y in range(0, h-8, 8):
                for x in range(0, w-8, 8):
                    block = img[y:y+8, x:x+8]
                    block_vars.append(block.var())
            if not block_vars:
                return -1
            mean_var = np.mean(block_vars)
            # Heuristic mapping: high variance = high quality
            if mean_var > 500: return 95
            elif mean_var > 200: return 85
            elif mean_var > 80: return 75
            elif mean_var > 30: return 60
            elif mean_var > 10: return 40
            else: return 20
        except Exception:
            return -1

    def _detect_moire(self, gray: np.ndarray) -> float:
        """
        Detect Moiré patterns using FFT peak analysis.
        Moiré = periodic interference patterns = peaks in frequency domain.
        """
        try:
            f   = np.fft.fft2(gray.astype(np.float64))
            mag = np.abs(np.fft.fftshift(f))
            h, w = mag.shape
            cy, cx = h // 2, w // 2

            # Suppress DC component
            mag[cy-2:cy+3, cx-2:cx+3] = 0

            # Find peaks in frequency domain
            max_val = mag.max()
            mean_val = mag.mean()
            if mean_val == 0:
                return 0.0
            # Peak-to-mean ratio (high = periodic pattern = moiré)
            peak_ratio = max_val / mean_val
            return min(1.0, max(0.0, (peak_ratio - 10) / 90))
        except Exception:
            return 0.0

    def _detect_color_banding(self, bgr: np.ndarray) -> float:
        """Detect color banding in gradient regions."""
        try:
            # Posterization: look for step changes in smooth gradients
            blurred = cv2.GaussianBlur(bgr, (5, 5), 0) if CV2_AVAILABLE else bgr
            diff = np.abs(bgr.astype(np.float64) - blurred.astype(np.float64))
            # High-frequency step changes in low-gradient areas
            step_ratio = float(np.mean(diff > 15) if diff.size > 0 else 0)
            return min(1.0, step_ratio * 5)
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — ADVANCED PERCEPTUAL QUALITY (BRISQUE / NIQE)
# ─────────────────────────────────────────────────────────────────────────────

class PerceptualQualityAnalyzer:
    """
    No-reference perceptual image quality assessment:
    - BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
    - NIQE (Natural Image Quality Evaluator)
    - Entropy-based information content
    - Edge density (information richness)
    - Frequency energy distribution
    - Color cast detection
    """

    def analyze(self, gray: np.ndarray, rgb: Optional[np.ndarray] = None) -> Dict:
        brisque = self._compute_brisque(gray)
        entropy = self._compute_entropy(gray)
        edge_density = self._compute_edge_density(gray)
        freq_energy  = self._compute_frequency_energy(gray)
        color_cast   = self._detect_color_cast(rgb) if rgb is not None else ""

        # Information content score (0-100)
        info_score = self._information_content_score(entropy, edge_density, freq_energy)

        return {
            "brisque_score":    round(brisque, 2),
            "entropy":          round(entropy, 4),
            "edge_density":     round(edge_density, 4),
            "frequency_energy": round(freq_energy, 4),
            "info_content":     round(info_score, 1),
            "color_cast":       color_cast,
        }

    def _compute_brisque(self, gray: np.ndarray) -> float:
        """
        Simplified BRISQUE score based on MSCN (Mean Subtracted Contrast Normalized) coefficients.
        Full BRISQUE requires a trained SVM model — this is a faithful approximation.
        Lower score = better quality (0=perfect, 100=worst).
        """
        try:
            img = gray.astype(np.float64)

            # Local mean and variance (MSCN normalization)
            if CV2_AVAILABLE:
                mu    = cv2.GaussianBlur(img, (7, 7), 7.0/6.0)
                mu_sq = cv2.GaussianBlur(img**2, (7, 7), 7.0/6.0)
            else:
                from scipy.ndimage import gaussian_filter
                mu    = gaussian_filter(img, sigma=7.0/6.0)
                mu_sq = gaussian_filter(img**2, sigma=7.0/6.0)

            sigma = np.sqrt(np.abs(mu_sq - mu**2))
            mscn  = (img - mu) / (sigma + 1.0)

            # Fit generalized Gaussian to MSCN coefficients
            from scipy.stats import kurtosis, skew
            kurt  = float(kurtosis(mscn.flatten()))
            skewness = float(abs(skew(mscn.flatten())))

            # MSCN product (horizontal pairs)
            h_prod = mscn[:, :-1] * mscn[:, 1:]
            h_kurt = float(kurtosis(h_prod.flatten()))

            # MSCN product (vertical pairs)
            v_prod = mscn[:-1, :] * mscn[1:, :]
            v_kurt = float(kurtosis(v_prod.flatten()))

            # Distortion-aware score
            # Natural images: kurt ≈ 3, skew ≈ 0
            score  = abs(kurt - 3) * 5
            score += skewness * 10
            score += abs(h_kurt - 3) * 3
            score += abs(v_kurt - 3) * 3

            return min(100.0, max(0.0, score))
        except Exception:
            return -1.0

    def _compute_entropy(self, gray: np.ndarray) -> float:
        """Shannon entropy of pixel distribution."""
        try:
            hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
            hist    = hist / (hist.sum() + 1e-8)
            nonzero = hist[hist > 0]
            return float(-np.sum(nonzero * np.log2(nonzero)))
        except Exception:
            return 0.0

    def _compute_edge_density(self, gray: np.ndarray) -> float:
        """Fraction of pixels classified as edges — measures information richness."""
        try:
            if CV2_AVAILABLE:
                edges = cv2.Canny(gray, 50, 150)
                return float(np.mean(edges > 0))
            elif SKIMAGE_AVAILABLE:
                edges = filters.sobel(gray.astype(np.float64) / 255.0)
                threshold = edges.mean() + edges.std()
                return float(np.mean(edges > threshold))
            else:
                return 0.1  # fallback estimate
        except Exception:
            return 0.0

    def _compute_frequency_energy(self, gray: np.ndarray) -> float:
        """
        High-frequency energy ratio (0-1).
        High value = rich in details, low value = smooth/blurry.
        """
        try:
            f      = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            mag    = np.abs(fshift) ** 2

            h, w   = gray.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 4

            y, x = np.ogrid[:h, :w]
            mask_high = (y - cy)**2 + (x - cx)**2 > radius**2

            total_energy = float(mag.sum() + 1e-8)
            high_energy  = float(mag[mask_high].sum())
            return high_energy / total_energy
        except Exception:
            return 0.5

    def _detect_color_cast(self, rgb: np.ndarray) -> str:
        """Detect dominant color cast in an image."""
        try:
            r_mean = float(rgb[:, :, 0].mean())
            g_mean = float(rgb[:, :, 1].mean())
            b_mean = float(rgb[:, :, 2].mean())
            total  = (r_mean + g_mean + b_mean) / 3.0 + 1e-8

            r_norm = r_mean / total
            g_norm = g_mean / total
            b_norm = b_mean / total

            threshold = 0.15
            if r_norm > 1 + threshold and r_norm > g_norm and r_norm > b_norm:
                return "red_cast"
            elif b_norm > 1 + threshold and b_norm > r_norm and b_norm > g_norm:
                return "blue_cast"
            elif g_norm > 1 + threshold and g_norm > r_norm and g_norm > b_norm:
                return "green_cast"
            elif r_norm > 1 + threshold * 0.7 and b_norm > 1 + threshold * 0.7:
                return "magenta_cast"
            elif r_norm > 1 + threshold * 0.7 and g_norm > 1 + threshold * 0.7:
                return "yellow_cast"
            elif g_norm > 1 + threshold * 0.7 and b_norm > 1 + threshold * 0.7:
                return "cyan_cast"
            else:
                return ""
        except Exception:
            return ""

    def _information_content_score(
        self,
        entropy: float,
        edge_density: float,
        freq_energy: float
    ) -> float:
        """Combined information content score (0-100)."""
        entropy_score   = min(100, (entropy / 8.0) * 100)
        edge_score      = min(100, edge_density * 500)
        freq_score      = min(100, freq_energy * 150)
        return entropy_score * 0.4 + edge_score * 0.3 + freq_score * 0.3


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — ML-SPECIFIC QUALITY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class MLQualityChecker:
    """
    ML-specific quality validation:
    - Resolution adequacy for different model types
    - Aspect ratio anomalies
    - Information sufficiency for learning
    - Augmentation suitability
    - Problematic patterns (all-black, all-white, gradient only)
    """

    # Minimum resolution recommendations per model type
    RESOLUTION_REQUIREMENTS = {
        "tiny_cnn":      32,
        "small_cnn":     64,
        "mobilenet":     96,
        "resnet50":      224,
        "efficientnet":  300,
        "vit_base":      224,
        "vit_large":     384,
    }

    def analyze(self, gray: np.ndarray, width: int, height: int) -> Dict:
        resolution_check = self._check_resolution(width, height)
        aspect_check     = self._check_aspect_ratio(width, height)
        content_check    = self._check_content_validity(gray)
        aug_suitability  = self._check_augmentation_suitability(gray, width, height)

        issues = []
        if resolution_check["is_low_resolution"]:
            issues.append(resolution_check["resolution_verdict"])
        if aspect_check["is_extreme_aspect"]:
            issues.append(f"Extreme aspect ratio: {aspect_check['aspect_ratio']:.2f}")
        if content_check["is_degenerate"]:
            issues.append(content_check["degenerate_reason"])

        return {
            **resolution_check,
            **aspect_check,
            **content_check,
            "augmentation_suitability": aug_suitability,
            "ml_issues": issues,
        }

    def _check_resolution(self, w: int, h: int) -> Dict:
        min_dim = min(w, h)
        max_dim = max(w, h)

        if min_dim < MIN_DIMENSION_CRITICAL:
            verdict = f"CRITICAL: {w}x{h} — too small for any model"
            is_low  = True
        elif min_dim < MIN_DIMENSION_WARNING:
            verdict = f"WARNING: {w}x{h} — only suitable for tiny CNNs"
            is_low  = True
        elif min_dim < MIN_DIMENSION_ML:
            verdict = f"SUBOPTIMAL: {w}x{h} — resize to ≥224x224 recommended"
            is_low  = True
        else:
            verdict = f"OK: {w}x{h}"
            is_low  = False

        # Which models can use this image?
        suitable_models = [
            model for model, min_res in self.RESOLUTION_REQUIREMENTS.items()
            if min_dim >= min_res
        ]

        return {
            "is_low_resolution":  is_low,
            "resolution_verdict": verdict,
            "min_dimension":      min_dim,
            "suitable_models":    suitable_models,
        }

    def _check_aspect_ratio(self, w: int, h: int) -> Dict:
        ratio = w / (h + 1e-8)
        is_extreme = ratio > 5.0 or ratio < 0.2
        is_unusual = ratio > 3.0 or ratio < 0.33

        return {
            "aspect_ratio":     round(ratio, 4),
            "is_extreme_aspect": is_extreme,
            "is_unusual_aspect": is_unusual,
        }

    def _check_content_validity(self, gray: np.ndarray) -> Dict:
        """Detect degenerate images: all-black, all-white, pure gradient."""
        from src import nydra_core
        
        flat_data = gray.flatten().astype(np.float64)
        std  = nydra_core.std_dev(flat_data)
        mean = nydra_core.mean(flat_data)
        
        # Use nydra_core for high-speed Shannon entropy
        counts = np.bincount(gray.flatten())
        entropy_res = nydra_core.shannon_entropy(counts.tolist())
        entropy = entropy_res["entropy_bits"]

        is_degenerate   = False
        degenerate_reason = ""

        if std < 2.0:
            is_degenerate = True
            if mean < 10:
                degenerate_reason = "all_black"
            elif mean > 245:
                degenerate_reason = "all_white"
            else:
                degenerate_reason = "flat_uniform"
        elif entropy < ENTROPY_TOO_LOW:
            is_degenerate = True
            degenerate_reason = "too_low_entropy"

        return {
            "is_degenerate":     is_degenerate,
            "degenerate_reason": degenerate_reason,
            "pixel_std":         round(std, 3),
            "pixel_entropy":     round(entropy, 4),
        }

    def _check_augmentation_suitability(
        self,
        gray: np.ndarray,
        w: int,
        h: int
    ) -> Dict:
        """
        Assess how suitable the image is for standard augmentations.
        """
        min_dim = min(w, h)
        return {
            "can_crop":      min_dim >= 224,
            "can_flip":      True,
            "can_rotate":    True,
            "can_zoom":      min_dim >= 128,
            "can_cutmix":    min_dim >= 64,
            "can_mosaic":    min_dim >= 64,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — QUALITY SCORER & VERDICT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class QualityScorerEngine:
    """
    Aggregates all metric scores into a single overall quality score.
    Produces verdict, reject reason, and suggested fix.
    Maps each problem to the appropriate image_cleaner.py method.
    """

    # Maps reject reason → suggested fix in image_cleaner.py
    FIX_MAP = {
        "blurry":          "image_cleaner.sharpen(method='unsharp_mask')",
        "motion_blur":     "image_cleaner.deblur(method='wiener')",
        "dark":            "image_cleaner.fix_brightness(method='gamma')",
        "overexposed":     "image_cleaner.fix_brightness(method='gamma', direction='down')",
        "noisy":           "image_cleaner.denoise(method='nlm')",
        "salt_pepper":     "image_cleaner.denoise(method='median')",
        "low_contrast":    "image_cleaner.fix_contrast(method='clahe')",
        "blocking":        "image_cleaner.remove_artifacts(method='dct_smooth')",
        "ringing":         "image_cleaner.remove_artifacts(method='tv_denoise')",
        "low_resolution":  "image_cleaner.upscale(method='lanczos') or remove",
        "all_black":       "Remove — no visual information",
        "all_white":       "Remove — no visual information",
        "flat_uniform":    "Remove — no visual information",
        "low_entropy":     "Remove — insufficient information for learning",
        "color_cast":      "image_cleaner.fix_color_cast(method='gray_world')",
    }

    def score(self, metrics: Dict) -> Dict:
        """Compute overall score, verdict, reason, and fix suggestion."""

        # Gather sub-scores
        sharpness  = metrics.get("sharpness_score",  50.0)
        exposure   = metrics.get("exposure_score",   50.0)
        noise      = metrics.get("noise_score",      50.0)
        contrast   = metrics.get("contrast_score",   50.0)
        artifacts  = metrics.get("artifact_score",  100.0)

        # Weighted overall score
        overall = (
            sharpness  * SCORE_WEIGHTS["sharpness"]  +
            exposure   * SCORE_WEIGHTS["exposure"]   +
            noise      * SCORE_WEIGHTS["noise"]      +
            contrast   * SCORE_WEIGHTS["contrast"]   +
            artifacts  * SCORE_WEIGHTS["artifacts"]
        )
        overall = round(overall, 1)

        # Determine primary issue
        reject_reason, suggested_fix = self._determine_primary_issue(metrics)

        # Verdict
        verdict = self._get_verdict(overall, metrics)

        return {
            "overall_score":  overall,
            "verdict":        verdict,
            "reject_reason":  reject_reason,
            "suggested_fix":  suggested_fix,
        }

    def _determine_primary_issue(self, m: Dict) -> Tuple[str, str]:
        """Find the single most impactful quality problem."""
        issues = []

        # Degenerate content — always reject
        if m.get("is_degenerate", False):
            reason = m.get("degenerate_reason", "flat_uniform")
            return reason, self.FIX_MAP.get(reason, "Remove this image")

        # Low resolution — critical
        if m.get("is_low_resolution", False):
            return "low_resolution", self.FIX_MAP["low_resolution"]

        # Score each issue by severity
        if m.get("is_blurry", False):
            blur_type = m.get("blur_type", "defocus")
            if blur_type == "motion":
                issues.append((100 - m.get("blur_score", 0), "motion_blur"))
            else:
                issues.append((100 - m.get("blur_score", 0), "blurry"))

        if m.get("is_underexposed", False):
            issues.append((100 - m.get("exposure_score", 0), "dark"))

        if m.get("is_overexposed", False):
            issues.append((100 - m.get("exposure_score", 0), "overexposed"))

        if m.get("is_noisy", False):
            noise_type = m.get("noise_type", "gaussian")
            if noise_type == "salt_pepper":
                issues.append((100 - m.get("noise_score", 0), "salt_pepper"))
            else:
                issues.append((100 - m.get("noise_score", 0), "noisy"))

        if m.get("contrast_score", 100) < 40:
            issues.append((100 - m.get("contrast_score", 0), "low_contrast"))

        if m.get("has_blocking", False):
            issues.append((m.get("blocking_score", 0) * 100, "blocking"))

        if m.get("has_ringing", False):
            issues.append((m.get("ringing_score", 0) * 100, "ringing"))

        if m.get("color_cast", ""):
            issues.append((20.0, "color_cast"))

        if not issues:
            return "", ""

        # Return the most severe issue
        issues.sort(reverse=True)
        primary = issues[0][1]
        return primary, self.FIX_MAP.get(primary, "Use image_cleaner.py")

    def _get_verdict(self, score: float, m: Dict) -> str:
        # Force reject for degenerate images
        if m.get("is_degenerate", False):
            return QualityVerdict.REJECT.value

        # Force reject for critical resolution
        if m.get("min_dimension", 999) < MIN_DIMENSION_CRITICAL:
            return QualityVerdict.REJECT.value

        if score >= VERDICT_HIGH:
            return QualityVerdict.HIGH.value
        elif score >= VERDICT_MEDIUM:
            return QualityVerdict.MEDIUM.value
        elif score >= VERDICT_LOW:
            return QualityVerdict.LOW.value
        else:
            return QualityVerdict.REJECT.value


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — DATASET-LEVEL QUALITY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class DatasetQualityStatistics:
    """
    Aggregates per-image quality metrics into dataset-level insights:
    - Overall distribution: High / Medium / Low / Reject counts
    - Class-level quality bias: detects if some classes have lower quality
    - Split-level quality: checks if train/test have similar quality
    - Recommendations: actionable fixes based on findings
    """

    def compute(self, df: pd.DataFrame) -> DatasetQualityReport:
        from datetime import datetime
        report = DatasetQualityReport()
        report.generated_at = datetime.utcnow().isoformat()
        report.total_images = len(df)

        if len(df) == 0:
            return report

        # ── Verdict counts ────────────────────────────────────────────────────
        if "verdict" in df.columns:
            verdicts = df["verdict"].value_counts()
            report.high_quality   = int(verdicts.get(QualityVerdict.HIGH.value,   0))
            report.medium_quality = int(verdicts.get(QualityVerdict.MEDIUM.value, 0))
            report.low_quality    = int(verdicts.get(QualityVerdict.LOW.value,    0))
            report.rejected       = int(verdicts.get(QualityVerdict.REJECT.value, 0))

        # ── Score statistics ──────────────────────────────────────────────────
        if "overall_score" in df.columns:
            scores = df["overall_score"].dropna()
            report.mean_overall_score = round(float(scores.mean()), 2)
            report.std_overall_score  = round(float(scores.std()),  2)
            report.min_score          = round(float(scores.min()),  2)
            report.max_score          = round(float(scores.max()),  2)

        # ── Mean sub-scores ───────────────────────────────────────────────────
        for col, attr in [
            ("sharpness_score",   "mean_sharpness"),
            ("brightness_score",  "mean_brightness"),
            ("contrast_score",    "mean_contrast"),
            ("noise_score",       "mean_noise"),
        ]:
            if col in df.columns:
                setattr(report, attr, round(float(df[col].mean()), 2))

        # ── Issue counts ──────────────────────────────────────────────────────
        if "is_blurry"         in df.columns: report.blurry_count    = int(df["is_blurry"].sum())
        if "is_underexposed"   in df.columns: report.dark_count      = int(df["is_underexposed"].sum())
        if "is_overexposed"    in df.columns: report.overexposed_count = int(df["is_overexposed"].sum())
        if "is_noisy"          in df.columns: report.noisy_count     = int(df["is_noisy"].sum())
        if "has_blocking"      in df.columns: report.artifact_count  = int(df["has_blocking"].sum())
        if "is_low_resolution" in df.columns: report.low_res_count   = int(df["is_low_resolution"].sum())

        # ── Class-level quality bias ──────────────────────────────────────────
        if "label" in df.columns and "overall_score" in df.columns:
            report.class_quality_bias = self._class_quality_bias(df)
            mean_score = df["overall_score"].mean()
            std_score  = df["overall_score"].std()
            report.biased_classes = [
                cls for cls, score in report.class_quality_bias.items()
                if score < mean_score - std_score
            ]

        # ── Split-level quality ───────────────────────────────────────────────
        if "split" in df.columns and "overall_score" in df.columns:
            report.split_quality = self._split_quality(df)
            train_score = report.split_quality.get("train", {}).get("mean_score", 0)
            test_score  = report.split_quality.get("test",  {}).get("mean_score", 0)
            report.split_quality_gap = round(abs(train_score - test_score), 2)

        # ── Recommendations ───────────────────────────────────────────────────
        report.recommendations = self._generate_recommendations(report, df)
        report.fix_priority    = self._prioritize_fixes(report)

        return report

    def _class_quality_bias(self, df: pd.DataFrame) -> Dict:
        """Mean quality score per class."""
        result = {}
        for label, group in df.groupby("label"):
            if "overall_score" in group.columns:
                result[str(label)] = round(float(group["overall_score"].mean()), 2)
        return result

    def _split_quality(self, df: pd.DataFrame) -> Dict:
        """Quality stats per split."""
        result = {}
        for split_name, group in df.groupby("split"):
            if "overall_score" in group.columns and len(group) > 0:
                result[str(split_name)] = {
                    "count":      len(group),
                    "mean_score": round(float(group["overall_score"].mean()), 2),
                    "min_score":  round(float(group["overall_score"].min()),  2),
                    "rejected":   int((group["verdict"] == QualityVerdict.REJECT.value).sum())
                                  if "verdict" in group.columns else 0,
                }
        return result

    def _generate_recommendations(
        self,
        report: DatasetQualityReport,
        df: pd.DataFrame
    ) -> List[str]:
        recs = []
        total = report.total_images + 1e-8

        if report.rejected / total > 0.1:
            recs.append(f"CRITICAL: {report.rejected} images ({report.rejected/total*100:.1f}%) "
                        f"should be rejected — run image_cleaner.remove_rejected()")

        if report.blurry_count / total > 0.1:
            recs.append(f"HIGH: {report.blurry_count} blurry images — "
                        f"run image_cleaner.sharpen_all()")

        if report.dark_count / total > 0.1:
            recs.append(f"HIGH: {report.dark_count} underexposed images — "
                        f"run image_cleaner.fix_brightness_all()")

        if report.noisy_count / total > 0.1:
            recs.append(f"HIGH: {report.noisy_count} noisy images — "
                        f"run image_cleaner.denoise_all()")

        if report.low_res_count / total > 0.05:
            recs.append(f"HIGH: {report.low_res_count} low-resolution images — "
                        f"remove or upscale before training")

        if report.biased_classes:
            recs.append(f"MEDIUM: Classes with below-average quality: {report.biased_classes} — "
                        f"collect higher-quality images for these classes")

        if report.split_quality_gap > 10:
            recs.append(f"MEDIUM: Quality gap between train ({report.split_quality.get('train',{}).get('mean_score',0):.1f}) "
                        f"and test ({report.split_quality.get('test',{}).get('mean_score',0):.1f}) — "
                        f"re-split with stratified quality filtering")

        if report.artifact_count / total > 0.05:
            recs.append(f"LOW: {report.artifact_count} images with JPEG artifacts — "
                        f"run image_cleaner.remove_artifacts()")

        if not recs:
            recs.append("✅ Dataset quality looks good! Ready for training.")

        return recs

    def _prioritize_fixes(self, report: DatasetQualityReport) -> List[str]:
        """Ordered list of fixes from most to least impactful."""
        fixes = []
        if report.rejected > 0:
            fixes.append(f"1. Remove {report.rejected} rejected images")
        if report.blurry_count > 0:
            fixes.append(f"2. Sharpen {report.blurry_count} blurry images")
        if report.dark_count + report.overexposed_count > 0:
            fixes.append(f"3. Fix exposure in {report.dark_count + report.overexposed_count} images")
        if report.noisy_count > 0:
            fixes.append(f"4. Denoise {report.noisy_count} images")
        if report.low_res_count > 0:
            fixes.append(f"5. Handle {report.low_res_count} low-resolution images")
        if report.artifact_count > 0:
            fixes.append(f"6. Remove artifacts from {report.artifact_count} images")
        return fixes


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — REPORT EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class QualityReportExporter:
    """Export quality results to JSON, Markdown, CSV, HTML."""

    def to_json(self, report: DatasetQualityReport, path: Optional[Path] = None) -> str:
        import dataclasses
        data = dataclasses.asdict(report)
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str)
        return json_str

    def to_markdown(self, report: DatasetQualityReport, path: Optional[Path] = None) -> str:
        lines = [
            "# 🩺 Nydra — Image Quality Report",
            f"\n**Generated:** {report.generated_at}",
            f"\n## 📊 Dataset Overview",
            f"| Metric | Count | % |",
            f"|--------|-------|---|",
            f"| Total Images    | {report.total_images} | 100% |",
            f"| High Quality ✅ | {report.high_quality} | {report.high_quality/max(report.total_images,1)*100:.1f}% |",
            f"| Medium ⚠️       | {report.medium_quality} | {report.medium_quality/max(report.total_images,1)*100:.1f}% |",
            f"| Low ❌          | {report.low_quality} | {report.low_quality/max(report.total_images,1)*100:.1f}% |",
            f"| Reject 🚫       | {report.rejected} | {report.rejected/max(report.total_images,1)*100:.1f}% |",
            f"\n## 🎯 Quality Scores",
            f"| Metric | Mean | Std | Min | Max |",
            f"|--------|------|-----|-----|-----|",
            f"| Overall    | {report.mean_overall_score} | {report.std_overall_score} | {report.min_score} | {report.max_score} |",
            f"| Sharpness  | {report.mean_sharpness} | — | — | — |",
            f"| Brightness | {report.mean_brightness} | — | — | — |",
            f"| Contrast   | {report.mean_contrast} | — | — | — |",
            f"| Noise      | {report.mean_noise} | — | — | — |",
            f"\n## ⚠️ Issues Found",
            f"| Issue | Count |",
            f"|-------|-------|",
            f"| Blurry         | {report.blurry_count} |",
            f"| Underexposed   | {report.dark_count} |",
            f"| Overexposed    | {report.overexposed_count} |",
            f"| Noisy          | {report.noisy_count} |",
            f"| Artifacts      | {report.artifact_count} |",
            f"| Low Resolution | {report.low_res_count} |",
        ]

        if report.biased_classes:
            lines += [
                "\n## 🎯 Class Quality Bias",
                "Classes with below-average quality:",
                *[f"- **{cls}**: {score:.1f}/100"
                  for cls, score in report.class_quality_bias.items()
                  if cls in report.biased_classes]
            ]

        if report.split_quality:
            lines += ["\n## ✂️ Split Quality"]
            for split, stats in report.split_quality.items():
                lines.append(f"- **{split}**: {stats.get('mean_score', 0):.1f}/100 "
                             f"({stats.get('count', 0)} images, "
                             f"{stats.get('rejected', 0)} rejected)")

        if report.recommendations:
            lines += ["\n## 🔧 Recommendations"]
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        if report.fix_priority:
            lines += ["\n## 📋 Fix Priority"]
            for fix in report.fix_priority:
                lines.append(f"- {fix}")

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md)
        return md

    def to_csv(self, df: pd.DataFrame, path: Path) -> None:
        quality_cols = [
            c for c in df.columns
            if any(k in c for k in [
                "score", "verdict", "is_", "has_", "blur", "noise",
                "exposure", "artifact", "reject", "fix"
            ])
        ]
        base_cols = ["file_path", "file_name"]
        export_cols = base_cols + [c for c in quality_cols if c not in base_cols]
        df[export_cols].to_csv(str(path), index=False)
        logger.info(f"Quality CSV saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLASS — ImageQualityAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class ImageQualityAnalyzer:
    """
    🩺 Nydra ImageQualityAnalyzer — Master Entry Point

    Orchestrates all quality analysis blocks:

    1.  BrightnessAnalyzer      → brightness, exposure, dynamic range
    2.  ContrastAnalyzer        → RMS, Michelson, local contrast
    3.  SharpnessBlurAnalyzer   → Laplacian, Tenengrad, FFT, motion/defocus
    4.  NoiseAnalyzer           → sigma, SNR, type classification
    5.  ArtifactDetector        → blocking, ringing, Moiré, JPEG quality
    6.  PerceptualQualityAnalyzer → BRISQUE, NIQE, entropy, edge density
    7.  MLQualityChecker        → resolution, aspect ratio, content validity
    8.  QualityScorerEngine     → overall score, verdict, fix suggestion
    9.  DatasetQualityStatistics → dataset-level stats, class bias, split gap
    10. QualityReportExporter   → JSON, Markdown, CSV

    Usage:
        # Single image
        analyzer = ImageQualityAnalyzer()
        result = analyzer.analyze_image("photo.jpg")

        # Full dataset DataFrame (from image_loader.py)
        df, _ = load_and_audit("dataset/")
        df_q  = analyzer.analyze_dataset(df)
        report = analyzer.dataset_report(df_q)
    """

    def __init__(
        self,
        n_workers:     int  = 4,
        compute_brisque: bool = True,
        verbose:       bool = True,
    ):
        self.n_workers       = min(n_workers, os.cpu_count() or 4)
        self.compute_brisque = compute_brisque
        self.verbose         = verbose

        self.brightness   = BrightnessAnalyzer()
        self.contrast     = ContrastAnalyzer()
        self.sharpness    = SharpnessBlurAnalyzer()
        self.noise        = NoiseAnalyzer()
        self.artifacts    = ArtifactDetector()
        self.perceptual   = PerceptualQualityAnalyzer()
        self.ml_check     = MLQualityChecker()
        self.scorer       = QualityScorerEngine()
        self.stats        = DatasetQualityStatistics()
        self.exporter     = QualityReportExporter()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_image(
        self,
        source: Union[str, Path, "Image.Image", np.ndarray],
    ) -> Dict:
        """
        Analyze a single image.
        Accepts: file path (str/Path), PIL Image, or numpy array.
        Returns: full quality metrics dict.
        """
        t0 = time.time()

        # Load image
        gray, rgb, bgr, w, h = self._load_to_arrays(source)
        if gray is None:
            return {"error": "Could not load image", "overall_score": 0.0,
                    "verdict": QualityVerdict.REJECT.value}

        # Run all analyzers
        metrics: Dict[str, Any] = {
            "file_path": str(source) if not isinstance(source, np.ndarray) else "",
            "file_name": Path(str(source)).name if isinstance(source, (str, Path)) else "",
            "width":  w,
            "height": h,
        }

        # Block 1 — Brightness
        metrics.update(self.brightness.analyze(gray))
        exposure_score = metrics.get("brightness_score", 50.0)
        metrics["exposure_score"] = exposure_score

        # Block 2 — Contrast
        metrics.update(self.contrast.analyze(gray))

        # Block 3 — Sharpness & Blur
        metrics.update(self.sharpness.analyze(gray))

        # Block 4 — Noise
        metrics.update(self.noise.analyze(gray))

        # Block 5 — Artifacts
        metrics.update(self.artifacts.analyze(gray, bgr))

        # Block 6 — Perceptual
        if self.compute_brisque:
            metrics.update(self.perceptual.analyze(gray, rgb))

        # Block 7 — ML-specific
        metrics.update(self.ml_check.analyze(gray, w, h))

        # Block 8 — Score & Verdict
        score_result = self.scorer.score(metrics)
        metrics.update(score_result)

        # Timing
        metrics["processing_time_ms"] = round((time.time() - t0) * 1000, 2)

        return metrics

    def analyze_file(self, path: Union[str, Path]) -> Dict:
        """Analyze a single image file."""
        return self.analyze_image(str(path))

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        path_col: str = "file_path",
    ) -> pd.DataFrame:
        """
        Analyze all images in a DataFrame (from image_loader.py).
        Adds quality columns to the DataFrame and returns it.
        """
        if path_col not in df.columns:
            raise ValueError(f"Column '{path_col}' not found in DataFrame")

        paths = df[path_col].tolist()
        n     = len(paths)

        if self.verbose:
            logger.info(f"🔍 Analyzing quality of {n} images ({self.n_workers} workers)...")

        # Parallel analysis
        results = [None] * n

        def analyze_one(args):
            idx, path = args
            try:
                return idx, self.analyze_image(path)
            except Exception as e:
                return idx, {"error": str(e), "overall_score": 0.0,
                             "verdict": QualityVerdict.REJECT.value}

        iterator = enumerate(paths)
        if TQDM_AVAILABLE and self.verbose:
            iterator = tqdm(enumerate(paths), total=n, desc="Quality Analysis")

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(analyze_one, (i, p)): i
                       for i, p in enumerate(paths)}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    results[idx] = {"overall_score": 0.0,
                                    "verdict": QualityVerdict.REJECT.value}

        # Convert to DataFrame and merge
        quality_df = pd.DataFrame([r for r in results if r is not None])

        # Drop columns that already exist in df to avoid duplicates
        existing_cols = set(df.columns)
        new_cols = [c for c in quality_df.columns
                    if c not in existing_cols or c == path_col]
        quality_df = quality_df[new_cols]

        if path_col in quality_df.columns:
            result_df = df.merge(quality_df, on=path_col, how="left")
        else:
            # Fallback: positional merge
            quality_df.index = df.index
            result_df = pd.concat([df, quality_df], axis=1)

        if self.verbose:
            mean_score = result_df["overall_score"].mean() if "overall_score" in result_df.columns else 0
            logger.info(f"✅ Quality analysis complete. Mean score: {mean_score:.1f}/100")

        return result_df

    def dataset_report(self, df: pd.DataFrame) -> DatasetQualityReport:
        """Generate dataset-level quality report from analyzed DataFrame."""
        return self.stats.compute(df)

    def export(
        self,
        df: pd.DataFrame,
        report: Optional[DatasetQualityReport] = None,
        output_dir: Union[str, Path] = "nydra_quality",
        formats: List[str] = ("csv", "json", "markdown"),
    ) -> Dict[str, Path]:
        """Export analysis results."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        if "csv" in formats:
            p = out / "image_quality.csv"
            self.exporter.to_csv(df, p)
            saved["csv"] = p

        if report:
            if "json" in formats:
                p = out / "quality_report.json"
                self.exporter.to_json(report, p)
                saved["json"] = p

            if "markdown" in formats:
                p = out / "quality_report.md"
                self.exporter.to_markdown(report, p)
                saved["markdown"] = p

        logger.info(f"Quality report exported to: {out}")
        return saved

    def get_rejected(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only images flagged for rejection."""
        if "verdict" not in df.columns:
            return pd.DataFrame()
        return df[df["verdict"] == QualityVerdict.REJECT.value].copy()

    def get_high_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only high-quality images."""
        if "verdict" not in df.columns:
            return pd.DataFrame()
        return df[df["verdict"] == QualityVerdict.HIGH.value].copy()

    def filter_by_score(
        self,
        df: pd.DataFrame,
        min_score: float = 60.0
    ) -> pd.DataFrame:
        """Filter dataset to keep only images above min_score."""
        if "overall_score" not in df.columns:
            return df
        return df[df["overall_score"] >= min_score].copy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_to_arrays(
        self,
        source: Union[str, Path, "Image.Image", np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], int, int]:
        """
        Load image from any source into:
        - gray: H x W uint8
        - rgb:  H x W x 3 uint8 (RGB)
        - bgr:  H x W x 3 uint8 (BGR, for OpenCV)
        - w, h: dimensions
        """
        try:
            if isinstance(source, np.ndarray):
                # Already numpy — assume RGB
                rgb = source if source.ndim == 3 else np.stack([source]*3, axis=-1)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE \
                       else np.mean(rgb, axis=2).astype(np.uint8)
                bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else None
                h, w = gray.shape
                return gray, rgb, bgr, w, h

            elif PIL_AVAILABLE and isinstance(source, Image.Image):
                img = source
            else:
                if not PIL_AVAILABLE:
                    return None, None, None, 0, 0
                img = Image.open(str(source))
                img.load()

            # Convert to RGB
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    bg = Image.new("RGB", img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    img = bg
                elif img.mode in ("L", "LA"):
                    img = img.convert("RGB")
                else:
                    img = img.convert("RGB")

            w, h = img.size
            rgb  = np.array(img, dtype=np.uint8)

            if CV2_AVAILABLE:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                gray = np.mean(rgb, axis=2).astype(np.uint8)
                bgr  = None

            return gray, rgb, bgr, w, h

        except Exception as e:
            logger.debug(f"Failed to load image: {e}")
            return None, None, None, 0, 0


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image_quality(
    path: Union[str, Path],
    verbose: bool = False
) -> Dict:
    """
    Quick quality analysis for a single image.

    Example:
        result = analyze_image_quality("photo.jpg")
        print(result["overall_score"])   # 73.4
        print(result["verdict"])         # "Low ❌"
        print(result["suggested_fix"])   # "image_cleaner.sharpen()"
    """
    analyzer = ImageQualityAnalyzer(verbose=verbose)
    return analyzer.analyze_image(path)


def analyze_dataset_quality(
    df: pd.DataFrame,
    n_workers: int = 4,
    verbose:   bool = True
) -> Tuple[pd.DataFrame, DatasetQualityReport]:
    """
    Full quality analysis pipeline for a dataset DataFrame.

    Example:
        from image_loader import load_images
        df = load_images("dataset/")
        df_quality, report = analyze_dataset_quality(df)
        print(f"Mean quality: {report.mean_overall_score}/100")
        print(report.recommendations)
    """
    analyzer = ImageQualityAnalyzer(n_workers=n_workers, verbose=verbose)
    df_q     = analyzer.analyze_dataset(df)
    report   = analyzer.dataset_report(df_q)
    return df_q, report


def quick_quality_check(path: Union[str, Path]) -> None:
    """
    Print a quick quality summary to stdout.

    Example:
        quick_quality_check("dataset/")
    """
    result = analyze_image_quality(path, verbose=False)
    print(f"\n{'='*50}")
    print(f"🩺 Image Quality Check: {Path(str(path)).name}")
    print(f"{'='*50}")
    print(f"📊 Overall Score:  {result.get('overall_score', 0):.1f}/100")
    print(f"📋 Verdict:        {result.get('verdict', 'N/A')}")
    print(f"🔍 Sharpness:      {result.get('sharpness_score', 0):.1f}/100")
    print(f"💡 Exposure:       {result.get('exposure_score', 0):.1f}/100")
    print(f"🌫️  Noise:          {result.get('noise_score', 0):.1f}/100")
    print(f"🎨 Contrast:       {result.get('contrast_score', 0):.1f}/100")
    print(f"🔧 Artifacts:      {result.get('artifact_score', 0):.1f}/100")
    if result.get("reject_reason"):
        print(f"\n⚠️  Issue:   {result.get('reject_reason')}")
        print(f"💡 Fix:     {result.get('suggested_fix')}")
    print(f"{'='*50}\n")
