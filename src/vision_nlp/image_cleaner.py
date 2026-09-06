"""
image_cleaner.py — Nydra v0.6.0
=====================================
Full image dataset cleaning pipeline.

Fixes every quality issue detected by image_quality.py and image_analyzer.py:
  - Brightness & Exposure
  - Contrast
  - Sharpness / Blur
  - Noise
  - JPEG / Compression Artifacts
  - Color Cast
  - Resize & Normalization
  - Remove / Filter (duplicates, corrupted, rejected)
  - Batch Processing with multi-threading
  - Before/After Comparison & Report

Version: 0.6.0
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    HAS_CV2 = True
except (ImportError, Exception):
    # Mock cv2 with required constants for class definitions
    class MockCV2:
        INTER_LANCZOS4 = 4
        INTER_CUBIC = 2
        INTER_LINEAR = 1
        INTER_NEAREST = 0
        IMWRITE_JPEG_QUALITY = 1
        IMWRITE_PNG_COMPRESSION = 16
        IMWRITE_WEBP_QUALITY = 64
        CV_64F = 6
    cv2 = MockCV2()  # type: ignore
    HAS_CV2 = False

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from scipy import ndimage, signal

try:
    from skimage import restoration, exposure, util, filters
    from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
    HAS_SKIMAGE = True
except (ImportError, Exception):
    HAS_SKIMAGE = False
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="[Nydra] %(levelname)s — %(message)s")
logger = logging.getLogger("nydra.image_cleaner")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".heic"}
DEFAULT_OUTPUT_QUALITY = 95
DEFAULT_RESIZE = None         # None means no resize
DEFAULT_MIN_SCORE = 50        # images below this score are filtered out
MAX_WORKERS = min(8, os.cpu_count() or 4)


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — BRIGHTNESS & EXPOSURE FIXER  (~300 lines)
# ──────────────────────────────────────────────────────────────────────────────

class BrightnessFixer:
    """
    Fix dark, overexposed, and unevenly lit images.

    Strategies
    ----------
    - Gamma Correction          : auto-gamma from mean brightness
    - CLAHE                     : adaptive local histogram equalization (LAB L-channel)
    - Retinex (Multi-Scale)     : simulates human visual system
    - Auto Levels               : stretches histogram to 0-255
    - HDR Tone Mapping          : for high-dynamic-range scenes

    Smart selector
    --------------
    dark image      → Gamma (up) + CLAHE
    overexposed     → Gamma (down) + Auto Levels
    uneven lighting → Retinex
    """

    # Brightness thresholds
    DARK_THRESHOLD  = 80    # mean brightness below this → dark
    BRIGHT_THRESHOLD = 180  # mean brightness above this → overexposed

    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8):
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    # ── public ───────────────────────────────────────────────────────────────

    def fix(self, img: np.ndarray, method: str = "smart") -> np.ndarray:
        """
        Apply brightness correction.

        Parameters
        ----------
        img    : BGR uint8 image (OpenCV format)
        method : 'smart' | 'gamma' | 'clahe' | 'retinex' | 'auto_levels' | 'hdr'
        """
        if method == "smart":
            return self._smart_fix(img)
        elif method == "gamma":
            return self._gamma_correction(img)
        elif method == "clahe":
            return self._clahe(img)
        elif method == "retinex":
            return self._retinex(img)
        elif method == "auto_levels":
            return self._auto_levels(img)
        elif method == "hdr":
            return self._hdr_tonemap(img)
        else:
            raise ValueError(f"Unknown brightness method: {method}")

    # ── private ──────────────────────────────────────────────────────────────

    def _smart_fix(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())

        if mean < self.DARK_THRESHOLD:
            logger.debug(f"Brightness fix: dark (mean={mean:.1f}) → gamma + CLAHE")
            img = self._gamma_correction(img, auto=True)
            img = self._clahe(img)
        elif mean > self.BRIGHT_THRESHOLD:
            logger.debug(f"Brightness fix: overexposed (mean={mean:.1f}) → gamma down + auto levels")
            img = self._gamma_correction(img, gamma=1.6)
            img = self._auto_levels(img)
        else:
            # Check for uneven lighting via std-dev of local means
            if self._has_uneven_lighting(gray):
                logger.debug(f"Brightness fix: uneven lighting → retinex")
                img = self._retinex(img)
        return img

    def _has_uneven_lighting(self, gray: np.ndarray, n_blocks: int = 4) -> bool:
        """Split into NxN blocks, check if local means vary too much."""
        h, w = gray.shape
        bh, bw = h // n_blocks, w // n_blocks
        means = []
        for i in range(n_blocks):
            for j in range(n_blocks):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                means.append(float(block.mean()))
        return (max(means) - min(means)) > 60

    def _gamma_correction(self, img: np.ndarray, gamma: float = None, auto: bool = False) -> np.ndarray:
        if auto or gamma is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean = float(gray.mean()) / 255.0
            # optimal gamma so that corrected mean ≈ 0.5
            mean = max(mean, 1e-6)
            gamma = math.log(0.5) / math.log(mean)
            gamma = float(np.clip(gamma, 0.2, 5.0))

        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(img, table)

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE on L-channel of LAB colorspace."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Auto-tune clip_limit based on current contrast
        std = float(l.std())
        clip = max(1.0, min(4.0, self.clip_limit * (1.0 + (128 - std) / 128)))

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(self.tile_size, self.tile_size))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _retinex(self, img: np.ndarray, sigmas: Tuple = (15, 80, 250)) -> np.ndarray:
        """Multi-Scale Retinex (MSR) with color restoration."""
        img_f = img.astype(np.float64) + 1.0
        log_img = np.log(img_f)

        msr = np.zeros_like(log_img)
        for sigma in sigmas:
            blurred = ndimage.gaussian_filter(img_f, sigma=[sigma, sigma, 0])
            blurred = np.clip(blurred, 1.0, None)
            msr += log_img - np.log(blurred)

        msr /= len(sigmas)

        # Color restoration factor
        img_sum = img_f.sum(axis=2, keepdims=True)
        img_sum = np.clip(img_sum, 1.0, None)
        crf = np.log(125.0 * img_f) - np.log(img_sum)
        msrcp = msr * crf

        # Normalize to 0-255
        for c in range(3):
            ch = msrcp[:, :, c]
            lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            msrcp[:, :, c] = ch

        return np.clip(msrcp, 0, 255).astype(np.uint8)

    def _auto_levels(self, img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
        """Stretch histogram using percentile clipping."""
        result = np.zeros_like(img)
        for c in range(img.shape[2]):
            ch = img[:, :, c].astype(np.float32)
            lo = np.percentile(ch, p_low)
            hi = np.percentile(ch, p_high)
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            result[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        return result

    def _hdr_tonemap(self, img: np.ndarray, algorithm: str = "drago") -> np.ndarray:
        """HDR Tone Mapping using Drago / Reinhard / Mantiuk."""
        img_f = img.astype(np.float32) / 255.0
        hdr   = img_f.copy()

        if algorithm == "drago":
            tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.85)
        elif algorithm == "reinhard":
            tonemap = cv2.createTonemapReinhard(gamma=1.5, intensity=0.0, light_adapt=0.8, color_adapt=0.0)
        else:
            tonemap = cv2.createTonemapMantiuk(gamma=2.2, scale=0.85, saturation=1.2)

        ldr = tonemap.process(hdr)
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        return ldr

    def needs_fix(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        return mean < self.DARK_THRESHOLD or mean > self.BRIGHT_THRESHOLD or self._has_uneven_lighting(gray)


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — CONTRAST ENHANCER  (~200 lines)
# ──────────────────────────────────────────────────────────────────────────────

class ContrastEnhancer:
    """
    Fix low-contrast (flat/washed-out) images.

    Methods
    -------
    - CLAHE                    : primary, adaptive local contrast
    - Unsharp Masking          : local contrast enhancement
    - Sigmoid Contrast Stretch : S-curve adjustment
    - Linear Contrast Stretch  : p2-p98 percentile stretch
    - Per-channel Stretch      : independent RGB channels
    """

    LOW_CONTRAST_THRESHOLD = 40   # RMS contrast below this → needs fix

    def fix(self, img: np.ndarray, method: str = "smart",
            sharpness_score: float = None) -> np.ndarray:
        if method == "smart":
            return self._smart_fix(img, sharpness_score)
        elif method == "clahe":
            return self._clahe(img)
        elif method == "unsharp":
            return self._unsharp_masking(img, sharpness_score)
        elif method == "sigmoid":
            return self._sigmoid_stretch(img)
        elif method == "linear":
            return self._linear_stretch(img)
        elif method == "per_channel":
            return self._per_channel_stretch(img)
        else:
            raise ValueError(f"Unknown contrast method: {method}")

    def _smart_fix(self, img: np.ndarray, sharpness_score: float = None) -> np.ndarray:
        rms = self._rms_contrast(img)
        if rms >= self.LOW_CONTRAST_THRESHOLD:
            return img
        if sharpness_score is not None and sharpness_score < 40:
            img = self._unsharp_masking(img, sharpness_score)
        img = self._clahe(img)
        return img

    def _rms_contrast(self, img: np.ndarray) -> float:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(gray.std())

    def _clahe(self, img: np.ndarray, clip_limit: float = 3.0, tile: int = 8) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _unsharp_masking(self, img: np.ndarray, sharpness_score: float = None) -> np.ndarray:
        if sharpness_score is None:
            sharpness_score = 50.0
        amount = 0.5 + (100 - sharpness_score) / 100.0 * 1.5   # 0.5 – 2.0
        radius = max(1, int(1 + sharpness_score / 50))           # 1 – 3
        sigma  = radius / 3.0

        blurred = cv2.GaussianBlur(img, (0, 0), sigma)
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _sigmoid_stretch(self, img: np.ndarray, cutoff: float = 0.5, gain: float = 10.0) -> np.ndarray:
        img_f = img.astype(np.float32) / 255.0
        result = 1.0 / (1.0 + np.exp(gain * (cutoff - img_f)))
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def _linear_stretch(self, img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
        result = np.zeros_like(img)
        for c in range(3):
            ch = img[:, :, c].astype(np.float32)
            lo, hi = np.percentile(ch, p_low), np.percentile(ch, p_high)
            if hi > lo:
                ch = (ch - lo) / (hi - lo) * 255.0
            result[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        return result

    def _per_channel_stretch(self, img: np.ndarray) -> np.ndarray:
        return self._linear_stretch(img, p_low=1.0, p_high=99.0)

    def needs_fix(self, img: np.ndarray) -> bool:
        return self._rms_contrast(img) < self.LOW_CONTRAST_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — SHARPENER & DEBLURRER  (~350 lines)
# ──────────────────────────────────────────────────────────────────────────────

class SharpenerDeblurrer:
    """
    Fix blurry images.

    Methods
    -------
    - Unsharp Masking           : most common, auto-tuned
    - Laplacian Sharpening      : fast, for mild blur
    - Wiener Filter             : defocus deconvolution
    - Richardson-Lucy           : iterative deconvolution
    - Motion Blur Deconvolution : detects motion direction from FFT

    Smart selector
    --------------
    motion blur   → Motion Deconvolution
    defocus blur  → Wiener / Richardson-Lucy
    mild softness → Unsharp Masking
    unknown       → Unsharp Masking (safe default)
    """

    BLUR_THRESHOLD = 100.0    # Laplacian variance below this → blurry

    def fix(self, img: np.ndarray, method: str = "smart",
            blur_score: float = None, blur_type: str = None) -> np.ndarray:
        if method == "smart":
            return self._smart_fix(img, blur_score, blur_type)
        elif method == "unsharp":
            return self._unsharp(img, blur_score)
        elif method == "laplacian":
            return self._laplacian_sharpen(img, blur_score)
        elif method == "wiener":
            return self._wiener_deconvolution(img)
        elif method == "richardson_lucy":
            return self._richardson_lucy(img)
        elif method == "motion":
            return self._motion_deblur(img)
        else:
            raise ValueError(f"Unknown sharpen method: {method}")

    def _smart_fix(self, img: np.ndarray, blur_score: float, blur_type: str) -> np.ndarray:
        if blur_type == "motion":
            return self._motion_deblur(img)
        elif blur_type in ("defocus", "gaussian"):
            return self._wiener_deconvolution(img)
        else:
            return self._unsharp(img, blur_score)

    # ── Unsharp Masking ───────────────────────────────────────────────────────

    def _unsharp(self, img: np.ndarray, blur_score: float = None,
                 radius: float = None, amount: float = None) -> np.ndarray:
        if blur_score is None:
            blur_score = 50.0
        if radius is None:
            radius = max(0.5, 3.0 - blur_score / 50.0)   # 0.5 – 3.0
        if amount is None:
            amount = max(0.5, 2.0 - blur_score / 100.0)  # 0.5 – 2.0

        gaussian = cv2.GaussianBlur(img, (0, 0), radius)
        sharpened = cv2.addWeighted(img, 1.0 + amount, gaussian, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Laplacian ─────────────────────────────────────────────────────────────

    def _laplacian_sharpen(self, img: np.ndarray, blur_score: float = None) -> np.ndarray:
        if blur_score is None:
            blur_score = 50.0
        strength = max(0.3, 1.5 - blur_score / 100.0)

        lap_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        laplacian  = cv2.filter2D(img.astype(np.float32), -1, lap_kernel)
        sharpened  = img.astype(np.float32) + strength * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ── Wiener Deconvolution ──────────────────────────────────────────────────

    def _wiener_deconvolution(self, img: np.ndarray, radius: int = 3,
                               noise_power: float = 0.01) -> np.ndarray:
        """Apply Wiener filter in frequency domain per channel."""
        # Build Gaussian PSF
        size = 2 * radius + 1
        psf  = cv2.getGaussianKernel(size, radius / 2)
        psf  = psf @ psf.T
        psf /= psf.sum()

        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c].astype(np.float64) / 255.0
            # Pad to avoid border artifacts
            pad = size // 2
            ch_padded = np.pad(ch, pad, mode="reflect")
            # FFT
            H  = np.fft.fft2(psf, s=ch_padded.shape)
            G  = np.fft.fft2(ch_padded)
            # Wiener filter
            H_conj = np.conj(H)
            W = H_conj / (np.abs(H) ** 2 + noise_power)
            restored = np.real(np.fft.ifft2(W * G))
            # Unpad
            restored = restored[pad:-pad, pad:-pad]
            result[:, :, c] = np.clip(restored * 255, 0, 255)
        return result.astype(np.uint8)

    # ── Richardson-Lucy ───────────────────────────────────────────────────────

    def _richardson_lucy(self, img: np.ndarray, num_iter: int = None,
                          psf_radius: int = 3) -> np.ndarray:
        size = 2 * psf_radius + 1
        psf  = cv2.getGaussianKernel(size, psf_radius / 2)
        psf  = (psf @ psf.T).astype(np.float64)
        psf /= psf.sum()

        # Auto-tune iterations from blur severity
        if num_iter is None:
            num_iter = 15  # default

        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c].astype(np.float64) / 255.0
            ch = np.clip(ch, 1e-6, 1.0)
            restored = restoration.richardson_lucy(ch, psf, num_iter=num_iter)
            result[:, :, c] = np.clip(restored * 255, 0, 255)
        return result.astype(np.uint8)

    # ── Motion Deblur ─────────────────────────────────────────────────────────

    def _motion_deblur(self, img: np.ndarray, size: int = 15) -> np.ndarray:
        """Detect motion direction from FFT, build directional PSF, apply Wiener."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Estimate motion angle from FFT magnitude
        fft    = np.fft.fft2(gray)
        fft_sh = np.fft.fftshift(fft)
        mag    = np.log1p(np.abs(fft_sh))
        angle  = self._estimate_motion_angle(mag)

        # Build directional PSF
        psf = self._build_motion_psf(size, angle)

        # Apply Wiener per channel
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c].astype(np.float64) / 255.0
            pad = size // 2
            ch_p = np.pad(ch, pad, mode="reflect")
            H  = np.fft.fft2(psf, s=ch_p.shape)
            G  = np.fft.fft2(ch_p)
            H_conj = np.conj(H)
            W  = H_conj / (np.abs(H) ** 2 + 0.002)
            res = np.real(np.fft.ifft2(W * G))
            res = res[pad:-pad, pad:-pad]
            result[:, :, c] = np.clip(res * 255, 0, 255)
        return result.astype(np.uint8)

    def _estimate_motion_angle(self, mag: np.ndarray) -> float:
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        region = mag[cy - 20:cy + 20, cx - 20:cx + 20]
        region_blur = cv2.GaussianBlur(region, (5, 5), 0)
        lines = cv2.HoughLines(
            cv2.Canny(region_blur.astype(np.uint8), 50, 150), 1, np.pi / 180, 5
        )
        if lines is not None:
            angle_rad = lines[0][0][1]
            return float(np.degrees(angle_rad))
        return 0.0

    def _build_motion_psf(self, size: int, angle_deg: float) -> np.ndarray:
        psf = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        angle_rad = np.radians(angle_deg)
        for i in range(size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1.0
        s = psf.sum()
        if s > 0:
            psf /= s
        else:
            psf[center, center] = 1.0
        return psf

    def needs_fix(self, img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.BLUR_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — NOISE CLEANER  (~350 lines)
# ──────────────────────────────────────────────────────────────────────────────

class NoiseCleaner:
    """
    Remove image noise.

    Methods
    -------
    - Gaussian Filter   : simple, fast; for heavy noise
    - Median Filter     : best for salt & pepper
    - Non-Local Means   : best overall for Gaussian noise
    - BM3D              : state-of-the-art (optional dep)
    - Bilateral Filter  : edge-preserving smoothing
    - Wavelet Denoising : good for natural images
    - TV Denoising      : piecewise-smooth images

    Smart selector
    --------------
    salt_pepper → Median Filter
    gaussian    → NLM (default) or BM3D (high quality)
    poisson     → VST + NLM
    mixed       → Bilateral + Median
    """

    HIGH_NOISE_THRESHOLD = 20.0

    def fix(self, img: np.ndarray, method: str = "smart",
            noise_type: str = None, noise_sigma: float = None) -> np.ndarray:
        if method == "smart":
            return self._smart_fix(img, noise_type, noise_sigma)
        elif method == "gaussian":
            return self._gaussian_filter(img, noise_sigma)
        elif method == "median":
            return self._median_filter(img, noise_sigma)
        elif method == "nlm":
            return self._nlm(img, noise_sigma)
        elif method == "bm3d":
            return self._bm3d(img, noise_sigma)
        elif method == "bilateral":
            return self._bilateral(img, noise_sigma)
        elif method == "wavelet":
            return self._wavelet(img)
        elif method == "tv":
            return self._tv_denoise(img)
        else:
            raise ValueError(f"Unknown noise method: {method}")

    def _smart_fix(self, img: np.ndarray, noise_type: str, noise_sigma: float) -> np.ndarray:
        if noise_type is None:
            noise_type = "gaussian"
        if noise_type == "salt_pepper":
            return self._median_filter(img, noise_sigma)
        elif noise_type == "poisson":
            return self._vst_nlm(img)
        elif noise_type == "mixed":
            img = self._bilateral(img, noise_sigma)
            img = self._median_filter(img, noise_sigma)
            return img
        else:
            # Default: NLM for Gaussian
            return self._nlm(img, noise_sigma)

    def _estimate_kernel(self, sigma: float) -> int:
        if sigma is None:
            return 3
        k = max(3, int(sigma * 2 + 1))
        return k if k % 2 == 1 else k + 1

    def _gaussian_filter(self, img: np.ndarray, sigma: float = None) -> np.ndarray:
        if sigma is None:
            sigma = 1.0
        sigma = float(np.clip(sigma / 15.0, 0.5, 3.0))
        return cv2.GaussianBlur(img, (0, 0), sigma)

    def _median_filter(self, img: np.ndarray, noise_sigma: float = None) -> np.ndarray:
        ksize = 3
        if noise_sigma is not None:
            ksize = 3 if noise_sigma < 20 else 5
        return cv2.medianBlur(img, ksize)

    def _nlm(self, img: np.ndarray, noise_sigma: float = None) -> np.ndarray:
        if noise_sigma is None:
            noise_sigma = 15.0
        h = float(np.clip(noise_sigma * 0.8, 3.0, 30.0))
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=h,
            hColor=h,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _bm3d(self, img: np.ndarray, noise_sigma: float = None) -> np.ndarray:
        try:
            import bm3d
        except ImportError:
            logger.warning("bm3d not installed; falling back to NLM. pip install bm3d")
            return self._nlm(img, noise_sigma)

        if noise_sigma is None:
            noise_sigma = 25.0
        sigma_psd = float(np.clip(noise_sigma / 255.0, 0.01, 0.15))

        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c].astype(np.float32) / 255.0
            denoised = bm3d.bm3d(ch, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
            result[:, :, c] = np.clip(denoised * 255, 0, 255)
        return result.astype(np.uint8)

    def _bilateral(self, img: np.ndarray, noise_sigma: float = None) -> np.ndarray:
        if noise_sigma is None:
            noise_sigma = 15.0
        d       = 9
        sigma_c = float(np.clip(noise_sigma * 2, 30, 150))
        sigma_s = float(np.clip(noise_sigma, 10, 75))
        return cv2.bilateralFilter(img, d, sigma_c, sigma_s)

    def _wavelet(self, img: np.ndarray) -> np.ndarray:
        result = np.zeros_like(img, dtype=np.float32)
        for c in range(3):
            ch = img[:, :, c].astype(np.float64) / 255.0
            sigma_est = float(np.mean(estimate_sigma(ch, channel_axis=None)))
            denoised  = denoise_wavelet(
                ch,
                sigma=sigma_est,
                mode="soft",
                method="BayesShrink",
                rescale_sigma=True,
            )
            result[:, :, c] = np.clip(denoised * 255, 0, 255)
        return result.astype(np.uint8)

    def _tv_denoise(self, img: np.ndarray, weight: float = None) -> np.ndarray:
        if weight is None:
            weight = 0.1
        img_f  = img.astype(np.float64) / 255.0
        result = restoration.denoise_tv_chambolle(img_f, weight=weight, channel_axis=-1)
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def _vst_nlm(self, img: np.ndarray) -> np.ndarray:
        """Variance-stabilizing transform + NLM for Poisson noise."""
        img_f    = img.astype(np.float32)
        vst      = 2.0 * np.sqrt(np.clip(img_f, 0, None) + 3.0 / 8.0)
        vst_u8   = np.clip(vst / vst.max() * 255, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(vst_u8, None, h=10, hColor=10)
        # Inverse VST
        denoised_f = denoised.astype(np.float32) / 255.0 * vst.max()
        inv_vst    = (denoised_f / 2.0) ** 2 - 3.0 / 8.0
        return np.clip(inv_vst, 0, 255).astype(np.uint8)

    def needs_fix(self, img: np.ndarray, noise_sigma: float = None) -> bool:
        if noise_sigma is not None:
            return noise_sigma > self.HIGH_NOISE_THRESHOLD
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sigma = float(np.mean(estimate_sigma(gray / 255.0)))
        return sigma * 255 > self.HIGH_NOISE_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — ARTIFACT REMOVER  (~250 lines)
# ──────────────────────────────────────────────────────────────────────────────

class ArtifactRemover:
    """
    Remove JPEG blocking artifacts and ringing artifacts.

    Methods
    -------
    - DCT Smoothing     : smooth 8x8 block boundaries (JPEG)
    - TV Minimization   : remove ringing near edges
    - Guided Filter     : edge-preserving smoothing
    - Re-compression    : load → remove → save at high quality
    """

    def fix(self, img: np.ndarray, method: str = "smart",
            blocking_score: float = None, ringing_score: float = None) -> np.ndarray:
        if method == "smart":
            return self._smart_fix(img, blocking_score, ringing_score)
        elif method == "dct":
            return self._dct_smoothing(img)
        elif method == "tv":
            return self._tv_remove(img, ringing_score)
        elif method == "guided":
            return self._guided_filter(img, img)
        else:
            raise ValueError(f"Unknown artifact method: {method}")

    def _smart_fix(self, img: np.ndarray, blocking_score: float, ringing_score: float) -> np.ndarray:
        has_blocking = blocking_score is not None and blocking_score > 30
        has_ringing  = ringing_score  is not None and ringing_score  > 30

        if has_blocking:
            img = self._dct_smoothing(img)
        if has_ringing:
            img = self._tv_remove(img, ringing_score)
        if has_blocking:
            img = self._guided_filter(img, img)
        return img

    def _dct_smoothing(self, img: np.ndarray, block_size: int = 8,
                        sigma: float = 0.5) -> np.ndarray:
        """Smooth only at 8x8 JPEG block boundaries."""
        result = img.copy().astype(np.float32)
        h, w   = img.shape[:2]

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                # Smooth boundary rows/cols between blocks
                if y + block_size < h:
                    row_start = max(y + block_size - 1, 0)
                    row_end   = min(y + block_size + 2, h)
                    strip     = result[row_start:row_end, x:x + block_size]
                    smoothed  = cv2.GaussianBlur(strip, (1, 3), sigma)
                    result[row_start:row_end, x:x + block_size] = smoothed

                if x + block_size < w:
                    col_start = max(x + block_size - 1, 0)
                    col_end   = min(x + block_size + 2, w)
                    strip     = result[y:y + block_size, col_start:col_end]
                    smoothed  = cv2.GaussianBlur(strip, (3, 1), sigma)
                    result[y:y + block_size, col_start:col_end] = smoothed

        return np.clip(result, 0, 255).astype(np.uint8)

    def _tv_remove(self, img: np.ndarray, ringing_score: float = None) -> np.ndarray:
        weight = 0.05
        if ringing_score is not None:
            weight = float(np.clip(ringing_score / 200.0, 0.02, 0.2))
        img_f  = img.astype(np.float64) / 255.0
        result = restoration.denoise_tv_chambolle(img_f, weight=weight, channel_axis=-1)
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def _guided_filter(self, img: np.ndarray, guide: np.ndarray,
                        radius: int = 4, eps: float = 0.01) -> np.ndarray:
        """Fast guided filter implementation (no opencv_contrib required)."""
        img_f   = img.astype(np.float32) / 255.0
        guide_f = guide.astype(np.float32) / 255.0
        result  = np.zeros_like(img_f)

        ksize = (2 * radius + 1, 2 * radius + 1)
        for c in range(3):
            I = guide_f[:, :, c]
            p = img_f[:, :, c]

            mean_I  = cv2.boxFilter(I, -1, ksize)
            mean_p  = cv2.boxFilter(p, -1, ksize)
            mean_Ip = cv2.boxFilter(I * p, -1, ksize)
            cov_Ip  = mean_Ip - mean_I * mean_p

            mean_II = cv2.boxFilter(I * I, -1, ksize)
            var_I   = mean_II - mean_I * mean_I

            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            mean_a = cv2.boxFilter(a, -1, ksize)
            mean_b = cv2.boxFilter(b, -1, ksize)

            result[:, :, c] = mean_a * I + mean_b

        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def needs_fix(self, blocking_score: float = None, ringing_score: float = None) -> bool:
        if blocking_score is not None and blocking_score > 30:
            return True
        if ringing_score is not None and ringing_score > 30:
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — COLOR CAST CORRECTOR  (~200 lines)
# ──────────────────────────────────────────────────────────────────────────────

class ColorCastCorrector:
    """
    Fix color casts (wrong white balance).

    Methods
    -------
    - Gray World Assumption    : scales channels so means are equal
    - Max-RGB (White Patch)    : brightest pixel → white
    - Per-channel Histogram EQ : equalizes each RGB channel independently
    - Illuminant Estimation    : corrects to D65 illuminant
    """

    CAST_THRESHOLD = 15.0  # channel mean deviation from overall mean

    def fix(self, img: np.ndarray, method: str = "smart",
            cast_strength: float = None) -> np.ndarray:
        if method == "smart":
            return self._smart_fix(img, cast_strength)
        elif method == "gray_world":
            return self._gray_world(img)
        elif method == "max_rgb":
            return self._max_rgb(img)
        elif method == "histogram":
            return self._histogram_eq(img)
        elif method == "illuminant":
            return self._illuminant_estimation(img)
        else:
            raise ValueError(f"Unknown color cast method: {method}")

    def _smart_fix(self, img: np.ndarray, cast_strength: float) -> np.ndarray:
        if cast_strength is None:
            cast_strength = self._measure_cast(img)
        if cast_strength < self.CAST_THRESHOLD:
            return img
        if cast_strength < 40:
            return self._gray_world(img)
        else:
            img = self._histogram_eq(img)
            return img

    def _measure_cast(self, img: np.ndarray) -> float:
        means = [float(img[:, :, c].mean()) for c in range(3)]
        overall = sum(means) / 3
        return max(abs(m - overall) for m in means)

    def _gray_world(self, img: np.ndarray) -> np.ndarray:
        result  = img.astype(np.float32)
        means   = [float(result[:, :, c].mean()) for c in range(3)]
        overall = sum(means) / 3.0
        for c in range(3):
            if means[c] > 0:
                result[:, :, c] *= overall / means[c]
        return np.clip(result, 0, 255).astype(np.uint8)

    def _max_rgb(self, img: np.ndarray) -> np.ndarray:
        result = img.astype(np.float32)
        for c in range(3):
            ch_max = float(result[:, :, c].max())
            if ch_max > 0:
                result[:, :, c] *= 255.0 / ch_max
        return np.clip(result, 0, 255).astype(np.uint8)

    def _histogram_eq(self, img: np.ndarray) -> np.ndarray:
        result = np.zeros_like(img)
        for c in range(3):
            result[:, :, c] = cv2.equalizeHist(img[:, :, c])
        return result

    def _illuminant_estimation(self, img: np.ndarray) -> np.ndarray:
        """
        Estimate scene illuminant using the Grayworld + shades-of-gray model
        and correct to D65 (neutral).
        """
        img_f  = img.astype(np.float32) + 1e-6
        # Shades of gray (n=6 approximates Gray World for n=1, White Patch for n=inf)
        n = 6
        illum = np.zeros(3, dtype=np.float64)
        for c in range(3):
            illum[c] = (np.mean(img_f[:, :, c] ** n)) ** (1.0 / n)

        illum /= illum.mean()   # normalize so mean channel gain = 1
        result = img.astype(np.float32)
        for c in range(3):
            result[:, :, c] /= illum[c]
        return np.clip(result, 0, 255).astype(np.uint8)

    def needs_fix(self, img: np.ndarray) -> bool:
        return self._measure_cast(img) > self.CAST_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — IMAGE RESIZER & NORMALIZER  (~250 lines)
# ──────────────────────────────────────────────────────────────────────────────

class ImageResizer:
    """
    Resize and normalize images for ML training.

    Resize methods
    --------------
    - LANCZOS     : best quality downscaling
    - BICUBIC     : good quality
    - BILINEAR    : fast
    - NEAREST     : for masks / labels

    Padding modes
    -------------
    - letterbox   : YOLO-style, add black border
    - center_crop : resize then crop center
    - stretch     : ignore aspect ratio

    Normalization
    -------------
    - 0-1           : divide by 255
    - -1 to 1       : (pixel / 127.5) - 1
    - imagenet      : ImageNet mean/std per channel
    - standardize   : zero mean, unit std per image
    - per_channel   : per-channel min-max
    """

    INTERP_MAP = {
        "lanczos":  cv2.INTER_LANCZOS4,
        "bicubic":  cv2.INTER_CUBIC,
        "bilinear": cv2.INTER_LINEAR,
        "nearest":  cv2.INTER_NEAREST,
    }

    # ImageNet statistics (BGR order for OpenCV)
    IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)  # BGR
    IMAGENET_STD  = np.array([58.393,  57.12,   57.375], dtype=np.float32)

    def resize(self, img: np.ndarray, target: Tuple[int, int],
               method: str = "lanczos", mode: str = "letterbox") -> np.ndarray:
        """
        Resize image to target (width, height).

        Parameters
        ----------
        target : (width, height)
        method : 'lanczos' | 'bicubic' | 'bilinear' | 'nearest'
        mode   : 'letterbox' | 'center_crop' | 'stretch'
        """
        tw, th = target
        interp = self.INTERP_MAP.get(method, cv2.INTER_LANCZOS4)

        if mode == "stretch":
            return cv2.resize(img, (tw, th), interpolation=interp)
        elif mode == "letterbox":
            return self._letterbox(img, tw, th, interp)
        elif mode == "center_crop":
            return self._center_crop(img, tw, th, interp)
        else:
            raise ValueError(f"Unknown resize mode: {mode}")

    def _letterbox(self, img: np.ndarray, tw: int, th: int,
                   interp: int) -> np.ndarray:
        h, w    = img.shape[:2]
        scale   = min(tw / w, th / h)
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Pad to target size with black
        canvas  = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
        pad_top = (th - new_h) // 2
        pad_left = (tw - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas

    def _center_crop(self, img: np.ndarray, tw: int, th: int,
                     interp: int) -> np.ndarray:
        h, w  = img.shape[:2]
        scale = max(tw / w, th / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Center crop
        cy    = (new_h - th) // 2
        cx    = (new_w - tw) // 2
        return resized[cy:cy + th, cx:cx + tw]

    def normalize(self, img: np.ndarray, mode: str = "0_1") -> np.ndarray:
        """
        Normalize pixel values.

        Parameters
        ----------
        mode : '0_1' | '-1_1' | 'imagenet' | 'standardize' | 'per_channel'

        Returns numpy float32 array.
        """
        img_f = img.astype(np.float32)

        if mode == "0_1":
            return img_f / 255.0

        elif mode == "-1_1":
            return img_f / 127.5 - 1.0

        elif mode == "imagenet":
            # Subtract ImageNet mean, divide by std
            result = img_f.copy()
            for c in range(3):
                result[:, :, c] = (result[:, :, c] - self.IMAGENET_MEAN[c]) / self.IMAGENET_STD[c]
            return result

        elif mode == "standardize":
            mean = img_f.mean()
            std  = img_f.std()
            if std < 1e-6:
                std = 1.0
            return (img_f - mean) / std

        elif mode == "per_channel":
            result = img_f.copy()
            for c in range(3):
                ch   = result[:, :, c]
                lo   = ch.min()
                hi   = ch.max()
                if hi > lo:
                    result[:, :, c] = (ch - lo) / (hi - lo)
            return result

        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def convert_format(self, img: np.ndarray, path: str,
                        fmt: str = "jpg", quality: int = DEFAULT_OUTPUT_QUALITY) -> str:
        """Save image in target format. Returns new path."""
        p    = Path(path)
        ext  = ".jpg" if fmt in ("jpg", "jpeg") else f".{fmt}"
        new_path = str(p.parent / (p.stem + ext))

        if fmt in ("jpg", "jpeg"):
            cv2.imwrite(new_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif fmt == "png":
            cv2.imwrite(new_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        elif fmt == "webp":
            cv2.imwrite(new_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
        else:
            cv2.imwrite(new_path, img)
        return new_path


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — FILTER & REMOVE ENGINE  (~200 lines)
# ──────────────────────────────────────────────────────────────────────────────

class FilterEngine:
    """
    Filter and remove unwanted images from a dataset.

    Modes
    -----
    - Remove rejected       : verdict == 'Reject 🚫'
    - Remove duplicates     : is_duplicate == True
    - Remove corrupted      : load_status == 'corrupted' | 'error'
    - Remove low-res        : is_low_resolution == True
    - Quality threshold     : quality_score < min_score
    - Class balance         : max N samples per class (subfolder)
    """

    def filter(
        self,
        df: pd.DataFrame,
        remove_rejected: bool       = True,
        remove_duplicates: bool     = True,
        remove_corrupted: bool      = True,
        remove_low_res: bool        = False,
        min_score: float            = None,
        max_per_class: int          = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply all filters and return (filtered_df, removal_report).
        """
        original_count = len(df)
        removal_report: Dict[str, int] = {}
        df = df.copy()

        if remove_rejected and "verdict" in df.columns:
            before = len(df)
            df = df[~df["verdict"].astype(str).str.contains("Reject", na=False)]
            removal_report["rejected"] = before - len(df)

        if remove_corrupted and "load_status" in df.columns:
            before = len(df)
            df = df[~df["load_status"].astype(str).isin(["corrupted", "error", "failed"])]
            removal_report["corrupted"] = before - len(df)

        if remove_duplicates and "is_duplicate" in df.columns:
            before = len(df)
            df = df[df["is_duplicate"].astype(str).isin(["False", "false", "0", "", "nan"]) |
                    df["is_duplicate"].isna()]
            removal_report["duplicates"] = before - len(df)

        if remove_low_res and "is_low_resolution" in df.columns:
            before = len(df)
            df = df[~df["is_low_resolution"].astype(bool)]
            removal_report["low_resolution"] = before - len(df)

        if min_score is not None and "quality_score" in df.columns:
            before = len(df)
            df = df[df["quality_score"].astype(float) >= min_score]
            removal_report["below_min_score"] = before - len(df)

        if max_per_class is not None and "class" in df.columns:
            before = len(df)
            df = (
                df.groupby("class", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=42))
                  .reset_index(drop=True)
            )
            removal_report["class_balanced"] = before - len(df)

        removal_report["total_removed"] = original_count - len(df)
        removal_report["remaining"]     = len(df)
        return df, removal_report


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — BATCH PROCESSING ENGINE  (~200 lines)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CleaningTask:
    """Represents a single image cleaning job."""
    image_path: str
    output_path: str
    row: Dict[str, Any]
    fix_flags: Dict[str, Any]


@dataclass
class TaskResult:
    """Result of a single image cleaning task."""
    image_path: str
    output_path: str
    success: bool
    error: Optional[str]
    fixes_applied: List[str]
    processing_time: float


class BatchProcessor:
    """
    Multi-threaded batch image processor.

    Features
    --------
    - Parallel processing     : ThreadPoolExecutor (configurable workers)
    - Memory-safe batching    : processes N images at a time
    - Per-image error isolation: one failure doesn't stop the batch
    - Resume support          : skips already-processed images
    - Dry-run mode            : logs what would happen, saves nothing
    - Preview mode            : processes only first N images
    """

    def __init__(
        self,
        max_workers: int = MAX_WORKERS,
        batch_size:  int = 64,
        dry_run:     bool = False,
        resume:      bool = True,
        preview:     Optional[int] = None,
    ):
        self.max_workers = max_workers
        self.batch_size  = batch_size
        self.dry_run     = dry_run
        self.resume      = resume
        self.preview     = preview

    def process(
        self,
        tasks: List[CleaningTask],
        process_fn,
        desc: str = "Cleaning images",
    ) -> List[TaskResult]:
        """
        Process a list of CleaningTask objects using process_fn.

        Parameters
        ----------
        tasks      : list of CleaningTask
        process_fn : callable(task) → TaskResult
        """
        if self.preview:
            tasks = tasks[:self.preview]
            logger.info(f"Preview mode: processing first {self.preview} images only")

        if self.resume:
            tasks = [t for t in tasks if not os.path.exists(t.output_path)]
            logger.info(f"Resume mode: {len(tasks)} images left to process")

        if self.dry_run:
            logger.info(f"Dry-run mode: would process {len(tasks)} images (no files saved)")
            return [
                TaskResult(
                    image_path=t.image_path,
                    output_path=t.output_path,
                    success=True,
                    error=None,
                    fixes_applied=["[dry-run]"],
                    processing_time=0.0,
                )
                for t in tasks
            ]

        results: List[TaskResult] = []

        with tqdm(total=len(tasks), desc=desc, unit="img") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_fn, task): task for task in tasks}

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                    except Exception as exc:
                        task = futures[future]
                        result = TaskResult(
                            image_path=task.image_path,
                            output_path=task.output_path,
                            success=False,
                            error=str(exc),
                            fixes_applied=[],
                            processing_time=0.0,
                        )
                    results.append(result)
                    pbar.update(1)
                    if not result.success:
                        pbar.set_postfix({"errors": sum(1 for r in results if not r.success)})

        n_ok  = sum(1 for r in results if r.success)
        n_err = len(results) - n_ok
        logger.info(f"Batch done: {n_ok} ok / {n_err} errors out of {len(tasks)} images")
        return results


# ──────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — BEFORE / AFTER REPORTER  (~200 lines)
# ──────────────────────────────────────────────────────────────────────────────

class CleaningReporter:
    """
    Build before/after comparison and export cleaning report.

    Outputs
    -------
    - before_after DataFrame  (CSV)
    - cleaning_report.json
    - cleaning_report.md
    """

    def build_report(
        self,
        df_before: pd.DataFrame,
        df_after:  pd.DataFrame,
        results:   List[TaskResult],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Build complete before/after report.
        Joins on 'file_path' or 'filename' column.
        """
        report: Dict[str, Any] = {}

        # --- per-image comparison ----------------------------------------
        ba_rows = []
        for r in results:
            row: Dict[str, Any] = {
                "image_path":       r.image_path,
                "output_path":      r.output_path,
                "success":          r.success,
                "fixes_applied":    ", ".join(r.fixes_applied),
                "processing_time":  round(r.processing_time, 3),
                "error":            r.error or "",
            }
            # Merge quality scores if present
            if "quality_score" in df_before.columns:
                before_row = df_before[df_before["file_path"] == r.image_path]
                if not before_row.empty:
                    row["quality_score_before"] = float(before_row["quality_score"].iloc[0])
            if "quality_score" in df_after.columns:
                after_row = df_after[df_after["file_path"] == r.output_path]
                if not after_row.empty:
                    row["quality_score_after"] = float(after_row["quality_score"].iloc[0])

            if "quality_score_before" in row and "quality_score_after" in row:
                row["delta"] = round(row["quality_score_after"] - row["quality_score_before"], 2)

            ba_rows.append(row)

        df_ba = pd.DataFrame(ba_rows)

        # --- summary stats -----------------------------------------------
        n_total    = len(results)
        n_success  = sum(1 for r in results if r.success)
        n_error    = n_total - n_success

        report["total_images"]    = n_total
        report["successfully_cleaned"] = n_success
        report["errors"]          = n_error
        report["success_rate"]    = round(n_success / max(n_total, 1) * 100, 1)

        if "delta" in df_ba.columns:
            deltas = df_ba["delta"].dropna()
            report["mean_score_improvement"] = round(float(deltas.mean()), 2)
            report["images_improved"]   = int((deltas > 0).sum())
            report["images_unchanged"]  = int((deltas == 0).sum())
            report["images_worsened"]   = int((deltas < 0).sum())

        # Fix frequency
        all_fixes: List[str] = []
        for r in results:
            all_fixes.extend(r.fixes_applied)
        from collections import Counter
        fix_freq = dict(Counter(all_fixes).most_common())
        report["fix_frequency"] = fix_freq

        total_time = sum(r.processing_time for r in results)
        report["total_processing_time_s"] = round(total_time, 2)
        report["avg_time_per_image_s"]    = round(total_time / max(n_success, 1), 4)

        # --- export -------------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)

        # CSV
        ba_path = os.path.join(output_dir, "before_after.csv")
        df_ba.to_csv(ba_path, index=False)

        # JSON
        json_path = os.path.join(output_dir, "cleaning_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Markdown
        md_path = os.path.join(output_dir, "cleaning_report.md")
        self._write_markdown(report, df_ba, md_path)

        report["output_files"] = {
            "before_after_csv":   ba_path,
            "cleaning_report_json": json_path,
            "cleaning_report_md":   md_path,
        }

        logger.info(f"Cleaning report saved to: {output_dir}")
        return report

    def _write_markdown(self, report: Dict, df_ba: pd.DataFrame, path: str) -> None:
        lines = [
            "# 🧹 Nydra — Image Cleaning Report\n",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total images | {report.get('total_images', 'N/A')} |",
            f"| Successfully cleaned | {report.get('successfully_cleaned', 'N/A')} |",
            f"| Errors | {report.get('errors', 'N/A')} |",
            f"| Success rate | {report.get('success_rate', 'N/A')}% |",
        ]
        if "mean_score_improvement" in report:
            lines += [
                f"| Mean score improvement | {report['mean_score_improvement']:+.2f} pts |",
                f"| Images improved | {report.get('images_improved', 'N/A')} |",
                f"| Images unchanged | {report.get('images_unchanged', 'N/A')} |",
                f"| Images worsened | {report.get('images_worsened', 'N/A')} |",
            ]
        lines += [
            f"| Total processing time | {report.get('total_processing_time_s', 'N/A')} s |",
            f"| Avg time/image | {report.get('avg_time_per_image_s', 'N/A')} s |",
            "",
            "## Fix Frequency\n",
            "| Fix | Count |",
            "|-----|-------|",
        ]
        for fix, count in report.get("fix_frequency", {}).items():
            lines.append(f"| {fix} | {count} |")

        lines += [
            "",
            "## Sample Results (first 20)\n",
            df_ba.head(20).to_markdown(index=False) if hasattr(df_ba, "to_markdown") else df_ba.head(20).to_string(),
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# MASTER CLASS — ImageCleaner
# ──────────────────────────────────────────────────────────────────────────────

class ImageCleaner:
    """
    🧹 Nydra Image Cleaning Pipeline — Master Class

    Orchestrates all 10 cleaning blocks into one coherent pipeline.

    Usage
    -----
    >>> cleaner = ImageCleaner(input_dir="dataset/", output_dir="dataset_clean/")
    >>> df_clean = cleaner.clean(
    ...     df=df,
    ...     fix_brightness=True,
    ...     fix_contrast=True,
    ...     fix_blur=True,
    ...     fix_noise=True,
    ...     fix_artifacts=True,
    ...     fix_color_cast=True,
    ...     remove_rejected=True,
    ...     remove_duplicates=True,
    ...     resize=(224, 224),
    ...     normalize=True,
    ... )

    Parameters (clean method)
    -------------------------
    df                : DataFrame from image_quality.py / image_analyzer.py
    fix_brightness    : run BrightnessFixer
    fix_contrast      : run ContrastEnhancer
    fix_blur          : run SharpenerDeblurrer
    fix_noise         : run NoiseCleaner
    fix_artifacts     : run ArtifactRemover
    fix_color_cast    : run ColorCastCorrector
    remove_rejected   : skip images with 'Reject' verdict
    remove_duplicates : skip duplicate images
    remove_corrupted  : skip corrupted images
    remove_low_res    : skip low-resolution images
    min_score         : skip images below this quality score
    resize            : target (width, height) or None
    resize_method     : 'lanczos' | 'bicubic' | 'bilinear' | 'nearest'
    resize_mode       : 'letterbox' | 'center_crop' | 'stretch'
    normalize         : save float32 .npy OR just save cleaned uint8
    normalize_mode    : '0_1' | '-1_1' | 'imagenet' | 'standardize'
    output_format     : 'jpg' | 'png' | 'webp' | 'keep'
    output_quality    : JPEG / WebP quality (1-100)
    max_workers       : parallel threads
    dry_run           : log what would happen, don't save
    resume            : skip already-processed images
    preview           : process only first N images
    """

    def __init__(
        self,
        input_dir:  str,
        output_dir: str,
    ):
        self.input_dir  = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all blocks
        self.brightness_fixer    = BrightnessFixer()
        self.contrast_enhancer   = ContrastEnhancer()
        self.sharpener           = SharpenerDeblurrer()
        self.noise_cleaner       = NoiseCleaner()
        self.artifact_remover    = ArtifactRemover()
        self.color_corrector     = ColorCastCorrector()
        self.resizer             = ImageResizer()
        self.filter_engine       = FilterEngine()
        self.reporter            = CleaningReporter()

    # ── public ───────────────────────────────────────────────────────────────

    def clean(
        self,
        df: pd.DataFrame                    = None,
        fix_brightness:     bool            = True,
        fix_contrast:       bool            = True,
        fix_blur:           bool            = True,
        fix_noise:          bool            = True,
        fix_artifacts:      bool            = True,
        fix_color_cast:     bool            = True,
        remove_rejected:    bool            = True,
        remove_duplicates:  bool            = True,
        remove_corrupted:   bool            = True,
        remove_low_res:     bool            = False,
        min_score:          float           = None,
        max_per_class:      int             = None,
        resize:             Tuple           = None,
        resize_method:      str             = "lanczos",
        resize_mode:        str             = "letterbox",
        normalize:          bool            = False,
        normalize_mode:     str             = "0_1",
        output_format:      str             = "keep",
        output_quality:     int             = DEFAULT_OUTPUT_QUALITY,
        max_workers:        int             = MAX_WORKERS,
        dry_run:            bool            = False,
        resume:             bool            = True,
        preview:            Optional[int]   = None,
    ) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.

        Returns updated DataFrame with cleaning results and before/after scores.
        Cleaned images saved to output_dir/.
        """
        logger.info("=" * 60)
        logger.info("🧹 Nydra — Image Cleaner v0.6.0")
        logger.info("=" * 60)

        start_time = time.time()

        # ── Step 1: Filter & Remove ───────────────────────────────────────
        if df is not None:
            df, removal_report = self.filter_engine.filter(
                df                = df,
                remove_rejected   = remove_rejected,
                remove_duplicates = remove_duplicates,
                remove_corrupted  = remove_corrupted,
                remove_low_res    = remove_low_res,
                min_score         = min_score,
                max_per_class     = max_per_class,
            )
            logger.info(
                f"Filter step: removed {removal_report.get('total_removed', 0)} images, "
                f"{removal_report.get('remaining', len(df))} remaining"
            )
            image_paths = df["file_path"].tolist() if "file_path" in df.columns else []
        else:
            # No DataFrame → scan input_dir
            image_paths = self._scan_input_dir()
            df = pd.DataFrame({"file_path": image_paths})
            removal_report = {}

        if not image_paths:
            logger.warning("No images to process. Exiting.")
            return df

        # ── Step 2: Build tasks ───────────────────────────────────────────
        fix_flags = {
            "fix_brightness":  fix_brightness,
            "fix_contrast":    fix_contrast,
            "fix_blur":        fix_blur,
            "fix_noise":       fix_noise,
            "fix_artifacts":   fix_artifacts,
            "fix_color_cast":  fix_color_cast,
            "resize":          resize,
            "resize_method":   resize_method,
            "resize_mode":     resize_mode,
            "normalize":       normalize,
            "normalize_mode":  normalize_mode,
            "output_format":   output_format,
            "output_quality":  output_quality,
        }

        tasks = self._build_tasks(df, fix_flags, output_format)

        # ── Step 3: Batch process ─────────────────────────────────────────
        processor = BatchProcessor(
            max_workers = max_workers,
            dry_run     = dry_run,
            resume      = resume,
            preview     = preview,
        )

        results = processor.process(
            tasks      = tasks,
            process_fn = self._process_single_image,
            desc       = "🧹 Cleaning images",
        )

        # ── Step 4: Build report ──────────────────────────────────────────
        report = self.reporter.build_report(
            df_before  = df,
            df_after   = pd.DataFrame(),   # re-scoring would call image_quality externally
            results    = results,
            output_dir = str(self.output_dir),
        )

        # ── Step 5: Merge results back into DataFrame ─────────────────────
        result_map = {r.image_path: r for r in results}
        df["cleaned_path"]     = df["file_path"].map(lambda p: getattr(result_map.get(p), "output_path", ""))
        df["cleaning_success"] = df["file_path"].map(lambda p: getattr(result_map.get(p), "success", False))
        df["fixes_applied"]    = df["file_path"].map(lambda p: ", ".join(getattr(result_map.get(p), "fixes_applied", [])))
        df["cleaning_error"]   = df["file_path"].map(lambda p: getattr(result_map.get(p), "error", ""))

        elapsed = time.time() - start_time
        logger.info(f"✅ Cleaning complete in {elapsed:.1f}s — {len(image_paths)} images processed")
        logger.info(f"   Output folder: {self.output_dir}")
        logger.info(f"   Report:        {self.output_dir}/cleaning_report.md")

        return df

    # ── private — task builder ────────────────────────────────────────────────

    def _build_tasks(
        self,
        df:            pd.DataFrame,
        fix_flags:     Dict,
        output_format: str,
    ) -> List[CleaningTask]:
        tasks = []
        for _, row in df.iterrows():
            src_path = str(row.get("file_path", ""))
            if not src_path or not os.path.exists(src_path):
                continue

            # Determine output path
            rel  = Path(src_path).relative_to(self.input_dir) if self.input_dir.name in src_path else Path(Path(src_path).name)
            ext  = Path(src_path).suffix.lower()
            if output_format != "keep":
                ext = f".{output_format}" if output_format in ("jpg", "jpeg", "png", "webp") else ext
            out_path = str(self.output_dir / rel.parent / (rel.stem + ext))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            tasks.append(CleaningTask(
                image_path  = src_path,
                output_path = out_path,
                row         = row.to_dict(),
                fix_flags   = fix_flags,
            ))
        return tasks

    # ── private — single image processor ─────────────────────────────────────

    def _process_single_image(self, task: CleaningTask) -> TaskResult:
        t_start     = time.time()
        fixes_applied: List[str] = []

        try:
            img = cv2.imread(task.image_path)
            if img is None:
                raise ValueError("cv2.imread returned None — possibly corrupted")

            row   = task.row
            flags = task.fix_flags

            # ── Fix brightness ──────────────────────────────────────────────
            if flags.get("fix_brightness"):
                fixer = self.brightness_fixer
                if fixer.needs_fix(img):
                    img = fixer.fix(img, method="smart")
                    fixes_applied.append("brightness")

            # ── Fix contrast ────────────────────────────────────────────────
            if flags.get("fix_contrast"):
                enhancer = self.contrast_enhancer
                sharpness = row.get("sharpness_score")
                if enhancer.needs_fix(img):
                    img = enhancer.fix(img, method="smart",
                                       sharpness_score=float(sharpness) if sharpness else None)
                    fixes_applied.append("contrast")

            # ── Fix blur ────────────────────────────────────────────────────
            if flags.get("fix_blur"):
                blur_score = row.get("blur_score") or row.get("sharpness_score")
                blur_type  = row.get("blur_type")
                if self.sharpener.needs_fix(img):
                    img = self.sharpener.fix(
                        img,
                        method     = "smart",
                        blur_score = float(blur_score) if blur_score is not None else None,
                        blur_type  = str(blur_type) if blur_type else None,
                    )
                    fixes_applied.append("blur")

            # ── Fix noise ───────────────────────────────────────────────────
            if flags.get("fix_noise"):
                noise_sigma = row.get("noise_sigma") or row.get("noise_level")
                noise_type  = row.get("noise_type")
                if self.noise_cleaner.needs_fix(img, float(noise_sigma) if noise_sigma else None):
                    img = self.noise_cleaner.fix(
                        img,
                        method      = "smart",
                        noise_type  = str(noise_type) if noise_type else None,
                        noise_sigma = float(noise_sigma) if noise_sigma is not None else None,
                    )
                    fixes_applied.append("noise")

            # ── Fix artifacts ───────────────────────────────────────────────
            if flags.get("fix_artifacts"):
                blocking = row.get("blocking_score")
                ringing  = row.get("ringing_score")
                if self.artifact_remover.needs_fix(
                    blocking_score = float(blocking) if blocking else None,
                    ringing_score  = float(ringing)  if ringing  else None,
                ):
                    img = self.artifact_remover.fix(
                        img,
                        method         = "smart",
                        blocking_score = float(blocking) if blocking else None,
                        ringing_score  = float(ringing)  if ringing  else None,
                    )
                    fixes_applied.append("artifacts")

            # ── Fix color cast ──────────────────────────────────────────────
            if flags.get("fix_color_cast"):
                cast_strength = row.get("color_cast_score")
                if self.color_corrector.needs_fix(img):
                    img = self.color_corrector.fix(
                        img,
                        method        = "smart",
                        cast_strength = float(cast_strength) if cast_strength else None,
                    )
                    fixes_applied.append("color_cast")

            # ── Resize ──────────────────────────────────────────────────────
            resize = flags.get("resize")
            if resize:
                img = self.resizer.resize(
                    img,
                    target = resize,
                    method = flags.get("resize_method", "lanczos"),
                    mode   = flags.get("resize_mode",   "letterbox"),
                )
                fixes_applied.append(f"resize({resize[0]}x{resize[1]})")

            # ── Normalize & Save ────────────────────────────────────────────
            if flags.get("normalize"):
                norm_img = self.resizer.normalize(img, mode=flags.get("normalize_mode", "0_1"))
                npy_path = task.output_path.rsplit(".", 1)[0] + ".npy"
                np.save(npy_path, norm_img)
                fixes_applied.append(f"normalize({flags.get('normalize_mode')})")

            # Save cleaned image
            fmt     = flags.get("output_format", "keep")
            quality = flags.get("output_quality", DEFAULT_OUTPUT_QUALITY)

            if fmt == "keep":
                cv2.imwrite(task.output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                self.resizer.convert_format(img, task.output_path, fmt=fmt, quality=quality)

            return TaskResult(
                image_path      = task.image_path,
                output_path     = task.output_path,
                success         = True,
                error           = None,
                fixes_applied   = fixes_applied,
                processing_time = time.time() - t_start,
            )

        except Exception as exc:
            logger.error(f"Failed to clean {task.image_path}: {exc}")
            return TaskResult(
                image_path      = task.image_path,
                output_path     = task.output_path,
                success         = False,
                error           = str(exc),
                fixes_applied   = fixes_applied,
                processing_time = time.time() - t_start,
            )

    def _scan_input_dir(self) -> List[str]:
        paths = []
        for ext in SUPPORTED_FORMATS:
            paths.extend([str(p) for p in self.input_dir.rglob(f"*{ext}")])
            paths.extend([str(p) for p in self.input_dir.rglob(f"*{ext.upper()}")])
        return sorted(set(paths))


# ──────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def clean_image(
    path:        str,
    output_path: str,
    fix_brightness: bool = True,
    fix_contrast:   bool = True,
    fix_blur:       bool = True,
    fix_noise:      bool = True,
    fix_artifacts:  bool = True,
    fix_color_cast: bool = True,
    resize:         Tuple = None,
    normalize:      bool  = False,
    normalize_mode: str   = "0_1",
    output_quality: int   = DEFAULT_OUTPUT_QUALITY,
) -> Dict[str, Any]:
    """
    Clean a single image file.

    Parameters
    ----------
    path        : source image path
    output_path : where to save the cleaned image

    Returns dict with fixes_applied, success, error.
    """
    input_dir  = str(Path(path).parent)
    output_dir = str(Path(output_path).parent)

    df = pd.DataFrame({"file_path": [path]})
    cleaner = ImageCleaner(input_dir=input_dir, output_dir=output_dir)

    df_result = cleaner.clean(
        df             = df,
        fix_brightness = fix_brightness,
        fix_contrast   = fix_contrast,
        fix_blur       = fix_blur,
        fix_noise      = fix_noise,
        fix_artifacts  = fix_artifacts,
        fix_color_cast = fix_color_cast,
        resize         = resize,
        normalize      = normalize,
        normalize_mode = normalize_mode,
        output_quality = output_quality,
    )

    if len(df_result) == 0:
        return {"success": False, "error": "Image filtered or not found", "fixes_applied": []}

    row = df_result.iloc[0]
    return {
        "success":       bool(row.get("cleaning_success", False)),
        "fixes_applied": str(row.get("fixes_applied", "")).split(", "),
        "output_path":   str(row.get("cleaned_path", "")),
        "error":         str(row.get("cleaning_error", "")),
    }


def clean_dataset(
    df:              pd.DataFrame,
    output_dir:      str,
    input_dir:       str            = ".",
    fix_brightness:  bool           = True,
    fix_contrast:    bool           = True,
    fix_blur:        bool           = True,
    fix_noise:       bool           = True,
    fix_artifacts:   bool           = True,
    fix_color_cast:  bool           = True,
    remove_rejected: bool           = True,
    remove_duplicates: bool         = True,
    resize:          Optional[Tuple] = None,
    normalize:       bool           = False,
    normalize_mode:  str            = "0_1",
    output_format:   str            = "keep",
    output_quality:  int            = DEFAULT_OUTPUT_QUALITY,
    max_workers:     int            = MAX_WORKERS,
    dry_run:         bool           = False,
    resume:          bool           = True,
) -> pd.DataFrame:
    """
    Clean an entire image dataset given a DataFrame from image_quality.py.

    Parameters
    ----------
    df         : DataFrame with 'file_path' column + quality columns
    output_dir : where to save cleaned images
    input_dir  : root input folder

    Returns updated DataFrame with 'cleaned_path', 'fixes_applied' columns.
    """
    cleaner = ImageCleaner(input_dir=input_dir, output_dir=output_dir)
    return cleaner.clean(
        df                = df,
        fix_brightness    = fix_brightness,
        fix_contrast      = fix_contrast,
        fix_blur          = fix_blur,
        fix_noise         = fix_noise,
        fix_artifacts     = fix_artifacts,
        fix_color_cast    = fix_color_cast,
        remove_rejected   = remove_rejected,
        remove_duplicates = remove_duplicates,
        resize            = resize,
        normalize         = normalize,
        normalize_mode    = normalize_mode,
        output_format     = output_format,
        output_quality    = output_quality,
        max_workers       = max_workers,
        dry_run           = dry_run,
        resume            = resume,
    )


def quick_clean(
    input_dir:  str,
    output_dir: str,
    resize:     Optional[Tuple] = (224, 224),
) -> pd.DataFrame:
    """
    One-liner dataset cleaning with sensible defaults.

    Applies all fixes, removes rejected/duplicates/corrupted,
    resizes to 224x224 (default for most vision models).

    Example
    -------
    >>> from image_cleaner import quick_clean
    >>> df = quick_clean("raw_images/", "clean_images/")
    """
    cleaner = ImageCleaner(input_dir=input_dir, output_dir=output_dir)
    return cleaner.clean(
        fix_brightness    = True,
        fix_contrast      = True,
        fix_blur          = True,
        fix_noise         = True,
        fix_artifacts     = True,
        fix_color_cast    = True,
        remove_rejected   = True,
        remove_duplicates = True,
        remove_corrupted  = True,
        resize            = resize,
        normalize         = False,
        output_format     = "jpg",
        output_quality    = 95,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🧹 Nydra — Image Dataset Cleaner v0.6.0"
    )
    parser.add_argument("input_dir",  help="Source image folder")
    parser.add_argument("output_dir", help="Output folder for cleaned images")
    parser.add_argument("--df",         default=None, help="CSV from image_quality.py")
    parser.add_argument("--resize",     default=None, help="Resize target e.g. 224x224")
    parser.add_argument("--format",     default="keep", choices=["keep", "jpg", "png", "webp"])
    parser.add_argument("--quality",    default=95,   type=int)
    parser.add_argument("--no-brightness",  action="store_true")
    parser.add_argument("--no-contrast",    action="store_true")
    parser.add_argument("--no-blur",        action="store_true")
    parser.add_argument("--no-noise",       action="store_true")
    parser.add_argument("--no-artifacts",   action="store_true")
    parser.add_argument("--no-color",       action="store_true")
    parser.add_argument("--keep-rejected",  action="store_true")
    parser.add_argument("--keep-dupes",     action="store_true")
    parser.add_argument("--normalize",      action="store_true")
    parser.add_argument("--norm-mode",  default="0_1")
    parser.add_argument("--workers",    default=MAX_WORKERS, type=int)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--preview",    default=None, type=int)
    parser.add_argument("--min-score",  default=None, type=float)

    args = parser.parse_args()

    resize_target = None
    if args.resize:
        parts = args.resize.lower().split("x")
        resize_target = (int(parts[0]), int(parts[1]))

    df_input = None
    if args.df:
        df_input = pd.read_csv(args.df)

    cleaner = ImageCleaner(input_dir=args.input_dir, output_dir=args.output_dir)
    df_out = cleaner.clean(
        df                = df_input,
        fix_brightness    = not args.no_brightness,
        fix_contrast      = not args.no_contrast,
        fix_blur          = not args.no_blur,
        fix_noise         = not args.no_noise,
        fix_artifacts     = not args.no_artifacts,
        fix_color_cast    = not args.no_color,
        remove_rejected   = not args.keep_rejected,
        remove_duplicates = not args.keep_dupes,
        min_score         = args.min_score,
        resize            = resize_target,
        normalize         = args.normalize,
        normalize_mode    = args.norm_mode,
        output_format     = args.format,
        output_quality    = args.quality,
        max_workers       = args.workers,
        dry_run           = args.dry_run,
        preview           = args.preview,
    )

    print(f"\n✅ Done — {len(df_out)} images processed")
    print(f"   Output: {args.output_dir}")
