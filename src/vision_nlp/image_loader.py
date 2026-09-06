"""
image_loader.py — Nydra v0.6.0
=====================================
Advanced Image Dataset Loader for ML/AI Pipelines

Handles every edge case, format, and ML-specific problem:
- File system scanning & validation
- Format intelligence & conversion
- Deep metadata extraction
- Memory-safe batch loading
- Dataset structure detection
- ML problem detection (leakage, imbalance, drift)
- Duplicate & near-duplicate detection
- Smart train/val/test splitting
- Dataset versioning & fingerprinting
- Smart recovery from all errors
- ML Readiness Score & full audit report

Version: 0.6.0
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────

import os
import re
import sys
import csv
import json
import math
import time
import uuid
import shutil
import hashlib
import logging
import pathlib
import platform
import warnings
import traceback
import threading
import mimetypes
import struct
import random
import colorsys
from io import BytesIO
from enum import Enum
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Dict, Generator, Iterator, List,
    Optional, Set, Tuple, Union
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Pillow — core image library
try:
    from PIL import Image, ImageFile, ExifTags, TiffImagePlugin
    from PIL.ExifTags import TAGS, GPSTAGS
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # handle truncated gracefully
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("Pillow not installed. Run: pip install Pillow")

# OpenCV — supplementary format support
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False

# imagehash — perceptual hashing
try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    warnings.warn("imagehash not installed. Run: pip install imagehash")

# scipy — statistical tests (KS-test for drift)
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# psutil — memory detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# tqdm — progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("nydra.image_loader")


# ─────────────────────────────────────────────
# CONSTANTS & ENUMS
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tiff", ".tif", ".gif",
    ".heic", ".heif", ".jp2", ".j2k",
    ".jpc", ".jpf", ".jpx", ".j2c",
}

HIDDEN_FILES: Set[str] = {
    ".ds_store", "thumbs.db", "desktop.ini",
    "__macosx", ".spotlight-v100", ".trashes",
    ".fseventsd",
}

MAGIC_BYTES: Dict[str, bytes] = {
    "jpeg": b"\xFF\xD8\xFF",
    "png":  b"\x89PNG\r\n\x1a\n",
    "gif":  b"GIF8",
    "webp": b"RIFF",
    "bmp":  b"BM",
    "tiff_le": b"II\x2A\x00",
    "tiff_be": b"MM\x00\x2A",
    "heic": b"\x00\x00\x00",   # checked differently
    "jp2":  b"\x00\x00\x00\x0C\x6A\x50\x20\x20",
    "pdf":  b"%PDF",
    "zip":  b"PK\x03\x04",
}

# Minimum recommended images per class for training
MIN_IMAGES_PER_CLASS = 50
MIN_TOTAL_IMAGES     = 100
MAX_SAFE_IMAGE_DIM   = 20_000      # pixels — above this is suspicious
MAX_METADATA_SIZE_MB = 5           # EXIF bomb threshold
MAX_PATH_LENGTH      = 260         # Windows MAX_PATH

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH     = "HIGH"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_LOW      = "LOW"


class DatasetStructure(Enum):
    FLAT          = "flat"           # all images in one folder
    CLASS_FOLDERS = "class_folders"  # ImageFolder style
    CSV_ANNOT     = "csv"            # CSV with image_path + label
    JSON_COCO     = "json_coco"      # COCO JSON format
    YOLO          = "yolo"           # image + .txt label
    VOC           = "voc"            # image + .xml label
    UNKNOWN       = "unknown"


class LoadStatus(Enum):
    OK             = "ok"
    CORRUPTED      = "corrupted"
    TRUNCATED      = "truncated"
    ZERO_BYTES     = "zero_bytes"
    WRONG_FORMAT   = "wrong_format"
    TOO_LARGE      = "too_large"
    TOO_SMALL      = "too_small"
    PERMISSION     = "permission_denied"
    HIDDEN         = "hidden_file"
    UNSUPPORTED    = "unsupported_format"
    ENCRYPTED      = "encrypted"
    METADATA_BOMB  = "metadata_bomb"
    SKIPPED        = "skipped"


class SplitStrategy(Enum):
    RANDOM      = "random"
    STRATIFIED  = "stratified"
    GROUP       = "group"
    TEMPORAL    = "temporal"


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class ImageRecord:
    """One row in the output DataFrame — every known fact about one image."""
    # Identity
    file_path:        str  = ""
    file_name:        str  = ""
    file_stem:        str  = ""
    extension:        str  = ""
    detected_format:  str  = ""   # format from magic bytes (may differ from extension)

    # Size
    file_size_bytes:  int  = 0
    file_size_kb:     float = 0.0
    file_size_mb:     float = 0.0

    # Dimensions
    width:            int  = 0
    height:           int  = 0
    aspect_ratio:     float = 0.0
    orientation:      str  = ""   # landscape / portrait / square
    total_pixels:     int  = 0
    megapixels:       float = 0.0
    compression_ratio: float = 0.0  # file_size / pixel_count

    # Color
    color_mode:       str  = ""   # RGB / RGBA / L / P / CMYK
    color_space:      str  = ""   # sRGB / Adobe RGB / etc.
    bit_depth:        int  = 0
    has_alpha:        bool = False
    is_grayscale:     bool = False
    is_animated:      bool = False
    n_frames:         int  = 1

    # Quality flags
    is_valid:         bool = True
    load_status:      str  = LoadStatus.OK.value
    error_message:    str  = ""
    error_severity:   str  = ""
    is_truncated:     bool = False
    is_corrupted:     bool = False
    is_hidden:        bool = False
    is_duplicate:     bool = False
    duplicate_of:     str  = ""

    # ML
    label:            str  = ""
    label_source:     str  = ""   # folder / csv / json / none
    split:            str  = ""   # train / val / test

    # Hashes (for deduplication)
    sha256:           str  = ""
    phash:            str  = ""
    dhash:            str  = ""
    ahash:            str  = ""
    whash:            str  = ""

    # EXIF
    exif_camera_make:   str  = ""
    exif_camera_model:  str  = ""
    exif_datetime:      str  = ""
    exif_iso:           str  = ""
    exif_focal_length:  str  = ""
    exif_gps_lat:       float = 0.0
    exif_gps_lon:       float = 0.0
    exif_dpi_x:         float = 0.0
    exif_dpi_y:         float = 0.0
    exif_size_bytes:    int   = 0

    # OS metadata
    created_at:       str  = ""
    modified_at:      str  = ""
    accessed_at:      str  = ""


@dataclass
class DatasetIssue:
    """A single detected problem in the dataset."""
    issue_id:    str
    severity:    str
    category:    str
    description: str
    affected:    List[str] = field(default_factory=list)
    fix:         str = ""


@dataclass
class MLReadinessReport:
    """Full ML readiness audit result."""
    total_images:        int   = 0
    valid_images:        int   = 0
    corrupted_images:    int   = 0
    duplicate_images:    int   = 0
    zero_byte_files:     int   = 0
    hidden_files:        int   = 0
    wrong_format_files:  int   = 0
    n_classes:           int   = 0
    class_distribution:  Dict  = field(default_factory=dict)
    imbalance_ratio:     float = 0.0
    minority_classes:    List  = field(default_factory=list)
    leakage_detected:    bool  = False
    leakage_pairs:       int   = 0
    split_valid:         bool  = False
    dimension_consistent: bool = False
    colorspace_consistent: bool = False
    has_covariate_shift: bool  = False
    file_health_score:   float = 0.0
    balance_score:       float = 0.0
    duplicate_score:     float = 0.0
    leakage_score:       float = 0.0
    consistency_score:   float = 0.0
    overall_score:       float = 0.0
    verdict:             str   = ""
    issues:              List[DatasetIssue] = field(default_factory=list)
    fix_checklist:       List[str]          = field(default_factory=list)
    dataset_fingerprint: str   = ""
    generated_at:        str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — FILE SYSTEM SCANNER
# ─────────────────────────────────────────────────────────────────────────────

class FileSystemScanner:
    """
    Scans file system paths and returns raw file paths,
    filtering out all non-image, hidden, system, and broken entries.
    Handles: symlinks, permission errors, long paths, special characters,
             hidden files, empty folders, OS-specific quirks.
    """

    def __init__(self, recursive: bool = True, follow_symlinks: bool = False):
        self.recursive      = recursive
        self.follow_symlinks = follow_symlinks
        self._scan_log: List[Dict] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def scan(self, path: Union[str, Path]) -> Tuple[List[Path], Dict]:
        """
        Scan a path (file or directory) and return:
        - List of valid image Paths
        - Scan summary dict
        """
        path = Path(path)
        valid_paths: List[Path] = []
        summary = {
            "scanned": 0, "accepted": 0, "rejected": 0,
            "hidden": 0, "permission_denied": 0, "symlink_broken": 0,
            "non_image": 0, "empty_folders": 0, "long_paths": 0,
            "special_chars": 0, "rejections": []
        }

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            result = self._validate_path(path)
            if result["valid"]:
                valid_paths.append(path)
                summary["accepted"] += 1
            else:
                summary["rejected"] += 1
                summary["rejections"].append(result)
            summary["scanned"] += 1
            return valid_paths, summary

        if path.is_dir():
            all_files = self._walk(path)
            for fp in all_files:
                summary["scanned"] += 1
                result = self._validate_path(fp)
                if result["valid"]:
                    valid_paths.append(fp)
                    summary["accepted"] += 1
                else:
                    summary["rejected"] += 1
                    reason = result.get("reason", "unknown")
                    if reason == "hidden":        summary["hidden"] += 1
                    if reason == "permission":    summary["permission_denied"] += 1
                    if reason == "non_image":     summary["non_image"] += 1
                    if reason == "long_path":     summary["long_paths"] += 1
                    if reason == "special_chars": summary["special_chars"] += 1
                    summary["rejections"].append(result)

            # Check for empty subdirectories
            summary["empty_folders"] = self._count_empty_dirs(path)

        return valid_paths, summary

    # ── Private ───────────────────────────────────────────────────────────────

    def _walk(self, root: Path) -> List[Path]:
        """Walk directory, respecting recursive flag and symlink policy."""
        results = []
        try:
            if self.recursive:
                walker = os.walk(str(root), followlinks=self.follow_symlinks)
            else:
                walker = [(str(root), [], os.listdir(str(root)))]

            for dirpath, dirnames, filenames in walker:
                # Filter out hidden/system dirs in place (affects os.walk traversal)
                dirnames[:] = [
                    d for d in dirnames
                    if not self._is_hidden_name(d)
                    and not self._is_system_dir(d)
                ]
                for fname in filenames:
                    fp = Path(dirpath) / fname
                    # Skip broken symlinks
                    if fp.is_symlink() and not fp.exists():
                        continue
                    results.append(fp)
        except PermissionError as e:
            logger.warning(f"Permission denied scanning directory: {root} — {e}")
        return results

    def _validate_path(self, fp: Path) -> Dict:
        """Validate a single file path. Returns dict with valid flag + reason."""
        base = {"path": str(fp), "valid": False, "reason": ""}

        # Long path check (Windows MAX_PATH)
        if len(str(fp)) > MAX_PATH_LENGTH and platform.system() == "Windows":
            return {**base, "reason": "long_path"}

        # Hidden file check
        if self._is_hidden(fp):
            return {**base, "reason": "hidden"}

        # Permission check
        if not os.access(str(fp), os.R_OK):
            return {**base, "reason": "permission"}

        # Special characters in filename that could cause issues
        if self._has_dangerous_chars(fp.name):
            logger.debug(f"Special chars in filename: {fp.name}")
            # Don't reject — just log. Many valid files have unicode names.

        # Extension check
        ext = fp.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {**base, "reason": "non_image"}

        return {**base, "valid": True, "reason": "ok"}

    def _is_hidden(self, fp: Path) -> bool:
        """Check if file is hidden (unix dot-files or Windows hidden attribute)."""
        # Unix hidden files
        if fp.name.startswith("."):
            return True
        # Known system files
        if self._is_hidden_name(fp.name):
            return True
        # Windows hidden attribute
        if platform.system() == "Windows":
            try:
                import ctypes
                attrs = ctypes.windll.kernel32.GetFileAttributesW(str(fp))
                return bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN = 2
            except Exception:
                pass
        return False

    def _is_hidden_name(self, name: str) -> bool:
        return name.lower() in HIDDEN_FILES or name.startswith(".")

    def _is_system_dir(self, name: str) -> bool:
        return name.lower() in {"__pycache__", "__macosx", ".git", ".svn", "node_modules"}

    def _has_dangerous_chars(self, name: str) -> bool:
        """Detect filenames with null bytes or other truly problematic chars."""
        return "\x00" in name

    def _count_empty_dirs(self, root: Path) -> int:
        count = 0
        for dirpath, dirnames, filenames in os.walk(str(root)):
            if not dirnames and not filenames:
                count += 1
        return count


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — DEEP FILE VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class FileValidator:
    """
    Deep validation of individual image files.
    Covers: zero-byte, truncated, wrong format, magic bytes,
            metadata bombs, encrypted files, dimension overflow.
    """

    def validate(self, fp: Path) -> Tuple[LoadStatus, str, str]:
        """
        Returns: (LoadStatus, error_message, severity)
        """
        if not PIL_AVAILABLE:
            return LoadStatus.OK, "", ""

        # ── Zero byte check ───────────────────────────────────────────────────
        try:
            size = fp.stat().st_size
        except OSError as e:
            return LoadStatus.PERMISSION, str(e), SEVERITY_HIGH

        if size == 0:
            return LoadStatus.ZERO_BYTES, "File is empty (0 bytes)", SEVERITY_CRITICAL

        # ── Magic bytes check (real format vs extension) ──────────────────────
        detected = self._detect_format_from_bytes(fp)
        declared = fp.suffix.lower().lstrip(".")
        declared = "jpeg" if declared == "jpg" else declared

        mismatch = self._is_format_mismatch(detected, declared)
        if mismatch and detected not in ("unknown", ""):
            msg = f"Extension '.{declared}' but magic bytes say '{detected}'"
            # Not fatal — PIL can still open it; just flag it
            logger.debug(f"Format mismatch: {fp.name} — {msg}")

        # ── EXIF metadata bomb check ──────────────────────────────────────────
        exif_size = self._estimate_exif_size(fp)
        if exif_size > MAX_METADATA_SIZE_MB * 1024 * 1024:
            return (
                LoadStatus.CORRUPTED,
                f"Metadata bomb: EXIF is {exif_size / 1024 / 1024:.1f} MB",
                SEVERITY_HIGH
            )

        # ── PIL open + verify + load ──────────────────────────────────────────
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                img = Image.open(str(fp))

            # verify() checks structural integrity (headers, markers)
            try:
                img.verify()
            except Exception as e:
                return LoadStatus.CORRUPTED, f"verify() failed: {e}", SEVERITY_CRITICAL

            # Re-open (verify() invalidates the pointer)
            img = Image.open(str(fp))

            # Check for absurd dimensions before loading into memory
            w, h = img.size
            if w > MAX_SAFE_IMAGE_DIM or h > MAX_SAFE_IMAGE_DIM:
                return (
                    LoadStatus.TOO_LARGE,
                    f"Dimension overflow: {w}x{h} exceeds {MAX_SAFE_IMAGE_DIM}px",
                    SEVERITY_HIGH
                )
            if w <= 1 or h <= 1:
                return (
                    LoadStatus.TOO_SMALL,
                    f"Trivial image: {w}x{h} pixels",
                    SEVERITY_MEDIUM
                )

            # load() catches pixel-level corruption and truncation
            try:
                img.load()
            except OSError as e:
                err = str(e).lower()
                if "truncated" in err:
                    return LoadStatus.TRUNCATED, f"Truncated: {e}", SEVERITY_HIGH
                return LoadStatus.CORRUPTED, f"load() failed: {e}", SEVERITY_CRITICAL

        except Exception as e:
            err = str(e).lower()
            if "cannot identify" in err:
                return LoadStatus.WRONG_FORMAT, f"Cannot identify format: {e}", SEVERITY_HIGH
            if "encrypted" in err or "password" in err:
                return LoadStatus.ENCRYPTED, f"Encrypted or password-protected: {e}", SEVERITY_HIGH
            return LoadStatus.CORRUPTED, f"Unexpected error: {e}", SEVERITY_HIGH

        return LoadStatus.OK, "", ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_format_from_bytes(self, fp: Path) -> str:
        """Read first 16 bytes and match against known magic bytes."""
        try:
            with open(str(fp), "rb") as f:
                header = f.read(16)
        except OSError:
            return "unknown"

        if header[:3] == MAGIC_BYTES["jpeg"]:
            return "jpeg"
        if header[:8] == MAGIC_BYTES["png"]:
            return "png"
        if header[:4] in (MAGIC_BYTES["gif"], b"GIF89"):
            return "gif"
        if header[:4] == MAGIC_BYTES["webp"] and header[8:12] == b"WEBP":
            return "webp"
        if header[:2] == MAGIC_BYTES["bmp"]:
            return "bmp"
        if header[:4] in (MAGIC_BYTES["tiff_le"], MAGIC_BYTES["tiff_be"]):
            return "tiff"
        if header[:8] == MAGIC_BYTES["jp2"]:
            return "jp2"
        if header[:4] == MAGIC_BYTES["pdf"]:
            return "pdf"
        if header[:4] == MAGIC_BYTES["zip"]:
            return "zip"
        return "unknown"

    def _is_format_mismatch(self, detected: str, declared: str) -> bool:
        """Fuzzy match between detected and declared formats."""
        aliases = {
            "jpeg": {"jpg", "jpeg"},
            "tiff": {"tif", "tiff"},
            "jp2":  {"jp2", "j2k", "jpf", "jpx"},
        }
        for canonical, group in aliases.items():
            if detected == canonical and declared in group:
                return False
        return detected != declared and detected not in ("unknown", "")

    def _estimate_exif_size(self, fp: Path) -> int:
        """Estimate EXIF block size by reading APP1 marker in JPEG."""
        try:
            ext = fp.suffix.lower()
            if ext not in (".jpg", ".jpeg"):
                return 0
            with open(str(fp), "rb") as f:
                data = f.read(65536)   # read first 64KB
            # Find APP1 marker (FF E1)
            idx = data.find(b"\xFF\xE1")
            if idx == -1:
                return 0
            length = struct.unpack(">H", data[idx + 2: idx + 4])[0]
            return length
        except Exception:
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — FORMAT INTELLIGENCE & CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

class FormatIntelligence:
    """
    Handles all format-specific edge cases and normalizes images
    to a consistent RGB/RGBA representation for ML pipelines.
    """

    def normalize(self, img: "Image.Image", target_mode: str = "RGB") -> "Image.Image":
        """
        Normalize any PIL image to target_mode (RGB by default).
        Handles: RGBA, Palette, Grayscale, CMYK, LAB, animated, 16/32-bit.
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow not available")

        original_mode = img.mode

        # Animated — get first frame
        if hasattr(img, "is_animated") and img.is_animated:
            img.seek(0)

        # 16-bit or 32-bit TIFF (medical / scientific)
        if img.mode in ("I", "I;16", "I;16B", "F"):
            img = self._convert_high_depth(img)

        # CMYK → RGB
        if img.mode == "CMYK":
            img = img.convert("RGB")

        # LAB / YCbCr / HSV → RGB
        if img.mode in ("LAB", "YCbCr", "HSV"):
            img = img.convert("RGB")

        # Palette (P) → RGBA first to preserve transparency
        if img.mode == "P":
            img = img.convert("RGBA")

        # RGBA → RGB (composite on white background)
        if img.mode == "RGBA" and target_mode == "RGB":
            img = self._rgba_to_rgb(img)

        # Grayscale → RGB (replicate channel)
        if img.mode in ("L", "LA") and target_mode == "RGB":
            img = img.convert("RGB")

        # Final conversion if still not target
        if img.mode != target_mode:
            try:
                img = img.convert(target_mode)
            except Exception as e:
                logger.warning(f"Could not convert {original_mode} → {target_mode}: {e}")

        return img

    def get_format_info(self, img: "Image.Image", fp: Path) -> Dict:
        """Extract all format-related information from a PIL Image."""
        info = {
            "color_mode":   img.mode,
            "color_space":  self._detect_color_space(img),
            "bit_depth":    self._get_bit_depth(img),
            "has_alpha":    img.mode in ("RGBA", "LA", "PA"),
            "is_grayscale": img.mode in ("L", "LA"),
            "is_animated":  getattr(img, "n_frames", 1) > 1,
            "n_frames":     getattr(img, "n_frames", 1),
            "detected_format": self._get_pil_format(img),
            "compression_type": img.info.get("compression", ""),
        }
        return info

    # ── Private ───────────────────────────────────────────────────────────────

    def _rgba_to_rgb(self, img: "Image.Image") -> "Image.Image":
        """Composite RGBA onto a white background → RGB."""
        background = Image.new("RGB", img.size, (255, 255, 255))
        try:
            background.paste(img, mask=img.split()[3])
        except Exception:
            background.paste(img)
        return background

    def _convert_high_depth(self, img: "Image.Image") -> "Image.Image":
        """Convert 16/32-bit images to 8-bit RGB safely."""
        try:
            arr = np.array(img, dtype=np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min) * 255
            arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L")
            return Image.fromarray(arr, mode="RGB")
        except Exception:
            return img.convert("RGB")

    def _detect_color_space(self, img: "Image.Image") -> str:
        """Try to detect ICC profile / color space from image metadata."""
        icc = img.info.get("icc_profile", b"")
        if not icc:
            return "sRGB"  # default assumption
        icc_str = icc.decode("latin-1", errors="ignore")
        if "Adobe RGB" in icc_str or "AdobeRGB" in icc_str:
            return "AdobeRGB"
        if "ProPhoto" in icc_str:
            return "ProPhoto"
        if "Display P3" in icc_str:
            return "DisplayP3"
        if "CMYK" in icc_str:
            return "CMYK"
        return "sRGB"

    def _get_bit_depth(self, img: "Image.Image") -> int:
        mode_to_depth = {
            "1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8,
            "CMYK": 8, "YCbCr": 8, "LAB": 8, "HSV": 8,
            "I": 32, "F": 32, "I;16": 16, "I;16B": 16,
            "LA": 8, "PA": 8, "RGBa": 8, "La": 8,
        }
        return mode_to_depth.get(img.mode, 8)

    def _get_pil_format(self, img: "Image.Image") -> str:
        fmt = img.format or ""
        return fmt.lower() if fmt else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — DEEP METADATA EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class MetadataExtractor:
    """
    Extracts all available metadata from an image file:
    - Dimensions, size, aspect ratio, orientation
    - EXIF: camera, ISO, focal length, GPS, datetime, DPI
    - OS: created/modified/accessed timestamps
    - Compression ratio, megapixels, bit depth
    """

    def extract(self, fp: Path, img: "Image.Image") -> Dict:
        """Full metadata extraction — returns flat dict."""
        meta = {}
        meta.update(self._file_meta(fp))
        meta.update(self._dimension_meta(img))
        meta.update(self._exif_meta(fp, img))
        return meta

    # ── File-level metadata ───────────────────────────────────────────────────

    def _file_meta(self, fp: Path) -> Dict:
        meta = {
            "file_path":     str(fp),
            "file_name":     fp.name,
            "file_stem":     fp.stem,
            "extension":     fp.suffix.lower(),
            "file_size_bytes": 0,
            "file_size_kb":  0.0,
            "file_size_mb":  0.0,
            "created_at":    "",
            "modified_at":   "",
            "accessed_at":   "",
        }
        try:
            stat = fp.stat()
            meta["file_size_bytes"] = stat.st_size
            meta["file_size_kb"]    = round(stat.st_size / 1024, 3)
            meta["file_size_mb"]    = round(stat.st_size / 1024 / 1024, 6)
            meta["modified_at"]     = datetime.fromtimestamp(stat.st_mtime).isoformat()
            meta["accessed_at"]     = datetime.fromtimestamp(stat.st_atime).isoformat()
            # created_at is platform-specific
            if platform.system() == "Windows":
                meta["created_at"] = datetime.fromtimestamp(stat.st_ctime).isoformat()
            else:
                # Linux: st_ctime is last metadata change, not creation
                meta["created_at"] = datetime.fromtimestamp(
                    getattr(stat, "st_birthtime", stat.st_mtime)
                ).isoformat()
        except OSError:
            pass
        return meta

    # ── Dimension metadata ────────────────────────────────────────────────────

    def _dimension_meta(self, img: "Image.Image") -> Dict:
        w, h = img.size
        total_px = w * h
        file_bytes = 0  # will be filled from file_meta
        ratio = round(w / h, 4) if h > 0 else 0.0
        if ratio > 1.05:
            orient = "landscape"
        elif ratio < 0.95:
            orient = "portrait"
        else:
            orient = "square"

        return {
            "width":            w,
            "height":           h,
            "aspect_ratio":     ratio,
            "orientation":      orient,
            "total_pixels":     total_px,
            "megapixels":       round(total_px / 1_000_000, 4),
        }

    # ── EXIF metadata ─────────────────────────────────────────────────────────

    def _exif_meta(self, fp: Path, img: "Image.Image") -> Dict:
        meta = {
            "exif_camera_make":   "",
            "exif_camera_model":  "",
            "exif_datetime":      "",
            "exif_iso":           "",
            "exif_focal_length":  "",
            "exif_gps_lat":       0.0,
            "exif_gps_lon":       0.0,
            "exif_dpi_x":         0.0,
            "exif_dpi_y":         0.0,
            "exif_size_bytes":    0,
        }

        if not PIL_AVAILABLE:
            return meta

        # DPI from image info (non-EXIF path)
        dpi = img.info.get("dpi", (0, 0))
        if dpi and len(dpi) == 2:
            meta["exif_dpi_x"] = round(float(dpi[0]), 2)
            meta["exif_dpi_y"] = round(float(dpi[1]), 2)

        # EXIF data
        try:
            raw_exif = img._getexif() if hasattr(img, "_getexif") else None
            if raw_exif is None:
                return meta

            exif_size = len(str(raw_exif).encode("utf-8"))
            meta["exif_size_bytes"] = exif_size

            exif = {
                TAGS.get(k, k): v
                for k, v in raw_exif.items()
                if k in TAGS
            }

            meta["exif_camera_make"]  = str(exif.get("Make", "")).strip()
            meta["exif_camera_model"] = str(exif.get("Model", "")).strip()
            meta["exif_datetime"]     = str(exif.get("DateTimeOriginal",
                                               exif.get("DateTime", ""))).strip()
            meta["exif_iso"]          = str(exif.get("ISOSpeedRatings", "")).strip()

            fl = exif.get("FocalLength")
            if fl:
                try:
                    meta["exif_focal_length"] = f"{float(fl):.1f}mm"
                except Exception:
                    meta["exif_focal_length"] = str(fl)

            # DPI from EXIF XResolution / YResolution
            xres = exif.get("XResolution")
            yres = exif.get("YResolution")
            if xres and meta["exif_dpi_x"] == 0:
                try:
                    meta["exif_dpi_x"] = round(float(xres), 2)
                except Exception:
                    pass
            if yres and meta["exif_dpi_y"] == 0:
                try:
                    meta["exif_dpi_y"] = round(float(yres), 2)
                except Exception:
                    pass

            # GPS
            gps_info = exif.get("GPSInfo")
            if gps_info:
                lat, lon = self._parse_gps(gps_info)
                meta["exif_gps_lat"] = lat
                meta["exif_gps_lon"] = lon

        except Exception as e:
            logger.debug(f"EXIF extraction failed for {fp.name}: {e}")

        return meta

    def _parse_gps(self, gps_info: Dict) -> Tuple[float, float]:
        """Convert raw EXIF GPS tuples to decimal lat/lon."""
        try:
            gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

            def to_decimal(coords, ref):
                d, m, s = [float(x) for x in coords]
                decimal = d + m / 60 + s / 3600
                if ref in ("S", "W"):
                    decimal = -decimal
                return round(decimal, 7)

            lat = to_decimal(gps["GPSLatitude"],  gps.get("GPSLatitudeRef",  "N"))
            lon = to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
            return lat, lon
        except Exception:
            return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — MEMORY & PERFORMANCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class MemoryEngine:
    """
    Memory-safe batch loading engine.
    Auto-detects available RAM, splits large datasets into manageable batches,
    supports generator mode, prefetching, and multi-threading.
    """

    def __init__(
        self,
        memory_budget_mb: Optional[float] = None,
        n_workers: int = 4,
        batch_size: Optional[int] = None,
        use_generator: bool = False,
    ):
        self.n_workers      = min(n_workers, os.cpu_count() or 4)
        self.use_generator  = use_generator
        self._cache: Dict[str, Any] = {}

        # Auto-detect memory budget
        if memory_budget_mb is None:
            self.memory_budget_mb = self._detect_safe_memory_mb()
        else:
            self.memory_budget_mb = memory_budget_mb

        # Auto-calculate batch size if not specified
        if batch_size is None:
            self.batch_size = self._estimate_batch_size()
        else:
            self.batch_size = batch_size

        logger.info(
            f"MemoryEngine: budget={self.memory_budget_mb:.0f}MB, "
            f"batch={self.batch_size}, workers={self.n_workers}"
        )

    def _detect_safe_memory_mb(self) -> float:
        """Use 50% of available RAM as safe budget."""
        if PSUTIL_AVAILABLE:
            available = psutil.virtual_memory().available / 1024 / 1024
            return available * 0.5
        return 2048.0  # safe fallback: 2GB

    def _estimate_batch_size(self) -> int:
        """Estimate how many images fit in memory budget (assume ~3MB avg)."""
        avg_image_mb = 3.0
        return max(1, int(self.memory_budget_mb / avg_image_mb))

    def load_parallel(
        self,
        paths: List[Path],
        load_fn,
        desc: str = "Loading images"
    ) -> List[Any]:
        """
        Load images in parallel using ThreadPoolExecutor.
        Returns list of results in original order.
        """
        results = [None] * len(paths)

        def _load_one(args):
            idx, fp = args
            return idx, load_fn(fp)

        iterator = enumerate(paths)
        if TQDM_AVAILABLE:
            iterator = tqdm(enumerate(paths), total=len(paths), desc=desc)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(_load_one, (i, fp)): i
                       for i, fp in enumerate(paths)}
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    idx = futures[future]
                    logger.debug(f"Worker error at index {idx}: {e}")
                    results[idx] = None

        return results

    def batch_iterator(self, paths: List[Path]) -> Iterator[List[Path]]:
        """Yield batches of paths according to batch_size."""
        for i in range(0, len(paths), self.batch_size):
            yield paths[i: i + self.batch_size]

    def is_cached(self, key: str) -> bool:
        return key in self._cache

    def get_cache(self, key: str) -> Any:
        return self._cache.get(key)

    def set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def clear_cache(self) -> None:
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — DATASET STRUCTURE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DatasetStructureDetector:
    """
    Auto-detects how the dataset is organized and extracts labels accordingly.
    Supports: ImageFolder, CSV, COCO JSON, YOLO, Pascal VOC, flat (no labels).
    """

    def detect(self, root: Path) -> Tuple[DatasetStructure, Dict]:
        """
        Returns: (DatasetStructure enum, label_map dict {path: label})
        """
        # Priority order: JSON COCO → CSV → YOLO → VOC → Class folders → Flat
        if self._has_coco_json(root):
            return DatasetStructure.JSON_COCO, self._load_coco(root)

        if self._has_csv_annotation(root):
            return DatasetStructure.CSV_ANNOT, self._load_csv(root)

        if self._has_yolo_labels(root):
            return DatasetStructure.YOLO, self._load_yolo(root)

        if self._has_voc_labels(root):
            return DatasetStructure.VOC, self._load_voc(root)

        if self._has_class_folders(root):
            return DatasetStructure.CLASS_FOLDERS, self._load_class_folders(root)

        return DatasetStructure.FLAT, {}

    def build_label_map(self, labels: List[str]) -> Tuple[Dict, Dict]:
        """Build class_to_idx and idx_to_class from a list of label strings."""
        unique = sorted(set(labels))
        class_to_idx = {cls: i for i, cls in enumerate(unique)}
        idx_to_class = {i: cls for cls, i in class_to_idx.items()}
        return class_to_idx, idx_to_class

    def validate_class_names(self, classes: List[str]) -> List[Dict]:
        """Check class names for potential issues."""
        issues = []
        reserved = {"train", "test", "val", "validation", "images", "labels"}
        for cls in classes:
            if " " in cls:
                issues.append({"class": cls, "issue": "contains spaces"})
            if cls.lower() in reserved:
                issues.append({"class": cls, "issue": f"reserved name '{cls}'"})
            if not cls.isascii():
                issues.append({"class": cls, "issue": "non-ASCII characters"})
        return issues

    # ── Detection helpers ─────────────────────────────────────────────────────

    def _has_coco_json(self, root: Path) -> bool:
        return any(root.glob("*.json"))

    def _has_csv_annotation(self, root: Path) -> bool:
        return any(root.glob("*.csv"))

    def _has_yolo_labels(self, root: Path) -> bool:
        label_dir = root / "labels"
        return label_dir.exists() and any(label_dir.glob("**/*.txt"))

    def _has_voc_labels(self, root: Path) -> bool:
        ann_dir = root / "Annotations"
        return ann_dir.exists() and any(ann_dir.glob("**/*.xml"))

    def _has_class_folders(self, root: Path) -> bool:
        subdirs = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        return len(subdirs) >= 2

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_class_folders(self, root: Path) -> Dict:
        label_map = {}
        for cls_dir in sorted(root.iterdir()):
            if not cls_dir.is_dir() or cls_dir.name.startswith("."):
                continue
            for ext in SUPPORTED_EXTENSIONS:
                for fp in cls_dir.glob(f"**/*{ext}"):
                    label_map[str(fp)] = cls_dir.name
        return label_map

    def _load_csv(self, root: Path) -> Dict:
        label_map = {}
        for csv_file in root.glob("*.csv"):
            try:
                df = pd.read_csv(str(csv_file))
                # Try common column names
                path_col  = next((c for c in df.columns if "path" in c.lower() or "file" in c.lower()), None)
                label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
                if path_col and label_col:
                    for _, row in df.iterrows():
                        fp = Path(row[path_col])
                        if not fp.is_absolute():
                            fp = root / fp
                        label_map[str(fp)] = str(row[label_col])
            except Exception as e:
                logger.warning(f"Failed to read CSV {csv_file.name}: {e}")
        return label_map

    def _load_coco(self, root: Path) -> Dict:
        label_map = {}
        for json_file in root.glob("*.json"):
            try:
                with open(str(json_file), "r") as f:
                    data = json.load(f)
                if "images" not in data or "annotations" not in data:
                    continue
                categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
                image_id_to_path = {img["id"]: img["file_name"] for img in data["images"]}
                for ann in data["annotations"]:
                    img_path = root / image_id_to_path.get(ann["image_id"], "")
                    cat_name = categories.get(ann["category_id"], "unknown")
                    label_map[str(img_path)] = cat_name
            except Exception as e:
                logger.warning(f"Failed to read COCO JSON {json_file.name}: {e}")
        return label_map

    def _load_yolo(self, root: Path) -> Dict:
        """Load YOLO class labels from labels/*.txt files."""
        label_map = {}
        classes_file = root / "classes.txt"
        class_names = []
        if classes_file.exists():
            with open(str(classes_file)) as f:
                class_names = [line.strip() for line in f if line.strip()]

        labels_dir = root / "labels"
        images_dir = root / "images"
        if not images_dir.exists():
            images_dir = root

        for txt_file in labels_dir.glob("**/*.txt"):
            try:
                with open(str(txt_file)) as f:
                    lines = f.readlines()
                if not lines:
                    continue
                # First token is class_id
                class_ids = list(set(int(line.split()[0]) for line in lines if line.strip()))
                labels = [class_names[cid] if cid < len(class_names) else str(cid) for cid in class_ids]
                # Find corresponding image
                for ext in SUPPORTED_EXTENSIONS:
                    img_path = images_dir / (txt_file.stem + ext)
                    if img_path.exists():
                        label_map[str(img_path)] = "|".join(labels)  # multi-label
                        break
            except Exception:
                pass
        return label_map

    def _load_voc(self, root: Path) -> Dict:
        """Load Pascal VOC XML annotations."""
        import xml.etree.ElementTree as ET
        label_map = {}
        ann_dir = root / "Annotations"
        img_dir = root / "JPEGImages"
        if not img_dir.exists():
            img_dir = root

        for xml_file in ann_dir.glob("**/*.xml"):
            try:
                tree = ET.parse(str(xml_file))
                rt = tree.getroot()
                filename = rt.findtext("filename", "")
                objects = rt.findall("object")
                labels = list(set(obj.findtext("name", "") for obj in objects))
                img_path = img_dir / filename
                if img_path.exists():
                    label_map[str(img_path)] = "|".join(labels)
            except Exception:
                pass
        return label_map


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — ML PROBLEM DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class MLProblemDetector:
    """
    Detects all ML-specific dataset problems:
    - Class imbalance & minority classes
    - Insufficient data per class
    - Dimension inconsistency
    - Color space inconsistency
    - Covariate shift between splits
    - Domain shift between datasets
    - Hidden stratification
    - Temporal & patient leakage
    """

    def detect_all(self, df: pd.DataFrame) -> List[DatasetIssue]:
        """Run all ML detectors. Returns list of DatasetIssue."""
        issues = []
        issues.extend(self._check_class_imbalance(df))
        issues.extend(self._check_insufficient_data(df))
        issues.extend(self._check_dimension_consistency(df))
        issues.extend(self._check_colorspace_consistency(df))
        issues.extend(self._check_covariate_shift(df))
        issues.extend(self._check_min_images_total(df))
        return issues

    # ── Checks ────────────────────────────────────────────────────────────────

    def _check_class_imbalance(self, df: pd.DataFrame) -> List[DatasetIssue]:
        issues = []
        if "label" not in df.columns or df["label"].isna().all():
            return issues

        counts = df[df["label"].notna() & (df["label"] != "")]["label"].value_counts()
        if len(counts) < 2:
            return issues

        majority = counts.iloc[0]
        minority = counts.iloc[-1]
        ratio = majority / minority if minority > 0 else float("inf")

        if ratio > 10:
            minority_classes = counts[counts < majority / 10].index.tolist()
            issues.append(DatasetIssue(
                issue_id="ML-001",
                severity=SEVERITY_CRITICAL,
                category="Class Imbalance",
                description=f"Severe class imbalance: ratio {ratio:.1f}:1 "
                            f"(max={majority}, min={minority})",
                affected=minority_classes,
                fix="Use oversampling (SMOTE), undersampling, or class weights."
            ))
        elif ratio > 3:
            issues.append(DatasetIssue(
                issue_id="ML-002",
                severity=SEVERITY_MEDIUM,
                category="Class Imbalance",
                description=f"Moderate class imbalance: ratio {ratio:.1f}:1",
                fix="Consider stratified sampling or class weights."
            ))
        return issues

    def _check_insufficient_data(self, df: pd.DataFrame) -> List[DatasetIssue]:
        issues = []
        if "label" not in df.columns:
            return issues

        counts = df[df["label"].notna() & (df["label"] != "")]["label"].value_counts()
        poor_classes = counts[counts < MIN_IMAGES_PER_CLASS].index.tolist()

        if poor_classes:
            issues.append(DatasetIssue(
                issue_id="ML-003",
                severity=SEVERITY_HIGH,
                category="Insufficient Data",
                description=f"{len(poor_classes)} class(es) have fewer than "
                            f"{MIN_IMAGES_PER_CLASS} images",
                affected=poor_classes,
                fix="Collect more images or use data augmentation / synthetic generation."
            ))
        return issues

    def _check_min_images_total(self, df: pd.DataFrame) -> List[DatasetIssue]:
        issues = []
        valid = df[df["is_valid"] == True] if "is_valid" in df.columns else df
        if len(valid) < MIN_TOTAL_IMAGES:
            issues.append(DatasetIssue(
                issue_id="ML-004",
                severity=SEVERITY_CRITICAL,
                category="Insufficient Data",
                description=f"Only {len(valid)} valid images. "
                            f"Minimum recommended: {MIN_TOTAL_IMAGES}",
                fix="Collect more data or use synthetic data generation."
            ))
        return issues

    def _check_dimension_consistency(self, df: pd.DataFrame) -> List[DatasetIssue]:
        issues = []
        if "width" not in df.columns or "height" not in df.columns:
            return issues

        valid = df[(df["width"] > 0) & (df["height"] > 0)]
        if len(valid) == 0:
            return issues

        unique_sizes = valid[["width", "height"]].drop_duplicates()
        n_unique = len(unique_sizes)

        if n_unique > 1:
            w_range = f"{valid['width'].min()}–{valid['width'].max()}"
            h_range = f"{valid['height'].min()}–{valid['height'].max()}"
            sev = SEVERITY_HIGH if n_unique > 10 else SEVERITY_MEDIUM
            issues.append(DatasetIssue(
                issue_id="ML-005",
                severity=sev,
                category="Dimension Inconsistency",
                description=f"{n_unique} unique image sizes. "
                            f"Width: {w_range}, Height: {h_range}",
                fix="Resize all images to a consistent target size before training."
            ))
        return issues

    def _check_colorspace_consistency(self, df: pd.DataFrame) -> List[DatasetIssue]:
        issues = []
        if "color_mode" not in df.columns:
            return issues

        modes = df["color_mode"].dropna().unique().tolist()
        if len(modes) > 1:
            issues.append(DatasetIssue(
                issue_id="ML-006",
                severity=SEVERITY_MEDIUM,
                category="Color Space Inconsistency",
                description=f"Mixed color modes detected: {modes}",
                fix="Normalize all images to RGB before training."
            ))
        return issues

    def _check_covariate_shift(self, df: pd.DataFrame) -> List[DatasetIssue]:
        """
        Compare pixel value distributions between train and test splits.
        Uses Kolmogorov-Smirnov test on image brightness/size distributions.
        """
        issues = []
        if not SCIPY_AVAILABLE:
            return issues
        if "split" not in df.columns or "total_pixels" not in df.columns:
            return issues

        train = df[df["split"] == "train"]
        test  = df[df["split"] == "test"]

        if len(train) < 10 or len(test) < 10:
            return issues

        # KS test on image sizes (proxy for content distribution)
        ks_stat, p_value = stats.ks_2samp(
            train["total_pixels"].dropna(),
            test["total_pixels"].dropna()
        )

        if p_value < 0.05:
            issues.append(DatasetIssue(
                issue_id="ML-007",
                severity=SEVERITY_HIGH,
                category="Covariate Shift",
                description=f"Significant distribution difference between train and test "
                            f"(KS statistic={ks_stat:.3f}, p={p_value:.4f})",
                fix="Ensure train and test come from the same distribution. "
                    "Use stratified splitting."
            ))
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — DUPLICATE & LEAKAGE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DuplicateLeakageDetector:
    """
    Detects exact and near-duplicate images using multiple hashing strategies.
    Also detects cross-split data leakage.

    Methods:
    - SHA256: exact duplicates
    - pHash / dHash / aHash / wHash: near-duplicates
    - Ensemble: combines all hash methods for maximum accuracy
    """

    def __init__(
        self,
        phash_threshold: int = 8,
        dhash_threshold: int = 8,
        use_ensemble:    bool = True,
    ):
        self.phash_threshold = phash_threshold
        self.dhash_threshold = dhash_threshold
        self.use_ensemble    = use_ensemble

    def compute_hashes(self, fp: Path, img: "Image.Image") -> Dict[str, str]:
        """Compute all available hashes for one image."""
        hashes = {"sha256": "", "phash": "", "dhash": "", "ahash": "", "whash": ""}

        # SHA256 — exact duplicate detection
        try:
            h = hashlib.sha256()
            with open(str(fp), "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            hashes["sha256"] = h.hexdigest()
        except OSError:
            pass

        # Perceptual hashes — near-duplicate detection
        if IMAGEHASH_AVAILABLE and img is not None:
            try:
                hashes["phash"] = str(imagehash.phash(img))
                hashes["dhash"] = str(imagehash.dhash(img))
                hashes["ahash"] = str(imagehash.average_hash(img))
                hashes["whash"] = str(imagehash.whash(img))
            except Exception as e:
                logger.debug(f"Hash computation failed for {fp.name}: {e}")

        return hashes

    def find_exact_duplicates(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Find exact duplicates by SHA256.
        Returns: {sha256_hash: [list of duplicate file paths]}
        """
        if "sha256" not in df.columns:
            return {}

        groups = df.groupby("sha256")["file_path"].apply(list).to_dict()
        return {h: paths for h, paths in groups.items() if len(paths) > 1}

    def find_near_duplicates(
        self,
        df: pd.DataFrame,
        method: str = "phash",
        threshold: int = 8
    ) -> List[Tuple[str, str, int]]:
        """
        Find near-duplicate pairs using perceptual hashing.
        Returns: list of (path_a, path_b, hamming_distance)
        """
        if not IMAGEHASH_AVAILABLE:
            return []
        if method not in df.columns:
            return []

        valid = df[df[method].notna() & (df[method] != "")].copy()
        if len(valid) < 2:
            return []

        pairs = []
        paths  = valid["file_path"].tolist()
        hashes = valid[method].tolist()

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                try:
                    h1 = imagehash.hex_to_hash(hashes[i])
                    h2 = imagehash.hex_to_hash(hashes[j])
                    dist = h1 - h2
                    if dist <= threshold:
                        pairs.append((paths[i], paths[j], dist))
                except Exception:
                    continue
        return pairs

    def detect_cross_split_leakage(
        self,
        df: pd.DataFrame,
        split_a: str = "train",
        split_b: str = "test"
    ) -> List[Tuple[str, str]]:
        """
        Detect images present in both split_a and split_b via SHA256.
        Returns: list of (path_in_train, path_in_test) leaking pairs.
        """
        if "split" not in df.columns or "sha256" not in df.columns:
            return []

        a_hashes = dict(zip(
            df[df["split"] == split_a]["sha256"],
            df[df["split"] == split_a]["file_path"]
        ))
        b_hashes = dict(zip(
            df[df["split"] == split_b]["sha256"],
            df[df["split"] == split_b]["file_path"]
        ))

        leaked = []
        for h, path_a in a_hashes.items():
            if h and h in b_hashes:
                leaked.append((path_a, b_hashes[h]))
        return leaked

    def detect_near_duplicate_leakage(
        self,
        df: pd.DataFrame,
        split_a: str = "train",
        split_b: str = "test",
        method: str = "phash",
        threshold: int = 8
    ) -> List[Tuple[str, str, int]]:
        """
        Detect near-duplicate images across splits (augmentation leakage).
        """
        if not IMAGEHASH_AVAILABLE or method not in df.columns:
            return []

        a_df = df[df["split"] == split_a]
        b_df = df[df["split"] == split_b]

        if len(a_df) == 0 or len(b_df) == 0:
            return []

        leaked = []
        for _, row_a in a_df.iterrows():
            h_a_str = row_a.get(method, "")
            if not h_a_str:
                continue
            try:
                h_a = imagehash.hex_to_hash(h_a_str)
            except Exception:
                continue
            for _, row_b in b_df.iterrows():
                h_b_str = row_b.get(method, "")
                if not h_b_str:
                    continue
                try:
                    h_b = imagehash.hex_to_hash(h_b_str)
                    dist = h_a - h_b
                    if dist <= threshold:
                        leaked.append((row_a["file_path"], row_b["file_path"], dist))
                except Exception:
                    continue
        return leaked

    def mark_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark duplicates in the DataFrame.
        Keeps first occurrence, marks the rest as duplicate.
        """
        df = df.copy()
        df["is_duplicate"] = False
        df["duplicate_of"] = ""

        # Exact duplicates via SHA256
        if "sha256" in df.columns:
            seen_sha: Dict[str, str] = {}
            for idx, row in df.iterrows():
                h = row.get("sha256", "")
                if not h:
                    continue
                if h in seen_sha:
                    df.at[idx, "is_duplicate"] = True
                    df.at[idx, "duplicate_of"] = seen_sha[h]
                else:
                    seen_sha[h] = row["file_path"]

        return df


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — SMART SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class SmartSplitter:
    """
    Splits dataset into Train / Val / Test with multiple strategies:
    - Random: for large balanced datasets
    - Stratified: preserves class proportions
    - Group: prevents same entity in multiple splits
    - Temporal: respects time ordering

    Validates splits for leakage and balance after creation.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
        test_ratio:  float = 0.15,
        seed:        int   = 42,
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed

    def split(
        self,
        df: pd.DataFrame,
        strategy: SplitStrategy = SplitStrategy.STRATIFIED,
        group_col: Optional[str] = None,
        time_col:  Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add 'split' column to df.
        Returns df with 'split' assigned: 'train' / 'val' / 'test'.
        """
        df = df.copy()
        n = len(df)
        if n == 0:
            df["split"] = ""
            return df

        if strategy == SplitStrategy.RANDOM:
            df["split"] = self._random_split(n)

        elif strategy == SplitStrategy.STRATIFIED:
            if "label" in df.columns and df["label"].notna().any():
                df["split"] = self._stratified_split(df["label"].tolist())
            else:
                df["split"] = self._random_split(n)

        elif strategy == SplitStrategy.GROUP and group_col:
            df["split"] = self._group_split(df, group_col)

        elif strategy == SplitStrategy.TEMPORAL and time_col:
            df["split"] = self._temporal_split(df, time_col)

        else:
            df["split"] = self._random_split(n)

        return df

    # ── Strategy implementations ──────────────────────────────────────────────

    def _random_split(self, n: int) -> List[str]:
        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)
        splits = [""] * n
        for i in indices[:n_train]:
            splits[i] = "train"
        for i in indices[n_train: n_train + n_val]:
            splits[i] = "val"
        for i in indices[n_train + n_val:]:
            splits[i] = "test"
        return splits

    def _stratified_split(self, labels: List[str]) -> List[str]:
        """
        Preserve class proportion in each split.
        """
        from collections import defaultdict
        rng = random.Random(self.seed)

        # Group indices by label
        label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            label_to_indices[lbl].append(i)

        splits = ["train"] * len(labels)

        for lbl, idxs in label_to_indices.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_val  = max(1, int(n * self.val_ratio))
            n_test = max(1, int(n * self.test_ratio))
            for i in idxs[-n_test:]:
                splits[i] = "test"
            for i in idxs[-(n_test + n_val): -n_test]:
                splits[i] = "val"

        return splits

    def _group_split(self, df: pd.DataFrame, group_col: str) -> List[str]:
        """
        Assign entire groups to a single split.
        Prevents same patient/subject from appearing in multiple splits.
        """
        rng = random.Random(self.seed)
        groups = df[group_col].dropna().unique().tolist()
        rng.shuffle(groups)
        n_g = len(groups)
        n_train = int(n_g * self.train_ratio)
        n_val   = int(n_g * self.val_ratio)

        group_to_split = {}
        for i, g in enumerate(groups):
            if i < n_train:
                group_to_split[g] = "train"
            elif i < n_train + n_val:
                group_to_split[g] = "val"
            else:
                group_to_split[g] = "test"

        return [group_to_split.get(g, "train") for g in df[group_col]]

    def _temporal_split(self, df: pd.DataFrame, time_col: str) -> List[str]:
        """
        Sort by time and split sequentially.
        Train = earliest, Test = latest.
        """
        df_sorted = df.sort_values(time_col).copy()
        n = len(df_sorted)
        n_train = int(n * self.train_ratio)
        n_val   = int(n * self.val_ratio)

        splits = ["test"] * n
        for i in range(n_train):
            splits[i] = "train"
        for i in range(n_train, n_train + n_val):
            splits[i] = "val"

        # Re-align with original index
        df_sorted["split"] = splits
        return df_sorted.loc[df.index]["split"].tolist()

    def validate_split(self, df: pd.DataFrame) -> List[DatasetIssue]:
        """Check split health: balance, leakage, minimum size."""
        issues = []
        if "split" not in df.columns:
            return issues

        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name]
            if len(split_df) == 0:
                issues.append(DatasetIssue(
                    issue_id="SPLIT-001",
                    severity=SEVERITY_CRITICAL,
                    category="Empty Split",
                    description=f"Split '{split_name}' is empty",
                    fix="Check splitting ratios or increase dataset size."
                ))
        return issues


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — DATASET VERSIONING & FINGERPRINT
# ─────────────────────────────────────────────────────────────────────────────

class DatasetVersioner:
    """
    Tracks dataset versions using SHA256 fingerprints.
    Supports: immutable snapshots, version diff, audit trail.
    """

    def __init__(self, version_dir: Optional[Path] = None):
        self.version_dir = version_dir or Path(".nydra_versions")
        self.version_dir.mkdir(exist_ok=True)
        self._history_file = self.version_dir / "history.json"
        self._history = self._load_history()

    def compute_fingerprint(self, paths: List[Path]) -> str:
        """
        SHA256 fingerprint of the entire dataset.
        Based on: sorted list of (relative_path + file_size + mtime).
        """
        h = hashlib.sha256()
        for fp in sorted(paths, key=lambda p: str(p)):
            try:
                stat = fp.stat()
                entry = f"{fp.name}:{stat.st_size}:{stat.st_mtime}"
                h.update(entry.encode("utf-8"))
            except OSError:
                pass
        return h.hexdigest()[:16]  # short fingerprint (16 hex chars)

    def compute_split_fingerprints(self, df: pd.DataFrame) -> Dict[str, str]:
        """Fingerprint each split separately."""
        fps = {}
        if "split" not in df.columns:
            return fps
        for split_name in ["train", "val", "test"]:
            split_paths = [
                Path(p) for p in df[df["split"] == split_name]["file_path"].tolist()
            ]
            if split_paths:
                fps[split_name] = self.compute_fingerprint(split_paths)
        return fps

    def save_snapshot(
        self,
        fingerprint: str,
        stats: Dict,
        version: Optional[str] = None
    ) -> str:
        """Save an immutable snapshot of the current dataset state."""
        if version is None:
            version = self._next_version()

        snapshot = {
            "version":     version,
            "fingerprint": fingerprint,
            "timestamp":   datetime.utcnow().isoformat(),
            "stats":       stats,
        }
        self._history.append(snapshot)
        self._save_history()

        snap_file = self.version_dir / f"snapshot_{version}.json"
        with open(str(snap_file), "w") as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"Saved dataset snapshot v{version} — fingerprint: {fingerprint}")
        return version

    def diff(self, v1: str, v2: str) -> Dict:
        """Compare two snapshot versions."""
        snap1 = self._find_snapshot(v1)
        snap2 = self._find_snapshot(v2)
        if not snap1 or not snap2:
            return {"error": "One or both versions not found"}

        return {
            "v1": v1, "v2": v2,
            "fingerprint_changed": snap1["fingerprint"] != snap2["fingerprint"],
            "total_images_v1": snap1["stats"].get("total_images", 0),
            "total_images_v2": snap2["stats"].get("total_images", 0),
            "delta_images": (
                snap2["stats"].get("total_images", 0) -
                snap1["stats"].get("total_images", 0)
            ),
        }

    def get_history(self) -> List[Dict]:
        return self._history

    # ── Private ───────────────────────────────────────────────────────────────

    def _next_version(self) -> str:
        """Auto-increment semantic version."""
        if not self._history:
            return "1.0.0"
        last = self._history[-1].get("version", "0.0.0")
        parts = last.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)

    def _find_snapshot(self, version: str) -> Optional[Dict]:
        return next((s for s in self._history if s["version"] == version), None)

    def _load_history(self) -> List[Dict]:
        if self._history_file.exists():
            try:
                with open(str(self._history_file)) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save_history(self) -> None:
        with open(str(self._history_file), "w") as f:
            json.dump(self._history, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 11 — SMART RECOVERY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SmartRecovery:
    """
    Handles all error conditions gracefully.
    Never crashes — always falls back to a safe state.
    Classifies errors by severity and logs everything.
    """

    def __init__(self):
        self._error_log: List[Dict] = []

    def safe_open(self, fp: Path) -> Tuple[Optional["Image.Image"], LoadStatus, str]:
        """
        Safely open an image with multiple fallback strategies.
        Returns: (PIL Image or None, status, error_message)
        """
        if not PIL_AVAILABLE:
            return None, LoadStatus.UNSUPPORTED, "Pillow not installed"

        # Strategy 1: Normal PIL open
        try:
            img = Image.open(str(fp))
            img.load()
            return img, LoadStatus.OK, ""
        except OSError as e:
            err = str(e).lower()
            if "truncated" in err:
                # Strategy 2: Force-load truncated (LOAD_TRUNCATED_IMAGES=True)
                try:
                    img = Image.open(str(fp))
                    img.load()
                    self._log_error(fp, "truncated", SEVERITY_HIGH, str(e))
                    return img, LoadStatus.TRUNCATED, str(e)
                except Exception as e2:
                    pass

            # Strategy 3: Try OpenCV if PIL fails
            if CV2_AVAILABLE:
                try:
                    arr = cv2.imread(str(fp))
                    if arr is not None:
                        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(arr_rgb)
                        self._log_error(fp, "pil_failed_cv2_ok", SEVERITY_MEDIUM, str(e))
                        return img, LoadStatus.OK, ""
                except Exception:
                    pass

            self._log_error(fp, "corrupted", SEVERITY_CRITICAL, str(e))
            return None, LoadStatus.CORRUPTED, str(e)

        except Exception as e:
            self._log_error(fp, "unknown_error", SEVERITY_HIGH, str(e))
            return None, LoadStatus.CORRUPTED, str(e)

    def safe_hash(self, fp: Path, img: Optional["Image.Image"]) -> Dict:
        """Compute hashes safely — return empty strings on failure."""
        hashes = {"sha256": "", "phash": "", "dhash": "", "ahash": "", "whash": ""}
        try:
            h = hashlib.sha256()
            with open(str(fp), "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            hashes["sha256"] = h.hexdigest()
        except Exception:
            pass

        if IMAGEHASH_AVAILABLE and img is not None:
            for name, fn in [
                ("phash", imagehash.phash),
                ("dhash", imagehash.dhash),
                ("ahash", imagehash.average_hash),
                ("whash", imagehash.whash),
            ]:
                try:
                    hashes[name] = str(fn(img))
                except Exception:
                    pass

        return hashes

    def classify_error(self, status: LoadStatus) -> str:
        critical = {LoadStatus.ZERO_BYTES, LoadStatus.CORRUPTED}
        high     = {LoadStatus.TRUNCATED, LoadStatus.WRONG_FORMAT,
                    LoadStatus.TOO_LARGE, LoadStatus.ENCRYPTED}
        medium   = {LoadStatus.TOO_SMALL, LoadStatus.UNSUPPORTED}
        low      = {LoadStatus.HIDDEN, LoadStatus.SKIPPED}
        if status in critical: return SEVERITY_CRITICAL
        if status in high:     return SEVERITY_HIGH
        if status in medium:   return SEVERITY_MEDIUM
        if status in low:      return SEVERITY_LOW
        return SEVERITY_LOW

    def get_error_log(self) -> List[Dict]:
        return self._error_log

    def export_error_log(self, output_path: Path) -> None:
        with open(str(output_path), "w") as f:
            json.dump(self._error_log, f, indent=2)
        logger.info(f"Error log saved to: {output_path}")

    def _log_error(self, fp: Path, error_type: str, severity: str, message: str) -> None:
        self._error_log.append({
            "file":      str(fp),
            "type":      error_type,
            "severity":  severity,
            "message":   message,
            "timestamp": datetime.utcnow().isoformat(),
        })


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 12 — ML READINESS SCORER & REPORTER
# ─────────────────────────────────────────────────────────────────────────────

class MLReadinessScorer:
    """
    Aggregates all audit results into a single ML Readiness Score (0-100).
    Generates a full structured report with issues, fix checklist, and verdict.
    """

    WEIGHTS = {
        "file_health":   0.20,
        "balance":       0.20,
        "duplicate":     0.20,
        "leakage":       0.25,
        "consistency":   0.15,
    }

    def score(self, df: pd.DataFrame, issues: List[DatasetIssue]) -> MLReadinessReport:
        report = MLReadinessReport()
        report.generated_at = datetime.utcnow().isoformat()
        report.issues = issues

        # ── Basic counts ──────────────────────────────────────────────────────
        report.total_images      = len(df)
        report.valid_images      = int(df["is_valid"].sum()) if "is_valid" in df.columns else len(df)
        report.corrupted_images  = int((df["load_status"] == LoadStatus.CORRUPTED.value).sum()) \
                                   if "load_status" in df.columns else 0
        report.duplicate_images  = int(df["is_duplicate"].sum()) if "is_duplicate" in df.columns else 0
        report.zero_byte_files   = int((df["load_status"] == LoadStatus.ZERO_BYTES.value).sum()) \
                                   if "load_status" in df.columns else 0
        report.hidden_files      = int((df["load_status"] == LoadStatus.HIDDEN.value).sum()) \
                                   if "load_status" in df.columns else 0
        report.wrong_format_files = int((df["load_status"] == LoadStatus.WRONG_FORMAT.value).sum()) \
                                    if "load_status" in df.columns else 0

        # ── Class info ────────────────────────────────────────────────────────
        if "label" in df.columns:
            counts = df[df["label"].notna() & (df["label"] != "")]["label"].value_counts()
            report.n_classes         = len(counts)
            report.class_distribution = counts.to_dict()
            if len(counts) >= 2:
                report.imbalance_ratio = counts.iloc[0] / counts.iloc[-1]
                report.minority_classes = counts[counts < MIN_IMAGES_PER_CLASS].index.tolist()

        # ── Leakage ───────────────────────────────────────────────────────────
        leakage_issues = [i for i in issues if "leakage" in i.issue_id.lower() or
                                               "leakage" in i.category.lower()]
        report.leakage_detected = len(leakage_issues) > 0

        # ── Consistency ───────────────────────────────────────────────────────
        dim_issues = [i for i in issues if "dimension" in i.category.lower()]
        cs_issues  = [i for i in issues if "color space" in i.category.lower()]
        report.dimension_consistent  = len(dim_issues) == 0
        report.colorspace_consistent = len(cs_issues) == 0

        # ── Sub-scores ────────────────────────────────────────────────────────
        report.file_health_score = self._file_health_score(report)
        report.balance_score     = self._balance_score(report)
        report.duplicate_score   = self._duplicate_score(report)
        report.leakage_score     = self._leakage_score(report, issues)
        report.consistency_score = self._consistency_score(report)

        # ── Overall score ─────────────────────────────────────────────────────
        report.overall_score = round(
            report.file_health_score   * self.WEIGHTS["file_health"]   +
            report.balance_score       * self.WEIGHTS["balance"]        +
            report.duplicate_score     * self.WEIGHTS["duplicate"]      +
            report.leakage_score       * self.WEIGHTS["leakage"]        +
            report.consistency_score   * self.WEIGHTS["consistency"],
            1
        )

        # ── Verdict ───────────────────────────────────────────────────────────
        critical_issues = [i for i in issues if i.severity == SEVERITY_CRITICAL]
        if report.overall_score >= 80 and not critical_issues:
            report.verdict = "Ready ✅"
        elif report.overall_score >= 50 or not critical_issues:
            report.verdict = "Needs Work ⚠️"
        else:
            report.verdict = "Not Ready ❌"

        # ── Fix checklist ─────────────────────────────────────────────────────
        report.fix_checklist = self._build_checklist(issues)

        return report

    # ── Sub-scorers ───────────────────────────────────────────────────────────

    def _file_health_score(self, r: MLReadinessReport) -> float:
        if r.total_images == 0: return 0.0
        bad = r.corrupted_images + r.zero_byte_files + r.wrong_format_files
        return round(max(0, 100 - (bad / r.total_images) * 100), 1)

    def _balance_score(self, r: MLReadinessReport) -> float:
        if r.imbalance_ratio <= 0: return 100.0
        if r.imbalance_ratio <= 2: return 100.0
        if r.imbalance_ratio <= 5: return 75.0
        if r.imbalance_ratio <= 10: return 50.0
        return 25.0

    def _duplicate_score(self, r: MLReadinessReport) -> float:
        if r.total_images == 0: return 0.0
        dup_rate = r.duplicate_images / r.total_images
        return round(max(0, 100 - dup_rate * 200), 1)

    def _leakage_score(self, r: MLReadinessReport, issues: List[DatasetIssue]) -> float:
        leakage = [i for i in issues if "leakage" in i.category.lower() or
                                        "leakage" in i.issue_id.lower()]
        if not leakage: return 100.0
        critical_leakage = [i for i in leakage if i.severity == SEVERITY_CRITICAL]
        return 0.0 if critical_leakage else 40.0

    def _consistency_score(self, r: MLReadinessReport) -> float:
        score = 100.0
        if not r.dimension_consistent:  score -= 40.0
        if not r.colorspace_consistent: score -= 30.0
        if r.has_covariate_shift:       score -= 30.0
        return max(0.0, score)

    def _build_checklist(self, issues: List[DatasetIssue]) -> List[str]:
        """Build prioritized fix checklist from issues."""
        order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1,
                 SEVERITY_MEDIUM: 2, SEVERITY_LOW: 3}
        sorted_issues = sorted(issues, key=lambda i: order.get(i.severity, 4))
        return [
            f"[{i.severity}] {i.description} → {i.fix}"
            for i in sorted_issues if i.fix
        ]


class ReportExporter:
    """Exports MLReadinessReport to multiple formats."""

    def to_dict(self, report: MLReadinessReport) -> Dict:
        d = report.__dict__.copy()
        d["issues"] = [
            {"id": i.issue_id, "severity": i.severity,
             "category": i.category, "description": i.description,
             "fix": i.fix, "affected": i.affected}
            for i in report.issues
        ]
        return d

    def to_json(self, report: MLReadinessReport, path: Optional[Path] = None) -> str:
        data = self.to_dict(report)
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            with open(str(path), "w") as f:
                f.write(json_str)
        return json_str

    def to_markdown(self, report: MLReadinessReport, path: Optional[Path] = None) -> str:
        lines = [
            "# 🩺 Nydra — Image Dataset ML Readiness Report",
            f"\n**Generated:** {report.generated_at}",
            f"\n## 🎯 Overall Score: {report.overall_score}/100 — {report.verdict}",
            "\n## 📊 Sub-scores",
            f"| Dimension | Score |",
            f"|-----------|-------|",
            f"| File Health | {report.file_health_score}/100 |",
            f"| Class Balance | {report.balance_score}/100 |",
            f"| Duplicate Check | {report.duplicate_score}/100 |",
            f"| Leakage Check | {report.leakage_score}/100 |",
            f"| Consistency | {report.consistency_score}/100 |",
            "\n## 📁 Dataset Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Images | {report.total_images} |",
            f"| Valid Images | {report.valid_images} |",
            f"| Corrupted | {report.corrupted_images} |",
            f"| Duplicates | {report.duplicate_images} |",
            f"| Classes | {report.n_classes} |",
            f"| Imbalance Ratio | {report.imbalance_ratio:.1f}:1 |",
        ]

        if report.issues:
            lines += ["\n## ⚠️ Issues Found", "| Severity | Category | Description |",
                      "|----------|----------|-------------|"]
            for issue in report.issues:
                lines.append(f"| {issue.severity} | {issue.category} | {issue.description} |")

        if report.fix_checklist:
            lines += ["\n## 🔧 Fix Checklist"]
            for item in report.fix_checklist:
                lines.append(f"- {item}")

        md = "\n".join(lines)
        if path:
            with open(str(path), "w") as f:
                f.write(md)
        return md

    def to_csv(self, df: pd.DataFrame, path: Path) -> None:
        df.to_csv(str(path), index=False)
        logger.info(f"DataFrame saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CLASS — ImageLoader
# ─────────────────────────────────────────────────────────────────────────────

class ImageLoader:
    """
    🩺 Nydra ImageLoader — Master Entry Point

    Orchestrates all 12 blocks into a single, coherent pipeline:

    1. FileSystemScanner     → discover all files
    2. FileValidator         → deep validation per file
    3. FormatIntelligence    → format normalization
    4. MetadataExtractor     → extract all metadata
    5. MemoryEngine          → memory-safe parallel loading
    6. DatasetStructureDetector → detect labels & structure
    7. MLProblemDetector     → detect ML-specific problems
    8. DuplicateLeakageDetector → find duplicates & leakage
    9. SmartSplitter         → train/val/test splitting
    10. DatasetVersioner     → versioning & fingerprinting
    11. SmartRecovery        → error handling & fallbacks
    12. MLReadinessScorer    → final score & report

    Usage:
        loader = ImageLoader("dataset/")
        df, report = loader.load()

        # Or step by step:
        df = loader.scan_and_load("dataset/")
        df = loader.split(df)
        report = loader.audit(df)
    """

    def __init__(
        self,
        path: Union[str, Path, None] = None,
        recursive:         bool  = True,
        follow_symlinks:   bool  = False,
        memory_budget_mb:  Optional[float] = None,
        n_workers:         int   = 4,
        batch_size:        Optional[int] = None,
        use_generator:     bool  = False,
        compute_hashes:    bool  = True,
        normalize_images:  bool  = False,
        split_strategy:    SplitStrategy = SplitStrategy.STRATIFIED,
        train_ratio:       float = 0.70,
        val_ratio:         float = 0.15,
        test_ratio:        float = 0.15,
        seed:              int   = 42,
        version_dir:       Optional[Path] = None,
        verbose:           bool  = True,
    ):
        self.path             = Path(path) if path else None
        self.recursive        = recursive
        self.follow_symlinks  = follow_symlinks
        self.compute_hashes   = compute_hashes
        self.normalize_images = normalize_images
        self.split_strategy   = split_strategy
        self.verbose          = verbose

        # Initialize all blocks
        self.scanner    = FileSystemScanner(recursive, follow_symlinks)
        self.validator  = FileValidator()
        self.fmt        = FormatIntelligence()
        self.meta       = MetadataExtractor()
        self.memory     = MemoryEngine(memory_budget_mb, n_workers, batch_size, use_generator)
        self.struct     = DatasetStructureDetector()
        self.ml_detect  = MLProblemDetector()
        self.dedup      = DuplicateLeakageDetector()
        self.splitter   = SmartSplitter(train_ratio, val_ratio, test_ratio, seed)
        self.versioner  = DatasetVersioner(version_dir)
        self.recovery   = SmartRecovery()
        self.scorer     = MLReadinessScorer()
        self.exporter   = ReportExporter()

        self._scan_summary: Dict = {}
        self._dataset_structure: DatasetStructure = DatasetStructure.UNKNOWN
        self._label_map: Dict[str, str] = {}

    # ── Main API ──────────────────────────────────────────────────────────────

    def load(
        self,
        path: Union[str, Path, None] = None,
        auto_split:   bool = True,
        auto_audit:   bool = True,
        save_version: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[MLReadinessReport]]:
        """
        Full pipeline: scan → validate → load → label → hash → split → audit.
        Returns (DataFrame, MLReadinessReport).
        """
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("No path provided to ImageLoader.load()")

        if self.verbose:
            logger.info(f"🩺 ImageLoader starting — path: {target}")

        # Step 1: Scan
        valid_paths, self._scan_summary = self.scanner.scan(target)
        if self.verbose:
            logger.info(
                f"📁 Found {self._scan_summary['scanned']} files → "
                f"{self._scan_summary['accepted']} accepted, "
                f"{self._scan_summary['rejected']} rejected"
            )

        if not valid_paths:
            logger.warning("No valid image files found.")
            return pd.DataFrame(), None

        # Step 2: Detect dataset structure & labels
        self._dataset_structure, self._label_map = self.struct.detect(target)
        if self.verbose:
            logger.info(f"📂 Dataset structure: {self._dataset_structure.value}")

        # Step 3: Load all images (parallel, batch-safe)
        df = self._load_images(valid_paths)

        # Step 4: Attach labels
        df = self._attach_labels(df)

        # Step 5: Mark duplicates
        df = self.dedup.mark_duplicates(df)

        # Step 6: Auto-split
        if auto_split:
            df = self.splitter.split(df, strategy=self.split_strategy)
            if self.verbose:
                if "split" in df.columns:
                    counts = df["split"].value_counts().to_dict()
                    logger.info(f"✂️ Split: {counts}")

        # Step 7: Audit
        report = None
        if auto_audit:
            report = self.audit(df)
            if self.verbose:
                logger.info(
                    f"🎯 ML Readiness: {report.overall_score}/100 — {report.verdict}"
                )

        # Step 8: Versioning
        if save_version:
            fp_str = self.versioner.compute_fingerprint(valid_paths)
            self.versioner.save_snapshot(fp_str, {
                "total_images": len(df),
                "valid_images": int(df.get("is_valid", pd.Series([True]*len(df))).sum()),
                "n_classes":    report.n_classes if report else 0,
            })

        return df, report

    def scan_and_load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load only — no split, no audit. For quick inspection."""
        df, _ = self.load(path, auto_split=False, auto_audit=False)
        return df

    def split(
        self,
        df: pd.DataFrame,
        strategy: Optional[SplitStrategy] = None,
        group_col: Optional[str] = None,
        time_col:  Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply splitting to an existing DataFrame."""
        strat = strategy or self.split_strategy
        return self.splitter.split(df, strat, group_col, time_col)

    def audit(self, df: pd.DataFrame) -> MLReadinessReport:
        """Run full ML audit on a DataFrame."""
        issues: List[DatasetIssue] = []

        # ML problem detection
        issues.extend(self.ml_detect.detect_all(df))

        # Split validation
        issues.extend(self.splitter.validate_split(df))

        # Leakage detection
        if "split" in df.columns:
            leaked_exact = self.dedup.detect_cross_split_leakage(df)
            if leaked_exact:
                issues.append(DatasetIssue(
                    issue_id="LEAK-001",
                    severity=SEVERITY_CRITICAL,
                    category="Data Leakage",
                    description=f"{len(leaked_exact)} exact duplicate(s) found between train and test splits",
                    affected=[p[0] for p in leaked_exact[:10]],
                    fix="Remove or re-split — these images appear in both train and test."
                ))

            leaked_near = self.dedup.detect_near_duplicate_leakage(df)
            if leaked_near:
                issues.append(DatasetIssue(
                    issue_id="LEAK-002",
                    severity=SEVERITY_HIGH,
                    category="Data Leakage",
                    description=f"{len(leaked_near)} near-duplicate pair(s) span train/test splits",
                    fix="Check for augmented versions of test images in training set."
                ))

        # Compute fingerprint
        report = self.scorer.score(df, issues)
        if self.path:
            all_paths = [Path(p) for p in df["file_path"].tolist()]
            report.dataset_fingerprint = self.versioner.compute_fingerprint(all_paths)

        return report

    # ── Export API ────────────────────────────────────────────────────────────

    def export(
        self,
        df: pd.DataFrame,
        report: Optional[MLReadinessReport],
        output_dir: Union[str, Path] = "nydra_output",
        formats: List[str] = ("csv", "json", "markdown"),
    ) -> Dict[str, Path]:
        """Export DataFrame and report to multiple formats."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}

        if "csv" in formats:
            p = out / "image_dataset.csv"
            self.exporter.to_csv(df, p)
            saved["csv"] = p

        if report:
            if "json" in formats:
                p = out / "ml_readiness_report.json"
                self.exporter.to_json(report, p)
                saved["json"] = p

            if "markdown" in formats:
                p = out / "ml_readiness_report.md"
                self.exporter.to_markdown(report, p)
                saved["markdown"] = p

        logger.info(f"Exported to: {out}")
        return saved

    # ── Private core loader ───────────────────────────────────────────────────

    def _load_images(self, paths: List[Path]) -> pd.DataFrame:
        """
        Core loading loop — validates, opens, extracts metadata, computes hashes.
        Uses MemoryEngine for batch/parallel processing.
        """
        records: List[Dict] = []

        def process_one(fp: Path) -> Dict:
            rec = ImageRecord()
            rec.file_path = str(fp)
            rec.file_name = fp.name
            rec.file_stem = fp.stem
            rec.extension = fp.suffix.lower()
            rec.is_hidden = fp.name.startswith(".")

            # Deep file validation
            status, error_msg, severity = self.validator.validate(fp)
            rec.load_status    = status.value
            rec.error_message  = error_msg
            rec.error_severity = severity
            rec.is_valid       = status == LoadStatus.OK
            rec.is_truncated   = status == LoadStatus.TRUNCATED
            rec.is_corrupted   = status == LoadStatus.CORRUPTED

            # Safe image open
            img, open_status, open_err = self.recovery.safe_open(fp)

            if img is not None:
                # Format info
                fmt_info = self.fmt.get_format_info(img, fp)
                for k, v in fmt_info.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

                # Full metadata extraction
                meta = self.meta.extract(fp, img)
                for k, v in meta.items():
                    if hasattr(rec, k):
                        setattr(rec, k, v)

                # Compression ratio
                if rec.file_size_bytes > 0 and rec.total_pixels > 0:
                    rec.compression_ratio = round(
                        rec.file_size_bytes / rec.total_pixels, 6
                    )

                # Hashes
                if self.compute_hashes:
                    h = self.recovery.safe_hash(fp, img)
                    rec.sha256 = h.get("sha256", "")
                    rec.phash  = h.get("phash", "")
                    rec.dhash  = h.get("dhash", "")
                    rec.ahash  = h.get("ahash", "")
                    rec.whash  = h.get("whash", "")

                img.close()

            return rec.__dict__

        # Process in parallel batches
        if self.verbose:
            logger.info(f"⚙️  Processing {len(paths)} images...")

        results = self.memory.load_parallel(paths, process_one, desc="Loading")
        records = [r for r in results if r is not None]

        df = pd.DataFrame(records)
        return df

    def _attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach labels from detected label_map to the DataFrame."""
        df = df.copy()
        if self._label_map:
            df["label"] = df["file_path"].map(self._label_map).fillna("")
            df["label_source"] = self._dataset_structure.value
        else:
            df["label"] = ""
            df["label_source"] = "none"
        return df

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_scan_summary(self) -> Dict:
        return self._scan_summary

    def get_error_log(self) -> List[Dict]:
        return self.recovery.get_error_log()

    def get_dataset_structure(self) -> DatasetStructure:
        return self._dataset_structure

    def get_version_history(self) -> List[Dict]:
        return self.versioner.get_history()

    def get_class_distribution(self, df: pd.DataFrame) -> pd.Series:
        if "label" not in df.columns:
            return pd.Series(dtype=int)
        return df[df["label"] != ""]["label"].value_counts()

    def get_split_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if "split" not in df.columns:
            return pd.DataFrame()
        stats = df.groupby("split").agg(
            count=("file_path", "count"),
            valid=("is_valid", "sum"),
            duplicates=("is_duplicate", "sum"),
        ).reset_index()
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_images(
    path: Union[str, Path],
    recursive:    bool = True,
    compute_hashes: bool = True,
    n_workers:    int  = 4,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Quick load — returns DataFrame only, no split, no audit.

    Example:
        df = load_images("dataset/")
        print(df[["file_name", "width", "height", "label"]].head())
    """
    loader = ImageLoader(
        path=path, recursive=recursive,
        compute_hashes=compute_hashes,
        n_workers=n_workers, verbose=verbose
    )
    return loader.scan_and_load(path)


def load_and_audit(
    path: Union[str, Path],
    split_strategy: SplitStrategy = SplitStrategy.STRATIFIED,
    n_workers:      int  = 4,
    verbose:        bool = True,
) -> Tuple[pd.DataFrame, MLReadinessReport]:
    """
    Full pipeline: load + split + audit.

    Example:
        df, report = load_and_audit("dataset/")
        print(f"Score: {report.overall_score}/100 — {report.verdict}")
        print(report.fix_checklist)
    """
    loader = ImageLoader(
        path=path, split_strategy=split_strategy,
        n_workers=n_workers, verbose=verbose
    )
    return loader.load(path, auto_split=True, auto_audit=True)


def quick_check(path: Union[str, Path]) -> None:
    """
    Print a quick summary to stdout — no DataFrame returned.

    Example:
        quick_check("dataset/")
    """
    df, report = load_and_audit(path, verbose=False)

    print(f"\n{'='*50}")
    print(f"🩺 Nydra — Image Dataset Quick Check")
    print(f"{'='*50}")
    print(f"📁 Path:           {path}")
    print(f"📸 Total images:   {report.total_images}")
    print(f"✅ Valid:          {report.valid_images}")
    print(f"❌ Corrupted:      {report.corrupted_images}")
    print(f"🔁 Duplicates:     {report.duplicate_images}")
    print(f"🏷️  Classes:        {report.n_classes}")
    print(f"⚖️  Imbalance ratio: {report.imbalance_ratio:.1f}:1")
    print(f"\n🎯 ML Readiness Score: {report.overall_score}/100")
    print(f"📋 Verdict: {report.verdict}")

    if report.fix_checklist:
        print(f"\n🔧 Fix Checklist ({len(report.fix_checklist)} items):")
        for item in report.fix_checklist[:5]:
            print(f"   • {item}")
        if len(report.fix_checklist) > 5:
            print(f"   ... and {len(report.fix_checklist) - 5} more.")

    print(f"{'='*50}\n")
