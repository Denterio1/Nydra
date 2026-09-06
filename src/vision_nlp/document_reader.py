"""
document_reader.py — Nydra v0.6.0
========================================
Advanced Document Reading & Extraction Pipeline.

  Block 1  — FormatRouter         : magic-bytes + extension detection, dep checking
  Block 2  — EncodingHandler      : chardet 7, BOM, RTL, mojibake repair, fallback chain
  Block 3  — PDFReader            : pdfplumber + PyMuPDF dual-library, OCR fallback, multi-column
  Block 4  — DOCXReader           : python-docx + docx2python, tracked changes, footnotes, comments
  Block 5  — PlainTextReader      : TXT/MD/HTML with full stripping, OpenGraph, JSON-LD, RTL
  Block 6  — MetadataExtractor    : unified schema across all formats
  Block 7  — DataFrameBuilder     : unified output DataFrame, NFC normalization
  Block 8  — BatchFolderReader    : parallel multi-file reading with error isolation
  Block 9  — TextCleaner          : broken hyphenation, repeated headers, watermarks, control chars
  Block 10 — DocumentReader Master + convenience functions

Author  : Nydra Team
GitHub  : https://github.com/Denterio1/Nydra
Version : 0.6.0
Date    : April 2026
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import io
import json
import logging
import os
import re
import sys
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ── tqdm ──────────────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings("ignore")

logging.basicConfig(
    level  = logging.INFO,
    format = "[Nydra:reader] %(levelname)s — %(message)s",
)
logger = logging.getLogger("nydra.document_reader")

# ── supported formats ─────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}

# ── magic bytes signatures ────────────────────────────────────────────────────
MAGIC_BYTES: Dict[bytes, str] = {
    b"%PDF":           "pdf",
    b"PK\x03\x04":    "docx",   # ZIP container (DOCX, XLSX, PPTX)
    b"\xd0\xcf\x11\xe0": "doc", # OLE2 (legacy .doc)
    b"<!DOCTYPE":      "html",
    b"<html":          "html",
    b"<HTML":          "html",
}

# ── BOM signatures ────────────────────────────────────────────────────────────
BOM_MAP: Dict[bytes, str] = {
    b"\xef\xbb\xbf":     "utf-8-sig",
    b"\xff\xfe\x00\x00": "utf-32-le",
    b"\x00\x00\xfe\xff": "utf-32-be",
    b"\xff\xfe":         "utf-16-le",
    b"\xfe\xff":         "utf-16-be",
}

# ── RTL Unicode ranges ─────────────────────────────────────────────────────────
RTL_RANGES = [
    (0x0590, 0x05FF),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0700, 0x074F),   # Syriac
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB1D, 0xFDFF),   # Hebrew / Arabic Presentation Forms
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
]

# ── reading speed (words per minute) ─────────────────────────────────────────
READING_WPM = 200


# ==============================================================================
# BLOCK 1 — FORMAT ROUTER + MAGIC BYTES DETECTOR
# ==============================================================================

class FormatRouter:
    """
    Detect document format from magic bytes AND file extension.

    Detection order
    ---------------
    1. Read first 8 bytes → check MAGIC_BYTES signatures
    2. Fall back to file extension if magic bytes are inconclusive
    3. Cross-validate: warn if magic bytes contradict extension

    Dependency checking
    -------------------
    Each format has required + optional deps.
    Missing required dep → raises ImportError with exact pip install command.
    Missing optional dep → logs a warning, falls back to alternative.
    """

    REQUIRED_DEPS: Dict[str, List[str]] = {
        "pdf":  ["pdfplumber"],
        "docx": ["docx"],
        "html": ["bs4"],
        "txt":  [],
        "md":   [],
    }

    OPTIONAL_DEPS: Dict[str, List[Tuple[str, str]]] = {
        "pdf":  [("fitz", "pymupdf"), ("pytesseract", "pytesseract")],
        "docx": [("docx2python", "docx2python")],
        "html": [("lxml", "lxml")],
    }

    PIP_NAMES: Dict[str, str] = {
        "pdfplumber": "pdfplumber",
        "docx":       "python-docx",
        "bs4":        "beautifulsoup4",
        "fitz":       "pymupdf",
        "pytesseract":"pytesseract",
        "docx2python":"docx2python",
        "lxml":       "lxml",
        "chardet":    "chardet",
    }

    def detect(self, path: str) -> str:
        """
        Returns format string: 'pdf'|'docx'|'doc'|'txt'|'md'|'html'
        Raises ValueError if format cannot be determined.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 1. Magic bytes
        fmt_magic = self._from_magic(path)

        # 2. Extension
        ext = p.suffix.lower()
        fmt_ext = {
            ".pdf":  "pdf",
            ".docx": "docx",
            ".doc":  "doc",
            ".txt":  "txt",
            ".md":   "md",
            ".html": "html",
            ".htm":  "html",
        }.get(ext)

        # 3. Reconcile
        if fmt_magic and fmt_ext and fmt_magic != fmt_ext:
            # Magic bytes win — but warn
            logger.warning(
                f"Magic bytes say '{fmt_magic}' but extension says '{fmt_ext}' "
                f"for {p.name} — trusting magic bytes"
            )
            return fmt_magic

        result = fmt_magic or fmt_ext
        if result is None:
            raise ValueError(
                f"Cannot determine format for '{p.name}'. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        # 4. Special case: ZIP container could be DOCX, XLSX, PPTX
        if result == "docx" and ext not in (".docx", ".doc"):
            logger.warning(f"ZIP container detected but extension is '{ext}' — assuming DOCX")

        return result

    def check_deps(self, fmt: str) -> Dict[str, bool]:
        """
        Check all required and optional deps for a format.
        Raises ImportError for missing required deps.
        Returns dict of {module: available} for optional deps.
        """
        # Required
        for mod in self.REQUIRED_DEPS.get(fmt, []):
            if not self._importable(mod):
                pip_name = self.PIP_NAMES.get(mod, mod)
                raise ImportError(
                    f"Missing required dependency '{mod}' for format '{fmt}'.\n"
                    f"Install it with:  pip install {pip_name}"
                )

        # Optional
        availability: Dict[str, bool] = {}
        for mod, pip_pkg in self.OPTIONAL_DEPS.get(fmt, []):
            available = self._importable(mod)
            availability[mod] = available
            if not available:
                logger.debug(
                    f"Optional dep '{mod}' not found (pip install {pip_pkg}). "
                    f"Some features will be limited."
                )
        return availability

    def _from_magic(self, path: str) -> Optional[str]:
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            for magic, fmt in MAGIC_BYTES.items():
                if header.startswith(magic):
                    return fmt
        except Exception:
            pass
        return None

    @staticmethod
    def _importable(module: str) -> bool:
        import importlib.util
        return importlib.util.find_spec(module) is not None


# ==============================================================================
# BLOCK 2 — ENCODING HANDLER
# ==============================================================================

class EncodingHandler:
    """
    Advanced encoding detection and decoding.

    Detection pipeline
    ------------------
    1. BOM detection (deterministic — highest priority)
    2. chardet 7 streaming detection (UniversalDetector, 32KB chunks)
    3. Confidence-based decision:
       >= 0.85 → use detected encoding
       0.50–0.85 → try detected, validate, fallback if mojibake
       < 0.50  → full fallback chain
    4. Fallback chain: UTF-8 → UTF-16 → CP1256 → ISO-8859-6 → Latin-1 → system
    5. Mojibake repair for double-encoded UTF-8

    RTL detection
    -------------
    Scans decoded text for Arabic/Hebrew Unicode codepoints.
    Sets is_rtl=True if RTL chars > 10% of non-ASCII chars.

    Returns EncodingResult dataclass.
    """

    FALLBACK_CHAIN = ["utf-8", "utf-16", "cp1256", "iso-8859-6", "latin-1"]
    CONFIDENCE_HIGH  = 0.85
    CONFIDENCE_LOW   = 0.50
    CHUNK_SIZE       = 32 * 1024   # 32 KB for streaming detection

    def detect_and_decode(self, raw: bytes) -> "EncodingResult":
        """Main entry point: raw bytes → EncodingResult."""
        # Step 1: BOM check
        bom_enc = self._detect_bom(raw)
        if bom_enc:
            text = self._try_decode(raw, bom_enc, strip_bom=True)
            if text is not None:
                return EncodingResult(
                    text           = text,
                    encoding       = bom_enc,
                    confidence     = 1.0,
                    bom_detected   = True,
                    is_rtl         = self._is_rtl(text),
                    language_hint  = self._lang_hint(text),
                    mojibake_fixed = False,
                )

        # Step 2: chardet streaming detection
        detected_enc, confidence = self._chardet_detect(raw)

        # Step 3: confidence-based decode
        if detected_enc and confidence >= self.CONFIDENCE_HIGH:
            text = self._try_decode(raw, detected_enc)
            if text is not None:
                text, fixed = self._repair_mojibake(text)
                return EncodingResult(
                    text           = text,
                    encoding       = detected_enc,
                    confidence     = confidence,
                    bom_detected   = False,
                    is_rtl         = self._is_rtl(text),
                    language_hint  = self._lang_hint(text),
                    mojibake_fixed = fixed,
                )

        # Step 4: medium confidence — try detected first, then fallback
        if detected_enc and confidence >= self.CONFIDENCE_LOW:
            text = self._try_decode(raw, detected_enc)
            if text is not None and not self._looks_garbled(text):
                text, fixed = self._repair_mojibake(text)
                return EncodingResult(
                    text           = text,
                    encoding       = detected_enc,
                    confidence     = confidence,
                    bom_detected   = False,
                    is_rtl         = self._is_rtl(text),
                    language_hint  = self._lang_hint(text),
                    mojibake_fixed = fixed,
                )

        # Step 5: full fallback chain
        for enc in self.FALLBACK_CHAIN:
            text = self._try_decode(raw, enc)
            if text is not None and not self._looks_garbled(text):
                text, fixed = self._repair_mojibake(text)
                return EncodingResult(
                    text           = text,
                    encoding       = enc,
                    confidence     = 0.3,
                    bom_detected   = False,
                    is_rtl         = self._is_rtl(text),
                    language_hint  = self._lang_hint(text),
                    mojibake_fixed = fixed,
                )

        # Last resort: decode with replacement characters
        text = raw.decode("utf-8", errors="replace")
        return EncodingResult(
            text           = text,
            encoding       = "utf-8 (lossy)",
            confidence     = 0.0,
            bom_detected   = False,
            is_rtl         = self._is_rtl(text),
            language_hint  = "unknown",
            mojibake_fixed = False,
        )

    def decode_file(self, path: str) -> "EncodingResult":
        """Read a file in binary mode and decode it."""
        with open(path, "rb") as f:
            raw = f.read()
        return self.detect_and_decode(raw)

    # ── internal ──────────────────────────────────────────────────────────────

    def _detect_bom(self, raw: bytes) -> Optional[str]:
        for bom, enc in BOM_MAP.items():
            if raw.startswith(bom):
                return enc
        return None

    def _chardet_detect(self, raw: bytes) -> Tuple[Optional[str], float]:
        try:
            import chardet
            # Use UniversalDetector for large data, direct detect for small
            if len(raw) > self.CHUNK_SIZE * 4:
                detector = chardet.UniversalDetector()
                for i in range(0, min(len(raw), self.CHUNK_SIZE * 8), self.CHUNK_SIZE):
                    detector.feed(raw[i: i + self.CHUNK_SIZE])
                    if detector.done:
                        break
                result = detector.close()
            else:
                result = chardet.detect(raw)

            enc  = result.get("encoding")
            conf = float(result.get("confidence", 0.0))
            return enc, conf
        except ImportError:
            logger.debug("chardet not installed; using heuristic fallback")
            return None, 0.0
        except Exception as exc:
            logger.debug(f"chardet error: {exc}")
            return None, 0.0

    def _try_decode(self, raw: bytes, encoding: str,
                    strip_bom: bool = False) -> Optional[str]:
        try:
            text = raw.decode(encoding, errors="strict")
            if strip_bom and text.startswith("\ufeff"):
                text = text[1:]
            return text
        except (UnicodeDecodeError, LookupError):
            return None

    def _looks_garbled(self, text: str, threshold: float = 0.15) -> bool:
        """
        Heuristic: if more than `threshold` fraction of characters are
        replacement chars or unrecognized control chars → probably garbled.
        """
        if not text:
            return True
        bad = sum(
            1 for c in text
            if c == "\ufffd" or (ord(c) < 32 and c not in "\n\r\t")
        )
        return bad / len(text) > threshold

    def _repair_mojibake(self, text: str) -> Tuple[str, bool]:
        """
        Attempt to fix common double-encoded UTF-8 (mojibake).
        Returns (fixed_text, was_fixed).
        """
        # Common Latin-1 / Windows-1252 double-encoded UTF-8 patterns
        MOJIBAKE_FIXES = [
            ("\u00c3\u00a9", "é"),   # é
            ("\u00c3\u00a0", "à"),   # à
            ("\u00c3\u00bc", "ü"),   # ü
            ("\u00c3\u00b6", "ö"),   # ö
            ("\u00c3\u00a4", "ä"),   # ä
            ("\u00c3\u00af", "ï"),   # ï
            ("\u00c3\u00aa", "ê"),   # ê
            ("\u00c3\u00a8", "è"),   # è
            ("\u00c2\u00ab", "«"),
            ("\u00c2\u00bb", "»"),
        ]
        fixed = False
        for wrong, correct in MOJIBAKE_FIXES:
            if wrong in text:
                text  = text.replace(wrong, correct)
                fixed = True

        # Try UTF-8 bytes re-interpreted from Latin-1
        if not fixed:
            try:
                repaired = text.encode("latin-1").decode("utf-8")
                # Only accept if it has fewer replacement chars
                if repaired.count("\ufffd") < text.count("\ufffd"):
                    return repaired, True
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass

        return text, fixed

    def _is_rtl(self, text: str, threshold: float = 0.10) -> bool:
        """True if RTL script chars exceed `threshold` of non-ASCII chars."""
        if not text:
            return False
        non_ascii = [c for c in text if ord(c) > 127]
        if not non_ascii:
            return False
        rtl_chars = sum(
            1 for c in non_ascii
            if any(lo <= ord(c) <= hi for lo, hi in RTL_RANGES)
        )
        return rtl_chars / len(non_ascii) > threshold

    def _lang_hint(self, text: str) -> str:
        """Simple language hint from Unicode script detection."""
        if not text:
            return "unknown"
        sample = text[:500]
        arabic  = sum(1 for c in sample if 0x0600 <= ord(c) <= 0x06FF)
        hebrew  = sum(1 for c in sample if 0x0590 <= ord(c) <= 0x05FF)
        latin   = sum(1 for c in sample if ord(c) < 128 and c.isalpha())
        cjk     = sum(1 for c in sample if 0x4E00 <= ord(c) <= 0x9FFF)
        cyrillic= sum(1 for c in sample if 0x0400 <= ord(c) <= 0x04FF)

        scores = {"arabic": arabic, "hebrew": hebrew, "latin": latin,
                  "cjk": cjk, "cyrillic": cyrillic}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "unknown"


@dataclass
class EncodingResult:
    text:           str
    encoding:       str
    confidence:     float
    bom_detected:   bool
    is_rtl:         bool
    language_hint:  str
    mojibake_fixed: bool


# ==============================================================================
# BLOCK 3 — PDF READER
# ==============================================================================

class PDFReader:
    """
    Advanced PDF text + table + metadata extraction.

    Dual-library strategy
    ---------------------
    pdfplumber : per-page text (layout-preserving), table extraction,
                 word bounding boxes, coordinate-based analysis
    PyMuPDF    : metadata dict, hyperlinks, TOC/bookmarks, images,
                 page rotation, encryption status

    Features
    --------
    - Multi-column layout detection via x-coordinate clustering
    - Footnote extraction via superscript font-size + page-bottom position
    - Scanned PDF detection + pytesseract OCR fallback (300 DPI)
    - Repeated header/footer detection across pages
    - Hyperlink extraction per page
    - Table reconstruction with column alignment
    - Per-page output with full metadata
    """

    SCANNED_THRESHOLD   = 50    # chars per page below this → likely scanned
    OCR_DPI             = 300
    COLUMN_GAP_RATIO    = 0.15  # gap > 15% of page width → multi-column
    FOOTNOTE_FONT_RATIO = 0.75  # font size < 75% of body → superscript/footnote
    FOOTER_ZONE_RATIO   = 0.88  # y > 88% of page height → footer zone

    def read(self, path: str) -> "DocumentData":
        """Read a PDF file. Returns DocumentData."""
        self._check_deps()
        import pdfplumber

        pages_data  : List[Dict] = []
        all_text    : List[str]  = []
        all_tables  : List[Any]  = []
        metadata    : Dict       = {}
        toc         : List       = []
        links_all   : List       = []

        # ── PyMuPDF pass: metadata, TOC, links, images ────────────────────────
        fitz_available = False
        try:
            import fitz  # PyMuPDF
            fitz_available = True
            doc_fitz = fitz.open(path)

            raw_meta = doc_fitz.metadata or {}
            metadata = self._normalize_pdf_meta(raw_meta, doc_fitz)

            # TOC / bookmarks
            toc = doc_fitz.get_toc()   # [(level, title, page_num), ...]

            # Per-page links
            for page_num, page in enumerate(doc_fitz):
                page_links = []
                for link in page.get_links():
                    uri = link.get("uri", "") or link.get("page", "")
                    if uri:
                        page_links.append({"page": page_num + 1, "uri": str(uri)})
                links_all.extend(page_links)

            doc_fitz.close()
        except ImportError:
            logger.debug("PyMuPDF not installed — metadata/links limited")
        except Exception as exc:
            logger.warning(f"PyMuPDF error on {path}: {exc}")

        # ── pdfplumber pass: text + tables ────────────────────────────────────
        with pdfplumber.open(path) as pdf:
            if not fitz_available:
                metadata["page_count"] = len(pdf.pages)

            # Detect repeated headers/footers across pages
            header_footer_patterns = self._detect_repeated_lines(pdf)

            for page_num, page in enumerate(pdf.pages):
                page_result = self._process_page(
                    page, page_num + 1,
                    header_footer_patterns,
                )
                pages_data.append(page_result)
                all_text.append(page_result["text"])
                all_tables.extend(page_result["tables"])

        full_text = "\n\n".join(t for t in all_text if t)
        metadata.setdefault("page_count", len(pages_data))
        metadata["has_tables"]      = len(all_tables) > 0
        metadata["has_links"]       = len(links_all) > 0
        metadata["table_count"]     = len(all_tables)
        metadata["toc"]             = toc
        metadata["links"]           = links_all[:50]   # cap at 50

        return DocumentData(
            file_path  = path,
            fmt        = "pdf",
            full_text  = full_text,
            pages      = pages_data,
            metadata   = metadata,
            tables     = all_tables,
        )

    # ── page processing ───────────────────────────────────────────────────────

    def _process_page(self, page, page_num: int,
                      skip_patterns: set) -> Dict:
        """Extract text + tables from one pdfplumber page."""
        result: Dict[str, Any] = {
            "page_num":    page_num,
            "text":        "",
            "tables":      [],
            "links":       [],
            "word_count":  0,
            "is_scanned":  False,
            "footnotes":   [],
            "layout":      "single",
        }

        # Raw text
        raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        # Remove repeated header/footer lines
        if skip_patterns:
            lines = raw_text.split("\n")
            lines = [l for l in lines if l.strip() not in skip_patterns]
            raw_text = "\n".join(lines)

        # Scanned page detection
        if len(raw_text.strip()) < self.SCANNED_THRESHOLD:
            result["is_scanned"] = True
            ocr_text = self._ocr_page(page)
            if ocr_text:
                raw_text = ocr_text

        # Multi-column detection + reordering
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        if words:
            n_cols, layout = self._detect_columns(words, page.width)
            result["layout"] = layout
            if n_cols > 1:
                raw_text = self._reorder_multicolumn(words, page.width, n_cols)

        # Footnote extraction
        result["footnotes"] = self._extract_footnotes(page)

        # Table extraction
        tables = page.extract_tables()
        if tables:
            for tbl in tables:
                if tbl:
                    result["tables"].append(self._table_to_dict(tbl))

        result["text"]       = raw_text.strip()
        result["word_count"] = len(raw_text.split())
        return result

    def _detect_columns(self, words: List[Dict],
                         page_width: float) -> Tuple[int, str]:
        """
        Cluster word x-coordinates to detect number of columns.
        Uses a gap analysis: finds x-gaps larger than COLUMN_GAP_RATIO * page_width.
        """
        if not words or page_width <= 0:
            return 1, "single"

        x_centers = sorted(set(
            int((w["x0"] + w["x1"]) / 2) for w in words
        ))
        if len(x_centers) < 4:
            return 1, "single"

        gaps = []
        for i in range(1, len(x_centers)):
            gap = x_centers[i] - x_centers[i - 1]
            if gap > page_width * self.COLUMN_GAP_RATIO:
                gaps.append((gap, x_centers[i - 1], x_centers[i]))

        n_cols = len(gaps) + 1
        if n_cols == 1:
            return 1, "single"
        elif n_cols == 2:
            return 2, "two-column"
        else:
            return n_cols, f"{n_cols}-column"

    def _reorder_multicolumn(self, words: List[Dict],
                              page_width: float, n_cols: int) -> str:
        """
        Re-order words from multi-column layout to reading order
        (left column top-to-bottom, then right column top-to-bottom).
        """
        col_width = page_width / n_cols
        columns: Dict[int, List] = {i: [] for i in range(n_cols)}

        for w in words:
            cx   = (w["x0"] + w["x1"]) / 2
            col  = min(int(cx / col_width), n_cols - 1)
            columns[col].append(w)

        text_parts = []
        for col_idx in range(n_cols):
            col_words = sorted(columns[col_idx], key=lambda w: (w["top"], w["x0"]))
            text_parts.append(" ".join(w["text"] for w in col_words))

        return "\n\n".join(text_parts)

    def _extract_footnotes(self, page) -> List[str]:
        """
        Detect footnotes: text in footer zone with smaller font size.
        """
        footnotes = []
        try:
            chars = page.chars
            if not chars:
                return []

            # Estimate body font size (mode)
            sizes = [c.get("size", 0) for c in chars if c.get("size", 0) > 0]
            if not sizes:
                return []
            from statistics import mode as stat_mode
            try:
                body_size = stat_mode(sizes)
            except Exception:
                body_size = sorted(sizes)[len(sizes) // 2]

            h = page.height
            footer_y = h * self.FOOTER_ZONE_RATIO

            # Collect small-font text in footer zone
            fn_chars = [
                c for c in chars
                if c.get("top", 0) > footer_y
                and c.get("size", body_size) < body_size * self.FOOTNOTE_FONT_RATIO
            ]
            if fn_chars:
                fn_text = "".join(c.get("text", "") for c in
                                  sorted(fn_chars, key=lambda c: (c.get("top", 0), c.get("x0", 0))))
                fn_text = fn_text.strip()
                if fn_text:
                    footnotes.append(fn_text)
        except Exception:
            pass
        return footnotes

    def _detect_repeated_lines(self, pdf) -> set:
        """
        Find lines that repeat across >= 50% of pages → headers/footers.
        Returns set of line strings to skip.
        """
        from collections import Counter
        n_pages = len(pdf.pages)
        if n_pages < 2:
            return set()

        line_counts: Counter = Counter()
        # Sample first 10 pages for speed
        sample = pdf.pages[:min(10, n_pages)]
        for page in sample:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if line and len(line) < 120:
                    line_counts[line] += 1

        threshold = max(2, int(len(sample) * 0.5))
        return {line for line, cnt in line_counts.items() if cnt >= threshold}

    def _ocr_page(self, page) -> str:
        """Render page to image and run pytesseract OCR."""
        try:
            import pytesseract
            from PIL import Image as PILImage
            img = page.to_image(resolution=self.OCR_DPI).original
            return pytesseract.image_to_string(img)
        except ImportError:
            logger.debug("pytesseract not installed; OCR unavailable for scanned pages")
            return ""
        except Exception as exc:
            logger.debug(f"OCR failed: {exc}")
            return ""

    def _table_to_dict(self, table: List[List]) -> Dict:
        """Convert pdfplumber table (list of lists) to serializable dict."""
        if not table:
            return {}
        headers = [str(h or "") for h in (table[0] or [])]
        rows    = []
        for row in table[1:]:
            if row:
                rows.append([str(c or "") for c in row])
        return {"headers": headers, "rows": rows, "n_rows": len(rows), "n_cols": len(headers)}

    def _normalize_pdf_meta(self, raw_meta: Dict, doc_fitz) -> Dict:
        """Normalize PyMuPDF metadata into unified schema."""
        def parse_date(s: str) -> Optional[str]:
            if not s:
                return None
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
            m = re.match(r"D:(\d{4})(\d{2})(\d{2})", s or "")
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            return s[:10] if len(s) >= 10 else s

        return {
            "author":       raw_meta.get("author", ""),
            "title":        raw_meta.get("title", ""),
            "subject":      raw_meta.get("subject", ""),
            "creator":      raw_meta.get("creator", ""),
            "producer":     raw_meta.get("producer", ""),
            "keywords":     raw_meta.get("keywords", ""),
            "created":      parse_date(raw_meta.get("creationDate", "")),
            "modified":     parse_date(raw_meta.get("modDate", "")),
            "page_count":   doc_fitz.page_count,
            "is_encrypted": doc_fitz.is_encrypted,
            "format":       raw_meta.get("format", "PDF"),
            "trapped":      raw_meta.get("trapped", ""),
        }

    def _check_deps(self) -> None:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF reading.\n"
                "Install: pip install pdfplumber"
            )


# ==============================================================================
# BLOCK 4 — DOCX READER
# ==============================================================================

class DOCXReader:
    """
    Advanced DOCX extraction.

    Dual-library strategy
    ---------------------
    python-docx  : paragraphs, styles, tables, core properties, runs/formatting
    docx2python  : headers, footers, footnotes, endnotes, comments, images,
                   nested list positions

    Additional features
    -------------------
    - Tracked changes detection via raw OOXML scan (w:ins / w:del tags)
    - Full document structure map (heading hierarchy tree)
    - Nested tables (tables inside table cells)
    - Comment extraction with author + date + text
    - Footnote and endnote extraction
    - Header/footer per section
    - Embedded image metadata (size, name, relationship ID)
    - Run-level formatting: bold, italic, underline, font name, font size, color
    """

    def read(self, path: str) -> "DocumentData":
        """Read a DOCX file. Returns DocumentData."""
        self._check_deps()
        from docx import Document

        doc = Document(path)
        sections_data: List[Dict] = []
        metadata     : Dict       = {}
        all_tables   : List       = []

        # ── Core properties ───────────────────────────────────────────────────
        metadata = self._extract_core_props(doc)

        # ── Paragraphs + structure ────────────────────────────────────────────
        structure_tree: List[Dict] = []
        for idx, para in enumerate(doc.paragraphs):
            text  = para.text.strip()
            style = para.style.name if para.style else "Normal"
            level = self._heading_level(style)

            section_dict = {
                "section_index": idx,
                "section_type":  "heading" if level else "paragraph",
                "heading_level": level,
                "text":          text,
                "style_name":    style,
                "word_count":    len(text.split()) if text else 0,
                "char_count":    len(text),
                "formatting":    self._extract_runs(para),
                "is_list_item":  self._is_list(para),
                "list_level":    self._list_level(para),
            }
            sections_data.append(section_dict)

            if level:
                structure_tree.append({"level": level, "title": text, "index": idx})

        # ── Tables ────────────────────────────────────────────────────────────
        for tbl_idx, table in enumerate(doc.tables):
            tbl_dict = self._extract_table(table, tbl_idx)
            all_tables.append(tbl_dict)
            sections_data.append({
                "section_index": len(sections_data),
                "section_type":  "table",
                "heading_level": None,
                "text":          tbl_dict["flat_text"],
                "style_name":    "Table",
                "word_count":    len(tbl_dict["flat_text"].split()),
                "char_count":    len(tbl_dict["flat_text"]),
                "formatting":    [],
                "table_data":    tbl_dict,
            })

        # ── docx2python extras ────────────────────────────────────────────────
        d2p_data = self._docx2python_extract(path)
        metadata.update(d2p_data.get("metadata", {}))

        # Add footnotes as sections
        for fn_idx, fn_text in enumerate(d2p_data.get("footnotes", [])):
            sections_data.append({
                "section_index": len(sections_data),
                "section_type":  "footnote",
                "heading_level": None,
                "text":          fn_text,
                "style_name":    "Footnote",
                "word_count":    len(fn_text.split()),
                "char_count":    len(fn_text),
                "formatting":    [],
            })

        # Add comments as sections
        for cmt in d2p_data.get("comments", []):
            sections_data.append({
                "section_index": len(sections_data),
                "section_type":  "comment",
                "heading_level": None,
                "text":          cmt.get("text", ""),
                "style_name":    "Comment",
                "word_count":    len(cmt.get("text", "").split()),
                "char_count":    len(cmt.get("text", "")),
                "formatting":    [],
                "comment_author": cmt.get("author", ""),
                "comment_date":   cmt.get("date", ""),
            })

        # ── Tracked changes ───────────────────────────────────────────────────
        tracked = self._detect_tracked_changes(path)
        metadata["tracked_insertions"] = tracked["insertions"]
        metadata["tracked_deletions"]  = tracked["deletions"]
        metadata["tracked_authors"]    = tracked["authors"]
        metadata["has_tracked_changes"] = tracked["insertions"] + tracked["deletions"] > 0

        # ── Headers / Footers ─────────────────────────────────────────────────
        metadata["headers"]  = d2p_data.get("headers", [])
        metadata["footers"]  = d2p_data.get("footers", [])
        metadata["has_tables"] = len(all_tables) > 0
        metadata["structure_tree"] = structure_tree

        full_text = "\n\n".join(
            s["text"] for s in sections_data
            if s.get("section_type") in ("paragraph", "heading") and s.get("text")
        )

        return DocumentData(
            file_path = path,
            fmt       = "docx",
            full_text = full_text,
            pages     = sections_data,
            metadata  = metadata,
            tables    = all_tables,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_core_props(self, doc) -> Dict:
        cp = doc.core_properties
        def _dt(v) -> Optional[str]:
            if v is None:
                return None
            return str(v)[:19] if hasattr(v, "isoformat") else str(v)[:19]
        return {
            "author":           cp.author or "",
            "title":            cp.title  or "",
            "subject":          cp.subject or "",
            "keywords":         cp.keywords or "",
            "last_modified_by": cp.last_modified_by or "",
            "revision":         str(cp.revision or ""),
            "created":          _dt(cp.created),
            "modified":         _dt(cp.modified),
            "category":         cp.category or "",
            "description":      cp.description or "",
        }

    def _heading_level(self, style_name: str) -> Optional[int]:
        """Extract heading level from style name, e.g. 'Heading 2' → 2."""
        m = re.match(r"[Hh]eading\s+(\d+)", style_name)
        return int(m.group(1)) if m else None

    def _extract_runs(self, para) -> List[Dict]:
        """Extract run-level formatting from a paragraph."""
        runs = []
        for run in para.runs:
            if not run.text:
                continue
            run_info: Dict[str, Any] = {"text": run.text}
            if run.bold:       run_info["bold"]      = True
            if run.italic:     run_info["italic"]    = True
            if run.underline:  run_info["underline"] = True
            if run.font.name:  run_info["font"]      = run.font.name
            if run.font.size:  run_info["size_pt"]   = run.font.size.pt
            if run.font.color and run.font.color.type is not None:
                try:
                    run_info["color"] = str(run.font.color.rgb)
                except Exception:
                    pass
            runs.append(run_info)
        return runs

    def _is_list(self, para) -> bool:
        style = para.style.name if para.style else ""
        return "List" in style or "Bullet" in style or "Number" in style

    def _list_level(self, para) -> int:
        try:
            return para._p.pPr.numPr.ilvl.val if para._p.pPr and para._p.pPr.numPr else 0
        except Exception:
            return 0

    def _extract_table(self, table, tbl_idx: int) -> Dict:
        """Extract a Word table including nested tables."""
        rows_data = []
        for row in table.rows:
            row_data = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                # Detect nested tables
                nested = [self._extract_table(t, 0) for t in cell.tables]
                row_data.append({
                    "text":           cell_text,
                    "nested_tables":  nested,
                })
            rows_data.append(row_data)

        # Build flat text for search/analysis
        flat = " | ".join(
            cell["text"]
            for row in rows_data
            for cell in row
            if cell["text"]
        )
        return {
            "table_index": tbl_idx,
            "n_rows":      len(rows_data),
            "n_cols":      len(rows_data[0]) if rows_data else 0,
            "rows":        rows_data,
            "flat_text":   flat,
        }

    def _docx2python_extract(self, path: str) -> Dict:
        """Use docx2python for headers, footers, footnotes, endnotes, comments."""
        result: Dict[str, Any] = {
            "headers": [], "footers": [], "footnotes": [],
            "endnotes": [], "comments": [], "metadata": {},
        }
        try:
            from docx2python import docx2python as d2p
            with d2p(path, html=False) as content:
                # Headers
                for section in (content.header or []):
                    for row in section:
                        for cell in row:
                            for para in cell:
                                if isinstance(para, str) and para.strip():
                                    result["headers"].append(para.strip())

                # Footers
                for section in (content.footer or []):
                    for row in section:
                        for cell in row:
                            for para in cell:
                                if isinstance(para, str) and para.strip():
                                    result["footers"].append(para.strip())

                # Footnotes
                for section in (content.footnotes or []):
                    for row in section:
                        for cell in row:
                            for para in cell:
                                if isinstance(para, str) and para.strip():
                                    result["footnotes"].append(para.strip())

                # Properties
                props = content.properties or {}
                if props:
                    result["metadata"] = {k: str(v) for k, v in props.items()}

        except ImportError:
            logger.debug("docx2python not installed — headers/footers/footnotes skipped")
        except Exception as exc:
            logger.debug(f"docx2python error: {exc}")
        return result

    def _detect_tracked_changes(self, path: str) -> Dict:
        """
        Scan raw OOXML (word/document.xml inside the ZIP) for
        tracked change tags: <w:ins> (insertions) and <w:del> (deletions).
        Also extracts author names from w:author attributes.
        """
        result = {"insertions": 0, "deletions": 0, "authors": []}
        try:
            import zipfile
            with zipfile.ZipFile(path, "r") as z:
                if "word/document.xml" not in z.namelist():
                    return result
                xml_bytes = z.read("word/document.xml")
            xml_str = xml_bytes.decode("utf-8", errors="replace")

            ins_count = len(re.findall(r"<w:ins\b", xml_str))
            del_count = len(re.findall(r"<w:del\b", xml_str))
            authors   = list(set(re.findall(r'w:author="([^"]+)"', xml_str)))

            result["insertions"] = ins_count
            result["deletions"]  = del_count
            result["authors"]    = authors
        except Exception as exc:
            logger.debug(f"Tracked changes detection error: {exc}")
        return result

    def _check_deps(self) -> None:
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX reading.\n"
                "Install: pip install python-docx"
            )


# ==============================================================================
# BLOCK 5 — PLAIN TEXT / MARKDOWN / HTML READER
# ==============================================================================

class PlainTextReader:
    """
    Read and extract text from TXT, Markdown, and HTML files.

    TXT
    ---
    - Encoding via EncodingHandler (Block 2)
    - Paragraph splitting on double newlines
    - Line ending detection (CRLF / LF / CR)
    - Reading time estimation

    Markdown
    --------
    - Full regex stripping pipeline:
      fenced code blocks, inline code, headers, bold/italic,
      links (keep text), images (keep alt), blockquotes, HR
    - Extracts: code_snippets, extracted_links, heading_hierarchy

    HTML
    ----
    - BeautifulSoup + lxml parser (fallback: html.parser)
    - Removes: <script>, <style>, <noscript>, <iframe>, comments
    - Extracts: visible text, <meta> tags (description/keywords/author/OG),
      <title>, all <a href> links, JSON-LD structured data,
      lang attribute, dir="rtl" detection
    - UnicodeDammit for secondary encoding confirmation
    """

    def read_txt(self, path: str) -> "DocumentData":
        enc_handler = EncodingHandler()
        enc_result  = enc_handler.decode_file(path)
        text        = enc_result.text

        paragraphs  = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        line_ending = self._detect_line_ending(path)
        word_count  = len(text.split())

        meta = {
            "encoding":        enc_result.encoding,
            "confidence":      enc_result.confidence,
            "is_rtl":          enc_result.is_rtl,
            "language_hint":   enc_result.language_hint,
            "bom_detected":    enc_result.bom_detected,
            "mojibake_fixed":  enc_result.mojibake_fixed,
            "line_ending":     line_ending,
            "line_count":      text.count("\n") + 1,
            "blank_line_ratio": self._blank_line_ratio(text),
            "word_count":      word_count,
            "reading_time_min": round(word_count / READING_WPM, 1),
            "has_tables":      False,
            "has_images":      False,
        }

        pages = []
        for idx, para in enumerate(paragraphs):
            pages.append({
                "section_index": idx,
                "section_type":  "paragraph",
                "heading_level": None,
                "text":          para,
                "word_count":    len(para.split()),
                "char_count":    len(para),
            })

        return DocumentData(
            file_path = path,
            fmt       = "txt",
            full_text = text,
            pages     = pages,
            metadata  = meta,
            tables    = [],
        )

    def read_markdown(self, path: str) -> "DocumentData":
        enc_handler = EncodingHandler()
        enc_result  = enc_handler.decode_file(path)
        raw_md      = enc_result.text

        # Extract before stripping
        code_snippets  = self._md_extract_code(raw_md)
        extracted_links= self._md_extract_links(raw_md)
        heading_tree   = self._md_extract_headings(raw_md)

        # Strip to plain text
        clean_text = self._md_strip(raw_md)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", clean_text) if p.strip()]
        word_count = len(clean_text.split())

        meta = {
            "encoding":       enc_result.encoding,
            "is_rtl":         enc_result.is_rtl,
            "language_hint":  enc_result.language_hint,
            "code_snippets":  len(code_snippets),
            "links_count":    len(extracted_links),
            "heading_tree":   heading_tree,
            "word_count":     word_count,
            "reading_time_min": round(word_count / READING_WPM, 1),
            "has_tables":     "| " in raw_md,
            "has_images":     "![" in raw_md,
        }

        pages = []
        for idx, para in enumerate(paragraphs):
            pages.append({
                "section_index": idx,
                "section_type":  "paragraph",
                "heading_level": None,
                "text":          para,
                "word_count":    len(para.split()),
                "char_count":    len(para),
            })

        return DocumentData(
            file_path = path,
            fmt       = "md",
            full_text = clean_text,
            pages     = pages,
            metadata  = meta,
            tables    = [],
        )

    def read_html(self, path: str) -> "DocumentData":
        with open(path, "rb") as f:
            raw = f.read()

        # Encoding: try BeautifulSoup UnicodeDammit first
        try:
            from bs4 import UnicodeDammit
            dammit   = UnicodeDammit(raw)
            detected_enc = dammit.original_encoding or "utf-8"
        except ImportError:
            detected_enc = "utf-8"

        # Also run our handler for confidence + RTL
        enc_handler = EncodingHandler()
        enc_result  = enc_handler.detect_and_decode(raw)

        # Parse HTML
        try:
            from bs4 import BeautifulSoup, Comment
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML reading.\n"
                              "Install: pip install beautifulsoup4")

        try:
            soup = BeautifulSoup(raw, "lxml")
        except Exception:
            soup = BeautifulSoup(raw, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "noscript", "iframe",
                          "head", "meta", "link"]):
            tag.decompose()
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            comment.extract()

        # Extract visible text
        visible_text = soup.get_text(separator=" ", strip=True)
        visible_text = re.sub(r" {2,}", " ", visible_text)
        visible_text = re.sub(r"\n{3,}", "\n\n", visible_text)

        # Re-parse with original soup for metadata (before decompose)
        try:
            soup_meta = BeautifulSoup(raw, "lxml")
        except Exception:
            soup_meta = BeautifulSoup(raw, "html.parser")

        # Extract metadata
        meta_tags   = self._html_meta_tags(soup_meta)
        title       = soup_meta.title.string.strip() if soup_meta.title else ""
        links       = self._html_links(soup_meta)
        json_ld     = self._html_json_ld(soup_meta)
        lang        = (soup_meta.html or {}).get("lang", "") if soup_meta.html else ""
        has_rtl     = bool(soup_meta.find(attrs={"dir": "rtl"})) or enc_result.is_rtl

        word_count = len(visible_text.split())

        meta = {
            "title":           title,
            "encoding":        detected_enc,
            "is_rtl":          has_rtl,
            "language":        lang or enc_result.language_hint,
            "description":     meta_tags.get("description", ""),
            "keywords":        meta_tags.get("keywords", ""),
            "author":          meta_tags.get("author", ""),
            "og_title":        meta_tags.get("og:title", ""),
            "og_description":  meta_tags.get("og:description", ""),
            "og_type":         meta_tags.get("og:type", ""),
            "links_count":     len(links),
            "links":           links[:30],
            "json_ld":         json_ld,
            "word_count":      word_count,
            "reading_time_min": round(word_count / READING_WPM, 1),
            "has_tables":      bool(soup_meta.find("table")),
            "has_images":      bool(soup_meta.find("img")),
        }

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", visible_text) if p.strip()]
        pages = []
        for idx, para in enumerate(paragraphs):
            pages.append({
                "section_index": idx,
                "section_type":  "paragraph",
                "heading_level": None,
                "text":          para,
                "word_count":    len(para.split()),
                "char_count":    len(para),
            })

        return DocumentData(
            file_path = path,
            fmt       = "html",
            full_text = visible_text,
            pages     = pages,
            metadata  = meta,
            tables    = [],
        )

    # ── Markdown helpers ──────────────────────────────────────────────────────

    def _md_strip(self, text: str) -> str:
        """Full Markdown → plain text stripping pipeline."""
        # Fenced code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"~~~[\s\S]*?~~~", "", text)
        # Inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Headers → keep text
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Bold + italic
        text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
        # Links [text](url) → text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Images ![alt](url) → alt
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)
        # Blockquotes
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        # Horizontal rules
        text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        # HTML tags in MD
        text = re.sub(r"<[^>]+>", "", text)
        # Collapse spaces
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def _md_extract_code(self, text: str) -> List[str]:
        blocks = re.findall(r"```[\w]*\n([\s\S]*?)```", text)
        inline = re.findall(r"`([^`]+)`", text)
        return blocks + inline

    def _md_extract_links(self, text: str) -> List[str]:
        return re.findall(r"\]\(([^\)]+)\)", text)

    def _md_extract_headings(self, text: str) -> List[Dict]:
        headings = []
        for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            headings.append({"level": len(m.group(1)), "title": m.group(2).strip()})
        return headings

    # ── HTML helpers ──────────────────────────────────────────────────────────

    def _html_meta_tags(self, soup) -> Dict[str, str]:
        meta: Dict[str, str] = {}
        for tag in soup.find_all("meta"):
            name    = tag.get("name",     "").lower()
            prop    = tag.get("property", "").lower()
            content = tag.get("content",  "")
            key     = prop or name
            if key and content:
                meta[key] = content
        return meta

    def _html_links(self, soup) -> List[str]:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href and not href.startswith("#") and len(href) > 2:
                links.append(href)
        return list(dict.fromkeys(links))  # deduplicate, preserve order

    def _html_json_ld(self, soup) -> List[Dict]:
        results = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "{}")
                results.append(data)
            except Exception:
                pass
        return results

    # ── TXT helpers ───────────────────────────────────────────────────────────

    def _detect_line_ending(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                sample = f.read(4096)
            if b"\r\n" in sample:
                return "CRLF"
            elif b"\r" in sample:
                return "CR"
            return "LF"
        except Exception:
            return "LF"

    def _blank_line_ratio(self, text: str) -> float:
        lines = text.split("\n")
        if not lines:
            return 0.0
        blank = sum(1 for l in lines if not l.strip())
        return round(blank / len(lines), 3)


# ==============================================================================
# BLOCK 6 — METADATA EXTRACTOR + NORMALIZER
# ==============================================================================

class MetadataExtractor:
    """
    Normalize format-specific metadata into one unified schema.

    Unified Schema
    --------------
    author, title, subject, keywords, created, modified,
    page_count, word_count, char_count, language, encoding,
    is_encrypted, is_scanned, has_tables, has_images, has_links,
    reading_time_min, format, file_size_kb, file_path
    """

    def normalize(self, raw_meta: Dict, fmt: str, path: str,
                  full_text: str) -> Dict:
        """Build unified metadata dict from raw format metadata + text stats."""
        p = Path(path)
        file_size_kb = round(p.stat().st_size / 1024, 2) if p.exists() else 0

        word_count = len(full_text.split()) if full_text else 0
        char_count = len(full_text) if full_text else 0
        sent_count = len(re.findall(r"[.!?]+", full_text)) if full_text else 0

        unified = {
            # Identity
            "file_path":       path,
            "file_name":       p.name,
            "format":          fmt,
            "file_size_kb":    file_size_kb,
            # Authorship
            "author":          str(raw_meta.get("author", "") or ""),
            "title":           str(raw_meta.get("title",  "") or "") or p.stem,
            "subject":         str(raw_meta.get("subject","") or ""),
            "keywords":        str(raw_meta.get("keywords","") or ""),
            "description":     str(raw_meta.get("description","") or ""),
            # Dates
            "created":         str(raw_meta.get("created", "") or ""),
            "modified":        str(raw_meta.get("modified","") or ""),
            # Document stats
            "page_count":      int(raw_meta.get("page_count", 1) or 1),
            "word_count":      word_count,
            "char_count":      char_count,
            "sentence_count":  sent_count,
            "reading_time_min":round(word_count / READING_WPM, 1),
            # Language
            "language":        str(raw_meta.get("language", "") or
                                   raw_meta.get("lang",     "") or
                                   raw_meta.get("language_hint","") or ""),
            "encoding":        str(raw_meta.get("encoding", "utf-8") or "utf-8"),
            "is_rtl":          bool(raw_meta.get("is_rtl", False)),
            # Flags
            "is_encrypted":    bool(raw_meta.get("is_encrypted", False)),
            "is_scanned":      bool(raw_meta.get("is_scanned",   False)),
            "has_tables":      bool(raw_meta.get("has_tables",   False)),
            "has_images":      bool(raw_meta.get("has_images",   False)),
            "has_links":       bool(raw_meta.get("has_links",    False)),
            # Extra (format-specific passthrough)
            "extra":           {k: v for k, v in raw_meta.items()
                                if k not in self._base_keys()},
        }
        return unified

    def _base_keys(self) -> set:
        return {
            "author","title","subject","keywords","description",
            "created","modified","page_count","word_count","char_count",
            "language","lang","language_hint","encoding","is_rtl",
            "is_encrypted","is_scanned","has_tables","has_images","has_links",
        }


# ==============================================================================
# BLOCK 7 — UNIFIED DATAFRAME BUILDER
# ==============================================================================

class DataFrameBuilder:
    """
    Convert DocumentData into a standard pandas DataFrame.

    Output columns
    --------------
    file_path | format | section_index | section_type | heading_level
    text | word_count | char_count | sentence_count | is_rtl
    has_table | table_data | code_snippet | links | metadata_json

    All text is NFC-normalized.
    Table data stored as JSON string.
    Metadata dict stored as JSON string in metadata_json column.
    """

    BASE_COLUMNS = [
        "file_path", "format", "section_index", "section_type",
        "heading_level", "text", "word_count", "char_count",
        "sentence_count", "is_rtl", "has_table", "table_data",
        "style_name", "metadata_json",
    ]

    def build(self, doc: "DocumentData") -> "pd.DataFrame":
        import pandas as pd

        rows = []
        meta_json = json.dumps(doc.metadata, ensure_ascii=False, default=str)

        for section in doc.pages:
            text      = unicodedata.normalize("NFC", section.get("text", "") or "")
            word_cnt  = section.get("word_count")  or len(text.split())
            char_cnt  = section.get("char_count")  or len(text)
            sent_cnt  = len(re.findall(r"[.!?]+", text))
            has_table = section.get("section_type") == "table"
            tbl_data  = json.dumps(
                section.get("table_data", {}), ensure_ascii=False, default=str
            ) if has_table else ""

            row = {
                "file_path":     doc.file_path,
                "format":        doc.fmt,
                "section_index": section.get("section_index", 0),
                "section_type":  section.get("section_type",  "paragraph"),
                "heading_level": section.get("heading_level"),
                "text":          text,
                "word_count":    word_cnt,
                "char_count":    char_cnt,
                "sentence_count":sent_cnt,
                "is_rtl":        doc.metadata.get("is_rtl", False),
                "has_table":     has_table,
                "table_data":    tbl_data,
                "style_name":    section.get("style_name", ""),
                "metadata_json": meta_json,
            }
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=self.BASE_COLUMNS)

        df = pd.DataFrame(rows)
        # Ensure all base columns exist
        for col in self.BASE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[self.BASE_COLUMNS]

    def combine(self, dfs: List["pd.DataFrame"]) -> "pd.DataFrame":
        """Combine multiple DataFrames (from multiple files) into one."""
        import pandas as pd
        if not dfs:
            return pd.DataFrame(columns=self.BASE_COLUMNS)
        combined = pd.concat(dfs, ignore_index=True)
        combined["section_index"] = range(len(combined))
        return combined


# ==============================================================================
# BLOCK 8 — BATCH FOLDER READER
# ==============================================================================

class BatchFolderReader:
    """
    Read all supported documents in a folder (recursively).

    Features
    --------
    - Recursive directory walk
    - Parallel processing via ThreadPoolExecutor
    - Per-file error isolation (one bad file never crashes batch)
    - Progress bar via tqdm
    - Summary stats per format
    - Skips unsupported extensions silently
    """

    def __init__(self, max_workers: int = 4, recursive: bool = True):
        self.max_workers = max_workers
        self.recursive   = recursive
        self._reader     = None  # DocumentReader — set by master class

    def read_folder(self, folder: str,
                    reader: "DocumentReader") -> Tuple["pd.DataFrame", Dict]:
        """
        Read all documents in folder.

        Returns
        -------
        (combined_df, summary_dict)
        """
        import pandas as pd
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        # Discover files
        if self.recursive:
            all_files = [
                str(f) for f in folder_path.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        else:
            all_files = [
                str(f) for f in folder_path.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

        if not all_files:
            logger.warning(f"No supported documents found in {folder}")
            return pd.DataFrame(), {"total": 0, "success": 0, "failed": 0}

        logger.info(f"Found {len(all_files)} documents in {folder}")

        results: List["pd.DataFrame"] = []
        summary = {
            "total":   len(all_files),
            "success": 0,
            "failed":  0,
            "by_format": {},
            "failed_files": [],
        }

        def _read_one(path: str) -> Optional["pd.DataFrame"]:
            try:
                return reader.read(path)
            except Exception as exc:
                logger.warning(f"Failed to read {path}: {exc}")
                return None

        with tqdm(total=len(all_files), desc="📄 Reading documents", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(_read_one, f): f for f in all_files}
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        df = future.result(timeout=120)
                    except Exception as exc:
                        logger.error(f"Error processing {path}: {exc}")
                        df = None

                    if df is not None and not df.empty:
                        results.append(df)
                        summary["success"] += 1
                        fmt = Path(path).suffix.lower().lstrip(".")
                        summary["by_format"][fmt] = summary["by_format"].get(fmt, 0) + 1
                    else:
                        summary["failed"] += 1
                        summary["failed_files"].append(path)
                    pbar.update(1)

        builder  = DataFrameBuilder()
        combined = builder.combine(results) if results else pd.DataFrame()
        logger.info(
            f"Batch complete: {summary['success']} ok / "
            f"{summary['failed']} failed / {summary['total']} total"
        )
        return combined, summary


# ==============================================================================
# BLOCK 9 — TEXT CLEANER & NORMALIZER
# ==============================================================================

class TextCleaner:
    """
    Post-processing for extracted text before passing to text_analyzer.py.

    Cleaning steps
    --------------
    1. Unicode NFC normalization
    2. Control character removal (except \\n \\t \\r)
    3. Multiple spaces → single space
    4. 3+ newlines → double newline (preserve paragraph boundaries)
    5. Broken PDF hyphenation repair (word-\\nnext → wordnext)
    6. Repeated page header/footer removal
    7. Watermark text detection and removal
    8. Non-printable ASCII removal
    9. RTL mark preservation (doesn't strip Arabic/Hebrew directional marks)
    10. Sentence boundary normalization

    RTL-safe
    --------
    Preserves Unicode directional formatting characters:
    U+200F (RLM), U+200E (LRM), U+202B (RLE), U+202C (PDF), U+202A (LRE)
    """

    RTL_PRESERVE = {"\u200f", "\u200e", "\u202b", "\u202c", "\u202a", "\u202d"}

    def clean(self, text: str, is_rtl: bool = False) -> str:
        """Apply all cleaning steps. Returns cleaned text."""
        if not text:
            return ""
        text = self._nfc(text)
        text = self._remove_control_chars(text, is_rtl)
        text = self._fix_pdf_hyphenation(text)
        text = self._collapse_spaces(text)
        text = self._collapse_newlines(text)
        text = self._remove_watermarks(text)
        text = self._normalize_quotes(text)
        text = self._normalize_sentence_ends(text)
        return text.strip()

    def clean_sections(self, sections: List[Dict],
                        is_rtl: bool = False) -> List[Dict]:
        """Apply cleaning to each section's text field."""
        result = []
        for sec in sections:
            sec = dict(sec)
            if sec.get("text"):
                sec["text"] = self.clean(sec["text"], is_rtl)
                sec["word_count"] = len(sec["text"].split())
                sec["char_count"] = len(sec["text"])
            result.append(sec)
        return result

    # ── cleaning steps ────────────────────────────────────────────────────────

    def _nfc(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _remove_control_chars(self, text: str, is_rtl: bool) -> str:
        result = []
        for ch in text:
            cp = ord(ch)
            # Always keep: printable, newline, tab, carriage return
            if ch in "\n\r\t":
                result.append(ch)
            elif is_rtl and ch in self.RTL_PRESERVE:
                result.append(ch)
            elif cp < 32 or cp == 127:
                pass  # strip control char
            elif cp == 0xFFFD:
                pass  # strip replacement char
            else:
                result.append(ch)
        return "".join(result)

    def _fix_pdf_hyphenation(self, text: str) -> str:
        """
        Fix words broken across lines in PDF extraction:
        'hyphen-\\nnation' → 'hyphenation'
        'soft-\\nbreak'    → 'soft-break' (keep if hyphen is semantic)
        """
        # Soft hyphen at end of line + next word starts lowercase → join
        text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
        # Line break mid-word without hyphen (PDF artifact)
        text = re.sub(r"([a-z])\n([a-z])", r"\1 \2", text)
        return text

    def _collapse_spaces(self, text: str) -> str:
        """Collapse multiple spaces/tabs into single space (within lines)."""
        lines  = text.split("\n")
        result = []
        for line in lines:
            line = re.sub(r"[ \t]{2,}", " ", line)
            result.append(line)
        return "\n".join(result)

    def _collapse_newlines(self, text: str) -> str:
        """Collapse 3+ consecutive newlines into exactly 2."""
        return re.sub(r"\n{3,}", "\n\n", text)

    def _remove_watermarks(self, text: str) -> str:
        """
        Detect and remove watermark text:
        single-word lines that repeat 3+ times across the document.
        """
        lines = text.split("\n")
        line_counts: Dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped.split()) == 1 and stripped.isalpha():
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        watermarks = {w for w, cnt in line_counts.items() if cnt >= 3}
        if not watermarks:
            return text

        result = []
        for line in lines:
            if line.strip() not in watermarks:
                result.append(line)
        return "\n".join(result)

    def _normalize_quotes(self, text: str) -> str:
        """Normalize smart/curly quotes to straight ASCII."""
        quote_map = {
            "\u2018": "'", "\u2019": "'",   # ' '
            "\u201c": '"', "\u201d": '"',   # " "
            "\u201a": ",", "\u201e": ",,",  # ‚ „
            "\u2032": "'", "\u2033": '"',   # ′ ″
        }
        for curly, straight in quote_map.items():
            text = text.replace(curly, straight)
        return text

    def _normalize_sentence_ends(self, text: str) -> str:
        """Ensure single space after sentence-ending punctuation."""
        text = re.sub(r"([.!?])\s{2,}", r"\1 ", text)
        return text


# ==============================================================================
# BLOCK 10 — DOCUMENT DATA + MASTER CLASS + CONVENIENCE FUNCTIONS
# ==============================================================================

@dataclass
class DocumentData:
    """Unified output container for any document format."""
    file_path:  str
    fmt:        str
    full_text:  str
    pages:      List[Dict]
    metadata:   Dict
    tables:     List[Dict]
    extra:      Dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.full_text.split()) if self.full_text else 0

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def to_dict(self) -> Dict:
        return {
            "file_path":  self.file_path,
            "format":     self.fmt,
            "word_count": self.word_count,
            "page_count": self.page_count,
            "metadata":   self.metadata,
        }


class DocumentReader:
    """
    🗂️ Nydra Document Reader — Master Class

    Auto-detects format from magic bytes + extension.
    Routes to the correct reader.
    Returns a unified pandas DataFrame or DocumentData.

    Supported formats
    -----------------
    PDF   (.pdf)              — pdfplumber + PyMuPDF
    DOCX  (.docx)             — python-docx + docx2python
    DOC   (.doc)              — flags as legacy, extraction limited
    TXT   (.txt)              — chardet encoding handler
    MD    (.md)               — full Markdown stripping
    HTML  (.html / .htm)      — BeautifulSoup + OpenGraph + JSON-LD

    Usage
    -----
    >>> reader = DocumentReader()
    >>> df = reader.read("report.pdf")
    >>> df = reader.read_folder("docs/")
    >>> text = reader.quick_read("paper.docx")
    >>> meta = reader.get_metadata("article.html")
    """

    def __init__(self, clean_text: bool = True, max_workers: int = 4):
        self.clean_text  = clean_text
        self._router     = FormatRouter()
        self._pdf        = PDFReader()
        self._docx       = DOCXReader()
        self._plain      = PlainTextReader()
        self._meta_ext   = MetadataExtractor()
        self._df_builder = DataFrameBuilder()
        self._cleaner    = TextCleaner()
        self._batch      = BatchFolderReader(max_workers=max_workers)

    # ── primary API ───────────────────────────────────────────────────────────

    def read(self, path: str) -> "pd.DataFrame":
        """
        Auto-detect format and read document.
        Returns unified DataFrame (one row per section/paragraph/page).
        """
        doc = self._read_raw(path)
        return self._df_builder.build(doc)

    def read_raw(self, path: str) -> DocumentData:
        """Same as read() but returns DocumentData instead of DataFrame."""
        return self._read_raw(path)

    def read_folder(self, folder: str,
                    recursive: bool = True) -> Tuple["pd.DataFrame", Dict]:
        """
        Read all supported documents in a folder.
        Returns (combined_DataFrame, summary_dict).
        """
        self._batch.recursive = recursive
        return self._batch.read_folder(folder, reader=self)

    def quick_read(self, path: str) -> str:
        """Read document and return full plain text string."""
        doc = self._read_raw(path)
        return doc.full_text

    def get_metadata(self, path: str) -> Dict:
        """Read document and return only the metadata dict."""
        doc = self._read_raw(path)
        return doc.metadata

    # ── internal ──────────────────────────────────────────────────────────────

    def _read_raw(self, path: str) -> DocumentData:
        """Core: detect format, dispatch reader, clean, normalize metadata."""
        fmt = self._router.detect(path)
        self._router.check_deps(fmt)

        logger.info(f"Reading [{fmt.upper()}] {Path(path).name}")
        t0 = time.time()

        if fmt == "pdf":
            doc = self._pdf.read(path)
        elif fmt in ("docx", "doc"):
            if fmt == "doc":
                logger.warning(
                    f"Legacy .doc format detected for {Path(path).name}. "
                    f"Extraction may be incomplete. "
                    f"Convert to .docx for best results."
                )
            doc = self._docx.read(path)
        elif fmt == "txt":
            doc = self._plain.read_txt(path)
        elif fmt == "md":
            doc = self._plain.read_markdown(path)
        elif fmt == "html":
            doc = self._plain.read_html(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # Text cleaning
        if self.clean_text:
            is_rtl   = doc.metadata.get("is_rtl", False)
            doc.pages = self._cleaner.clean_sections(doc.pages, is_rtl)
            doc.full_text = self._cleaner.clean(doc.full_text, is_rtl)

        # Metadata normalization
        doc.metadata = self._meta_ext.normalize(
            doc.metadata, fmt, path, doc.full_text
        )

        elapsed = time.time() - t0
        logger.info(
            f"Done in {elapsed:.2f}s — "
            f"{doc.word_count} words, {doc.page_count} sections"
        )
        return doc


# ── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────────

_default_reader = None


def _get_reader() -> DocumentReader:
    global _default_reader
    if _default_reader is None:
        _default_reader = DocumentReader()
    return _default_reader


def read(path: str) -> "pd.DataFrame":
    """Auto-detect format and read document → DataFrame."""
    return _get_reader().read(path)


def read_folder(folder: str, recursive: bool = True) -> Tuple["pd.DataFrame", Dict]:
    """Read all supported documents in a folder → (DataFrame, summary)."""
    return _get_reader().read_folder(folder, recursive=recursive)


def quick_read(path: str) -> str:
    """Read document → plain text string."""
    return _get_reader().quick_read(path)


def get_metadata(path: str) -> Dict:
    """Read document → metadata dict only."""
    return _get_reader().get_metadata(path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    import argparse, pandas as pd

    parser = argparse.ArgumentParser(
        prog        = "document_reader",
        description = "📄 Nydra — Document Reader v0.6.0",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path",      help="Document file or folder to read")
    parser.add_argument("--folder",  action="store_true",
                        help="Read entire folder (all supported docs)")
    parser.add_argument("--output",  default=None, metavar="CSV",
                        help="Save output DataFrame to CSV")
    parser.add_argument("--meta",    action="store_true",
                        help="Print metadata only")
    parser.add_argument("--text",    action="store_true",
                        help="Print plain text only")
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip text cleaning")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for folder mode")

    args   = parser.parse_args()
    reader = DocumentReader(
        clean_text  = not args.no_clean,
        max_workers = args.workers,
    )

    if args.folder or Path(args.path).is_dir():
        df, summary = reader.read_folder(args.path)
        print(f"\n📂 Folder: {args.path}")
        print(f"   Total   : {summary['total']}")
        print(f"   Success : {summary['success']}")
        print(f"   Failed  : {summary['failed']}")
        print(f"   By format: {summary.get('by_format', {})}")
        if not df.empty and args.output:
            df.to_csv(args.output, index=False, encoding="utf-8")
            print(f"\n✅ Saved → {args.output}")
    else:
        if args.meta:
            meta = reader.get_metadata(args.path)
            print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        elif args.text:
            text = reader.quick_read(args.path)
            print(text)
        else:
            df = reader.read(args.path)
            print(df.to_string(index=False))
            if args.output:
                df.to_csv(args.output, index=False, encoding="utf-8")
                print(f"\n✅ Saved → {args.output}")


