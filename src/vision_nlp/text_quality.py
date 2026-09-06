"""
text_quality.py — Nydra v0.6.0
=====================================
Advanced Text Quality Analysis Engine

Detects and scores:
  • Duplicate content       (exact + near-duplicate + semantic)
  • Empty / near-empty text (entropy, token ratio, meaningful content)
  • Noise                   (HTML, URLs, special chars, boilerplate, code)
  • PII                     (emails, phones, SSN, CC, IPs, NER-based names)
  • Coherence               (embedding chain, lexical, topic drift)
  • Consistency             (NLI contradictions, numerical, tense, entities)
  • Encoding issues         (chardet, mojibake, BOM, unicode normalization)
  • Language quality        (spelling, grammar, vocabulary richness, formality)

Version : 0.6.0
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
import os
import json
import math
import hashlib
import unicodedata
import warnings
import logging
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# ── Optional heavy imports (graceful degradation) ────────────────────────────
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_quality")

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & SEVERITY
# ─────────────────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH     = "High"
    MEDIUM   = "Medium"
    LOW      = "Low"
    OK       = "OK"


class PIIType(str, Enum):
    EMAIL       = "Email"
    PHONE       = "Phone"
    CREDIT_CARD = "Credit Card"
    SSN         = "SSN"
    IP_ADDRESS  = "IP Address"
    PASSPORT    = "Passport"
    NATIONAL_ID = "National ID"
    DATE_OF_BIRTH = "Date of Birth"
    PERSON_NAME = "Person Name"
    ORGANIZATION = "Organization"
    URL         = "URL"


class PIISeverity(str, Enum):
    CRITICAL = "Critical"   # SSN, Credit Card, Passport
    HIGH     = "High"       # Email, Phone, IP
    MEDIUM   = "Medium"     # Names, Organizations, DOB
    LOW      = "Low"        # Generic URLs


class DuplicateLevel(str, Enum):
    EXACT        = "Exact"           # byte-identical
    NEAR_EXACT   = "Near-Exact"      # 1-3 word difference
    PARAPHRASE   = "Paraphrase"      # same meaning, different words
    SIMILAR      = "Similar"         # partial overlap


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextQualityConfig:
    """Central configuration for all quality checkers."""

    # Duplicate thresholds
    simhash_distance_threshold: int   = 3      # bits different (0-64)
    minhash_threshold: float          = 0.8    # Jaccard similarity
    cosine_threshold: float           = 0.85   # TF-IDF cosine
    rouge_l_threshold: float          = 0.75   # ROUGE-L F1

    # Empty content thresholds
    empty_char_threshold: int         = 10     # fewer chars = empty
    near_empty_token_ratio: float     = 0.05   # <5% meaningful tokens
    entropy_threshold: float          = 1.5    # Shannon entropy minimum
    min_unique_words: int             = 3      # fewer = near-empty

    # Noise thresholds
    html_tag_ratio_threshold: float   = 0.05   # >5% = noisy
    special_char_ratio_threshold: float = 0.15 # >15% = noisy
    url_density_threshold: float      = 0.10   # >10% = noisy
    boilerplate_repeat_threshold: int = 3      # phrase repeated N times

    # PII
    pii_detection_mode: str           = "all"  # "all" | "critical" | "fast"
    mask_pii_in_output: bool          = False
    pii_mask_char: str                = "[REDACTED]"

    # Coherence
    coherence_window: int             = 3      # adjacent sentence window
    coherence_min_sentences: int      = 5      # need at least N sentences
    topic_drift_threshold: float      = 0.3    # cosine drop = drift

    # Consistency
    nli_contradiction_threshold: float = 0.7   # NLI contradiction score
    numerical_tolerance: float        = 0.01   # 1% for numerical contradictions

    # Encoding
    encoding_confidence_threshold: float = 0.7 # chardet confidence
    mojibake_ratio_threshold: float   = 0.02   # >2% garbled chars

    # Language quality
    max_spelling_error_rate: float    = 0.05   # >5% = poor quality
    min_ttr: float                    = 0.2    # minimum type-token ratio
    min_mtld: float                   = 40.0   # minimum MTLD score
    passive_voice_threshold: float    = 0.4    # >40% passive = flag

    # Scoring weights (must sum to 1.0)
    weight_duplicate: float           = 0.20
    weight_noise: float               = 0.20
    weight_pii: float                 = 0.15
    weight_coherence: float           = 0.20
    weight_language: float            = 0.15
    weight_encoding: float            = 0.10

    # Output
    report_format: str                = "markdown"  # "markdown"|"json"|"html"
    include_examples: bool            = True
    max_examples_per_issue: int       = 3


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES — Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityIssue:
    """A single quality problem found in the text."""
    checker: str
    issue_type: str
    severity: Severity
    description: str
    location: Optional[str] = None      # sentence / paragraph index
    evidence: Optional[str] = None      # the actual problematic text
    suggestion: Optional[str] = None
    score_impact: float = 0.0           # how much this drops the score


@dataclass
class DuplicateResult:
    level: DuplicateLevel
    similarity_score: float
    text_a: str
    text_b: str
    method: str
    position_a: Optional[int] = None
    position_b: Optional[int] = None


@dataclass
class PIIMatch:
    pii_type: PIIType
    severity: PIISeverity
    value: str                         # the actual PII (or masked version)
    masked_value: str
    start_char: int
    end_char: int
    context: str                       # surrounding text
    sentence_index: Optional[int] = None


@dataclass
class CoherenceResult:
    local_coherence: float             # adjacent sentence similarity avg
    global_coherence: float            # topic consistency score
    discourse_score: float             # discourse connective density
    problem_transitions: List[Dict]    # where coherence drops
    overall_coherence: float           # weighted composite


@dataclass
class ConsistencyResult:
    contradiction_pairs: List[Dict]    # sentence pairs that contradict
    numerical_inconsistencies: List[Dict]
    tense_inconsistencies: List[Dict]
    entity_inconsistencies: List[Dict]
    overall_consistency_score: float


@dataclass
class EncodingResult:
    detected_encoding: str
    confidence: float
    has_bom: bool
    has_null_bytes: bool
    mojibake_ratio: float
    non_printable_ratio: float
    unicode_issues: List[str]
    is_clean: bool
    encoding_score: float


@dataclass
class LanguageQualityResult:
    spelling_error_rate: float
    spelling_errors: List[str]
    estimated_grammar_score: float
    ttr: float                         # Type-Token Ratio
    mtld: float                        # Measure of Textual Lexical Diversity
    hdd: float                         # HD-D vocabulary richness
    formality_score: float
    passive_voice_ratio: float
    avg_sentence_length: float
    sentence_length_variance: float
    overall_language_score: float


@dataclass
class TextQualityReport:
    """Full quality report for a piece of text."""
    text_length: int
    sentence_count: int
    word_count: int

    # Sub-scores (0-100, higher = better)
    duplicate_score: float
    empty_score: float
    noise_score: float
    pii_score: float
    coherence_score: float
    consistency_score: float
    encoding_score: float
    language_score: float

    # Composite
    overall_score: float
    verdict: str                       # "Excellent" | "Good" | "Needs Work" | "Poor"

    # Findings
    issues: List[QualityIssue]
    duplicate_findings: List[DuplicateResult]
    pii_findings: List[PIIMatch]
    coherence_findings: CoherenceResult
    consistency_findings: ConsistencyResult
    encoding_findings: EncodingResult
    language_findings: LanguageQualityResult

    # Meta
    processing_time_sec: float
    timestamp: str


# ─────────────────────────────────────────────────────────────────────────────
# BASE CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class BaseTextChecker(ABC):
    """Abstract base for all quality checkers."""

    def __init__(self, config: TextQualityConfig):
        self.config = config

    @abstractmethod
    def check(self, text: str) -> Dict[str, Any]:
        pass

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize into sentences with fallback."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        # Fallback: split on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize into words with fallback."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except Exception:
                pass
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _get_stopwords(self) -> set:
        """Get English stopwords."""
        if NLTK_AVAILABLE:
            try:
                return set(stopwords.words('english'))
            except Exception:
                pass
        return {
            'the','a','an','is','it','in','on','at','to','for',
            'of','and','or','but','not','with','this','that',
            'are','was','were','be','been','being','have','has',
            'had','do','does','did','will','would','could','should',
            'may','might','shall','can','need','dare','ought'
        }

    def _score_to_severity(self, score: float) -> Severity:
        """Convert 0-100 score to severity level."""
        if score >= 80:
            return Severity.OK
        elif score >= 60:
            return Severity.LOW
        elif score >= 40:
            return Severity.MEDIUM
        elif score >= 20:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def _truncate(self, text: str, max_chars: int = 100) -> str:
        """Truncate text for display."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUPLICATE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class DuplicateDetector(BaseTextChecker):
    """
    4-level duplicate detection:
      L1 — Exact hash         (SHA256) — byte-identical
      L2 — SimHash            (Charikar) — near-identical (1-3 word changes)
      L3 — MinHash + LSH      (Datasketch) — paraphrase-level
      L4 — TF-IDF cosine      — semantic near-duplicates

    Works at sentence, paragraph, and full-document level.
    """

    def __init__(self, config: TextQualityConfig):
        super().__init__(config)
        self._seen_hashes: Dict[str, int] = {}   # hash → first seen index

    # ── Public API ───────────────────────────────────────────────────────────

    def check(self, text: str) -> Dict[str, Any]:
        """Check a single text for internal duplicates."""
        sentences   = self._tokenize_sentences(text)
        paragraphs  = self._split_paragraphs(text)

        sentence_dupes = self._find_duplicates_in_list(sentences, unit="sentence")
        paragraph_dupes = self._find_duplicates_in_list(paragraphs, unit="paragraph")

        all_dupes = sentence_dupes + paragraph_dupes
        score = self._compute_score(all_dupes, len(sentences) + len(paragraphs))

        return {
            "duplicate_findings": all_dupes,
            "sentence_duplicate_count": len(sentence_dupes),
            "paragraph_duplicate_count": len(paragraph_dupes),
            "total_duplicate_count": len(all_dupes),
            "duplicate_rate": len(all_dupes) / max(len(sentences), 1),
            "score": score,
            "severity": self._score_to_severity(score).value,
        }

    def check_corpus(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check a list of documents for cross-document duplicates."""
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        results = {
            "exact_duplicates":     [],
            "near_duplicates":      [],
            "paraphrase_duplicates":[],
            "total_checked":        len(texts),
            "duplicate_pairs":      0,
        }

        # L1: Exact
        exact = self._find_exact_duplicates(texts, ids)
        results["exact_duplicates"] = exact

        # L2: SimHash
        near = self._find_simhash_duplicates(texts, ids)
        results["near_duplicates"] = near

        # L3: MinHash LSH
        if DATASKETCH_AVAILABLE:
            paraphrase = self._find_minhash_duplicates(texts, ids)
            results["paraphrase_duplicates"] = paraphrase

        # L4: TF-IDF cosine (on full texts)
        if SKLEARN_AVAILABLE and len(texts) > 1:
            semantic = self._find_cosine_duplicates(texts, ids)
            results["paraphrase_duplicates"] += semantic

        # Deduplicate results
        results["paraphrase_duplicates"] = self._deduplicate_pairs(
            results["paraphrase_duplicates"]
        )

        total_pairs = (
            len(results["exact_duplicates"]) +
            len(results["near_duplicates"]) +
            len(results["paraphrase_duplicates"])
        )
        results["duplicate_pairs"] = total_pairs
        results["score"] = max(0, 100 - (total_pairs / max(len(texts), 1)) * 50)

        return results

    # ── Level 1: Exact SHA256 ─────────────────────────────────────────────────

    def _sha256(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def _find_exact_duplicates(
        self, texts: List[str], ids: List[str]
    ) -> List[DuplicateResult]:
        seen: Dict[str, str] = {}
        dupes = []
        for i, text in enumerate(texts):
            h = self._sha256(text)
            if h in seen:
                dupes.append(DuplicateResult(
                    level=DuplicateLevel.EXACT,
                    similarity_score=1.0,
                    text_a=self._truncate(texts[int(seen[h])]),
                    text_b=self._truncate(text),
                    method="SHA256",
                    position_a=int(seen[h]),
                    position_b=i,
                ))
            else:
                seen[h] = str(i)
        return dupes

    # ── Level 2: SimHash (Charikar's) ─────────────────────────────────────────

    def _simhash(self, text: str, bits: int = 64) -> int:
        """Compute 64-bit SimHash of text."""
        tokens = self._tokenize_words(text)
        if not tokens:
            return 0

        v = [0] * bits
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(bits):
                bit = (h >> i) & 1
                v[i] += 1 if bit else -1

        fingerprint = 0
        for i in range(bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def _hamming_distance(self, a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def _find_simhash_duplicates(
        self, texts: List[str], ids: List[str]
    ) -> List[DuplicateResult]:
        hashes = [(i, self._simhash(t)) for i, t in enumerate(texts)]
        dupes = []
        threshold = self.config.simhash_distance_threshold

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                dist = self._hamming_distance(hashes[i][1], hashes[j][1])
                if 0 < dist <= threshold:
                    similarity = 1.0 - (dist / 64)
                    dupes.append(DuplicateResult(
                        level=DuplicateLevel.NEAR_EXACT,
                        similarity_score=similarity,
                        text_a=self._truncate(texts[i]),
                        text_b=self._truncate(texts[j]),
                        method=f"SimHash (hamming={dist})",
                        position_a=i,
                        position_b=j,
                    ))
        return dupes

    # ── Level 3: MinHash + LSH ────────────────────────────────────────────────

    def _text_to_shingles(self, text: str, k: int = 3) -> set:
        """Convert text to k-shingles (character n-grams)."""
        text = text.lower().strip()
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _find_minhash_duplicates(
        self, texts: List[str], ids: List[str]
    ) -> List[DuplicateResult]:
        if not DATASKETCH_AVAILABLE:
            return []

        threshold = self.config.minhash_threshold
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = []

        for i, text in enumerate(texts):
            m = MinHash(num_perm=128)
            for shingle in self._text_to_shingles(text):
                m.update(shingle.encode('utf-8'))
            minhashes.append(m)
            try:
                lsh.insert(str(i), m)
            except Exception:
                pass

        dupes = []
        seen_pairs = set()

        for i, m in enumerate(minhashes):
            try:
                neighbors = lsh.query(m)
            except Exception:
                continue
            for neighbor in neighbors:
                j = int(neighbor)
                if j == i:
                    continue
                pair = tuple(sorted((i, j)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Estimate Jaccard from MinHash
                jaccard = m.jaccard(minhashes[j])
                if jaccard >= threshold:
                    dupes.append(DuplicateResult(
                        level=DuplicateLevel.PARAPHRASE,
                        similarity_score=jaccard,
                        text_a=self._truncate(texts[i]),
                        text_b=self._truncate(texts[j]),
                        method=f"MinHash LSH (jaccard={jaccard:.2f})",
                        position_a=i,
                        position_b=j,
                    ))
        return dupes

    # ── Level 4: TF-IDF Cosine ────────────────────────────────────────────────

    def _find_cosine_duplicates(
        self, texts: List[str], ids: List[str]
    ) -> List[DuplicateResult]:
        if not SKLEARN_AVAILABLE or len(texts) < 2:
            return []

        threshold = self.config.cosine_threshold
        try:
            vectorizer = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2), min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            sim_matrix = cosine_similarity(tfidf_matrix)
        except Exception:
            return []

        dupes = []
        seen_pairs = set()
        n = len(texts)

        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])
                if score >= threshold:
                    pair = (i, j)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        dupes.append(DuplicateResult(
                            level=DuplicateLevel.SIMILAR,
                            similarity_score=score,
                            text_a=self._truncate(texts[i]),
                            text_b=self._truncate(texts[j]),
                            method=f"TF-IDF Cosine (score={score:.2f})",
                            position_a=i,
                            position_b=j,
                        ))
        return dupes

    # ── ROUGE-L for sentence-level ────────────────────────────────────────────

    def _rouge_l(self, a: str, b: str) -> float:
        """Compute ROUGE-L F1 score between two strings."""
        tokens_a = a.lower().split()
        tokens_b = b.lower().split()
        if not tokens_a or not tokens_b:
            return 0.0
        lcs = self._lcs_length(tokens_a, tokens_b)
        precision = lcs / len(tokens_b)
        recall    = lcs / len(tokens_a)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs_length(self, a: List[str], b: List[str]) -> int:
        """Longest common subsequence length."""
        m, n = len(a), len(b)
        # Space-optimized O(min(m,n))
        if m < n:
            a, b = b, a
            m, n = n, m
        prev = [0] * (n + 1)
        for token in a:
            curr = [0] * (n + 1)
            for j, t in enumerate(b):
                if token == t:
                    curr[j + 1] = prev[j] + 1
                else:
                    curr[j + 1] = max(curr[j], prev[j + 1])
            prev = curr
        return prev[n]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_duplicates_in_list(
        self, units: List[str], unit: str = "sentence"
    ) -> List[DuplicateResult]:
        """Find duplicates within a list of text units."""
        if len(units) < 2:
            return []

        dupes = []
        # L1: Exact
        hash_map: Dict[str, int] = {}
        for i, u in enumerate(units):
            h = self._sha256(u)
            if h in hash_map:
                dupes.append(DuplicateResult(
                    level=DuplicateLevel.EXACT,
                    similarity_score=1.0,
                    text_a=self._truncate(units[hash_map[h]]),
                    text_b=self._truncate(u),
                    method=f"SHA256 ({unit})",
                    position_a=hash_map[h],
                    position_b=i,
                ))
            else:
                hash_map[h] = i

        # L2: ROUGE-L for near-exact (efficient pairwise for small sets)
        threshold = self.config.rouge_l_threshold
        seen = set()
        if len(units) <= 500:
            for i in range(len(units)):
                for j in range(i + 1, len(units)):
                    pair = (i, j)
                    if pair in seen:
                        continue
                    rl = self._rouge_l(units[i], units[j])
                    if rl >= threshold and rl < 1.0:
                        seen.add(pair)
                        dupes.append(DuplicateResult(
                            level=DuplicateLevel.NEAR_EXACT,
                            similarity_score=rl,
                            text_a=self._truncate(units[i]),
                            text_b=self._truncate(units[j]),
                            method=f"ROUGE-L ({unit}, score={rl:.2f})",
                            position_a=i,
                            position_b=j,
                        ))
        return dupes

    def _split_paragraphs(self, text: str) -> List[str]:
        paras = re.split(r'\n{2,}', text.strip())
        return [p.strip() for p in paras if len(p.strip()) > 20]

    def _deduplicate_pairs(self, dupes: List[DuplicateResult]) -> List[DuplicateResult]:
        seen = set()
        result = []
        for d in dupes:
            key = (min(d.position_a or 0, d.position_b or 0),
                   max(d.position_a or 0, d.position_b or 0))
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    def _compute_score(self, dupes: List[DuplicateResult], total_units: int) -> float:
        if total_units == 0:
            return 100.0
        exact    = sum(1 for d in dupes if d.level == DuplicateLevel.EXACT)
        near     = sum(1 for d in dupes if d.level == DuplicateLevel.NEAR_EXACT)
        paraph   = sum(1 for d in dupes if d.level == DuplicateLevel.PARAPHRASE)
        similar  = sum(1 for d in dupes if d.level == DuplicateLevel.SIMILAR)

        penalty = (exact * 10 + near * 7 + paraph * 4 + similar * 2)
        penalty_pct = penalty / max(total_units, 1)
        score = max(0.0, 100.0 - penalty_pct * 100)
        return round(score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMPTY CONTENT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class EmptyContentDetector(BaseTextChecker):
    """
    Detects empty or near-empty content using:
      • Character count threshold
      • Shannon entropy
      • Meaningful token ratio (after stopword removal)
      • Unique word count
      • Whitespace ratio
      • Average sentence length
    """

    def check(self, text: str) -> Dict[str, Any]:
        if text is None:
            return self._empty_result("null input")

        issues = []
        metrics = self._compute_metrics(text)

        # Check 1: Hard empty
        if metrics["char_count"] < self.config.empty_char_threshold:
            issues.append(QualityIssue(
                checker="EmptyContentDetector",
                issue_type="empty_text",
                severity=Severity.CRITICAL,
                description=f"Text is empty or nearly empty ({metrics['char_count']} chars).",
                suggestion="Remove or fill this entry.",
                score_impact=100.0,
            ))

        # Check 2: Entropy too low
        elif metrics["entropy"] < self.config.entropy_threshold:
            issues.append(QualityIssue(
                checker="EmptyContentDetector",
                issue_type="low_entropy",
                severity=Severity.HIGH,
                description=f"Shannon entropy is very low ({metrics['entropy']:.2f}). "
                            f"Text may be repetitive or meaningless.",
                suggestion="Check if content is repeated characters or garbage.",
                score_impact=60.0,
            ))

        # Check 3: Meaningful token ratio
        if metrics["meaningful_token_ratio"] < self.config.near_empty_token_ratio:
            issues.append(QualityIssue(
                checker="EmptyContentDetector",
                issue_type="near_empty_content",
                severity=Severity.HIGH,
                description=f"Only {metrics['meaningful_token_ratio']*100:.1f}% of tokens "
                            f"are meaningful (non-stopword, non-punctuation).",
                suggestion="Text may be mostly stopwords or punctuation.",
                score_impact=50.0,
            ))

        # Check 4: Too few unique words
        if 0 < metrics["unique_word_count"] < self.config.min_unique_words:
            issues.append(QualityIssue(
                checker="EmptyContentDetector",
                issue_type="low_vocabulary",
                severity=Severity.MEDIUM,
                description=f"Only {metrics['unique_word_count']} unique words in text.",
                suggestion="Text lacks sufficient vocabulary diversity.",
                score_impact=30.0,
            ))

        # Check 5: Whitespace ratio
        if metrics["whitespace_ratio"] > 0.5:
            issues.append(QualityIssue(
                checker="EmptyContentDetector",
                issue_type="excessive_whitespace",
                severity=Severity.MEDIUM,
                description=f"{metrics['whitespace_ratio']*100:.1f}% of text is whitespace.",
                suggestion="Strip or normalize whitespace.",
                score_impact=20.0,
            ))

        score = self._compute_score(metrics, issues)

        return {
            "metrics": metrics,
            "issues": issues,
            "score": score,
            "severity": self._score_to_severity(score).value,
            "is_empty": metrics["char_count"] < self.config.empty_char_threshold,
            "is_near_empty": metrics["meaningful_token_ratio"] < self.config.near_empty_token_ratio,
        }

    def _compute_metrics(self, text: str) -> Dict[str, Any]:
        char_count = len(text)
        words = self._tokenize_words(text)
        stopwords_set = self._get_stopwords()
        sentences = self._tokenize_sentences(text)

        meaningful_tokens = [
            w for w in words
            if w not in stopwords_set
            and re.match(r'^[a-zA-Z]+$', w)
            and len(w) > 1
        ]

        unique_words = set(words)
        whitespace_count = sum(1 for c in text if c.isspace())
        entropy = self._shannon_entropy(text)

        avg_sentence_len = (
            sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        )

        return {
            "char_count": char_count,
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "sentence_count": len(sentences),
            "meaningful_token_count": len(meaningful_tokens),
            "meaningful_token_ratio": len(meaningful_tokens) / max(len(words), 1),
            "whitespace_ratio": whitespace_count / max(char_count, 1),
            "entropy": entropy,
            "avg_sentence_length": avg_sentence_len,
        }

    def _shannon_entropy(self, text: str) -> float:
        """Shannon entropy of character distribution."""
        if not text:
            return 0.0
        counter = Counter(text)
        total = len(text)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return round(entropy, 4)

    def _compute_score(
        self, metrics: Dict, issues: List[QualityIssue]
    ) -> float:
        if metrics["char_count"] < self.config.empty_char_threshold:
            return 0.0

        score = 100.0
        # Entropy contribution
        if metrics["entropy"] < self.config.entropy_threshold:
            ratio = metrics["entropy"] / self.config.entropy_threshold
            score -= (1 - ratio) * 40

        # Meaningful token ratio contribution
        mtr = metrics["meaningful_token_ratio"]
        if mtr < 0.3:
            score -= (0.3 - mtr) / 0.3 * 30

        # Whitespace penalty
        wr = metrics["whitespace_ratio"]
        if wr > 0.3:
            score -= (wr - 0.3) * 30

        return round(max(0.0, min(100.0, score)), 2)

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        return {
            "metrics": {"char_count": 0},
            "issues": [QualityIssue(
                checker="EmptyContentDetector",
                issue_type="null_input",
                severity=Severity.CRITICAL,
                description=f"No text provided: {reason}",
                score_impact=100.0,
            )],
            "score": 0.0,
            "severity": Severity.CRITICAL.value,
            "is_empty": True,
            "is_near_empty": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. NOISE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class NoiseDetector(BaseTextChecker):
    """
    Detects text noise including:
      • HTML / XML tags
      • URLs and email addresses
      • Special / control characters
      • Boilerplate phrases (repeated N times)
      • Code snippets
      • Watermark-like patterns
      • Excessive punctuation
      • Junk / garbage text
    """

    # Pre-compiled patterns
    HTML_TAG_PATTERN    = re.compile(r'<[^>]+>', re.MULTILINE)
    URL_PATTERN         = re.compile(
        r'https?://\S+|www\.\S+|ftp://\S+', re.IGNORECASE
    )
    EMAIL_PATTERN_NOISE = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    EXCESSIVE_PUNCT      = re.compile(r'[!?.,;:]{3,}')
    EMOJI_PATTERN        = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F9FF"
        u"\U00002600-\U000027BF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    CODE_INDICATORS = re.compile(
        r'(?:def |class |import |from |#include|<\?php|<script|</script|'
        r'function\s*\(|var\s+\w+\s*=|console\.log|System\.out|'
        r'\$\w+\s*=|^\s*\{|\}\s*$|==|!=|<=|>=|\|\|&&)',
        re.MULTILINE
    )
    WATERMARK_PATTERNS = [
        re.compile(r'(?:confidential|proprietary|draft|do\s+not\s+distribute)', re.IGNORECASE),
        re.compile(r'page\s+\d+\s+of\s+\d+', re.IGNORECASE),
        re.compile(r'copyright\s+©?\s*\d{4}', re.IGNORECASE),
        re.compile(r'all\s+rights\s+reserved', re.IGNORECASE),
    ]

    def check(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"score": 100.0, "issues": [], "metrics": {}}

        metrics = self._compute_noise_metrics(text)
        issues  = self._detect_issues(text, metrics)
        score   = self._compute_noise_score(metrics)

        return {
            "metrics": metrics,
            "issues": issues,
            "score": score,
            "severity": self._score_to_severity(score).value,
            "clean_text_estimate": self._estimate_clean_text(text, metrics),
        }

    def _compute_noise_metrics(self, text: str) -> Dict[str, Any]:
        total_chars = max(len(text), 1)

        # HTML tags
        html_matches = self.HTML_TAG_PATTERN.findall(text)
        html_chars   = sum(len(m) for m in html_matches)
        html_ratio   = html_chars / total_chars

        # URLs
        url_matches  = self.URL_PATTERN.findall(text)
        url_chars    = sum(len(u) for u in url_matches)
        url_ratio    = url_chars / total_chars

        # Control characters
        ctrl_matches = self.CONTROL_CHAR_PATTERN.findall(text)
        ctrl_ratio   = len(ctrl_matches) / total_chars

        # Special characters (non-alphanumeric, non-standard punctuation)
        special_chars = sum(
            1 for c in text
            if not c.isalnum()
            and c not in ' \t\n\r.,!?;:\'"()-–—'
        )
        special_ratio = special_chars / total_chars

        # Excessive punctuation
        exc_punct = self.EXCESSIVE_PUNCT.findall(text)

        # Emoji count
        emoji_matches = self.EMOJI_PATTERN.findall(text)

        # Code snippets
        code_lines = self.CODE_INDICATORS.findall(text)
        is_code_heavy = len(code_lines) >= 3

        # Boilerplate detection
        boilerplate_phrases = self._detect_boilerplate(text)

        # Watermark
        watermark_matches = []
        for pattern in self.WATERMARK_PATTERNS:
            watermark_matches.extend(pattern.findall(text))

        # Numeric ratio
        digit_count  = sum(1 for c in text if c.isdigit())
        numeric_ratio = digit_count / total_chars

        return {
            "total_chars": total_chars,
            "html_tag_count": len(html_matches),
            "html_ratio": round(html_ratio, 4),
            "url_count": len(url_matches),
            "url_ratio": round(url_ratio, 4),
            "control_char_count": len(ctrl_matches),
            "control_char_ratio": round(ctrl_ratio, 4),
            "special_char_count": special_chars,
            "special_char_ratio": round(special_ratio, 4),
            "excessive_punctuation_count": len(exc_punct),
            "emoji_count": len(emoji_matches),
            "is_code_heavy": is_code_heavy,
            "code_indicator_count": len(code_lines),
            "boilerplate_phrases": boilerplate_phrases,
            "watermark_matches": watermark_matches,
            "numeric_ratio": round(numeric_ratio, 4),
        }

    def _detect_boilerplate(self, text: str) -> List[str]:
        """Detect repeated phrases that appear N or more times."""
        sentences = self._tokenize_sentences(text)
        phrase_counter: Counter = Counter()

        # Build 3-grams of words across sentences
        words = self._tokenize_words(text)
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i+3])
            phrase_counter[trigram] += 1

        threshold = self.config.boilerplate_repeat_threshold
        boilerplate = [
            phrase for phrase, count in phrase_counter.items()
            if count >= threshold and len(phrase) > 10
        ]
        return boilerplate[:10]  # top 10

    def _detect_issues(
        self, text: str, metrics: Dict[str, Any]
    ) -> List[QualityIssue]:
        issues = []

        if metrics["html_ratio"] > self.config.html_tag_ratio_threshold:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="html_noise",
                severity=Severity.HIGH if metrics["html_ratio"] > 0.15 else Severity.MEDIUM,
                description=f"Found {metrics['html_tag_count']} HTML tags "
                            f"({metrics['html_ratio']*100:.1f}% of text).",
                suggestion="Strip HTML tags before processing.",
                score_impact=metrics["html_ratio"] * 100,
            ))

        if metrics["url_ratio"] > self.config.url_density_threshold:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="url_noise",
                severity=Severity.MEDIUM,
                description=f"Found {metrics['url_count']} URLs "
                            f"({metrics['url_ratio']*100:.1f}% of text).",
                suggestion="Remove or replace URLs with domain names if not needed.",
                score_impact=metrics["url_ratio"] * 50,
            ))

        if metrics["control_char_count"] > 0:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="control_characters",
                severity=Severity.HIGH,
                description=f"Found {metrics['control_char_count']} control characters.",
                suggestion="Remove all control characters (except \\n, \\t).",
                score_impact=30.0,
            ))

        if metrics["special_char_ratio"] > self.config.special_char_ratio_threshold:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="special_characters",
                severity=Severity.MEDIUM,
                description=f"Special character ratio is {metrics['special_char_ratio']*100:.1f}%.",
                suggestion="Clean or normalize special characters.",
                score_impact=metrics["special_char_ratio"] * 80,
            ))

        if metrics["is_code_heavy"]:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="code_content",
                severity=Severity.MEDIUM,
                description=f"Text appears to contain code snippets "
                            f"({metrics['code_indicator_count']} code indicators found).",
                suggestion="If this is a text corpus, remove code blocks.",
                score_impact=20.0,
            ))

        if metrics["boilerplate_phrases"]:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="boilerplate",
                severity=Severity.MEDIUM,
                description=f"Found {len(metrics['boilerplate_phrases'])} boilerplate phrase(s) "
                            f"repeated ≥{self.config.boilerplate_repeat_threshold}x.",
                evidence=str(metrics["boilerplate_phrases"][:3]),
                suggestion="Remove repeated boilerplate phrases.",
                score_impact=15.0,
            ))

        if metrics["watermark_matches"]:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="watermark_text",
                severity=Severity.LOW,
                description=f"Found watermark / legal text: {metrics['watermark_matches'][:2]}",
                suggestion="Remove watermarks and legal footers from training data.",
                score_impact=10.0,
            ))

        if metrics["excessive_punctuation_count"] > 0:
            issues.append(QualityIssue(
                checker="NoiseDetector",
                issue_type="excessive_punctuation",
                severity=Severity.LOW,
                description=f"Found {metrics['excessive_punctuation_count']} excessive punctuation sequences.",
                suggestion="Normalize punctuation.",
                score_impact=5.0,
            ))

        return issues

    def _compute_noise_score(self, metrics: Dict[str, Any]) -> float:
        score = 100.0

        # HTML penalty
        html_pen = min(metrics["html_ratio"] / self.config.html_tag_ratio_threshold, 3.0) * 20
        score -= html_pen

        # URL penalty
        url_pen = min(metrics["url_ratio"] / self.config.url_density_threshold, 2.0) * 10
        score -= url_pen

        # Control char penalty
        if metrics["control_char_count"] > 0:
            score -= min(metrics["control_char_ratio"] * 1000, 30)

        # Special char penalty
        sp_pen = min(metrics["special_char_ratio"] / self.config.special_char_ratio_threshold, 2.0) * 15
        score -= sp_pen

        # Code penalty
        if metrics["is_code_heavy"]:
            score -= 15

        # Boilerplate penalty
        score -= len(metrics["boilerplate_phrases"]) * 2

        return round(max(0.0, min(100.0, score)), 2)

    def _estimate_clean_text(self, text: str, metrics: Dict) -> str:
        """Return estimated cleaned text (strips HTML + URLs)."""
        clean = self.HTML_TAG_PATTERN.sub('', text)
        clean = self.URL_PATTERN.sub('[URL]', clean)
        clean = self.CONTROL_CHAR_PATTERN.sub('', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean[:500]  # preview only


# ─────────────────────────────────────────────────────────────────────────────
# 4. PII DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class PIIDetector(BaseTextChecker):
    """
    Comprehensive PII detection with:
      • Regex patterns: email, phone, credit card (+ Luhn), SSN, IP v4/v6,
        passport, national ID, date of birth
      • spaCy NER: PERSON, ORG, GPE
      • Severity classification: Critical / High / Medium
      • Masking / redaction support
      • Per-match location tracking
    """

    # ── Regex patterns ─────────────────────────────────────────────────────

    _PATTERNS: Dict[PIIType, Tuple[re.Pattern, PIISeverity]] = {
        PIIType.EMAIL: (
            re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
            PIISeverity.HIGH,
        ),
        PIIType.PHONE: (
            re.compile(
                r'(?:\+?\d{1,3}[\s\-.])?'
                r'(?:\(?\d{1,4}\)?[\s\-.])?'
                r'\d{1,4}[\s\-.]\d{1,4}[\s\-.]\d{1,9}'
            ),
            PIISeverity.HIGH,
        ),
        PIIType.SSN: (
            re.compile(r'\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0{4})\d{4}\b'),
            PIISeverity.CRITICAL,
        ),
        PIIType.IP_ADDRESS: (
            re.compile(
                r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
                r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
                r'|'
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
            ),
            PIISeverity.HIGH,
        ),
        PIIType.DATE_OF_BIRTH: (
            re.compile(
                r'\b(?:DOB|date\s+of\s+birth|born\s+on|birthday)[:\s]+'
                r'(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
                re.IGNORECASE,
            ),
            PIISeverity.MEDIUM,
        ),
        PIIType.PASSPORT: (
            re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
            PIISeverity.CRITICAL,
        ),
        PIIType.NATIONAL_ID: (
            re.compile(r'\b\d{8,12}\b'),
            PIISeverity.MEDIUM,
        ),
        PIIType.URL: (
            re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            PIISeverity.LOW,
        ),
    }

    # Credit card: separate for Luhn validation
    _CC_PATTERN = re.compile(
        r'\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2})|3[47]\d{2}|3(?:0[0-5]|[68]\d)\d)'
        r'[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
    )

    def __init__(self, config: TextQualityConfig):
        super().__init__(config)
        self._nlp = None
        self._nlp_loaded = False

    def _load_spacy(self) -> bool:
        if self._nlp_loaded:
            return self._nlp is not None
        self._nlp_loaded = True
        if not SPACY_AVAILABLE:
            return False
        try:
            import spacy as sp
            self._nlp = sp.load("en_core_web_sm")
            return True
        except Exception:
            try:
                import spacy as sp
                self._nlp = sp.load("en_core_web_md")
                return True
            except Exception:
                self._nlp = None
                return False

    # ── Luhn Algorithm ────────────────────────────────────────────────────────

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = re.sub(r'\D', '', card_number)
        if len(digits) < 13 or len(digits) > 19:
            return False
        total = 0
        reverse_digits = digits[::-1]
        for i, d in enumerate(reverse_digits):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 == 0

    # ── Main detection ────────────────────────────────────────────────────────

    def check(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"pii_found": [], "pii_count": 0, "score": 100.0, "issues": []}

        matches: List[PIIMatch] = []
        sentences = self._tokenize_sentences(text)

        # Regex-based detection
        matches.extend(self._detect_regex_pii(text, sentences))

        # Credit card with Luhn validation
        matches.extend(self._detect_credit_cards(text, sentences))

        # NER-based detection (names, orgs)
        if self.config.pii_detection_mode in ("all",):
            matches.extend(self._detect_ner_pii(text, sentences))

        # Mask if configured
        if self.config.mask_pii_in_output:
            for m in matches:
                m.value = m.masked_value

        score = self._compute_pii_score(matches, len(text))
        issues = self._build_issues(matches)

        return {
            "pii_found": matches,
            "pii_count": len(matches),
            "pii_by_type": self._group_by_type(matches),
            "critical_count": sum(1 for m in matches if m.severity == PIISeverity.CRITICAL),
            "high_count": sum(1 for m in matches if m.severity == PIISeverity.HIGH),
            "score": score,
            "severity": self._score_to_severity(score).value,
            "issues": issues,
        }

    def _detect_regex_pii(
        self, text: str, sentences: List[str]
    ) -> List[PIIMatch]:
        matches = []
        for pii_type, (pattern, severity) in self._PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group()
                start = match.start()
                end   = match.end()

                # Find which sentence
                sent_idx = self._find_sentence_index(text, sentences, start)
                context  = text[max(0, start-30): min(len(text), end+30)]

                masked = self.config.pii_mask_char

                matches.append(PIIMatch(
                    pii_type=pii_type,
                    severity=severity,
                    value=value,
                    masked_value=masked,
                    start_char=start,
                    end_char=end,
                    context=context.strip(),
                    sentence_index=sent_idx,
                ))
        return matches

    def _detect_credit_cards(
        self, text: str, sentences: List[str]
    ) -> List[PIIMatch]:
        matches = []
        for match in self._CC_PATTERN.finditer(text):
            value = match.group()
            if self._luhn_check(value):
                start = match.start()
                end   = match.end()
                sent_idx = self._find_sentence_index(text, sentences, start)
                context  = text[max(0, start-30): min(len(text), end+30)]
                matches.append(PIIMatch(
                    pii_type=PIIType.CREDIT_CARD,
                    severity=PIISeverity.CRITICAL,
                    value=value,
                    masked_value=self.config.pii_mask_char,
                    start_char=start,
                    end_char=end,
                    context=context.strip(),
                    sentence_index=sent_idx,
                ))
        return matches

    def _detect_ner_pii(
        self, text: str, sentences: List[str]
    ) -> List[PIIMatch]:
        if not self._load_spacy():
            return []

        matches = []
        try:
            # Process in chunks to handle long texts
            chunk_size = 100000
            for chunk_start in range(0, len(text), chunk_size):
                chunk = text[chunk_start:chunk_start + chunk_size]
                doc = self._nlp(chunk)
                for ent in doc.ents:
                    if ent.label_ in ("PERSON",):
                        pii_type = PIIType.PERSON_NAME
                        sev      = PIISeverity.MEDIUM
                    elif ent.label_ in ("ORG",):
                        pii_type = PIIType.ORGANIZATION
                        sev      = PIISeverity.MEDIUM
                    else:
                        continue

                    abs_start = chunk_start + ent.start_char
                    abs_end   = chunk_start + ent.end_char
                    context   = text[max(0, abs_start-30): min(len(text), abs_end+30)]
                    sent_idx  = self._find_sentence_index(text, sentences, abs_start)

                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        severity=sev,
                        value=ent.text,
                        masked_value=self.config.pii_mask_char,
                        start_char=abs_start,
                        end_char=abs_end,
                        context=context.strip(),
                        sentence_index=sent_idx,
                    ))
        except Exception as e:
            logger.warning(f"NER failed: {e}")

        return matches

    def _find_sentence_index(
        self, text: str, sentences: List[str], char_pos: int
    ) -> int:
        pos = 0
        for i, sent in enumerate(sentences):
            pos = text.find(sent, pos)
            if pos == -1:
                break
            if pos <= char_pos <= pos + len(sent):
                return i
            pos += len(sent)
        return -1

    def _group_by_type(self, matches: List[PIIMatch]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for m in matches:
            counts[m.pii_type.value] += 1
        return dict(counts)

    def _compute_pii_score(self, matches: List[PIIMatch], text_len: int) -> float:
        if not matches:
            return 100.0

        critical = sum(1 for m in matches if m.severity == PIISeverity.CRITICAL)
        high     = sum(1 for m in matches if m.severity == PIISeverity.HIGH)
        medium   = sum(1 for m in matches if m.severity == PIISeverity.MEDIUM)
        low      = sum(1 for m in matches if m.severity == PIISeverity.LOW)

        penalty = (critical * 25 + high * 15 + medium * 8 + low * 2)
        score = max(0.0, 100.0 - penalty)
        return round(score, 2)

    def _build_issues(self, matches: List[PIIMatch]) -> List[QualityIssue]:
        if not matches:
            return []
        by_type = self._group_by_type(matches)
        issues = []
        for pii_type_str, count in by_type.items():
            # Find severity from first match of this type
            sev_map = {PIISeverity.CRITICAL: Severity.CRITICAL,
                       PIISeverity.HIGH: Severity.HIGH,
                       PIISeverity.MEDIUM: Severity.MEDIUM,
                       PIISeverity.LOW: Severity.LOW}
            sample = next(m for m in matches if m.pii_type.value == pii_type_str)
            severity = sev_map.get(sample.severity, Severity.MEDIUM)
            issues.append(QualityIssue(
                checker="PIIDetector",
                issue_type=f"pii_{pii_type_str.lower().replace(' ','_')}",
                severity=severity,
                description=f"Found {count} {pii_type_str} instance(s) in text.",
                suggestion=f"Anonymize or remove {pii_type_str} before using in training.",
                score_impact=min(count * 10, 50),
            ))
        return issues

    def redact_text(self, text: str) -> str:
        """Return text with all detected PII replaced by mask."""
        result = self.check(text)
        redacted = text
        # Replace from end to start to preserve positions
        pii_list = sorted(result["pii_found"], key=lambda m: m.start_char, reverse=True)
        for match in pii_list:
            redacted = (
                redacted[:match.start_char] +
                f"[{match.pii_type.value.upper()}]" +
                redacted[match.end_char:]
            )
        return redacted


# ─────────────────────────────────────────────────────────────────────────────
# 5. COHERENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

class CoherenceScorer(BaseTextChecker):
    """
    Measures text coherence using:
      • Local coherence  : cosine similarity between adjacent sentence pairs
                          (using TF-IDF or sentence embeddings)
      • Global coherence : topic consistency via LDA-style vocabulary overlap
      • Discourse score  : density of discourse connectives (however, therefore...)
      • Lexical chain    : noun chain continuity across sentences
      • Topic drift      : detects abrupt topic changes
    """

    DISCOURSE_CONNECTIVES = {
        "addition":    {"furthermore", "moreover", "additionally", "also", "besides",
                        "in addition", "likewise", "similarly"},
        "contrast":    {"however", "nevertheless", "nonetheless", "although", "though",
                        "whereas", "on the other hand", "in contrast", "but", "yet"},
        "causation":   {"therefore", "thus", "hence", "consequently", "as a result",
                        "because", "since", "so", "accordingly"},
        "sequence":    {"first", "second", "third", "finally", "next", "then",
                        "subsequently", "afterwards", "lastly"},
        "example":     {"for example", "for instance", "such as", "namely",
                        "specifically", "in particular"},
        "conclusion":  {"in conclusion", "in summary", "to summarize", "overall",
                        "in short", "ultimately"},
    }

    def __init__(self, config: TextQualityConfig):
        super().__init__(config)
        self._vectorizer: Optional[Any] = None

    def check(self, text: str) -> Dict[str, Any]:
        sentences = self._tokenize_sentences(text)
        min_sents = self.config.coherence_min_sentences

        if len(sentences) < min_sents:
            return {
                "local_coherence": 1.0,
                "global_coherence": 1.0,
                "discourse_score": 1.0,
                "overall_coherence": 1.0,
                "score": 100.0,
                "issues": [],
                "problem_transitions": [],
                "note": f"Too few sentences ({len(sentences)}) for coherence analysis.",
            }

        local_coh, problem_trans = self._compute_local_coherence(sentences)
        global_coh               = self._compute_global_coherence(sentences)
        discourse_score          = self._compute_discourse_score(text, sentences)
        lexical_score            = self._compute_lexical_chain_score(sentences)

        overall = (
            local_coh * 0.40 +
            global_coh * 0.30 +
            discourse_score * 0.20 +
            lexical_score * 0.10
        )

        score   = round(overall * 100, 2)
        issues  = self._build_issues(score, problem_trans, discourse_score)

        return {
            "local_coherence": round(local_coh, 4),
            "global_coherence": round(global_coh, 4),
            "discourse_score": round(discourse_score, 4),
            "lexical_chain_score": round(lexical_score, 4),
            "overall_coherence": round(overall, 4),
            "score": score,
            "severity": self._score_to_severity(score).value,
            "problem_transitions": problem_trans,
            "issues": issues,
        }

    def _compute_local_coherence(
        self, sentences: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Cosine similarity between adjacent sentence pairs.
        Uses a sliding window of size coherence_window.
        """
        if not SKLEARN_AVAILABLE or len(sentences) < 2:
            return 1.0, []

        try:
            vectorizer = TfidfVectorizer(
                max_features=3000, ngram_range=(1, 2), min_df=1
            )
            tfidf = vectorizer.fit_transform(sentences)
        except Exception:
            return 1.0, []

        window = self.config.coherence_window
        scores = []
        problem_transitions = []
        threshold = self.config.topic_drift_threshold

        for i in range(len(sentences) - 1):
            j = min(i + 1, len(sentences) - 1)
            sim = cosine_similarity(tfidf[i], tfidf[j])[0][0]
            scores.append(float(sim))

            # Check for topic drift in window
            if i > 0 and len(scores) >= 2:
                prev_sim = scores[-2]
                drop = prev_sim - sim
                if drop > threshold:
                    problem_transitions.append({
                        "position": i,
                        "sentence_a": self._truncate(sentences[i], 80),
                        "sentence_b": self._truncate(sentences[j], 80),
                        "similarity_drop": round(drop, 3),
                        "similarity_score": round(float(sim), 3),
                    })

        return (sum(scores) / max(len(scores), 1), problem_transitions)

    def _compute_global_coherence(self, sentences: List[str]) -> float:
        """
        Global coherence: vocabulary overlap between first half and second half.
        A coherent text maintains similar vocabulary throughout.
        """
        if len(sentences) < 4:
            return 1.0

        mid = len(sentences) // 2
        first_half  = set(self._tokenize_words(" ".join(sentences[:mid])))
        second_half = set(self._tokenize_words(" ".join(sentences[mid:])))
        stopwords   = self._get_stopwords()

        first_half  -= stopwords
        second_half -= stopwords

        if not first_half or not second_half:
            return 1.0

        intersection = first_half & second_half
        union        = first_half | second_half
        jaccard      = len(intersection) / max(len(union), 1)

        # Scale: 0.1 Jaccard → 0.5 coherence, 0.3+ → good
        scaled = min(jaccard / 0.3, 1.0)
        return round(scaled, 4)

    def _compute_discourse_score(self, text: str, sentences: List[str]) -> float:
        """
        Score based on discourse connective density.
        A well-structured text uses connectives to link ideas.
        """
        text_lower = text.lower()
        all_connectives = set()
        for group in self.DISCOURSE_CONNECTIVES.values():
            all_connectives |= group

        found = sum(1 for c in all_connectives if c in text_lower)
        total_sents = max(len(sentences), 1)

        # Ideal: ~1 connective per 4 sentences
        ideal_rate = total_sents / 4
        rate       = found / ideal_rate if ideal_rate > 0 else 0

        # Score peaks at ideal_rate, penalizes both too few and too many
        score = 1.0 - abs(min(rate, 2.0) - 1.0) * 0.5
        return round(max(0.0, score), 4)

    def _compute_lexical_chain_score(self, sentences: List[str]) -> float:
        """
        Lexical chain: nouns that appear in adjacent sentences form chains.
        Higher chain continuity = better coherence.
        """
        stopwords = self._get_stopwords()

        # Extract content words per sentence
        sentence_words = []
        for sent in sentences:
            words = set(self._tokenize_words(sent)) - stopwords
            # Keep only alpha words of length > 3
            words = {w for w in words if w.isalpha() and len(w) > 3}
            sentence_words.append(words)

        if len(sentence_words) < 2:
            return 1.0

        chain_scores = []
        for i in range(len(sentence_words) - 1):
            a = sentence_words[i]
            b = sentence_words[i + 1]
            if not a or not b:
                chain_scores.append(0.0)
                continue
            overlap = len(a & b) / max(len(a | b), 1)
            chain_scores.append(overlap)

        avg = sum(chain_scores) / max(len(chain_scores), 1)
        # Normalize: 0.1 overlap is decent, scale to 0.5 coherence
        scaled = min(avg / 0.15, 1.0)
        return round(scaled, 4)

    def _build_issues(
        self,
        score: float,
        problem_transitions: List[Dict],
        discourse_score: float,
    ) -> List[QualityIssue]:
        issues = []

        if score < 60:
            issues.append(QualityIssue(
                checker="CoherenceScorer",
                issue_type="low_coherence",
                severity=Severity.HIGH if score < 40 else Severity.MEDIUM,
                description=f"Overall coherence score is {score:.1f}/100. "
                            f"Text lacks logical flow between ideas.",
                suggestion="Add discourse connectives and ensure topic continuity.",
                score_impact=60 - score,
            ))

        if problem_transitions:
            issues.append(QualityIssue(
                checker="CoherenceScorer",
                issue_type="topic_drift",
                severity=Severity.MEDIUM,
                description=f"Found {len(problem_transitions)} abrupt topic transition(s).",
                evidence=str([t["position"] for t in problem_transitions[:3]]),
                suggestion="Review transitions at detected positions for smoother flow.",
                score_impact=len(problem_transitions) * 5,
            ))

        if discourse_score < 0.3:
            issues.append(QualityIssue(
                checker="CoherenceScorer",
                issue_type="low_discourse_connectivity",
                severity=Severity.LOW,
                description="Very few discourse connectives found (however, therefore, etc.).",
                suggestion="Add logical connectives to improve text structure.",
                score_impact=10.0,
            ))

        return issues


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONSISTENCY CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyChecker(BaseTextChecker):
    """
    Detects internal inconsistencies:
      • NLI-based contradiction detection (transformers)
      • Numerical fact inconsistency
      • Tense consistency (POS tagging)
      • Negation detection
      • Entity alias detection
      • Date / time contradiction
    """

    # Negation words
    NEGATION_WORDS = {
        "not", "never", "no", "neither", "nor", "without",
        "hardly", "barely", "scarcely", "cannot", "can't",
        "won't", "isn't", "aren't", "wasn't", "weren't",
        "doesn't", "don't", "didn't", "hasn't", "haven't", "hadn't",
    }

    # Tense indicators
    PAST_TENSE_PATTERNS    = re.compile(r'\b(?:was|were|had|did|went|said|told|made)\b', re.I)
    PRESENT_TENSE_PATTERNS = re.compile(r'\b(?:is|are|has|does|goes|says|tells|makes)\b', re.I)
    FUTURE_TENSE_PATTERNS  = re.compile(r'\b(?:will|shall|going\s+to|would)\b', re.I)

    # Number extraction
    NUMBER_PATTERN = re.compile(
        r'\b(\d+(?:[.,]\d+)?)\s*'
        r'(%|percent|million|billion|thousand|hundred|'
        r'km|kg|lb|mph|kph|degrees?|°)?\b',
        re.IGNORECASE,
    )

    def __init__(self, config: TextQualityConfig):
        super().__init__(config)
        self._nli_pipeline = None
        self._nli_loaded   = False

    def _load_nli(self) -> bool:
        if self._nli_loaded:
            return self._nli_pipeline is not None
        self._nli_loaded = True
        if not TRANSFORMERS_AVAILABLE:
            return False
        try:
            self._nli_pipeline = hf_pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=-1,  # CPU
            )
            return True
        except Exception:
            try:
                self._nli_pipeline = hf_pipeline(
                    "text-classification",
                    model="typeform/distilbert-base-uncased-mnli",
                    device=-1,
                )
                return True
            except Exception:
                self._nli_pipeline = None
                return False

    def check(self, text: str) -> Dict[str, Any]:
        sentences = self._tokenize_sentences(text)

        result = ConsistencyResult(
            contradiction_pairs=[],
            numerical_inconsistencies=[],
            tense_inconsistencies=[],
            entity_inconsistencies=[],
            overall_consistency_score=100.0,
        )

        # Numerical consistency
        result.numerical_inconsistencies = self._check_numerical_consistency(
            text, sentences
        )

        # Tense consistency
        result.tense_inconsistencies = self._check_tense_consistency(sentences)

        # NLI contradictions (only if model available and sentences > 4)
        if len(sentences) >= 4:
            result.contradiction_pairs = self._check_nli_contradictions(sentences)

        # Negation patterns
        negation_issues = self._check_negation_patterns(sentences)

        # Compute score
        score = self._compute_score(result, negation_issues)
        result.overall_consistency_score = score
        issues = self._build_issues(result, negation_issues)

        return {
            "contradiction_pairs": result.contradiction_pairs,
            "numerical_inconsistencies": result.numerical_inconsistencies,
            "tense_inconsistencies": result.tense_inconsistencies,
            "negation_issues": negation_issues,
            "score": score,
            "severity": self._score_to_severity(score).value,
            "issues": issues,
        }

    def _check_numerical_consistency(
        self, text: str, sentences: List[str]
    ) -> List[Dict]:
        """
        Extract numbers in context and detect contradictions.
        Example: "The study had 100 participants" vs "50 participants were surveyed".
        """
        inconsistencies = []

        # Extract numbers with their context per sentence
        sentence_numbers: List[List[Tuple[float, str, str]]] = []
        for sent in sentences:
            nums = []
            for match in self.NUMBER_PATTERN.finditer(sent):
                try:
                    val = float(match.group(1).replace(',', ''))
                    unit = (match.group(2) or "").lower()
                    nums.append((val, unit, sent))
                except Exception:
                    pass
            sentence_numbers.append(nums)

        # Look for same unit appearing with very different values
        unit_values: Dict[str, List[Tuple[float, str, int]]] = defaultdict(list)
        for i, nums in enumerate(sentence_numbers):
            for val, unit, sent in nums:
                if unit:
                    unit_values[unit].append((val, sent, i))

        tol = self.config.numerical_tolerance
        for unit, entries in unit_values.items():
            if len(entries) < 2:
                continue
            values = [e[0] for e in entries]
            mean_val = sum(values) / len(values)
            if mean_val == 0:
                continue
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    a_val, a_sent, a_idx = entries[i]
                    b_val, b_sent, b_idx = entries[j]
                    relative_diff = abs(a_val - b_val) / max(abs(mean_val), 1e-9)
                    if relative_diff > 0.5 and a_val != b_val:
                        inconsistencies.append({
                            "unit": unit,
                            "value_a": a_val,
                            "value_b": b_val,
                            "sentence_a_idx": a_idx,
                            "sentence_b_idx": b_idx,
                            "sentence_a": self._truncate(a_sent, 100),
                            "sentence_b": self._truncate(b_sent, 100),
                            "relative_diff": round(relative_diff, 3),
                        })
        return inconsistencies[:10]

    def _check_tense_consistency(self, sentences: List[str]) -> List[Dict]:
        """
        Detect major tense switches that may indicate inconsistency.
        """
        inconsistencies = []
        if len(sentences) < 3:
            return []

        def detect_tense(sent: str) -> str:
            if self.FUTURE_TENSE_PATTERNS.search(sent):
                return "future"
            if self.PRESENT_TENSE_PATTERNS.search(sent):
                return "present"
            if self.PAST_TENSE_PATTERNS.search(sent):
                return "past"
            return "unknown"

        tenses = [detect_tense(s) for s in sentences]
        known  = [t for t in tenses if t != "unknown"]

        if len(known) < 3:
            return []

        # Find dominant tense
        tense_counts = Counter(known)
        dominant = tense_counts.most_common(1)[0][0]
        dominant_pct = tense_counts[dominant] / len(known)

        if dominant_pct < 0.6:
            # Mixed tenses
            for i, tense in enumerate(tenses):
                if tense != "unknown" and tense != dominant:
                    inconsistencies.append({
                        "sentence_idx": i,
                        "detected_tense": tense,
                        "dominant_tense": dominant,
                        "sentence": self._truncate(sentences[i], 100),
                    })

        return inconsistencies[:5]

    def _check_nli_contradictions(self, sentences: List[str]) -> List[Dict]:
        """
        Use NLI to detect contradictions between sentence pairs.
        Checks first/last sentences against all others (O(n) not O(n²)).
        """
        if not self._load_nli():
            return self._heuristic_contradictions(sentences)

        contradictions = []
        # Sample: check every 3rd sentence against its neighbors
        indices = list(range(0, len(sentences), 3))[:10]

        for i in indices:
            sent_a = sentences[i]
            if len(sent_a.split()) < 5:
                continue
            for j in range(max(0, i+1), min(len(sentences), i+5)):
                sent_b = sentences[j]
                if len(sent_b.split()) < 5:
                    continue
                try:
                    result = self._nli_pipeline(
                        f"{sent_a} [SEP] {sent_b}",
                        candidate_labels=["contradiction", "entailment", "neutral"],
                        multi_label=False,
                    )
                    if isinstance(result, dict) and "labels" in result:
                        top_label = result["labels"][0]
                        top_score = result["scores"][0]
                        if (top_label == "contradiction" and
                                top_score >= self.config.nli_contradiction_threshold):
                            contradictions.append({
                                "sentence_a_idx": i,
                                "sentence_b_idx": j,
                                "sentence_a": self._truncate(sent_a, 100),
                                "sentence_b": self._truncate(sent_b, 100),
                                "contradiction_score": round(top_score, 3),
                                "method": "NLI",
                            })
                except Exception:
                    continue

        return contradictions

    def _heuristic_contradictions(self, sentences: List[str]) -> List[Dict]:
        """
        Lightweight heuristic: detect negation flips between sentence pairs.
        E.g., "X is Y" vs "X is not Y".
        """
        contradictions = []
        neg_words = self.NEGATION_WORDS

        for i in range(len(sentences) - 1):
            for j in range(i + 1, min(i + 4, len(sentences))):
                words_a = set(self._tokenize_words(sentences[i]))
                words_b = set(self._tokenize_words(sentences[j]))
                content_a = words_a - self._get_stopwords()
                content_b = words_b - self._get_stopwords()
                overlap = content_a & content_b
                if len(overlap) < 3:
                    continue

                has_neg_a = bool(words_a & neg_words)
                has_neg_b = bool(words_b & neg_words)
                if has_neg_a != has_neg_b:
                    contradictions.append({
                        "sentence_a_idx": i,
                        "sentence_b_idx": j,
                        "sentence_a": self._truncate(sentences[i], 100),
                        "sentence_b": self._truncate(sentences[j], 100),
                        "contradiction_score": 0.6,
                        "method": "negation-heuristic",
                    })
        return contradictions[:5]

    def _check_negation_patterns(self, sentences: List[str]) -> List[Dict]:
        """
        Find suspicious negation-heavy sentences.
        """
        issues = []
        neg_words = self.NEGATION_WORDS
        for i, sent in enumerate(sentences):
            words = self._tokenize_words(sent)
            neg_count = sum(1 for w in words if w in neg_words)
            if neg_count >= 3:
                issues.append({
                    "sentence_idx": i,
                    "negation_count": neg_count,
                    "sentence": self._truncate(sent, 100),
                })
        return issues

    def _compute_score(
        self,
        result: ConsistencyResult,
        negation_issues: List[Dict],
    ) -> float:
        score = 100.0
        score -= len(result.contradiction_pairs) * 15
        score -= len(result.numerical_inconsistencies) * 10
        score -= len(result.tense_inconsistencies) * 5
        score -= len(negation_issues) * 3
        return round(max(0.0, min(100.0, score)), 2)

    def _build_issues(
        self,
        result: ConsistencyResult,
        negation_issues: List[Dict],
    ) -> List[QualityIssue]:
        issues = []

        if result.contradiction_pairs:
            issues.append(QualityIssue(
                checker="ConsistencyChecker",
                issue_type="nli_contradiction",
                severity=Severity.HIGH,
                description=f"Found {len(result.contradiction_pairs)} potential contradiction(s) "
                            f"between sentences.",
                suggestion="Review and resolve contradictory statements.",
                score_impact=len(result.contradiction_pairs) * 15,
            ))

        if result.numerical_inconsistencies:
            issues.append(QualityIssue(
                checker="ConsistencyChecker",
                issue_type="numerical_inconsistency",
                severity=Severity.MEDIUM,
                description=f"Found {len(result.numerical_inconsistencies)} numerical "
                            f"inconsistency(ies) (same unit, very different values).",
                suggestion="Verify all numerical facts for consistency.",
                score_impact=len(result.numerical_inconsistencies) * 10,
            ))

        if result.tense_inconsistencies:
            issues.append(QualityIssue(
                checker="ConsistencyChecker",
                issue_type="tense_inconsistency",
                severity=Severity.LOW,
                description=f"Found {len(result.tense_inconsistencies)} tense switch(es). "
                            f"Text mixes past/present/future tenses.",
                suggestion="Standardize tense usage throughout the text.",
                score_impact=len(result.tense_inconsistencies) * 5,
            ))

        return issues


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENCODING CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class EncodingChecker(BaseTextChecker):
    """
    Detects text encoding issues:
      • Auto-detect encoding via chardet
      • Mojibake detection (garbled UTF-8 decoded as Latin-1)
      • Unicode NFC / NFD normalization issues
      • BOM (Byte Order Mark) detection
      • Null byte detection
      • Non-printable character ratio
      • Replacement character (U+FFFD) detection
    """

    # Common mojibake sequences (UTF-8 bytes read as Latin-1)
    MOJIBAKE_INDICATORS = [
        'Ã©', 'Ã¨', 'Ã ', 'Ã¢', 'Ã§', 'Ã¹', 'Ã»', 'Ã®', 'Ã¯',  # French
        'Ã¤', 'Ã¶', 'Ã¼', 'Ã', 'â€™', 'â€œ', 'â€', 'â€"', 'â€"',  # German / Smart quotes
        'Ð', 'Ñ', 'ÑÐ',  # Cyrillic
        'Ø', 'Ù', 'Ú', 'Û',  # Arabic
        '\ufffd',  # Unicode replacement char
    ]

    # BOM markers
    BOM_MARKERS = {
        b'\xef\xbb\xbf': 'UTF-8 BOM',
        b'\xff\xfe':      'UTF-16 LE BOM',
        b'\xfe\xff':      'UTF-16 BE BOM',
        b'\xff\xfe\x00\x00': 'UTF-32 LE BOM',
        b'\x00\x00\xfe\xff': 'UTF-32 BE BOM',
    }

    def check(self, text: str) -> Dict[str, Any]:
        if not text:
            return EncodingResult(
                detected_encoding="unknown",
                confidence=0.0,
                has_bom=False,
                has_null_bytes=False,
                mojibake_ratio=0.0,
                non_printable_ratio=0.0,
                unicode_issues=[],
                is_clean=True,
                encoding_score=100.0,
            ).__dict__

        raw = text.encode('utf-8', errors='replace')
        result = self._analyze(text, raw)
        issues = self._build_issues(result)
        return {**result.__dict__, "issues": issues}

    def _analyze(self, text: str, raw: bytes) -> EncodingResult:
        # 1. Chardet encoding detection
        detected_enc = "utf-8"
        confidence   = 1.0
        if CHARDET_AVAILABLE:
            try:
                det = chardet.detect(raw[:50000])
                detected_enc = det.get("encoding") or "utf-8"
                confidence   = det.get("confidence") or 0.0
            except Exception:
                pass

        # 2. BOM detection
        has_bom = False
        for bom, _ in self.BOM_MARKERS.items():
            if raw.startswith(bom):
                has_bom = True
                break

        # 3. Null bytes
        has_null_bytes = b'\x00' in raw

        # 4. Mojibake detection
        mojibake_char_count = 0
        for indicator in self.MOJIBAKE_INDICATORS:
            mojibake_char_count += text.count(indicator)
        mojibake_ratio = min(
            mojibake_char_count / max(len(text), 1), 1.0
        )

        # 5. Non-printable character ratio
        non_printable = sum(
            1 for c in text
            if not c.isprintable() and c not in '\n\t\r'
        )
        non_printable_ratio = non_printable / max(len(text), 1)

        # 6. Unicode normalization issues
        unicode_issues = self._detect_unicode_issues(text)

        # 7. Compute score
        score = self._compute_encoding_score(
            confidence, mojibake_ratio, non_printable_ratio,
            has_bom, has_null_bytes, unicode_issues
        )

        is_clean = (
            mojibake_ratio < self.config.mojibake_ratio_threshold and
            not has_null_bytes and
            non_printable_ratio < 0.01 and
            len(unicode_issues) == 0
        )

        return EncodingResult(
            detected_encoding=detected_enc,
            confidence=round(confidence, 3),
            has_bom=has_bom,
            has_null_bytes=has_null_bytes,
            mojibake_ratio=round(mojibake_ratio, 4),
            non_printable_ratio=round(non_printable_ratio, 4),
            unicode_issues=unicode_issues,
            is_clean=is_clean,
            encoding_score=round(score, 2),
        )

    def _detect_unicode_issues(self, text: str) -> List[str]:
        """Check for Unicode normalization inconsistencies."""
        issues = []

        # Check NFC vs NFD
        nfc_text = unicodedata.normalize('NFC', text)
        if nfc_text != text:
            issues.append("Text is not NFC-normalized (may cause comparison issues)")

        # Replacement characters
        if '\ufffd' in text:
            count = text.count('\ufffd')
            issues.append(f"Found {count} Unicode replacement character(s) (U+FFFD)")

        # Zero-width characters
        zero_width = ['\u200b', '\u200c', '\u200d', '\ufeff', '\u00ad']
        for zw in zero_width:
            if zw in text:
                issues.append(f"Found zero-width/soft character: U+{ord(zw):04X}")

        # Mixed scripts (basic check)
        scripts_found = set()
        for char in text[:1000]:  # sample first 1000 chars
            name = unicodedata.name(char, '')
            if 'ARABIC' in name:
                scripts_found.add('Arabic')
            elif 'CYRILLIC' in name:
                scripts_found.add('Cyrillic')
            elif 'LATIN' in name:
                scripts_found.add('Latin')
            elif 'CJK' in name or 'CHINESE' in name:
                scripts_found.add('CJK')

        if len(scripts_found) > 2:
            issues.append(f"Mixed scripts detected: {', '.join(scripts_found)}")

        return issues

    def _compute_encoding_score(
        self,
        confidence: float,
        mojibake_ratio: float,
        non_printable_ratio: float,
        has_bom: bool,
        has_null_bytes: bool,
        unicode_issues: List[str],
    ) -> float:
        score = 100.0

        # Low chardet confidence
        if confidence < self.config.encoding_confidence_threshold:
            score -= (1 - confidence) * 20

        # Mojibake penalty
        if mojibake_ratio > self.config.mojibake_ratio_threshold:
            score -= (mojibake_ratio / self.config.mojibake_ratio_threshold) * 30

        # Non-printable chars
        score -= non_printable_ratio * 200

        # BOM present
        if has_bom:
            score -= 5

        # Null bytes
        if has_null_bytes:
            score -= 20

        # Unicode issues
        score -= len(unicode_issues) * 5

        return max(0.0, min(100.0, score))

    def _build_issues(self, result: EncodingResult) -> List[QualityIssue]:
        issues = []

        if result.mojibake_ratio > self.config.mojibake_ratio_threshold:
            issues.append(QualityIssue(
                checker="EncodingChecker",
                issue_type="mojibake",
                severity=Severity.HIGH,
                description=f"Mojibake detected ({result.mojibake_ratio*100:.2f}% of text). "
                            f"Text may have been decoded with wrong encoding.",
                suggestion="Re-decode source bytes with the correct encoding (likely UTF-8).",
                score_impact=30.0,
            ))

        if result.has_null_bytes:
            issues.append(QualityIssue(
                checker="EncodingChecker",
                issue_type="null_bytes",
                severity=Severity.HIGH,
                description="Null bytes (\\x00) found in text.",
                suggestion="Strip null bytes before processing.",
                score_impact=20.0,
            ))

        if result.has_bom:
            issues.append(QualityIssue(
                checker="EncodingChecker",
                issue_type="bom_present",
                severity=Severity.LOW,
                description="Byte Order Mark (BOM) found at start of text.",
                suggestion="Strip the BOM before processing.",
                score_impact=5.0,
            ))

        if result.non_printable_ratio > 0.01:
            issues.append(QualityIssue(
                checker="EncodingChecker",
                issue_type="non_printable_chars",
                severity=Severity.MEDIUM,
                description=f"Non-printable character ratio: {result.non_printable_ratio*100:.2f}%.",
                suggestion="Remove or replace non-printable characters.",
                score_impact=10.0,
            ))

        for issue_text in result.unicode_issues:
            issues.append(QualityIssue(
                checker="EncodingChecker",
                issue_type="unicode_issue",
                severity=Severity.LOW,
                description=issue_text,
                suggestion="Normalize text to NFC and remove zero-width characters.",
                score_impact=5.0,
            ))

        return issues

    def fix_encoding(self, text: str) -> str:
        """
        Attempt to fix common encoding issues:
          1. Strip BOM
          2. NFC normalize
          3. Remove null bytes and non-printable chars
        """
        # Strip BOM
        for bom_bytes, _ in self.BOM_MARKERS.items():
            bom_str = bom_bytes.decode('utf-8', errors='replace')
            if text.startswith(bom_str):
                text = text[len(bom_str):]
                break

        # NFC normalize
        text = unicodedata.normalize('NFC', text)

        # Remove null bytes
        text = text.replace('\x00', '')

        # Remove non-printable chars (keep \n, \t, \r)
        text = ''.join(
            c for c in text
            if c.isprintable() or c in '\n\t\r'
        )

        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# 8. LANGUAGE QUALITY SCORER
# ─────────────────────────────────────────────────────────────────────────────

class LanguageQualityScorer(BaseTextChecker):
    """
    Measures language quality using:
      • Spelling error rate (pyspellchecker)
      • Grammar score estimate (pattern-based heuristics)
      • TTR — Type-Token Ratio (vocabulary richness, simple)
      • MTLD — Measure of Textual Lexical Diversity (length-robust)
      • HD-D — Hypergeometric Distribution D (probabilistic vocabulary)
      • Formality score (formal vs informal vocabulary)
      • Passive voice ratio
      • Average / variance of sentence length
    """

    # Informal / colloquial words (lower formality)
    INFORMAL_WORDS = {
        "gonna", "wanna", "gotta", "kinda", "sorta", "lotta",
        "hafta", "shoulda", "coulda", "woulda", "dunno", "lemme",
        "gimme", "ain't", "yeah", "nope", "yep", "hey", "ok",
        "okay", "cool", "awesome", "stuff", "thing", "guy", "guys",
        "like", "just", "really", "very", "so", "totally", "literally",
    }

    # Formal words (boost formality)
    FORMAL_WORDS = {
        "therefore", "however", "moreover", "furthermore", "consequently",
        "nevertheless", "notwithstanding", "subsequently", "henceforth",
        "aforementioned", "hereinafter", "pursuant", "whereby", "thereof",
        "inasmuch", "insofar", "accordingly", "hitherto", "heretofore",
    }

    # Passive voice indicators
    PASSIVE_PATTERN = re.compile(
        r'\b(?:is|are|was|were|be|been|being)\s+'
        r'(?:\w+\s+)?'
        r'(?:\w+ed|built|done|made|known|seen|found|used|given|shown|'
        r'taken|kept|put|set|told|written|spoken|chosen|driven)\b',
        re.IGNORECASE,
    )

    def __init__(self, config: TextQualityConfig):
        super().__init__(config)
        self._spellchecker = None
        self._spell_loaded = False

    def _load_spellchecker(self) -> bool:
        if self._spell_loaded:
            return self._spellchecker is not None
        self._spell_loaded = True
        if not SPELLCHECKER_AVAILABLE:
            return False
        try:
            self._spellchecker = SpellChecker()
            return True
        except Exception:
            return False

    def check(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 20:
            return self._minimal_result()

        words     = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)
        alpha_words = [w for w in words if w.isalpha()]

        # Core metrics
        spelling_rate, spelling_errors = self._check_spelling(alpha_words)
        ttr    = self._compute_ttr(alpha_words)
        mtld   = self._compute_mtld(alpha_words)
        hdd    = self._compute_hdd(alpha_words)
        formal = self._compute_formality(alpha_words)
        passive_ratio = self._compute_passive_ratio(text, sentences)
        avg_len, len_var = self._compute_sentence_length_stats(sentences)
        grammar_score = self._estimate_grammar_score(text, sentences, spelling_rate)

        # Composite score
        overall = self._compute_language_score(
            spelling_rate, ttr, mtld, formal, passive_ratio, grammar_score
        )

        result = LanguageQualityResult(
            spelling_error_rate=round(spelling_rate, 4),
            spelling_errors=spelling_errors[:20],
            estimated_grammar_score=round(grammar_score, 2),
            ttr=round(ttr, 4),
            mtld=round(mtld, 2),
            hdd=round(hdd, 4),
            formality_score=round(formal, 4),
            passive_voice_ratio=round(passive_ratio, 4),
            avg_sentence_length=round(avg_len, 2),
            sentence_length_variance=round(len_var, 2),
            overall_language_score=round(overall, 2),
        )

        issues = self._build_issues(result)

        return {
            **result.__dict__,
            "score": overall,
            "severity": self._score_to_severity(overall).value,
            "issues": issues,
        }

    # ── Spelling ──────────────────────────────────────────────────────────────

    def _check_spelling(
        self, words: List[str]
    ) -> Tuple[float, List[str]]:
        if not self._load_spellchecker() or not words:
            return 0.0, []

        # Filter: keep only reasonably-sized alpha words
        check_words = [
            w for w in words
            if 2 < len(w) < 20
        ]
        if not check_words:
            return 0.0, []

        try:
            misspelled = self._spellchecker.unknown(check_words)
            rate = len(misspelled) / max(len(check_words), 1)
            return rate, list(misspelled)[:20]
        except Exception:
            return 0.0, []

    # ── TTR (Type-Token Ratio) ────────────────────────────────────────────────

    def _compute_ttr(self, words: List[str]) -> float:
        """Simple Type-Token Ratio. Biased by length — use MTLD for accuracy."""
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    # ── MTLD ─────────────────────────────────────────────────────────────────

    def _compute_mtld(self, words: List[str], threshold: float = 0.72) -> float:
        """
        Measure of Textual Lexical Diversity.
        Not biased by text length unlike TTR.
        Computes average factor length where TTR stays above threshold.
        """
        if len(words) < 10:
            return 0.0

        def _mtld_pass(word_list: List[str]) -> float:
            factors = 0.0
            types: set = set()
            tokens = 0

            for word in word_list:
                types.add(word)
                tokens += 1
                ttr = len(types) / tokens
                if ttr <= threshold:
                    factors += 1
                    types = set()
                    tokens = 0

            # Partial factor at end
            if tokens > 0:
                ttr = len(types) / tokens
                partial = (1 - ttr) / (1 - threshold)
                factors += partial

            return len(word_list) / max(factors, 1)

        forward_mtld  = _mtld_pass(words)
        backward_mtld = _mtld_pass(list(reversed(words)))
        return (forward_mtld + backward_mtld) / 2

    # ── HD-D ─────────────────────────────────────────────────────────────────

    def _compute_hdd(self, words: List[str], sample_size: int = 42) -> float:
        """
        Hypergeometric Distribution D.
        Probability that a random sample of 42 words includes a given type.
        More robust than TTR for short texts.
        """
        n = len(words)
        if n < sample_size:
            return self._compute_ttr(words)

        type_counts = Counter(words)
        hdd = 0.0

        for word_type, count in type_counts.items():
            # P(type in sample) = 1 - P(type not in sample)
            # P(not in sample) = C(n-count, sample) / C(n, sample)
            # Approximation using hypergeometric
            p_not_in = 1.0
            for i in range(sample_size):
                p_not_in *= (n - count - i) / max(n - i, 1)
            hdd += (1 - p_not_in)

        return hdd / max(len(type_counts), 1)

    # ── Formality ─────────────────────────────────────────────────────────────

    def _compute_formality(self, words: List[str]) -> float:
        """
        Formality score 0-1 based on informal vs formal word ratio.
        """
        if not words:
            return 0.5

        word_set   = set(words)
        informal_count = len(word_set & self.INFORMAL_WORDS)
        formal_count   = len(word_set & self.FORMAL_WORDS)
        total      = len(words)

        informal_rate = informal_count / max(total, 1)
        formal_rate   = formal_count   / max(total, 1)

        # Score: 0.5 base, +formal, -informal
        score = 0.5 + formal_rate * 5 - informal_rate * 5
        return max(0.0, min(1.0, score))

    # ── Passive Voice ─────────────────────────────────────────────────────────

    def _compute_passive_ratio(self, text: str, sentences: List[str]) -> float:
        """Fraction of sentences containing passive constructions."""
        if not sentences:
            return 0.0
        passive_count = sum(
            1 for s in sentences
            if self.PASSIVE_PATTERN.search(s)
        )
        return passive_count / len(sentences)

    # ── Sentence Length Stats ─────────────────────────────────────────────────

    def _compute_sentence_length_stats(
        self, sentences: List[str]
    ) -> Tuple[float, float]:
        if not sentences:
            return 0.0, 0.0
        lengths = [len(s.split()) for s in sentences]
        avg = sum(lengths) / len(lengths)
        variance = (
            sum((l - avg) ** 2 for l in lengths) / max(len(lengths), 1)
        )
        return avg, variance

    # ── Grammar Score Estimate ────────────────────────────────────────────────

    def _estimate_grammar_score(
        self,
        text: str,
        sentences: List[str],
        spelling_rate: float,
    ) -> float:
        """
        Grammar score via heuristics:
          - Fragment detection (very short sentences)
          - Run-on detection (very long sentences)
          - Double punctuation
          - Missing capitalization
          - Repeated words
        """
        score = 100.0
        if not sentences:
            return score

        for sent in sentences:
            words_in_sent = sent.split()

            # Fragment: < 3 words
            if 0 < len(words_in_sent) < 3:
                score -= 2

            # Run-on: > 60 words
            if len(words_in_sent) > 60:
                score -= 3

            # Missing capitalization at sentence start
            if words_in_sent and words_in_sent[0][0].islower():
                score -= 2

            # Repeated consecutive words
            for i in range(len(words_in_sent) - 1):
                if words_in_sent[i].lower() == words_in_sent[i+1].lower():
                    score -= 3

        # Double punctuation
        double_punct = len(re.findall(r'[.!?]{2,}', text))
        score -= double_punct * 2

        # Spelling penalty
        score -= spelling_rate * 30

        return round(max(0.0, min(100.0, score)), 2)

    # ── Composite Language Score ──────────────────────────────────────────────

    def _compute_language_score(
        self,
        spelling_rate: float,
        ttr: float,
        mtld: float,
        formality: float,
        passive_ratio: float,
        grammar_score: float,
    ) -> float:
        score = 0.0

        # Spelling (30%)
        spell_score = max(0, 100 - spelling_rate * 500)
        score += spell_score * 0.30

        # Grammar (30%)
        score += grammar_score * 0.30

        # Vocabulary richness via MTLD (25%)
        # MTLD: 0-40 poor, 40-80 ok, 80+ good
        mtld_score = min(mtld / 80 * 100, 100)
        score += mtld_score * 0.25

        # Formality (10%)
        # Formality itself isn't bad, but very informal = low quality for ML
        formal_score = formality * 100
        score += formal_score * 0.10

        # Passive voice penalty (5%)
        passive_score = max(0, 100 - passive_ratio * 100)
        if passive_ratio > self.config.passive_voice_threshold:
            passive_score -= 20
        score += max(0, passive_score) * 0.05

        return round(max(0.0, min(100.0, score)), 2)

    def _build_issues(self, result: LanguageQualityResult) -> List[QualityIssue]:
        issues = []

        if result.spelling_error_rate > self.config.max_spelling_error_rate:
            issues.append(QualityIssue(
                checker="LanguageQualityScorer",
                issue_type="high_spelling_error_rate",
                severity=Severity.HIGH if result.spelling_error_rate > 0.1 else Severity.MEDIUM,
                description=f"Spelling error rate is {result.spelling_error_rate*100:.1f}%. "
                            f"Errors: {', '.join(result.spelling_errors[:5])}",
                suggestion="Run spell-check and fix misspelled words.",
                score_impact=result.spelling_error_rate * 100,
            ))

        if result.mtld < self.config.min_mtld:
            issues.append(QualityIssue(
                checker="LanguageQualityScorer",
                issue_type="low_vocabulary_diversity",
                severity=Severity.MEDIUM,
                description=f"MTLD score is {result.mtld:.1f} (minimum: {self.config.min_mtld}). "
                            f"Text has low lexical diversity.",
                suggestion="Expand vocabulary; avoid over-repetition of words.",
                score_impact=30.0,
            ))

        if result.passive_voice_ratio > self.config.passive_voice_threshold:
            issues.append(QualityIssue(
                checker="LanguageQualityScorer",
                issue_type="excessive_passive_voice",
                severity=Severity.LOW,
                description=f"Passive voice ratio is {result.passive_voice_ratio*100:.1f}% "
                            f"(threshold: {self.config.passive_voice_threshold*100:.0f}%).",
                suggestion="Consider converting passive constructions to active voice.",
                score_impact=10.0,
            ))

        if result.avg_sentence_length > 35:
            issues.append(QualityIssue(
                checker="LanguageQualityScorer",
                issue_type="long_sentences",
                severity=Severity.LOW,
                description=f"Average sentence length is {result.avg_sentence_length:.1f} words. "
                            f"Very long sentences reduce readability.",
                suggestion="Break long sentences into shorter, clearer ones.",
                score_impact=10.0,
            ))

        if result.ttr < self.config.min_ttr:
            issues.append(QualityIssue(
                checker="LanguageQualityScorer",
                issue_type="low_ttr",
                severity=Severity.LOW,
                description=f"Type-Token Ratio is {result.ttr:.3f} (minimum: {self.config.min_ttr}). "
                            f"High word repetition detected.",
                suggestion="Reduce repetition of the same words.",
                score_impact=15.0,
            ))

        return issues

    def _minimal_result(self) -> Dict[str, Any]:
        return {
            "spelling_error_rate": 0.0,
            "spelling_errors": [],
            "estimated_grammar_score": 100.0,
            "ttr": 0.0, "mtld": 0.0, "hdd": 0.0,
            "formality_score": 0.5,
            "passive_voice_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_length_variance": 0.0,
            "overall_language_score": 100.0,
            "score": 100.0,
            "severity": "OK",
            "issues": [],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 9. TEXT QUALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

class TextQualityScore:
    """
    Unified 0-100 quality score that combines all sub-checker scores.
    Weights are configurable via TextQualityConfig.

    Score breakdown:
      Duplicate Score   (20%) — penalizes repeated content
      Noise Score       (20%) — penalizes HTML, special chars, junk
      PII Score         (15%) — penalizes sensitive data
      Coherence Score   (20%) — penalizes incoherent flow
      Language Score    (15%) — spelling, grammar, vocabulary
      Encoding Score    (10%) — encoding issues
    """

    VERDICT_MAP = {
        (90, 101): "Excellent",
        (75,  90): "Good",
        (55,  75): "Needs Work",
        (35,  55): "Poor",
        (0,   35): "Very Poor",
    }

    def __init__(self, config: TextQualityConfig):
        self.config = config

    def compute(
        self,
        duplicate_score: float,
        noise_score: float,
        pii_score: float,
        coherence_score: float,
        language_score: float,
        encoding_score: float,
        empty_score: float = 100.0,
    ) -> Dict[str, Any]:

        cfg = self.config

        weighted_score = (
            duplicate_score  * cfg.weight_duplicate +
            noise_score      * cfg.weight_noise +
            pii_score        * cfg.weight_pii +
            coherence_score  * cfg.weight_coherence +
            language_score   * cfg.weight_language +
            encoding_score   * cfg.weight_encoding
        )

        # Empty content is a hard override
        if empty_score < 20:
            weighted_score = min(weighted_score, 20.0)

        overall = round(weighted_score, 2)
        verdict = self._get_verdict(overall)
        severity = self._get_severity(overall)

        return {
            "overall_score": overall,
            "verdict": verdict,
            "severity": severity,
            "breakdown": {
                "duplicate_score":   round(duplicate_score, 2),
                "noise_score":       round(noise_score, 2),
                "pii_score":         round(pii_score, 2),
                "coherence_score":   round(coherence_score, 2),
                "language_score":    round(language_score, 2),
                "encoding_score":    round(encoding_score, 2),
                "empty_score":       round(empty_score, 2),
            },
            "weights": {
                "duplicate":  cfg.weight_duplicate,
                "noise":      cfg.weight_noise,
                "pii":        cfg.weight_pii,
                "coherence":  cfg.weight_coherence,
                "language":   cfg.weight_language,
                "encoding":   cfg.weight_encoding,
            },
            "priority_issues": self._get_priority_issues({
                "Duplicate":  duplicate_score,
                "Noise":      noise_score,
                "PII":        pii_score,
                "Coherence":  coherence_score,
                "Language":   language_score,
                "Encoding":   encoding_score,
            }),
        }

    def _get_verdict(self, score: float) -> str:
        for (low, high), verdict in self.VERDICT_MAP.items():
            if low <= score < high:
                return verdict
        return "Very Poor"

    def _get_severity(self, score: float) -> str:
        if score >= 75:
            return Severity.OK.value
        elif score >= 55:
            return Severity.LOW.value
        elif score >= 35:
            return Severity.MEDIUM.value
        elif score >= 20:
            return Severity.HIGH.value
        else:
            return Severity.CRITICAL.value

    def _get_priority_issues(self, scores: Dict[str, float]) -> List[Dict]:
        """Return checkers sorted by lowest score (worst first)."""
        sorted_issues = sorted(scores.items(), key=lambda x: x[1])
        return [
            {
                "checker": name,
                "score": round(score, 2),
                "priority": i + 1,
                "action": "fix_immediately" if score < 40 else
                          "review" if score < 70 else "ok",
            }
            for i, (name, score) in enumerate(sorted_issues)
            if score < 80
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 10. TEXT QUALITY REPORTER
# ─────────────────────────────────────────────────────────────────────────────

class TextQualityReporter:
    """
    Generates full quality reports in:
      • Markdown  (default)
      • JSON
      • HTML      (with inline styling)

    Report includes:
      - Executive summary + score
      - Per-dimension breakdown table
      - All detected issues sorted by severity
      - Recommendations sorted by priority
      - PII summary (if found)
      - Example problem text (configurable)
    """

    SEVERITY_EMOJI = {
        "Critical": "🔴",
        "High":     "🟠",
        "Medium":   "🟡",
        "Low":      "🔵",
        "OK":       "✅",
    }

    VERDICT_EMOJI = {
        "Excellent":  "🏆",
        "Good":       "✅",
        "Needs Work": "⚠️",
        "Poor":       "❌",
        "Very Poor":  "💀",
    }

    def __init__(self, config: TextQualityConfig):
        self.config = config

    def generate(
        self,
        report: Dict[str, Any],
        output_format: str = "markdown",
    ) -> str:
        fmt = output_format.lower()
        if fmt == "json":
            return self._to_json(report)
        elif fmt == "html":
            return self._to_html(report)
        else:
            return self._to_markdown(report)

    def save(self, report: Dict[str, Any], output_path: str) -> str:
        ext = Path(output_path).suffix.lower()
        fmt_map = {".json": "json", ".html": "html", ".md": "markdown"}
        fmt = fmt_map.get(ext, "markdown")
        content = self.generate(report, fmt)
        Path(output_path).write_text(content, encoding="utf-8")
        return output_path

    # ── Markdown ──────────────────────────────────────────────────────────────

    def _to_markdown(self, report: Dict[str, Any]) -> str:
        lines = []
        score_info = report.get("score_info", {})
        overall    = score_info.get("overall_score", 0)
        verdict    = score_info.get("verdict", "Unknown")
        breakdown  = score_info.get("breakdown", {})
        all_issues = report.get("all_issues", [])

        verdict_icon = self.VERDICT_EMOJI.get(verdict, "❓")

        lines.append("# 📝 Text Quality Report — Nydra v0.6.0")
        lines.append("")
        lines.append(f"**Overall Score:** `{overall}/100`  ")
        lines.append(f"**Verdict:** {verdict_icon} **{verdict}**  ")
        lines.append(f"**Text Length:** {report.get('text_length', 0):,} chars  ")
        lines.append(f"**Words:** {report.get('word_count', 0):,}  ")
        lines.append(f"**Sentences:** {report.get('sentence_count', 0):,}  ")
        lines.append("")

        # Score breakdown table
        lines.append("## 📊 Score Breakdown")
        lines.append("")
        lines.append("| Dimension | Score | Status | Weight |")
        lines.append("|-----------|-------|--------|--------|")

        weight_map = score_info.get("weights", {})
        dim_map = {
            "duplicate_score": ("Duplicate", "duplicate"),
            "noise_score":     ("Noise",     "noise"),
            "pii_score":       ("PII",       "pii"),
            "coherence_score": ("Coherence", "coherence"),
            "language_score":  ("Language",  "language"),
            "encoding_score":  ("Encoding",  "encoding"),
        }
        for key, (label, wkey) in dim_map.items():
            s = breakdown.get(key, 100)
            status = self._score_status(s)
            weight = weight_map.get(wkey, 0)
            lines.append(f"| {label} | {s:.1f} | {status} | {weight*100:.0f}% |")

        lines.append("")

        # Priority issues
        priority = score_info.get("priority_issues", [])
        if priority:
            lines.append("## 🎯 Fix Priority List")
            lines.append("")
            for item in priority:
                action_icon = "🔥" if item["action"] == "fix_immediately" else "👀"
                lines.append(
                    f"**{item['priority']}.** {action_icon} **{item['checker']}** "
                    f"— Score: `{item['score']}/100`"
                )
            lines.append("")

        # Issues
        if all_issues:
            lines.append("## 🔍 Issues Found")
            lines.append("")
            sorted_issues = sorted(
                all_issues,
                key=lambda x: ["Critical","High","Medium","Low","OK"].index(
                    x.severity.value if hasattr(x.severity, "value") else x.severity
                )
            )
            for issue in sorted_issues:
                sev_val = issue.severity.value if hasattr(issue.severity, "value") else issue.severity
                icon = self.SEVERITY_EMOJI.get(sev_val, "❓")
                lines.append(f"### {icon} [{sev_val}] {issue.issue_type.replace('_', ' ').title()}")
                lines.append(f"> {issue.description}")
                if issue.evidence and self.config.include_examples:
                    lines.append(f"> **Evidence:** `{issue.evidence[:150]}`")
                if issue.suggestion:
                    lines.append(f"> **Fix:** {issue.suggestion}")
                lines.append("")

        # PII section
        pii_info = report.get("pii_info", {})
        pii_count = pii_info.get("pii_count", 0)
        if pii_count > 0:
            lines.append("## 🔐 PII Summary")
            lines.append("")
            lines.append(f"**Total PII found:** {pii_count}")
            lines.append(f"**Critical:** {pii_info.get('critical_count', 0)}")
            lines.append(f"**High:** {pii_info.get('high_count', 0)}")
            by_type = pii_info.get("pii_by_type", {})
            if by_type:
                lines.append("")
                lines.append("| PII Type | Count |")
                lines.append("|----------|-------|")
                for pii_type, count in sorted(by_type.items()):
                    lines.append(f"| {pii_type} | {count} |")
            lines.append("")

        # Recommendations
        lines.append("## 💡 Recommendations")
        lines.append("")
        recommendations = self._build_recommendations(score_info, all_issues)
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
        lines.append("---")
        lines.append(f"*Generated by Nydra v0.6.0 — Text Quality Module*")

        return "\n".join(lines)

    def _score_status(self, score: float) -> str:
        if score >= 80:
            return "✅ Good"
        elif score >= 60:
            return "⚠️ OK"
        elif score >= 40:
            return "🟠 Poor"
        else:
            return "🔴 Critical"

    def _build_recommendations(
        self, score_info: Dict, issues: List
    ) -> List[str]:
        recs = []
        breakdown = score_info.get("breakdown", {})

        if breakdown.get("duplicate_score", 100) < 70:
            recs.append("Deduplicate content at sentence and paragraph level "
                        "before training.")
        if breakdown.get("noise_score", 100) < 70:
            recs.append("Strip HTML tags, URLs, and special characters. "
                        "Apply a text cleaning pipeline.")
        if breakdown.get("pii_score", 100) < 90:
            recs.append("Anonymize all detected PII before releasing dataset.")
        if breakdown.get("coherence_score", 100) < 70:
            recs.append("Review incoherent sections and add discourse connectives.")
        if breakdown.get("language_score", 100) < 70:
            recs.append("Run spell-check and grammar correction. "
                        "Improve vocabulary diversity.")
        if breakdown.get("encoding_score", 100) < 80:
            recs.append("Fix encoding issues: normalize to UTF-8 NFC and "
                        "remove non-printable characters.")
        if breakdown.get("empty_score", 100) < 80:
            recs.append("Remove empty or near-empty text entries from dataset.")

        if not recs:
            recs.append("Text quality is good. Continue monitoring as dataset grows.")

        return recs

    # ── JSON ──────────────────────────────────────────────────────────────────

    def _to_json(self, report: Dict[str, Any]) -> str:
        def _serialize(obj):
            if isinstance(obj, (QualityIssue, PIIMatch, DuplicateResult)):
                d = asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
                for k, v in d.items():
                    if hasattr(v, "value"):
                        d[k] = v.value
                return d
            if hasattr(obj, "value"):
                return obj.value
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        return json.dumps(report, default=_serialize, indent=2, ensure_ascii=False)

    # ── HTML ──────────────────────────────────────────────────────────────────

    def _to_html(self, report: Dict[str, Any]) -> str:
        score_info = report.get("score_info", {})
        overall    = score_info.get("overall_score", 0)
        verdict    = score_info.get("verdict", "Unknown")
        breakdown  = score_info.get("breakdown", {})
        all_issues = report.get("all_issues", [])

        color_map = {"Excellent": "#27ae60", "Good": "#2ecc71",
                     "Needs Work": "#f39c12", "Poor": "#e74c3c",
                     "Very Poor": "#8e1111"}
        verdict_color = color_map.get(verdict, "#95a5a6")

        rows = ""
        for key, label in [
            ("duplicate_score", "Duplicate"), ("noise_score", "Noise"),
            ("pii_score", "PII"), ("coherence_score", "Coherence"),
            ("language_score", "Language"), ("encoding_score", "Encoding"),
        ]:
            s = breakdown.get(key, 100)
            bar_color = "#27ae60" if s >= 80 else "#f39c12" if s >= 60 else "#e74c3c"
            rows += f"""
            <tr>
              <td>{label}</td>
              <td>
                <div style="background:#eee;border-radius:4px;height:12px;width:200px">
                  <div style="background:{bar_color};width:{s:.0f}%;height:100%;
                              border-radius:4px"></div>
                </div>
              </td>
              <td><b>{s:.1f}</b>/100</td>
            </tr>"""

        issue_rows = ""
        sev_colors = {"Critical": "#e74c3c", "High": "#e67e22",
                      "Medium":   "#f1c40f", "Low": "#3498db", "OK": "#27ae60"}
        for issue in all_issues[:20]:
            sev = issue.severity.value if hasattr(issue.severity, "value") else issue.severity
            bg  = sev_colors.get(sev, "#95a5a6")
            issue_rows += f"""
            <tr>
              <td><span style="background:{bg};color:white;padding:2px 6px;
                border-radius:3px;font-size:12px">{sev}</span></td>
              <td>{issue.issue_type.replace('_',' ').title()}</td>
              <td style="font-size:13px">{issue.description[:120]}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Text Quality Report — Nydra</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; color: #333; }}
  h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  .score-badge {{ font-size: 48px; font-weight: bold; color: {verdict_color}; }}
  .verdict {{ font-size: 20px; font-weight: bold; color: {verdict_color}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th {{ background: #2c3e50; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f8f9fa; }}
  .meta {{ background: #f0f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
  footer {{ margin-top: 40px; color: #777; font-size: 12px; text-align: center; }}
</style>
</head>
<body>
<h1>🩺 Text Quality Report</h1>
<div class="meta">
  <span class="score-badge">{overall:.0f}</span><span style="font-size:24px">/100</span>
  &nbsp;&nbsp;
  <span class="verdict">{verdict}</span>
  <br><br>
  Words: <b>{report.get('word_count',0):,}</b> &nbsp;|&nbsp;
  Sentences: <b>{report.get('sentence_count',0):,}</b> &nbsp;|&nbsp;
  Chars: <b>{report.get('text_length',0):,}</b>
</div>

<h2>📊 Score Breakdown</h2>
<table>
  <tr><th>Dimension</th><th>Score Bar</th><th>Score</th></tr>
  {rows}
</table>

<h2>🔍 Issues Detected</h2>
<table>
  <tr><th>Severity</th><th>Type</th><th>Description</th></tr>
  {issue_rows if issue_rows else '<tr><td colspan="3">✅ No issues found</td></tr>'}
</table>

<footer>Generated by Nydra v0.6.0 — Text Quality Module</footer>
</body>
</html>"""
        return html


# ─────────────────────────────────────────────────────────────────────────────
# 11. SMART TEXT QUALITY CHECKER (Master Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class SmartTextQualityChecker:
    """
    Master orchestrator — runs all checkers in one call.

    Usage:
        checker = SmartTextQualityChecker()
        report  = checker.check("Your text here...")

        print(report["overall_score"])     # 73.4
        print(report["verdict"])           # "Needs Work"
        print(report["all_issues"])        # list of QualityIssue
        print(report["priority_issues"])   # what to fix first

    Corpus mode (multiple documents):
        report = checker.check_corpus(["doc1...", "doc2...", ...])
    """

    def __init__(self, config: Optional[TextQualityConfig] = None):
        self.config = config or TextQualityConfig()

        # Lazy-instantiate checkers
        self._duplicate    = DuplicateDetector(self.config)
        self._empty        = EmptyContentDetector(self.config)
        self._noise        = NoiseDetector(self.config)
        self._pii          = PIIDetector(self.config)
        self._coherence    = CoherenceScorer(self.config)
        self._consistency  = ConsistencyChecker(self.config)
        self._encoding     = EncodingChecker(self.config)
        self._language     = LanguageQualityScorer(self.config)
        self._scorer       = TextQualityScore(self.config)
        self._reporter     = TextQualityReporter(self.config)

    # ── Single text ───────────────────────────────────────────────────────────

    def check(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Run all quality checks on a single text.

        Args:
            text    : Input text to analyze.
            verbose : Print progress to stdout if True.

        Returns:
            Full quality report dictionary.
        """
        start_time = time.time()

        def _log(msg: str):
            if verbose:
                print(f"  [{time.time() - start_time:.1f}s] {msg}")

        _log("Checking empty content...")
        empty_result = self._empty.check(text)

        _log("Checking duplicates...")
        dup_result = self._duplicate.check(text)

        _log("Checking noise...")
        noise_result = self._noise.check(text)

        _log("Checking PII...")
        pii_result = self._pii.check(text)

        _log("Checking coherence...")
        coh_result = self._coherence.check(text)

        _log("Checking consistency...")
        con_result = self._consistency.check(text)

        _log("Checking encoding...")
        enc_result = self._encoding.check(text)

        _log("Checking language quality...")
        lang_result = self._language.check(text)

        # Compute composite score
        score_info = self._scorer.compute(
            duplicate_score  = dup_result.get("score", 100),
            noise_score      = noise_result.get("score", 100),
            pii_score        = pii_result.get("score", 100),
            coherence_score  = coh_result.get("score", 100),
            language_score   = lang_result.get("score", 100),
            encoding_score   = enc_result.get("encoding_score", 100),
            empty_score      = empty_result.get("score", 100),
        )

        # Collect all issues
        all_issues: List[QualityIssue] = []
        for result_dict in [
            empty_result, dup_result, noise_result, pii_result,
            coh_result, con_result, enc_result, lang_result
        ]:
            all_issues.extend(result_dict.get("issues", []))

        # Build basic text stats
        sentences  = self._empty._tokenize_sentences(text)
        words      = self._empty._tokenize_words(text)

        # Sort issues by severity
        sev_order = ["Critical", "High", "Medium", "Low", "OK"]
        all_issues.sort(
            key=lambda x: sev_order.index(
                x.severity.value if hasattr(x.severity, "value") else x.severity
            )
        )

        processing_time = time.time() - start_time

        report = {
            # Meta
            "text_length":      len(text),
            "word_count":       len(words),
            "sentence_count":   len(sentences),
            "processing_time":  round(processing_time, 3),

            # Score
            "overall_score":    score_info["overall_score"],
            "verdict":          score_info["verdict"],
            "severity":         score_info["severity"],
            "score_info":       score_info,

            # Issues
            "all_issues":       all_issues,
            "issue_count":      len(all_issues),
            "critical_issues":  [i for i in all_issues
                                 if (i.severity.value if hasattr(i.severity, "value")
                                     else i.severity) == "Critical"],

            # Sub-results
            "empty_result":     empty_result,
            "duplicate_result": dup_result,
            "noise_result":     noise_result,
            "pii_info":         pii_result,
            "coherence_result": coh_result,
            "consistency_result": con_result,
            "encoding_result":  enc_result,
            "language_result":  lang_result,
        }

        _log(f"Done. Score: {score_info['overall_score']}/100 ({score_info['verdict']})")
        return report

    # ── Corpus mode ───────────────────────────────────────────────────────────

    def check_corpus(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        verbose: bool = False,
        check_cross_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Run quality checks on a list of texts (corpus / dataset).

        Returns per-document scores + corpus-level statistics.
        """
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        per_doc_reports = []
        for i, text in enumerate(texts):
            if verbose:
                print(f"  Checking document {i+1}/{len(texts)}...")
            report = self.check(text, verbose=False)
            report["doc_id"] = ids[i]
            per_doc_reports.append(report)

        # Cross-document duplicate check
        cross_dupes = {}
        if check_cross_duplicates and len(texts) > 1:
            if verbose:
                print("  Checking cross-document duplicates...")
            cross_dupes = self._duplicate.check_corpus(texts, ids)

        # Corpus statistics
        scores = [r["overall_score"] for r in per_doc_reports]
        corpus_stats = {
            "total_documents":   len(texts),
            "mean_score":        round(sum(scores) / max(len(scores), 1), 2),
            "min_score":         round(min(scores), 2) if scores else 0,
            "max_score":         round(max(scores), 2) if scores else 0,
            "std_score":         round(
                (sum((s - sum(scores)/len(scores))**2 for s in scores) /
                 max(len(scores), 1)) ** 0.5, 2
            ) if scores else 0,
            "excellent_count":   sum(1 for s in scores if s >= 90),
            "good_count":        sum(1 for s in scores if 75 <= s < 90),
            "needs_work_count":  sum(1 for s in scores if 55 <= s < 75),
            "poor_count":        sum(1 for s in scores if s < 55),
            "verdict":           self._corpus_verdict(scores),
        }

        return {
            "corpus_stats":     corpus_stats,
            "per_document":     per_doc_reports,
            "cross_duplicates": cross_dupes,
        }

    def _corpus_verdict(self, scores: List[float]) -> str:
        if not scores:
            return "No data"
        mean = sum(scores) / len(scores)
        poor_pct = sum(1 for s in scores if s < 55) / len(scores)
        if poor_pct > 0.3 or mean < 55:
            return "Dataset Needs Major Cleaning"
        elif mean < 70:
            return "Dataset Needs Work"
        elif mean < 80:
            return "Dataset Quality Acceptable"
        else:
            return "High Quality Dataset"

    # ── Report generation ─────────────────────────────────────────────────────

    def generate_report(
        self,
        check_result: Dict[str, Any],
        output_format: str = "markdown",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted report from check() output."""
        content = self._reporter.generate(check_result, output_format)
        if output_path:
            self._reporter.save(check_result, output_path)
        return content

    def redact_pii(self, text: str) -> str:
        """Return text with all detected PII replaced by type labels."""
        return self._pii.redact_text(text)

    def fix_encoding(self, text: str) -> str:
        """Attempt to fix common encoding issues in text."""
        return self._encoding.fix_encoding(text)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def check_text_quality(
    text: str,
    config: Optional[TextQualityConfig] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    One-call quality check for a single text.

    Args:
        text    : Text to analyze.
        config  : Optional custom configuration.
        verbose : Print progress.

    Returns:
        Full report dict with overall_score, verdict, issues, etc.

    Example:
        report = check_text_quality("Your text here...")
        print(report["overall_score"])   # 82.4
        print(report["verdict"])         # "Good"
    """
    checker = SmartTextQualityChecker(config)
    return checker.check(text, verbose=verbose)


def detect_pii(
    text: str,
    config: Optional[TextQualityConfig] = None,
    redact: bool = False,
) -> Dict[str, Any]:
    """
    Detect PII in text with optional redaction.

    Args:
        text    : Text to scan.
        config  : Optional config (set pii_detection_mode, mask_pii_in_output, etc.)
        redact  : If True, also return redacted version.

    Returns:
        Dict with pii_found, pii_count, pii_by_type, score.

    Example:
        result = detect_pii("Call me at +1-555-123-4567 or john@email.com")
        print(result["pii_count"])       # 2
        print(result["pii_by_type"])     # {"Phone": 1, "Email": 1}
    """
    cfg = config or TextQualityConfig()
    detector = PIIDetector(cfg)
    result = detector.check(text)
    if redact:
        result["redacted_text"] = detector.redact_text(text)
    return result


def find_duplicates(
    texts: List[str],
    ids: Optional[List[str]] = None,
    config: Optional[TextQualityConfig] = None,
) -> Dict[str, Any]:
    """
    Find duplicate / near-duplicate texts in a list of documents.

    Args:
        texts   : List of text strings.
        ids     : Optional document identifiers.
        config  : Optional config (set thresholds).

    Returns:
        Dict with exact_duplicates, near_duplicates, paraphrase_duplicates.

    Example:
        result = find_duplicates(["Hello world", "Hello world!", "Goodbye"])
        print(result["exact_duplicates"])  # []
        print(result["near_duplicates"])   # [...] (Hello world ≈ Hello world!)
    """
    cfg = config or TextQualityConfig()
    detector = DuplicateDetector(cfg)
    return detector.check_corpus(texts, ids)


def check_coherence(
    text: str,
    config: Optional[TextQualityConfig] = None,
) -> Dict[str, Any]:
    """
    Measure coherence of a text document.

    Args:
        text    : Text to analyze.
        config  : Optional config.

    Returns:
        Dict with local_coherence, global_coherence, discourse_score,
        overall_coherence, score, problem_transitions.

    Example:
        result = check_coherence(long_article)
        print(result["score"])              # 78.5
        print(result["problem_transitions"]) # [{position: 12, ...}]
    """
    cfg = config or TextQualityConfig()
    scorer = CoherenceScorer(cfg)
    return scorer.check(text)


def full_text_quality_report(
    text: str,
    output_format: str = "markdown",
    output_path: Optional[str] = None,
    config: Optional[TextQualityConfig] = None,
    verbose: bool = False,
) -> str:
    """
    Run full quality check and return a formatted report.

    Args:
        text          : Text to analyze.
        output_format : "markdown" | "json" | "html"
        output_path   : Optional file path to save report.
        config        : Optional custom configuration.
        verbose       : Print progress.

    Returns:
        Report string in specified format.

    Example:
        report_md = full_text_quality_report(text, output_format="markdown")
        print(report_md)

        # Or save HTML report:
        full_text_quality_report(text, output_format="html",
                                  output_path="quality_report.html")
    """
    checker = SmartTextQualityChecker(config)
    result  = checker.check(text, verbose=verbose)
    return checker.generate_report(result, output_format, output_path)


def check_corpus_quality(
    texts: List[str],
    ids: Optional[List[str]] = None,
    config: Optional[TextQualityConfig] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Quality check for a full text corpus / dataset.

    Args:
        texts   : List of text documents.
        ids     : Optional document identifiers.
        config  : Optional configuration.
        verbose : Print progress.

    Returns:
        Dict with corpus_stats, per_document results, cross_duplicates.

    Example:
        result = check_corpus_quality(df["text"].tolist())
        print(result["corpus_stats"]["mean_score"])  # 74.2
        print(result["corpus_stats"]["verdict"])     # "Dataset Needs Work"
    """
    checker = SmartTextQualityChecker(config)
    return checker.check_corpus(texts, ids, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-USE ALIASES
# ─────────────────────────────────────────────────────────────────────────────

quick_quality_check  = check_text_quality
quick_pii_scan       = detect_pii
quick_dedup          = find_duplicates
quick_coherence      = check_coherence


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("🩺 Nydra — text_quality.py — Smoke Test")
    print("=" * 65)

    SAMPLE_TEXT = """
    Machine learning is a subset of artificial intelligence that enables
    systems to learn from data. The model was trained on 10,000 samples.
    However, the model was also evaluated on 5,000 samples.

    Deep learning uses neural networks with many layers. Neural networks
    are inspired by the human brain. The human brain contains approximately
    86 billion neurons. Contact the team at admin@example.com or call
    +1-555-987-6543 for more information.

    Machine learning is a subset of artificial intelligence that enables
    systems to learn from data. This sentence appears again to test
    duplicate detection functionality.

    The results show that accuracy is high. However the results show that
    precision is low. The study had 200 participants. Only 50 subjects
    were included in the final analysis.
    """

    print("\n📋 Running full quality check...")
    checker = SmartTextQualityChecker()
    result  = checker.check(SAMPLE_TEXT, verbose=True)

    print(f"\n{'─'*40}")
    print(f"Overall Score : {result['overall_score']}/100")
    print(f"Verdict       : {result['verdict']}")
    print(f"Issues found  : {result['issue_count']}")
    print(f"Processing    : {result['processing_time']}s")
    print(f"{'─'*40}")

    print("\n📊 Breakdown:")
    for dim, score in result["score_info"]["breakdown"].items():
        bar = "█" * int(score // 10) + "░" * (10 - int(score // 10))
        print(f"  {dim:<22} [{bar}] {score:.1f}")

    print("\n🔍 Top Issues:")
    for issue in result["all_issues"][:5]:
        sev = issue.severity.value if hasattr(issue.severity, "value") else issue.severity
        print(f"  [{sev}] {issue.issue_type}: {issue.description[:80]}...")

    print("\n✅ Smoke test complete.")
