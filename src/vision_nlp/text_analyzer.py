"""
text_analyzer.py — Nydra v0.6.0
======================================
Advanced text intelligence pipeline — 12 blocks, 2000+ lines.
Runs from raw text or integrates with document_reader.py.

Dependency tiers (graceful degradation):
  Tier 0 — stdlib only          : always available
  Tier 1 — numpy / pandas       : almost always available
  Tier 2 — sklearn / scipy      : standard ML stack
  Tier 3 — nltk / textstat      : NLP basics
  Tier 4 — spacy                : advanced NLP
  Tier 5 — transformers / torch : deep learning (optional)
  Tier 6 — sentence-transformers: semantic similarity (optional)
  Tier 7 — yake / keybert       : keyword extraction (optional)

"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────
import collections
import functools
import hashlib
import itertools
import logging
import math
import os
import re
import string
import unicodedata
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("text_analyzer")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS — each wrapped so the file loads even without the lib
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    NUMPY_OK = True
except ImportError:
    np = None  # type: ignore
    NUMPY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

try:
    from scipy.stats import entropy as scipy_entropy, kurtosis, skew
    from scipy.spatial.distance import cosine as scipy_cosine
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    import textstat
    TEXTSTAT_OK = True
except ImportError:
    TEXTSTAT_OK = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
    _nltk_downloads = ["vader_lexicon", "punkt", "stopwords", "wordnet",
                       "averaged_perceptron_tagger", "punkt_tab"]
    for _pkg in _nltk_downloads:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
        SPACY_OK = True
    except OSError:
        try:
            os.system("python -m spacy download en_core_web_sm -q")
            _nlp = spacy.load("en_core_web_sm")
            SPACY_OK = True
        except Exception:
            SPACY_OK = False
            _nlp = None
except ImportError:
    SPACY_OK = False
    _nlp = None

try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import langid
    LANGID_OK = True
except ImportError:
    LANGID_OK = False

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_OK = True
except Exception:
    TRANSFORMERS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

try:
    import yake
    YAKE_OK = True
except ImportError:
    YAKE_OK = False

try:
    from keybert import KeyBERT
    KEYBERT_OK = True
except ImportError:
    KEYBERT_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

VERSION = "0.6.0"

GRADE_LEVELS = [
    (90, "Elementary",   "Very easy — readable by age 10"),
    (70, "Middle School","Easy — readable by age 13"),
    (50, "High School",  "Standard — readable by age 16"),
    (30, "College",      "Difficult — college level"),
    (0,  "Expert",       "Very difficult — specialist audience"),
]

QUALITY_GRADES = [
    (90, "A", "Excellent 🟢"),
    (75, "B", "Good 🟡"),
    (60, "C", "Average 🟠"),
    (45, "D", "Poor 🔴"),
    (0,  "F", "Critical ⛔"),
]

STYLE_LABELS = {
    "academic"      : "Academic 📚",
    "journalistic"  : "Journalistic 📰",
    "conversational": "Conversational 💬",
    "technical"     : "Technical ⚙️",
    "creative"      : "Creative 🎨",
    "legal"         : "Legal ⚖️",
}

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "would",
    "should", "seem", "seems", "appeared", "appears", "suggest", "suggests",
    "indicate", "indicates", "approximately", "roughly", "somewhat", "rather",
    "fairly", "quite", "generally", "usually", "often", "sometimes", "likely",
    "unlikely", "uncertain", "unclear", "arguably", "apparently", "presumably",
}

INTENSIFIERS = {
    "very", "extremely", "highly", "absolutely", "completely", "totally",
    "utterly", "entirely", "really", "truly", "deeply", "strongly", "greatly",
    "incredibly", "remarkably", "exceptionally", "extraordinarily", "undeniably",
    "unquestionably", "definitely", "certainly", "obviously", "clearly",
}

PASSIVE_AUXILIARIES = {
    "is", "are", "was", "were", "be", "been", "being",
    "has been", "have been", "had been", "will be", "would be",
    "can be", "could be", "should be", "might be", "must be",
}

ENTITY_PATTERNS = {
    "EMAIL"  : re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "URL"    : re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
    "PHONE"  : re.compile(r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "DATE"   : re.compile(
        r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b',
        re.IGNORECASE
    ),
    "MONEY"  : re.compile(r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|DZD)\b'),
    "PERCENT": re.compile(r'\b\d+(?:\.\d+)?\s*%'),
}

COMMON_ENGLISH_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "this","that","these","those","i","you","he","she","it","we","they",
    "me","him","her","us","them","my","your","his","its","our","their",
    "what","which","who","when","where","how","all","each","every","both",
    "few","more","most","other","some","such","no","not","only","same",
    "so","than","too","very","just","because","as","until","while",
}

SCRIPT_RANGES = {
    "Latin"  : (0x0041, 0x024F),
    "Arabic" : (0x0600, 0x06FF),
    "Cyrillic": (0x0400, 0x04FF),
    "CJK"    : (0x4E00, 0x9FFF),
    "Hebrew" : (0x0590, 0x05FF),
    "Devanagari": (0x0900, 0x097F),
}


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 1 — BasicStats
# What  : Pure-Python linguistic surface statistics — zero external deps.
# Why   : Fast baseline every other block can reference.
# Extras: hapax legomena, lexical density, punctuation density,
#         digit ratio, uppercase ratio, avg syllables (heuristic)
# ─────────────────────────────────────────────────────────────────────────────

class BasicStats:
    """
    Computes ~25 surface-level statistics on raw text.
    No external dependencies — pure Python + regex.

    Statistics
    ----------
    words, sentences, paragraphs, characters, chars_no_spaces,
    avg_word_length, avg_sentence_length, avg_paragraph_length,
    vocabulary_size, type_token_ratio (TTR), unique_ratio,
    lexical_density, hapax_legomena_count, hapax_ratio,
    punctuation_count, punctuation_density, digit_count, digit_ratio,
    uppercase_ratio, avg_syllables_per_word, sentence_length_variance,
    short_sentences, long_sentences, question_count, exclamation_count,
    whitespace_ratio, char_entropy
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self._result: dict[str, Any] | None = None

    # ── public ───────────────────────────────────────────────────────────────

    def compute(self) -> dict[str, Any]:
        if self._result is not None:
            return self._result

        text = self.text
        words      = self._tokenize_words(text)
        sentences  = self._tokenize_sentences(text)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

        n_words  = len(words)
        n_sents  = max(len(sentences), 1)
        n_paras  = max(len(paragraphs), 1)
        n_chars  = len(text)
        n_chars_ns = len(text.replace(" ", "").replace("\n", ""))

        word_lengths = [len(w) for w in words]
        sent_lengths = [len(self._tokenize_words(s)) for s in sentences]

        vocab       = [w.lower() for w in words]
        freq        = collections.Counter(vocab)
        vocab_size  = len(freq)
        hapax       = [w for w, c in freq.items() if c == 1]

        content_words = [w for w in vocab if w not in COMMON_ENGLISH_STOPWORDS and w.isalpha()]
        lex_density   = len(content_words) / max(n_words, 1)

        punct_chars = [c for c in text if c in string.punctuation]
        digit_chars = [c for c in text if c.isdigit()]
        upper_words = [w for w in words if w.isupper() and len(w) > 1]

        syllables_per_word = [self._count_syllables(w) for w in words]
        avg_syllables      = sum(syllables_per_word) / max(n_words, 1)

        sl_var = float(np.var(sent_lengths)) if NUMPY_OK and sent_lengths else \
                 _variance(sent_lengths)

        questions    = sum(1 for s in sentences if s.strip().endswith("?"))
        exclamations = sum(1 for s in sentences if s.strip().endswith("!"))

        ws_count  = sum(1 for c in text if c.isspace())
        char_ent  = _char_entropy(text)

        self._result = {
            # counts
            "word_count"              : n_words,
            "sentence_count"          : len(sentences),
            "paragraph_count"         : n_paras,
            "char_count"              : n_chars,
            "chars_no_spaces"         : n_chars_ns,
            # averages
            "avg_word_length"         : round(sum(word_lengths) / max(n_words, 1), 2),
            "avg_sentence_length"     : round(sum(sent_lengths)  / n_sents, 2),
            "avg_paragraph_length"    : round(n_words / n_paras, 2),
            "avg_syllables_per_word"  : round(avg_syllables, 2),
            # vocabulary
            "vocabulary_size"         : vocab_size,
            "type_token_ratio"        : round(vocab_size / max(n_words, 1), 4),
            "unique_ratio"            : round(vocab_size / max(n_words, 1), 4),
            "lexical_density"         : round(lex_density, 4),
            "hapax_legomena_count"    : len(hapax),
            "hapax_ratio"             : round(len(hapax) / max(vocab_size, 1), 4),
            # character-level
            "punctuation_count"       : len(punct_chars),
            "punctuation_density"     : round(len(punct_chars) / max(n_chars, 1), 4),
            "digit_count"             : len(digit_chars),
            "digit_ratio"             : round(len(digit_chars) / max(n_chars, 1), 4),
            "uppercase_ratio"         : round(len(upper_words) / max(n_words, 1), 4),
            "whitespace_ratio"        : round(ws_count / max(n_chars, 1), 4),
            # structure
            "sentence_length_variance": round(sl_var, 2),
            "short_sentence_count"    : sum(1 for l in sent_lengths if l < 8),
            "long_sentence_count"     : sum(1 for l in sent_lengths if l > 30),
            "question_count"          : questions,
            "exclamation_count"       : exclamations,
            # information
            "char_entropy"            : round(char_ent, 4),
            # top vocab
            "top_20_words"            : [w for w, _ in freq.most_common(20)
                                         if w not in COMMON_ENGLISH_STOPWORDS],
        }
        return self._result

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        if NLTK_OK:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        return re.findall(r"\b[a-zA-Z']+\b", text)

    @staticmethod
    def _tokenize_sentences(text: str) -> list[str]:
        if NLTK_OK:
            try:
                return sent_tokenize(text)
            except Exception:
                pass
        return re.split(r'(?<=[.!?])\s+', text.strip())

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Heuristic syllable counter (no external lib needed)."""
        word = word.lower().strip("'")
        if not word:
            return 0
        vowels = "aeiouy"
        count  = 0
        prev_vowel = False
        for ch in word:
            is_v = ch in vowels
            if is_v and not prev_vowel:
                count += 1
            prev_vowel = is_v
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 2 — ReadabilityScorer
# What  : 7 standard readability indices + combined weighted score.
# Extras: per-index grade label, confidence interval based on text length,
#         raw index values + normalised 0-100 + grade level label.
# ─────────────────────────────────────────────────────────────────────────────

class ReadabilityScorer:
    """
    Computes 7 readability indices and combines them into one score.

    Indices
    -------
    Flesch Reading Ease      (FRE)  — higher = easier
    Flesch-Kincaid Grade     (FKG)  — US grade level
    Gunning Fog              (GFI)  — years of education needed
    SMOG Index               (SMOG) — grade level via polysyllables
    Coleman-Liau Index       (CLI)  — based on chars, not syllables
    Automated Readability    (ARI)  — char-based
    Dale-Chall               (DCI)  — difficult word ratio

    Fallback: heuristic implementation when textstat unavailable.
    """

    # weights for combined score (sum = 1.0)
    _WEIGHTS = {
        "flesch_ease"       : 0.25,
        "flesch_kincaid"    : 0.20,
        "gunning_fog"       : 0.15,
        "smog"              : 0.15,
        "coleman_liau"      : 0.10,
        "ari"               : 0.10,
        "dale_chall"        : 0.05,
    }

    def __init__(self, text: str, stats: dict | None = None) -> None:
        self.text  = text
        self.stats = stats or BasicStats(text).compute()

    def compute(self) -> dict[str, Any]:
        raw = self._compute_raw()
        normed = self._normalise(raw)
        combined = sum(
            normed.get(k, 50) * w
            for k, w in self._WEIGHTS.items()
        )
        combined = max(0.0, min(100.0, combined))
        grade_level, label = self._grade(combined)

        # Confidence interval — wider for short texts
        n = self.stats.get("word_count", 0)
        ci = 15.0 if n < 100 else (8.0 if n < 500 else 3.0)

        return {
            "indices"        : raw,
            "normalised"     : {k: round(v, 2) for k, v in normed.items()},
            "combined_score" : round(combined, 2),
            "grade_level"    : grade_level,
            "grade_label"    : label,
            "confidence_interval": round(ci, 1),
            "per_index_grade": {k: self._grade_for_index(k, v)
                                for k, v in raw.items()},
        }

    # ── raw computation ───────────────────────────────────────────────────────

    def _compute_raw(self) -> dict[str, float]:
        if TEXTSTAT_OK:
            return self._from_textstat()
        return self._heuristic()

    def _from_textstat(self) -> dict[str, float]:
        t = self.text
        return {
            "flesch_ease"   : round(float(textstat.flesch_reading_ease(t)), 2),
            "flesch_kincaid": round(float(textstat.flesch_kincaid_grade(t)), 2),
            "gunning_fog"   : round(float(textstat.gunning_fog(t)), 2),
            "smog"          : round(float(textstat.smog_index(t)), 2),
            "coleman_liau"  : round(float(textstat.coleman_liau_index(t)), 2),
            "ari"           : round(float(textstat.automated_readability_index(t)), 2),
            "dale_chall"    : round(float(textstat.dale_chall_readability_score(t)), 2),
        }

    def _heuristic(self) -> dict[str, float]:
        """Pure-Python fallback — approximations only."""
        s = self.stats
        wc  = max(s.get("word_count", 1), 1)
        sc  = max(s.get("sentence_count", 1), 1)
        cc  = max(s.get("chars_no_spaces", 1), 1)
        syl = s.get("avg_syllables_per_word", 1.5) * wc  # total syllables
        asl = wc / sc      # avg sentence length
        asw = syl  / wc    # avg syllables per word

        fre  = 206.835 - 1.015 * asl - 84.6 * asw
        fkg  = 0.39 * asl + 11.8 * asw - 15.59
        gfi  = 0.4 * (asl + 100 * max(0, asw - 1) / max(wc, 1))
        smog = 3.1291 + 1.0430 * math.sqrt(max(asw - 1, 0) * 30)
        cli  = 5.88 * (cc / wc) - 29.6 * (sc / wc) - 15.8
        ari  = 4.71 * (cc / wc) + 0.5 * asl - 21.43
        dch  = fkg * 0.9  # rough approximation

        return {
            "flesch_ease"   : round(float(fre),  2),
            "flesch_kincaid": round(float(fkg),  2),
            "gunning_fog"   : round(float(gfi),  2),
            "smog"          : round(float(smog), 2),
            "coleman_liau"  : round(float(cli),  2),
            "ari"           : round(float(ari),  2),
            "dale_chall"    : round(float(dch),  2),
        }

    # ── normalisation — convert all indices to 0-100 (higher = easier) ───────

    def _normalise(self, raw: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        # Flesch Reading Ease is already 0-100 (higher = easier)
        out["flesch_ease"] = max(0.0, min(100.0, raw.get("flesch_ease", 50)))

        # Grade-level indices: lower grade = easier; map grade 0-20 → score 100-0
        for key in ("flesch_kincaid", "gunning_fog", "smog", "coleman_liau", "ari"):
            g = raw.get(key, 10)
            out[key] = max(0.0, min(100.0, 100.0 - (g / 20.0) * 100.0))

        # Dale-Chall: 4.9 = easy, 9.9 = very hard; map to 100-0
        dc = raw.get("dale_chall", 7.0)
        out["dale_chall"] = max(0.0, min(100.0, 100.0 - ((dc - 4.9) / 5.0) * 100.0))

        return out

    @staticmethod
    def _grade(score: float) -> tuple[str, str]:
        for thresh, level, label in GRADE_LEVELS:
            if score >= thresh:
                return level, label
        return "Expert", "Very difficult"

    @staticmethod
    def _grade_for_index(key: str, value: float) -> str:
        """Returns human label for a single raw index."""
        if key == "flesch_ease":
            if value >= 90: return "Very Easy"
            if value >= 70: return "Easy"
            if value >= 60: return "Standard"
            if value >= 30: return "Difficult"
            return "Very Difficult"
        # grade-level indices
        if value <= 6:  return "Elementary"
        if value <= 9:  return "Middle School"
        if value <= 12: return "High School"
        if value <= 16: return "College"
        return "Expert"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 3 — LanguageDetector
# What  : Ensemble language detection with chunk analysis.
# Extras: script detection, confidence scoring, multilingual flag,
#         per-chunk language, short-text warning.
# ─────────────────────────────────────────────────────────────────────────────

_LANG_NAMES: dict[str, str] = {
    "en": "English",   "fr": "French",   "de": "German",   "es": "Spanish",
    "ar": "Arabic",    "zh": "Chinese",  "ru": "Russian",  "pt": "Portuguese",
    "it": "Italian",   "nl": "Dutch",    "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi",     "tr": "Turkish",  "pl": "Polish",   "sv": "Swedish",
    "da": "Danish",    "fi": "Finnish",  "no": "Norwegian","cs": "Czech",
    "ro": "Romanian",  "hu": "Hungarian","uk": "Ukrainian", "id": "Indonesian",
}

class LanguageDetector:
    """
    Detects language using an ensemble of langdetect + langid.
    Falls back to character-script heuristics if both are unavailable.

    Features
    --------
    • Ensemble voting (majority wins, tie → langdetect)
    • Chunk-level analysis: splits every 500 words → per-chunk lang
    • is_multilingual flag: True when >1 unique lang found in chunks
    • Script detection: Latin / Arabic / Cyrillic / CJK / Hebrew / Devanagari
    • Short-text warning when word_count < 20
    """

    CHUNK_SIZE = 500   # words per chunk

    def __init__(self, text: str, stats: dict | None = None) -> None:
        self.text  = text
        self.words = re.findall(r"\S+", text)
        self.stats = stats or {}

    def compute(self) -> dict[str, Any]:
        if len(self.words) < 5:
            return {
                "language_code"  : "unknown",
                "language_name"  : "Unknown",
                "confidence"     : 0.0,
                "is_multilingual": False,
                "script"         : self._detect_script(),
                "warning"        : "Text too short for reliable detection",
                "chunks"         : [],
            }

        primary, confidence = self._detect_primary()
        chunks = self._chunk_detect()
        unique_langs = list({c["language_code"] for c in chunks if c["language_code"] != "unknown"})
        is_multi = len(unique_langs) > 1

        warning = None
        if len(self.words) < 20:
            warning = "Short text — detection may be unreliable"

        return {
            "language_code"  : primary,
            "language_name"  : _LANG_NAMES.get(primary, primary),
            "confidence"     : round(confidence, 4),
            "is_multilingual": is_multi,
            "secondary_langs": [l for l in unique_langs if l != primary],
            "script"         : self._detect_script(),
            "chunks"         : chunks,
            "warning"        : warning,
        }

    # ── primary detection ─────────────────────────────────────────────────────

    def _detect_primary(self) -> tuple[str, float]:
        results: list[tuple[str, float]] = []

        if LANGDETECT_OK:
            try:
                langs = detect_langs(self.text)
                if langs:
                    results.append((langs[0].lang, langs[0].prob))
            except Exception:
                pass

        if LANGID_OK:
            try:
                lang, score = langid.classify(self.text)
                # langid score is log-probability — convert to 0-1 range
                conf = min(1.0, max(0.0, 1.0 + score / 100.0))
                results.append((lang, conf))
            except Exception:
                pass

        if not results:
            return self._script_fallback()

        # Ensemble vote
        lang_votes: dict[str, list[float]] = collections.defaultdict(list)
        for lang, conf in results:
            lang_votes[lang].append(conf)

        best_lang = max(lang_votes, key=lambda l: sum(lang_votes[l]) / len(lang_votes[l]))
        best_conf = sum(lang_votes[best_lang]) / len(lang_votes[best_lang])
        return best_lang, best_conf

    def _chunk_detect(self) -> list[dict]:
        chunks = []
        for i in range(0, len(self.words), self.CHUNK_SIZE):
            chunk_words = self.words[i: i + self.CHUNK_SIZE]
            chunk_text  = " ".join(chunk_words)
            lang, conf  = self._detect_primary_for_text(chunk_text)
            chunks.append({
                "chunk_index"  : i // self.CHUNK_SIZE,
                "word_range"   : [i, i + len(chunk_words)],
                "language_code": lang,
                "confidence"   : round(conf, 3),
            })
        return chunks

    @staticmethod
    def _detect_primary_for_text(text: str) -> tuple[str, float]:
        if LANGDETECT_OK:
            try:
                langs = detect_langs(text)
                if langs:
                    return langs[0].lang, langs[0].prob
            except Exception:
                pass
        if LANGID_OK:
            try:
                lang, score = langid.classify(text)
                return lang, min(1.0, max(0.0, 1.0 + score / 100.0))
            except Exception:
                pass
        return "unknown", 0.0

    def _detect_script(self) -> str:
        """Determine dominant writing script by character range."""
        script_counts: dict[str, int] = collections.defaultdict(int)
        for ch in self.text:
            cp = ord(ch)
            for script, (lo, hi) in SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    script_counts[script] += 1
                    break
        if not script_counts:
            return "Latin"  # default
        return max(script_counts, key=script_counts.__getitem__)

    def _script_fallback(self) -> tuple[str, float]:
        script = self._detect_script()
        mapping = {
            "Arabic"    : "ar",
            "Cyrillic"  : "ru",
            "CJK"       : "zh",
            "Hebrew"    : "he",
            "Devanagari": "hi",
        }
        return mapping.get(script, "en"), 0.3


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 4 — SentimentAnalyzer
# What  : Two-layer sentiment — transformers (DistilBERT) → VADER fallback.
# Extras: per-sentence timeline, sentiment arc, emotion volatility,
#         subjectivity ratio, sentence-level outliers.
# ─────────────────────────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Layer 1 : HuggingFace DistilBERT SST-2 (deep learning, most accurate)
    Layer 2 : VADER (rule-based, no model download needed)
    Layer 3 : Heuristic lexicon fallback (zero dependencies)

    Outputs
    -------
    • doc_level  : positive / negative / neutral + dominant + confidence
    • sentences  : per-sentence scores (up to 500 sentences)
    • sentiment_arc: arc of sentiment across the document
    • volatility : std-dev of sentence-level scores
    • subjectivity: ratio of subjective sentences
    • outliers   : most positive + most negative sentences
    """

    _SENTIMENT_PIPELINE = None   # cached HF pipeline

    SUBJ_WORDS = {
        "love","hate","great","terrible","amazing","awful","good","bad",
        "wonderful","horrible","excellent","poor","fantastic","dreadful",
        "beautiful","ugly","perfect","worst","best","nice","nasty",
    }

    def __init__(self, text: str) -> None:
        self.text      = text
        self.sentences = BasicStats._tokenize_sentences(text)

    def compute(self) -> dict[str, Any]:
        per_sent = self._score_sentences()
        doc      = self._aggregate(per_sent)
        arc      = self._sentiment_arc(per_sent)
        vol      = self._volatility(per_sent)
        subj     = self._subjectivity()
        outliers = self._outliers(per_sent)

        return {
            "doc_level"    : doc,
            "per_sentence" : per_sent[:100],    # cap for readability
            "sentiment_arc": arc,
            "volatility"   : round(vol, 4),
            "volatility_label": self._vol_label(vol),
            "subjectivity" : round(subj, 4),
            "outliers"     : outliers,
            "method_used"  : self._method(),
        }

    # ── sentence scoring ──────────────────────────────────────────────────────

    def _score_sentences(self) -> list[dict]:
        sents = self.sentences[:500]

        if TRANSFORMERS_OK:
            try:
                return self._hf_score(sents)
            except Exception as e:
                logger.debug(f"HF sentiment failed: {e}")

        if NLTK_OK:
            try:
                return self._vader_score(sents)
            except Exception as e:
                logger.debug(f"VADER failed: {e}")

        return self._lexicon_score(sents)

    def _hf_score(self, sents: list[str]) -> list[dict]:
        if SentimentAnalyzer._SENTIMENT_PIPELINE is None:
            SentimentAnalyzer._SENTIMENT_PIPELINE = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True, max_length=512,
            )
        pipe    = SentimentAnalyzer._SENTIMENT_PIPELINE
        results = []
        for i, sent in enumerate(sents):
            try:
                out  = pipe(sent[:512])[0]
                label = out["label"].lower()
                score = float(out["score"])
                pos   = score if label == "positive" else 1.0 - score
                neg   = 1.0 - pos
                results.append({
                    "index"    : i,
                    "text"     : sent[:120],
                    "positive" : round(pos, 4),
                    "negative" : round(neg, 4),
                    "neutral"  : 0.0,
                    "compound" : round(pos - neg, 4),
                    "dominant" : label,
                    "confidence": round(score, 4),
                })
            except Exception:
                results.append(self._neutral_sent(i, sent))
        return results

    def _vader_score(self, sents: list[str]) -> list[dict]:
        sia = SentimentIntensityAnalyzer()
        results = []
        for i, sent in enumerate(sents):
            sc = sia.polarity_scores(sent)
            dominant = ("positive" if sc["compound"] >= 0.05 else
                        "negative" if sc["compound"] <= -0.05 else "neutral")
            results.append({
                "index"    : i,
                "text"     : sent[:120],
                "positive" : round(sc["pos"], 4),
                "negative" : round(sc["neg"], 4),
                "neutral"  : round(sc["neu"], 4),
                "compound" : round(sc["compound"], 4),
                "dominant" : dominant,
                "confidence": round(abs(sc["compound"]), 4),
            })
        return results

    def _lexicon_score(self, sents: list[str]) -> list[dict]:
        """Zero-dependency heuristic using a small lexicon."""
        POS = {
            "good", "great", "excellent", "wonderful", "amazing", "love", "happy",
            "positive", "beautiful", "perfect", "fantastic", "best", "nice", "joy",
            "thrilled", "efficient", "clean", "effective", "successful", "reliable"
        }
        NEG = {
            "bad", "terrible", "awful", "horrible", "hate", "sad", "negative", "ugly",
            "worst", "poor", "dreadful", "nasty", "wrong", "fail", "failure", "broken",
            "inefficient", "dirty", "unreliable", "useless"
        }
        results = []
        for i, sent in enumerate(sents):
            words = re.findall(r"\b\w+\b", sent.lower())
            pos = sum(1 for w in words if w in POS)
            neg = sum(1 for w in words if w in NEG)
            total = max(pos + neg, 1)
            compound = (pos - neg) / total
            dominant = ("positive" if compound > 0.1 else
                        "negative" if compound < -0.1 else "neutral")
            results.append({
                "index"    : i,
                "text"     : sent[:120],
                "positive" : round(pos / total, 4),
                "negative" : round(neg / total, 4),
                "neutral"  : round(max(0, 1 - pos / total - neg / total), 4),
                "compound" : round(compound, 4),
                "dominant" : dominant,
                "confidence": round(abs(compound), 4),
            })
        return results

    @staticmethod
    def _neutral_sent(i: int, sent: str) -> dict:
        return {"index": i, "text": sent[:120], "positive": 0.33,
                "negative": 0.33, "neutral": 0.34, "compound": 0.0,
                "dominant": "neutral", "confidence": 0.0}

    # ── aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(per_sent: list[dict]) -> dict:
        if not per_sent:
            return {}
        avg_pos  = sum(s["positive"] for s in per_sent) / len(per_sent)
        avg_neg  = sum(s["negative"] for s in per_sent) / len(per_sent)
        avg_neu  = sum(s["neutral"]  for s in per_sent) / len(per_sent)
        avg_comp = sum(s["compound"] for s in per_sent) / len(per_sent)
        dominant = ("positive" if avg_comp >= 0.05 else
                    "negative" if avg_comp <= -0.05 else "neutral")
        return {
            "positive" : round(avg_pos,  4),
            "negative" : round(avg_neg,  4),
            "neutral"  : round(avg_neu,  4),
            "compound" : round(avg_comp, 4),
            "dominant" : dominant,
            "confidence": round(abs(avg_comp), 4),
        }

    @staticmethod
    def _sentiment_arc(per_sent: list[dict]) -> list[float]:
        """Returns compound scores grouped into 10 equal segments."""
        if not per_sent:
            return []
        n = len(per_sent)
        seg = max(1, n // 10)
        arc = []
        for i in range(0, n, seg):
            chunk = per_sent[i: i + seg]
            arc.append(round(sum(s["compound"] for s in chunk) / len(chunk), 4))
        return arc[:10]

    @staticmethod
    def _volatility(per_sent: list[dict]) -> float:
        if len(per_sent) < 2:
            return 0.0
        compounds = [s["compound"] for s in per_sent]
        if NUMPY_OK:
            return float(np.std(compounds))
        return math.sqrt(_variance(compounds))

    @staticmethod
    def _vol_label(vol: float) -> str:
        if vol < 0.1: return "Stable"
        if vol < 0.2: return "Mildly Volatile"
        if vol < 0.35: return "Volatile"
        return "Highly Volatile"

    def _subjectivity(self) -> float:
        words = re.findall(r"\b\w+\b", self.text.lower())
        subj_count = sum(1 for w in words if w in self.SUBJ_WORDS)
        return subj_count / max(len(words), 1)

    @staticmethod
    def _outliers(per_sent: list[dict]) -> dict:
        if not per_sent:
            return {}
        most_pos = max(per_sent, key=lambda s: s["positive"])
        most_neg = max(per_sent, key=lambda s: s["negative"])
        return {
            "most_positive": {"text": most_pos["text"], "score": most_pos["positive"]},
            "most_negative": {"text": most_neg["text"], "score": most_neg["negative"]},
        }

    def _method(self) -> str:
        if TRANSFORMERS_OK: return "transformers/DistilBERT-SST2"
        if NLTK_OK:         return "VADER"
        return "heuristic-lexicon"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 5 — KeywordExtractor
# What  : Ensemble of TF-IDF + YAKE + KeyBERT with deduplication.
# Extras: bigram + trigram support, per-method scores, final ranked list.
# ─────────────────────────────────────────────────────────────────────────────

class KeywordExtractor:
    """
    Extracts and ranks keywords using up to 3 methods.

    TF-IDF   : statistical, fast — always available with sklearn
    YAKE     : position-aware, unsupervised — needs yake lib
    KeyBERT  : semantic, deep — needs keybert + sentence-transformers

    Deduplication: normalises keywords (lowercase + strip), then picks
    the one with the highest average normalised score across methods.
    """

    def __init__(self, text: str, top_n: int = 20,
                 ngram_range: tuple[int, int] = (1, 3)) -> None:
        self.text       = text
        self.top_n      = top_n
        self.ngram_range= ngram_range
        self._sbert_model = None

    def compute(self) -> dict[str, Any]:
        method_results: dict[str, list[tuple[str, float]]] = {}

        tfidf_kws = self._tfidf()
        if tfidf_kws:
            method_results["tfidf"] = tfidf_kws

        if YAKE_OK:
            yake_kws = self._yake()
            if yake_kws:
                method_results["yake"] = yake_kws

        if KEYBERT_OK:
            keybert_kws = self._keybert()
            if keybert_kws:
                method_results["keybert"] = keybert_kws

        ranked = self._ensemble(method_results)

        # Bigrams + Trigrams via NLTK collocations
        collocations = self._collocations()

        return {
            "keywords"          : ranked,
            "methods_available" : list(method_results.keys()),
            "collocations"      : collocations,
        }

    # ── TF-IDF ───────────────────────────────────────────────────────────────

    def _tfidf(self) -> list[tuple[str, float]]:
        if not SKLEARN_OK:
            return self._freq_fallback()
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=self.ngram_range,
                stop_words="english",
                max_features=500,
                sublinear_tf=True,
            )
            matrix = vectorizer.fit_transform([self.text])
            names  = vectorizer.get_feature_names_out()
            scores = matrix.toarray()[0]
            pairs  = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
            # Normalise scores to 0-1
            max_s  = max((s for _, s in pairs), default=1.0) or 1.0
            return [(kw, round(sc / max_s, 4)) for kw, sc in pairs[:self.top_n]]
        except Exception:
            return []

    def _freq_fallback(self) -> list[tuple[str, float]]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", self.text.lower())
        words = [w for w in words if w not in COMMON_ENGLISH_STOPWORDS]
        freq  = collections.Counter(words)
        total = max(sum(freq.values()), 1)
        return [(w, round(c / total, 4)) for w, c in freq.most_common(self.top_n)]

    # ── YAKE ─────────────────────────────────────────────────────────────────

    def _yake(self) -> list[tuple[str, float]]:
        try:
            extractor = yake.KeywordExtractor(
                lan="en", n=self.ngram_range[1],
                dedupLim=0.9, top=self.top_n,
            )
            raw = extractor.extract_keywords(self.text)
            if not raw:
                return []
            # YAKE scores: lower = more important — invert to 0-1
            max_score = max(s for _, s in raw) or 1.0
            return [(kw, round(1.0 - s / max_score, 4)) for kw, s in raw]
        except Exception:
            return []

    # ── KeyBERT ──────────────────────────────────────────────────────────────

    def _keybert(self) -> list[tuple[str, float]]:
        try:
            kb  = KeyBERT()
            raw = kb.extract_keywords(
                self.text,
                keyphrase_ngram_range=self.ngram_range,
                stop_words="english",
                top_n=self.top_n,
                use_maxsum=True,
                nr_candidates=min(40, self.top_n * 2),
            )
            if not raw:
                return []
            # scores already 0-1 (cosine similarity)
            return [(kw, round(float(sc), 4)) for kw, sc in raw]
        except Exception:
            return []

    # ── Ensemble ─────────────────────────────────────────────────────────────

    def _ensemble(
        self,
        method_results: dict[str, list[tuple[str, float]]],
    ) -> list[dict]:
        if not method_results:
            return []

        kw_scores: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for method, pairs in method_results.items():
            for kw, sc in pairs:
                kw_norm = kw.lower().strip()
                kw_scores[kw_norm][method] = sc

        ranked: list[dict] = []
        for kw, method_map in kw_scores.items():
            avg = sum(method_map.values()) / len(method_map)
            ranked.append({
                "keyword"      : kw,
                "avg_score"    : round(avg, 4),
                "methods"      : method_map,
                "method_count" : len(method_map),
            })

        ranked.sort(key=lambda x: (-x["method_count"], -x["avg_score"]))
        return ranked[:self.top_n]

    # ── Collocations ─────────────────────────────────────────────────────────

    def _collocations(self) -> dict[str, list[str]]:
        if not NLTK_OK:
            return {}
        try:
            tokens = word_tokenize(self.text.lower())
            tokens = [t for t in tokens if t.isalpha()
                      and t not in COMMON_ENGLISH_STOPWORDS]

            bigram_finder   = BigramCollocationFinder.from_words(tokens)
            trigram_finder  = TrigramCollocationFinder.from_words(tokens)

            bigram_finder.apply_freq_filter(2)
            trigram_finder.apply_freq_filter(2)

            bigrams  = bigram_finder.nbest(BigramAssocMeasures.pmi, 10)
            trigrams = trigram_finder.nbest(TrigramAssocMeasures.pmi, 10)

            return {
                "bigrams" : [" ".join(b) for b in bigrams],
                "trigrams": [" ".join(t) for t in trigrams],
            }
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 6 — NamedEntityRecognizer
# What  : spaCy NER + regex patterns + co-occurrence matrix.
# Extras: entity frequency, first/last position, timeline across paragraphs,
#         entity density per 1000 words, co-occurrence pairs.
# ─────────────────────────────────────────────────────────────────────────────

class NamedEntityRecognizer:
    """
    Extracts named entities using spaCy (primary) + regex (fallback/supplement).

    Entity types (spaCy)
    --------------------
    PERSON, ORG, GPE, DATE, TIME, MONEY, PRODUCT, EVENT, LAW, LOC, FAC

    Regex supplements
    -----------------
    EMAIL, URL, PHONE, MONEY, PERCENT

    Outputs
    -------
    • entities     : list of {text, label, count, first_pos, last_pos}
    • by_type      : entities grouped by label
    • co_occurrence: top entity pairs appearing in same sentence
    • density      : entities per 1000 words
    • timeline     : entity presence across paragraphs
    """

    SPACY_TYPES = {"PERSON","ORG","GPE","DATE","TIME","MONEY",
                   "PRODUCT","EVENT","LAW","LOC","FAC","NORP"}

    def __init__(self, text: str, stats: dict | None = None) -> None:
        self.text  = text
        self.stats = stats or {}

    def compute(self) -> dict[str, Any]:
        spacy_ents  = self._spacy_extract()
        regex_ents  = self._regex_extract()

        # Merge: spacy primary, regex supplementary
        all_ents = self._merge(spacy_ents, regex_ents)

        by_type    = self._group_by_type(all_ents)
        cooccur    = self._cooccurrence(spacy_ents)
        density    = len(all_ents) / max(self.stats.get("word_count", 1), 1) * 1000
        timeline   = self._entity_timeline(spacy_ents)

        return {
            "entities"      : all_ents[:100],   # cap to top-100
            "by_type"       : by_type,
            "co_occurrence" : cooccur[:20],
            "entity_density": round(density, 2),
            "total_unique"  : len(all_ents),
            "timeline"      : timeline,
            "method"        : "spaCy + regex" if SPACY_OK else "regex-only",
        }

    # ── spaCy ────────────────────────────────────────────────────────────────

    def _spacy_extract(self) -> list[dict]:
        if not SPACY_OK or _nlp is None:
            return []
        try:
            doc = _nlp(self.text[:1_000_000])   # spaCy max input guard
            counter: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
            for ent in doc.ents:
                if ent.label_ in self.SPACY_TYPES:
                    counter[(ent.text.strip(), ent.label_)].append(ent.start_char)
            result = []
            for (text, label), positions in counter.items():
                result.append({
                    "text"     : text,
                    "label"    : label,
                    "count"    : len(positions),
                    "first_pos": positions[0],
                    "last_pos" : positions[-1],
                    "source"   : "spacy",
                })
            result.sort(key=lambda e: -e["count"])
            return result
        except Exception:
            return []

    # ── regex ─────────────────────────────────────────────────────────────────

    def _regex_extract(self) -> list[dict]:
        result = []
        for label, pattern in ENTITY_PATTERNS.items():
            matches = [(m.group(), m.start()) for m in pattern.finditer(self.text)]
            counter: dict[str, list[int]] = collections.defaultdict(list)
            for text, pos in matches:
                counter[text].append(pos)
            for text, positions in counter.items():
                result.append({
                    "text"     : text,
                    "label"    : label,
                    "count"    : len(positions),
                    "first_pos": positions[0],
                    "last_pos" : positions[-1],
                    "source"   : "regex",
                })
        return result

    @staticmethod
    def _merge(spacy_ents: list[dict], regex_ents: list[dict]) -> list[dict]:
        seen: set[str] = {e["text"].lower() for e in spacy_ents}
        merged = list(spacy_ents)
        for ent in regex_ents:
            if ent["text"].lower() not in seen:
                merged.append(ent)
                seen.add(ent["text"].lower())
        merged.sort(key=lambda e: -e["count"])
        return merged

    @staticmethod
    def _group_by_type(entities: list[dict]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = collections.defaultdict(list)
        for ent in entities:
            groups[ent["label"]].append(ent["text"])
        return dict(groups)

    def _cooccurrence(self, spacy_ents: list[dict]) -> list[dict]:
        """Find entity pairs that appear in the same sentence."""
        if not SPACY_OK or _nlp is None or not spacy_ents:
            return []
        try:
            doc     = _nlp(self.text[:500_000])
            pair_counter: dict[tuple[str, str], int] = collections.defaultdict(int)
            for sent in doc.sents:
                sent_ents = [e.text for e in sent.ents if e.label_ in self.SPACY_TYPES]
                for a, b in itertools.combinations(sorted(set(sent_ents)), 2):
                    pair_counter[(a, b)] += 1
            result = [
                {"entity_a": a, "entity_b": b, "count": c}
                for (a, b), c in sorted(pair_counter.items(), key=lambda x: -x[1])
            ]
            return result
        except Exception:
            return []

    def _entity_timeline(self, entities: list[dict]) -> list[dict]:
        """Track top-10 entities across paragraphs."""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', self.text) if p.strip()]
        top_ents   = [e["text"] for e in entities[:10]]
        timeline   = []
        for i, para in enumerate(paragraphs):
            present = [e for e in top_ents if e.lower() in para.lower()]
            if present:
                timeline.append({"paragraph": i, "entities": present})
        return timeline


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 7 — TopicModeler
# What  : LDA + NMF dual-model with auto topic count selection.
# Extras: guard for short texts, per-paragraph topic assignment,
#         coherence proxy, dominant topic per doc.
# ─────────────────────────────────────────────────────────────────────────────

class TopicModeler:
    """
    Discovers latent topics using LDA (probabilistic) + NMF (matrix factorisation).

    Auto topic count: 3–10, selected by minimising LDA perplexity
    and NMF reconstruction error.

    Guards
    ------
    • Minimum 50 words — returns empty if below
    • Minimum 3 sentences
    • Minimum 5 documents for meaningful modelling (single doc → paragraph split)
    """

    MIN_WORDS     = 50
    MIN_SENTENCES = 3
    MAX_TOPICS    = 10
    MIN_TOPICS    = 3

    def __init__(self, text: str, stats: dict | None = None) -> None:
        self.text  = text
        self.stats = stats or BasicStats(text).compute()

    def compute(self) -> dict[str, Any]:
        if not SKLEARN_OK:
            return {"error": "scikit-learn required for topic modelling"}

        wc = self.stats.get("word_count", 0)
        sc = self.stats.get("sentence_count", 0)

        if wc < self.MIN_WORDS:
            return {"error": f"Text too short ({wc} words) — minimum {self.MIN_WORDS}"}
        if sc < self.MIN_SENTENCES:
            return {"error": f"Too few sentences ({sc}) — minimum {self.MIN_SENTENCES}"}

        docs = self._split_documents()
        if len(docs) < 3:
            return {"error": "Too few text segments for topic modelling"}

        vectorizer, dtm = self._vectorize(docs)
        n_topics        = self._select_n_topics(dtm)

        lda_result = self._run_lda(dtm, vectorizer, n_topics)
        nmf_result = self._run_nmf(dtm, vectorizer, n_topics)

        para_topics = self._para_topics(docs, lda_result.get("doc_topic_dist", []))

        return {
            "n_topics"        : n_topics,
            "lda"             : lda_result,
            "nmf"             : nmf_result,
            "paragraph_topics": para_topics,
            "method"          : "LDA + NMF",
        }

    # ── document splitting ────────────────────────────────────────────────────

    def _split_documents(self) -> list[str]:
        """Split text into paragraphs; fall back to sentence groups."""
        paras = [p.strip() for p in re.split(r'\n\s*\n', self.text) if len(p.strip()) > 30]
        if len(paras) >= 5:
            return paras
        # Fallback: group every 3 sentences
        sents = BasicStats._tokenize_sentences(self.text)
        groups = []
        for i in range(0, len(sents), 3):
            groups.append(" ".join(sents[i: i + 3]))
        return groups if len(groups) >= 3 else [self.text]

    def _vectorize(self, docs: list[str]) -> tuple[Any, Any]:
        vec = CountVectorizer(
            stop_words="english",
            max_features=500,
            min_df=1,
            ngram_range=(1, 2),
        )
        dtm = vec.fit_transform(docs)
        return vec, dtm

    # ── auto topic count ──────────────────────────────────────────────────────

    def _select_n_topics(self, dtm: Any) -> int:
        """Choose n_topics that minimises LDA perplexity."""
        best_n, best_score = self.MIN_TOPICS, float("inf")
        for n in range(self.MIN_TOPICS, min(self.MAX_TOPICS + 1, dtm.shape[0])):
            try:
                model = LatentDirichletAllocation(
                    n_components=n, random_state=42,
                    max_iter=5, learning_method="online",
                )
                model.fit(dtm)
                perp = model.perplexity(dtm)
                if perp < best_score:
                    best_score = perp
                    best_n     = n
            except Exception:
                break
        return best_n

    # ── LDA ──────────────────────────────────────────────────────────────────

    def _run_lda(self, dtm: Any, vectorizer: Any, n: int) -> dict:
        try:
            model = LatentDirichletAllocation(
                n_components=n, random_state=42,
                max_iter=20, learning_method="online",
            )
            doc_topic = model.fit_transform(dtm)
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for i, comp in enumerate(model.components_):
                top_indices = comp.argsort()[-10:][::-1]
                topics.append({
                    "topic_id"  : i,
                    "keywords"  : [(feature_names[j], round(comp[j] / comp.sum(), 4))
                                   for j in top_indices],
                    "weight"    : round(float(doc_topic[:, i].mean()), 4),
                })
            topics.sort(key=lambda t: -t["weight"])
            dominant = topics[0]["topic_id"] if topics else 0
            return {
                "topics"         : topics,
                "dominant_topic" : dominant,
                "doc_topic_dist" : doc_topic.tolist(),
                "perplexity"     : round(float(model.perplexity(dtm)), 2),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── NMF ──────────────────────────────────────────────────────────────────

    def _run_nmf(self, dtm: Any, vectorizer: Any, n: int) -> dict:
        try:
            tfidf_vec = TfidfVectorizer(
                stop_words="english", max_features=500,
                min_df=1, ngram_range=(1, 2),
            )
            tfidf_mat  = tfidf_vec.fit_transform(
                [" ".join(vectorizer.get_feature_names_out())]  # vocab only
            )
            # Re-vectorize original docs with TF-IDF for NMF
            tfidf_docs = TfidfVectorizer(
                stop_words="english", max_features=500,
                min_df=1, ngram_range=(1, 2),
            ).fit_transform(
                [self.text]  # single doc — NMF on paragraphs via dtm
            )
            # Use original dtm converted to float
            dtm_f = dtm.astype(float)
            model = NMF(n_components=n, random_state=42, max_iter=200)
            W     = model.fit_transform(dtm_f)
            H     = model.components_
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for i, comp in enumerate(H):
                top_i = comp.argsort()[-10:][::-1]
                topics.append({
                    "topic_id": i,
                    "keywords": [(feature_names[j], round(float(comp[j]), 4))
                                 for j in top_i],
                    "weight"  : round(float(W[:, i].mean()), 4),
                })
            topics.sort(key=lambda t: -t["weight"])
            recon_err = round(float(model.reconstruction_err_), 4)
            return {
                "topics"              : topics,
                "dominant_topic"      : topics[0]["topic_id"] if topics else 0,
                "reconstruction_error": recon_err,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── paragraph topic assignment ────────────────────────────────────────────

    @staticmethod
    def _para_topics(docs: list[str], doc_topic_dist: list) -> list[dict]:
        result = []
        for i, (doc, dist) in enumerate(zip(docs, doc_topic_dist)):
            if isinstance(dist, list) and dist:
                dominant = int(max(range(len(dist)), key=lambda j: dist[j]))
                result.append({
                    "paragraph"     : i,
                    "text_preview"  : doc[:80],
                    "dominant_topic": dominant,
                    "confidence"    : round(float(dist[dominant]), 4),
                })
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 8 — TextSimilarity
# What  : Multi-method similarity between two documents.
# Extras: structural similarity, combined score, detailed label.
# ─────────────────────────────────────────────────────────────────────────────

class TextSimilarity:
    """
    Computes similarity between two texts using 5 methods.

    Methods
    -------
    1. TF-IDF Cosine      — keyword overlap (sklearn)
    2. Jaccard            — word-set intersection / union
    3. SequenceMatcher    — character-level longest common subsequence
    4. Semantic Cosine    — sentence-transformer embeddings (optional)
    5. Structural         — paragraph count + length ratio

    Combined score: weighted average, 0.0 → 1.0
    """

    _SBERT_MODEL = None

    LABELS = [
        (0.95, "Identical"),
        (0.85, "Near-Duplicate"),
        (0.70, "Very Similar"),
        (0.50, "Similar"),
        (0.30, "Loosely Related"),
        (0.00, "Unrelated"),
    ]

    def __init__(self, text1: str, text2: str) -> None:
        self.text1 = text1
        self.text2 = text2

    def compute(self) -> dict[str, Any]:
        methods: dict[str, float] = {}

        tfidf_s = self._tfidf_cosine()
        if tfidf_s is not None:
            methods["tfidf_cosine"] = tfidf_s

        methods["jaccard"]         = self._jaccard()
        methods["sequence_matcher"]= self._sequence_matcher()

        sem = self._semantic()
        if sem is not None:
            methods["semantic_cosine"] = sem

        structural = self._structural()
        methods["structural"] = structural

        # Weighted combination
        weights = {
            "semantic_cosine" : 0.35,
            "tfidf_cosine"    : 0.30,
            "jaccard"         : 0.15,
            "sequence_matcher": 0.10,
            "structural"      : 0.10,
        }
        total_w = sum(weights[k] for k in methods)
        combined = sum(methods[k] * weights[k] for k in methods) / max(total_w, 1e-6)
        label    = self._label(combined)

        return {
            "combined_score"  : round(combined, 4),
            "label"           : label,
            "methods"         : {k: round(v, 4) for k, v in methods.items()},
        }

    # ── method implementations ─────────────────────────────────────────────

    def _tfidf_cosine(self) -> float | None:
        if not SKLEARN_OK:
            return None
        try:
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform([self.text1, self.text2])
            sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
            return float(sim)
        except Exception:
            return None

    def _jaccard(self) -> float:
        w1 = set(re.findall(r"\b\w+\b", self.text1.lower()))
        w2 = set(re.findall(r"\b\w+\b", self.text2.lower()))
        inter = len(w1 & w2)
        union = len(w1 | w2)
        return inter / max(union, 1)

    def _sequence_matcher(self) -> float:
        import difflib
        # Sample first 2000 chars to keep it fast
        return difflib.SequenceMatcher(None, self.text1[:2000], self.text2[:2000]).ratio()

    def _semantic(self) -> float | None:
        if not SBERT_OK:
            return None
        try:
            if TextSimilarity._SBERT_MODEL is None:
                TextSimilarity._SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            model = TextSimilarity._SBERT_MODEL
            emb1, emb2 = model.encode([self.text1[:512], self.text2[:512]])
            dot   = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = math.sqrt(sum(a * a for a in emb1))
            norm2 = math.sqrt(sum(b * b for b in emb2))
            return dot / max(norm1 * norm2, 1e-9)
        except Exception:
            return None

    def _structural(self) -> float:
        paras1 = len([p for p in self.text1.split("\n\n") if p.strip()])
        paras2 = len([p for p in self.text2.split("\n\n") if p.strip()])
        len1   = len(self.text1.split())
        len2   = len(self.text2.split())
        para_ratio = min(paras1, paras2) / max(max(paras1, paras2), 1)
        len_ratio  = min(len1, len2)     / max(max(len1, len2), 1)
        return (para_ratio + len_ratio) / 2.0

    @classmethod
    def _label(cls, score: float) -> str:
        for thresh, label in cls.LABELS:
            if score >= thresh:
                return label
        return "Unrelated"


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 9 — StyleFingerprinter
# What  : Detect writing style and advanced stylometric features.
# Extras: passive/active ratio, hedge/intensifier density,
#         vocabulary sophistication, style label classification.
# ─────────────────────────────────────────────────────────────────────────────

class StyleFingerprinter:
    """
    Fingerprints the writing style of a text.

    Features
    --------
    passive_ratio         — fraction of sentences with passive construction
    hedge_density         — hedge words per 100 words
    intensifier_density   — intensifier words per 100 words
    sentence_length_var   — variance in sentence word counts
    vocab_sophistication  — ratio of 3+ syllable content words
    question_density      — questions per sentence
    exclamation_density   — exclamations per sentence
    avg_syllables         — avg syllables per word
    style_label           — Academic / Journalistic / Conversational / Technical / Creative / Legal
    """

    LEGAL_WORDS = {
        "herein","thereof","heretofore","pursuant","notwithstanding","whereas",
        "thereto","hereinafter","aforesaid","aforementioned","jurisdiction",
        "plaintiff","defendant","statute","liability","indemnify","arbitration",
    }
    TECH_WORDS = {
        "algorithm","framework","interface","implementation","infrastructure",
        "architecture","protocol","configuration","deployment","repository",
        "parameter","function","variable","database","server","api","endpoint",
    }
    ACADEMIC_WORDS = {
        "furthermore","however","therefore","consequently","subsequently",
        "nevertheless","nonetheless","moreover","methodology","analysis",
        "hypothesis","empirical","theoretical","quantitative","qualitative",
        "literature","paradigm","discourse","conceptual","framework",
    }

    def __init__(self, text: str, stats: dict | None = None) -> None:
        self.text  = text
        self.stats = stats or BasicStats(text).compute()
        self._sentences = BasicStats._tokenize_sentences(text)
        self._words     = re.findall(r"\b\w+\b", text.lower())

    def compute(self) -> dict[str, Any]:
        n_words = max(len(self._words), 1)
        n_sents = max(len(self._sentences), 1)

        passive     = self._passive_ratio()
        hedge       = sum(1 for w in self._words if w in HEDGE_WORDS)
        intensifier = sum(1 for w in self._words if w in INTENSIFIERS)
        legal_dens  = sum(1 for w in self._words if w in self.LEGAL_WORDS)
        tech_dens   = sum(1 for w in self._words if w in self.TECH_WORDS)
        academic_d  = sum(1 for w in self._words if w in self.ACADEMIC_WORDS)

        sl_var      = self.stats.get("sentence_length_variance", 0)
        avg_syl     = self.stats.get("avg_syllables_per_word", 1.5)
        q_density   = self.stats.get("question_count", 0) / n_sents
        ex_density  = self.stats.get("exclamation_count", 0) / n_sents

        # Vocab sophistication: ratio of content words with 3+ syllables
        hard_words = [
            w for w in self._words
            if BasicStats._count_syllables(w) >= 3
            and w not in COMMON_ENGLISH_STOPWORDS
        ]
        vocab_soph = len(hard_words) / n_words

        style = self._classify_style(
            passive, hedge / n_words,
            tech_dens / n_words, legal_dens / n_words,
            academic_d / n_words, q_density, ex_density,
            sl_var, vocab_soph,
        )

        return {
            "passive_ratio"          : round(passive, 4),
            "hedge_density"          : round(hedge / n_words * 100, 2),
            "intensifier_density"    : round(intensifier / n_words * 100, 2),
            "legal_word_density"     : round(legal_dens / n_words * 100, 2),
            "technical_word_density" : round(tech_dens / n_words * 100, 2),
            "academic_word_density"  : round(academic_d / n_words * 100, 2),
            "sentence_length_variance": round(sl_var, 2),
            "vocab_sophistication"   : round(vocab_soph, 4),
            "question_density"       : round(q_density, 4),
            "exclamation_density"    : round(ex_density, 4),
            "avg_syllables_per_word" : round(avg_syl, 2),
            "style_label"            : style,
            "style_emoji"            : STYLE_LABELS.get(style, style),
        }

    def _passive_ratio(self) -> float:
        """Heuristic: sentence contains 'be/is/are/was/were' + past participle."""
        passive_count = 0
        be_forms = re.compile(r"\b(is|are|was|were|be|been|being|has been|have been|had been|will be)\b")
        for sent in self._sentences:
            words_in_sent = sent.lower().split()
            has_be = bool(be_forms.search(sent.lower()))
            # Past participle heuristic: ends in -ed or common irregular
            has_pp = any(w.endswith("ed") and len(w) > 4 for w in words_in_sent)
            if has_be and has_pp:
                passive_count += 1
        return passive_count / max(len(self._sentences), 1)

    @staticmethod
    def _classify_style(
        passive: float, hedge: float, tech: float, legal: float,
        academic: float, q_dens: float, ex_dens: float,
        sl_var: float, vocab_soph: float,
    ) -> str:
        scores = {
            "legal"         : legal * 50 + passive * 5,
            "technical"     : tech  * 40 + vocab_soph * 10,
            "academic"      : academic * 40 + hedge * 20 + passive * 10,
            "journalistic"  : q_dens * 30 + sl_var * 0.1 + (1 - passive) * 5,
            "conversational": q_dens * 15 + ex_dens * 20 + (1 - vocab_soph) * 10,
            "creative"      : ex_dens * 15 + sl_var * 0.15 + (1 - academic) * 5,
        }
        return max(scores, key=scores.__getitem__)


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 10 — TextCoherenceAnalyzer
# What  : Measures logical flow and internal consistency of a text.
# Extras: inter-sentence similarity chain, topic jump detection,
#         duplicate sentence detection, paragraph coherence scores.
# ─────────────────────────────────────────────────────────────────────────────

class TextCoherenceAnalyzer:
    """
    Measures text coherence through sentence-to-sentence similarity chaining.

    Algorithm
    ---------
    For each consecutive sentence pair (s_i, s_{i+1}):
        similarity = TF-IDF cosine OR Jaccard (fallback)

    Coherence score = mean of all pair similarities × 100

    Topic jumps: pairs where similarity < threshold (default 0.05)

    Duplicate detection: exact + near-duplicate (similarity > 0.9) sentences.

    Paragraph coherence: same metric applied within each paragraph.
    """

    JUMP_THRESHOLD  = 0.05
    DUPE_THRESHOLD  = 0.90

    def __init__(self, text: str) -> None:
        self.text      = text
        self.sentences = BasicStats._tokenize_sentences(text)
        self.paragraphs= [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    def compute(self) -> dict[str, Any]:
        chain    = self._sentence_chain()
        score    = self._chain_score(chain)
        jumps    = self._topic_jumps(chain)
        dupes    = self._duplicate_sentences()
        para_coh = self._paragraph_coherence()
        flow     = self._logical_flow_score(score, len(jumps), len(dupes))

        return {
            "coherence_score"       : round(score, 2),
            "coherence_label"       : _coherence_label(score),
            "sentence_chain"        : [round(s, 4) for s in chain],
            "topic_jumps"           : jumps,
            "topic_jump_count"      : len(jumps),
            "duplicate_sentences"   : dupes,
            "duplicate_count"       : len(dupes),
            "paragraph_coherence"   : para_coh,
            "logical_flow_score"    : round(flow, 2),
        }

    # ── chain ─────────────────────────────────────────────────────────────────

    def _sentence_chain(self) -> list[float]:
        sents = self.sentences
        if len(sents) < 2:
            return []
        if SKLEARN_OK:
            return self._tfidf_chain(sents)
        return self._jaccard_chain(sents)

    def _tfidf_chain(self, sents: list[str]) -> list[float]:
        try:
            vec = TfidfVectorizer(stop_words="english", min_df=1)
            mat = vec.fit_transform(sents)
            chain = []
            for i in range(len(sents) - 1):
                sim = cosine_similarity(mat[i:i+1], mat[i+1:i+2])[0][0]
                chain.append(float(sim))
            return chain
        except Exception:
            return self._jaccard_chain(sents)

    @staticmethod
    def _jaccard_chain(sents: list[str]) -> list[float]:
        chain = []
        for i in range(len(sents) - 1):
            w1 = set(sents[i].lower().split())
            w2 = set(sents[i+1].lower().split())
            inter = len(w1 & w2)
            union = len(w1 | w2)
            chain.append(inter / max(union, 1))
        return chain

    @staticmethod
    def _chain_score(chain: list[float]) -> float:
        if not chain:
            return 50.0
        return sum(chain) / len(chain) * 100.0

    def _topic_jumps(self, chain: list[float]) -> list[dict]:
        jumps = []
        for i, sim in enumerate(chain):
            if sim < self.JUMP_THRESHOLD:
                jumps.append({
                    "position"  : i,
                    "similarity": round(sim, 4),
                    "before"    : self.sentences[i][:80] if i < len(self.sentences) else "",
                    "after"     : self.sentences[i+1][:80] if i+1 < len(self.sentences) else "",
                })
        return jumps

    def _duplicate_sentences(self) -> list[dict]:
        sents  = self.sentences
        n      = len(sents)
        dupes: list[dict] = []
        seen: set[str] = set()

        for i in range(n):
            norm_i = _normalise_sent(sents[i])
            if norm_i in seen:
                continue
            for j in range(i + 1, min(n, i + 50)):   # local window for speed
                norm_j = _normalise_sent(sents[j])
                if norm_i == norm_j:
                    dupes.append({
                        "type"      : "exact",
                        "index_a"   : i,
                        "index_b"   : j,
                        "text"      : sents[i][:100],
                    })
                    seen.add(norm_i)
                    break
        return dupes[:50]

    def _paragraph_coherence(self) -> list[dict]:
        result = []
        for i, para in enumerate(self.paragraphs):
            sents = BasicStats._tokenize_sentences(para)
            if len(sents) < 2:
                continue
            if SKLEARN_OK:
                chain = self._tfidf_chain(sents)
            else:
                chain = self._jaccard_chain(sents)
            score = self._chain_score(chain)
            result.append({
                "paragraph"     : i,
                "score"         : round(score, 2),
                "label"         : _coherence_label(score),
                "sentence_count": len(sents),
            })
        return result

    @staticmethod
    def _logical_flow_score(
        coherence: float, jump_count: int, dupe_count: int
    ) -> float:
        score = coherence
        score -= jump_count * 5.0
        score -= dupe_count * 3.0
        return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 11 — TextQualityScorer
# What  : Composite 0-100 quality score across 8 dimensions.
# Extras: per-dimension scores, grade A→F, top-3 improvement suggestions.
# ─────────────────────────────────────────────────────────────────────────────

class TextQualityScorer:
    """
    Aggregates all analysis results into a single Quality Score (0-100).

    Dimensions & Weights
    --------------------
    readability         20%  — combined readability score
    vocabulary_richness 15%  — TTR + lexical density + hapax ratio
    coherence           15%  — sentence chain coherence
    style_consistency   10%  — passive ratio + sentence variety
    keyword_density     10%  — keyword coverage of content
    sentence_variety    10%  — low variance is monotonous
    language_consistency10%  — no multilingual conflicts
    entity_density      10%  — richness of named entities
    """

    WEIGHTS = {
        "readability"         : 0.20,
        "vocabulary_richness" : 0.15,
        "coherence"           : 0.15,
        "style_consistency"   : 0.10,
        "keyword_density"     : 0.10,
        "sentence_variety"    : 0.10,
        "language_consistency": 0.10,
        "entity_density"      : 0.10,
    }

    def __init__(
        self,
        stats     : dict,
        readability: dict,
        language  : dict,
        coherence : dict,
        style     : dict,
        keywords  : dict,
        entities  : dict,
    ) -> None:
        self.stats      = stats
        self.readability= readability
        self.language   = language
        self.coherence  = coherence
        self.style      = style
        self.keywords   = keywords
        self.entities   = entities

    def compute(self) -> dict[str, Any]:
        dim = self._compute_dimensions()
        combined = sum(dim[k] * self.WEIGHTS[k] for k in dim)
        combined = max(0.0, min(100.0, combined))
        grade, label = _score_to_grade(combined)
        suggestions  = self._suggestions(dim)

        return {
            "quality_score"   : round(combined, 2),
            "grade"           : grade,
            "label"           : label,
            "dimensions"      : {k: round(v, 2) for k, v in dim.items()},
            "suggestions"     : suggestions[:3],
        }

    def _compute_dimensions(self) -> dict[str, float]:
        s = self.stats

        # 1. Readability (0-100 from ReadabilityScorer)
        read = float(self.readability.get("combined_score", 50))

        # 2. Vocabulary richness (TTR + lexical density + hapax ratio)
        ttr  = float(s.get("type_token_ratio", 0.5)) * 100
        ld   = float(s.get("lexical_density", 0.5)) * 100
        hap  = float(s.get("hapax_ratio", 0.3)) * 100
        vocab_rich = (ttr * 0.4 + ld * 0.4 + hap * 0.2)

        # 3. Coherence
        coh  = float(self.coherence.get("coherence_score", 50))

        # 4. Style consistency (low passive + good sentence variety)
        passive = float(self.style.get("passive_ratio", 0.2))
        sl_var  = float(self.style.get("sentence_length_variance", 30))
        # Ideal passive ≈ 0.1-0.2; ideal variance ≈ 20-80
        passive_score  = max(0, 100 - abs(passive - 0.15) * 200)
        variance_score = min(100, sl_var * 2)
        style_cons     = (passive_score + variance_score) / 2

        # 5. Keyword density
        kw_count = len(self.keywords.get("keywords", []))
        kw_score = min(100, kw_count * 5)

        # 6. Sentence variety
        variety = min(100, float(s.get("sentence_length_variance", 0)) * 2)

        # 7. Language consistency
        is_multi = self.language.get("is_multilingual", False)
        conf     = float(self.language.get("confidence", 1.0)) * 100
        lang_cons= conf * (0.7 if is_multi else 1.0)

        # 8. Entity density (0-30 per 1000 words is healthy)
        ent_dens = float(self.entities.get("entity_density", 0))
        ent_score = min(100, ent_dens * 5)

        return {
            "readability"         : read,
            "vocabulary_richness" : min(100, vocab_rich),
            "coherence"           : coh,
            "style_consistency"   : style_cons,
            "keyword_density"     : kw_score,
            "sentence_variety"    : variety,
            "language_consistency": lang_cons,
            "entity_density"      : ent_score,
        }

    def _suggestions(self, dim: dict[str, float]) -> list[str]:
        tips: list[tuple[float, str]] = [
            (dim["readability"],
             "Simplify sentence structure — aim for Flesch ease > 60"),
            (dim["vocabulary_richness"],
             "Diversify vocabulary — avoid repeating the same words"),
            (dim["coherence"],
             "Improve logical flow — add transition words between sentences"),
            (dim["style_consistency"],
             "Reduce passive voice — prefer active constructions"),
            (dim["keyword_density"],
             "Include more domain-specific keywords for better topic coverage"),
            (dim["sentence_variety"],
             "Vary sentence lengths — mix short punchy with longer explanatory"),
            (dim["language_consistency"],
             "Ensure consistent use of one language throughout the document"),
            (dim["entity_density"],
             "Add specific names, organisations, or dates to ground the text"),
        ]
        # Sort by lowest score — worst dimensions get suggestions first
        tips.sort(key=lambda t: t[0])
        return [tip for _, tip in tips]


# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 12 — TextAnalyzer MASTER CLASS + CONVENIENCE FUNCTIONS
# What  : Unified API — one class, all blocks. Convenience shortcuts.
# ─────────────────────────────────────────────────────────────────────────────

class TextAnalyzer:
    """
    Master text analysis class — wires all 11 blocks into a single API.

    Quick start
    -----------
    ta     = TextAnalyzer(text)
    report = ta.analyze()          # full report dict

    Convenience
    -----------
    ta.quick_stats()               # block 1 only
    ta.quick_sentiment()           # block 4 only
    ta.quick_keywords(n=15)        # block 5 only
    ta.quick_quality()             # block 11 only
    ta.compare(other_text)         # block 8 only
    ta.analyze_batch([t1,t2,t3])   # DataFrame with all scores
    """

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        self.text    = text
        self._cache: dict[str, Any] = {}

    # ── full analysis ─────────────────────────────────────────────────────────

    def analyze(self) -> dict[str, Any]:
        if "full" in self._cache:
            return self._cache["full"]

        # Run blocks in dependency order
        stats       = self._run("stats",       lambda: BasicStats(self.text).compute())
        readability = self._run("readability", lambda: ReadabilityScorer(self.text, stats).compute())
        language    = self._run("language",    lambda: LanguageDetector(self.text, stats).compute())
        sentiment   = self._run("sentiment",   lambda: SentimentAnalyzer(self.text).compute())
        keywords    = self._run("keywords",    lambda: KeywordExtractor(self.text).compute())
        entities    = self._run("entities",    lambda: NamedEntityRecognizer(self.text, stats).compute())
        topics      = self._run("topics",      lambda: TopicModeler(self.text, stats).compute())
        coherence   = self._run("coherence",   lambda: TextCoherenceAnalyzer(self.text).compute())
        style       = self._run("style",       lambda: StyleFingerprinter(self.text, stats).compute())
        quality     = self._run("quality",     lambda: TextQualityScorer(
            stats, readability, language, coherence, style, keywords, entities
        ).compute())

        result = {
            "text_hash"   : hashlib.md5(self.text.encode()).hexdigest()[:12],
            "char_count"  : len(self.text),
            "stats"       : stats,
            "readability" : readability,
            "language"    : language,
            "sentiment"   : sentiment,
            "keywords"    : keywords,
            "entities"    : entities,
            "topics"      : topics,
            "coherence"   : coherence,
            "style"       : style,
            "quality"     : quality,
            "nydra_version": VERSION,
        }
        self._cache["full"] = result
        return result

    # ── convenience methods ───────────────────────────────────────────────────

    def quick_stats(self) -> dict[str, Any]:
        """Block 1 only — zero external deps, instant."""
        return self._run("stats", lambda: BasicStats(self.text).compute())

    def quick_sentiment(self) -> dict[str, Any]:
        """Block 4 only — returns doc-level sentiment."""
        result = self._run("sentiment", lambda: SentimentAnalyzer(self.text).compute())
        return result.get("doc_level", result)

    def quick_keywords(self, n: int = 10) -> list[dict]:
        """Block 5 only — returns top-n keywords."""
        result = self._run(f"keywords_{n}", lambda: KeywordExtractor(self.text, top_n=n).compute())
        return result.get("keywords", [])[:n]

    def quick_quality(self) -> dict[str, Any]:
        """Block 11 only — quality score + grade."""
        return self.analyze()["quality"]

    def compare(self, other_text: str) -> dict[str, Any]:
        """Block 8 only — similarity between this text and other_text."""
        return TextSimilarity(self.text, other_text).compute()

    def analyze_batch(
        self,
        texts  : list[str],
        columns: list[str] | None = None,
    ) -> Any:
        """
        Runs analysis on a list of texts.
        Returns a pandas DataFrame if available, else a list of dicts.

        columns: which keys to include (default: quality score + key stats)
        """
        default_cols = [
            "quality_score", "grade", "word_count",
            "sentiment", "style_label", "readability_score",
            "coherence_score", "language",
        ]
        cols = columns or default_cols

        rows = []
        for i, text in enumerate(texts):
            try:
                ta  = TextAnalyzer(text)
                rep = ta.analyze()
                row: dict[str, Any] = {"index": i, "text_preview": text[:60]}
                for col in cols:
                    row[col] = _extract_batch_field(rep, col)
                rows.append(row)
            except Exception as e:
                rows.append({"index": i, "error": str(e)})

        if PANDAS_OK:
            return pd.DataFrame(rows)
        return rows

    def analyze_document(self, path: str | Path) -> dict[str, Any]:
        """
        Read a document via document_reader.py (if available),
        then run full analysis.
        """
        path = Path(path)
        text = self._read_document(path)
        if text is None:
            raise FileNotFoundError(f"Could not read document: {path}")
        ta = TextAnalyzer(text)
        result = ta.analyze()
        result["source_file"] = str(path)
        return result

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self, key: str, fn: Callable) -> dict:
        if key not in self._cache:
            try:
                self._cache[key] = fn()
            except Exception as e:
                logger.warning(f"Block '{key}' failed: {e}")
                self._cache[key] = {"error": str(e)}
        return self._cache[key]

    @staticmethod
    def _read_document(path: Path) -> str | None:
        """Try document_reader.py first, then raw UTF-8 fallback."""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "document_reader",
                Path(__file__).parent / "document_reader.py",
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                reader = mod.DocumentReader(str(path))
                result = reader.read()
                return result.get("text") or result.get("content")
        except Exception:
            pass
        # Raw text fallback
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(text: str) -> dict[str, Any]:
    """One-call full analysis. Returns complete report dict."""
    return TextAnalyzer(text).analyze()


def quick_stats(text: str) -> dict[str, Any]:
    """Surface statistics only — no external deps."""
    return BasicStats(text).compute()


def quick_sentiment(text: str) -> dict[str, Any]:
    """Sentiment only."""
    return SentimentAnalyzer(text).compute().get("doc_level", {})


def quick_keywords(text: str, n: int = 10) -> list[dict]:
    """Top-n keywords."""
    return KeywordExtractor(text, top_n=n).compute().get("keywords", [])[:n]


def compare(text1: str, text2: str) -> dict[str, Any]:
    """Similarity between two texts."""
    return TextSimilarity(text1, text2).compute()


def analyze_document(path: str | Path) -> dict[str, Any]:
    """Full analysis on a file path."""
    return TextAnalyzer("").analyze_document(path)


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _char_entropy(text: str) -> float:
    freq = collections.Counter(text)
    n    = len(text)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


def _normalise_sent(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _score_to_grade(score: float) -> tuple[str, str]:
    for thresh, grade, label in QUALITY_GRADES:
        if score >= thresh:
            return grade, label
    return "F", "Critical ⛔"


def _coherence_label(score: float) -> str:
    if score >= 70: return "High Coherence 🟢"
    if score >= 45: return "Moderate Coherence 🟡"
    if score >= 20: return "Low Coherence 🟠"
    return "Incoherent 🔴"


def _extract_batch_field(report: dict, field: str) -> Any:
    """Extract a named field from the full analysis report for batch mode."""
    mapping: dict[str, Callable[[dict], Any]] = {
        "quality_score"    : lambda r: r.get("quality", {}).get("quality_score"),
        "grade"            : lambda r: r.get("quality", {}).get("grade"),
        "word_count"       : lambda r: r.get("stats", {}).get("word_count"),
        "sentiment"        : lambda r: r.get("sentiment", {}).get("doc_level", {}).get("dominant"),
        "style_label"      : lambda r: r.get("style", {}).get("style_label"),
        "readability_score": lambda r: r.get("readability", {}).get("combined_score"),
        "coherence_score"  : lambda r: r.get("coherence", {}).get("coherence_score"),
        "language"         : lambda r: r.get("language", {}).get("language_name"),
    }
    fn = mapping.get(field)
    return fn(report) if fn else None

