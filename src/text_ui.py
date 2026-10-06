"""text_ui.py - JSON-safe Text Intelligence payload for the dashboard."""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Models are already cached locally; never download during a request.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MAX_CHARS = 30_000
_LOCK = threading.Lock()
_MODS: Dict[str, Any] = {}

_ENGLISH_ONLY_CHECKERS = {
    "PIIDetector", "CoherenceScorer", "LanguageQualityScorer",
    "EmptyContentDetector", "ConsistencyChecker",
}
_CONTACT_LABELS = {"EMAIL", "PHONE", "URL"}


def _clean(o: Any, depth: int = 0) -> Any:
    """Convert dataclasses, Enums, numpy values, NaN into plain JSON types."""
    if depth > 8:
        return str(o)
    if isinstance(o, enum.Enum):
        return _clean(o.value, depth + 1)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {f.name: _clean(getattr(o, f.name), depth + 1) for f in dataclasses.fields(o)}
    if isinstance(o, dict):
        return {str(k): _clean(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v, depth + 1) for v in o]
    if hasattr(o, "tolist"):
        try:
            return _clean(o.tolist(), depth + 1)
        except Exception:
            return str(o)
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if o is None or isinstance(o, (bool, int, str)):
        return o
    return str(o)


def _modules() -> Dict[str, Any]:
    if not _MODS:
        from src.vision_nlp import text_quality as tq
        # Disable the 568 MB NLI contradiction model (and its fallback download).
        tq.ConsistencyChecker._load_nli = lambda self: False
        from src.vision_nlp.text_analyzer import TextAnalyzer
        _MODS["analyzer"] = TextAnalyzer
        _MODS["checker"] = tq.SmartTextQualityChecker()
    return _MODS


def read_text(path: str) -> Tuple[str, str]:
    """Return (text, encoding_used) for txt/md/pdf/docx."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".txt", ".md", ".text", ".csv"):
        raw = p.read_bytes()
        for enc in ("utf-8-sig", "cp1256", "latin-1"):
            try:
                return raw.decode(enc), enc
            except UnicodeDecodeError:
                continue
    from src.vision_nlp.document_reader import quick_read
    return quick_read(str(p)), "document_reader"


def build_text_report(text: str, name: str = "document") -> Dict[str, Any]:
    text = (text or "").strip()
    if len(text) < 20:
        return {"status": "error", "message": "Text is too short to analyze."}
    truncated = len(text) > MAX_CHARS
    if truncated:
        text = text[:MAX_CHARS]

    t0 = time.time()
    with _LOCK:
        m = _modules()
        intel = _clean(m["analyzer"](text).analyze())
        qual = _clean(m["checker"].check(text, verbose=False))

    lang = intel.get("language") or {}
    code = lang.get("language_code") or "unknown"
    lang_name = lang.get("language_name") or code
    english = code == "en"

    st = intel.get("stats") or {}
    stats = {k: st.get(k) for k in (
        "word_count", "sentence_count", "paragraph_count", "char_count",
        "avg_word_length", "avg_sentence_length", "vocabulary_size",
        "type_token_ratio", "top_20_words")}

    bd = (qual.get("score_info") or {}).get("breakdown") or {}
    issues = qual.get("all_issues") or []
    if english:
        score, verdict, scored_on = qual.get("overall_score"), qual.get("verdict"), sorted(bd.keys())
    else:
        weights = {"duplicate_score": 0.4, "noise_score": 0.4, "encoding_score": 0.2}
        parts = {k: bd[k] for k in weights if isinstance(bd.get(k), (int, float))}
        tw = sum(weights[k] for k in parts)
        score = round(sum(parts[k] * weights[k] for k in parts) / tw, 1) if tw else None
        verdict = None if score is None else ("Good" if score >= 80 else "Needs Work" if score >= 50 else "Poor")
        scored_on = sorted(parts.keys())
        issues = [i for i in issues if isinstance(i, dict) and i.get("checker") not in _ENGLISH_ONLY_CHECKERS]
    quality = {
        "score": score, "verdict": verdict, "scored_on": scored_on,
        "breakdown": bd, "issue_count": len(issues), "issues": issues[:30],
    }

    ents = (intel.get("entities") or {}).get("entities") or []
    hidden: List[Dict[str, str]] = []
    if english:
        kws = (intel.get("keywords") or {}).get("keywords") or []
        topics = intel.get("topics") or {}
        rd = intel.get("readability") or {}
        wc = st.get("word_count") or 0
        pii = qual.get("pii_info") or {}
        langq = qual.get("language_result") or {}
        intelligence = {
            "sentiment": (intel.get("sentiment") or {}).get("doc_level"),
            "readability": {"score": rd.get("combined_score"), "grade": rd.get("grade_level"),
                            "indices": rd.get("indices")},
            "keywords": [{"keyword": k.get("keyword"), "score": round(float(k.get("avg_score") or 0), 3)}
                         for k in kws[:12]],
            "entities": (intel.get("entities") or {}).get("by_type"),
            "topics": ({"note": topics["error"]} if "error" in topics else topics),
            "style": (intel.get("style") or {}).get("style_label"),
            "coherence": {"score": (intel.get("coherence") or {}).get("coherence_score"),
                          "reliable": wc >= 100},
            "pii": {"by_type": pii.get("pii_by_type"), "score": pii.get("score")},
            "spelling": {"error_rate": langq.get("spelling_error_rate"),
                         "errors": (langq.get("spelling_errors") or [])[:20]},
        }
    else:
        why = "Built for English; detected language: %s." % lang_name
        intelligence = {
            "contact_info": [{"text": e.get("text"), "label": e.get("label")}
                             for e in ents if e.get("label") in _CONTACT_LABELS],
        }
        for sec in ("Sentiment", "Readability", "Keywords", "Spelling and grammar",
                    "Coherence", "Names and places", "Person/organization PII"):
            hidden.append({"section": sec, "reason": why})

    return {
        "status": "success", "name": name,
        "language": {"code": code, "name": lang_name, "script": lang.get("script"),
                     "confidence": lang.get("confidence"), "fully_supported": english},
        "stats": stats, "quality": quality, "intelligence": intelligence,
        "hidden": hidden, "truncated": truncated,
        "seconds": round(time.time() - t0, 2),
    }


def build_text_report_from_file(path: str, name: str | None = None) -> Dict[str, Any]:
    p = Path(path)
    try:
        text, enc = read_text(str(p))
    except Exception as e:
        return {"status": "error", "message": "Could not read file: %s" % type(e).__name__}
    rep = build_text_report(text, name or p.name)
    if rep.get("status") == "success":
        rep["file"] = {"name": p.name, "type": p.suffix.lower().lstrip("."), "encoding": enc}
    return rep
