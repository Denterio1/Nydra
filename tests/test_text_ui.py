"""Tests for src/text_ui.py. The heavy NLP models are faked (via _modules), so these
run in about a second and need no torch/spaCy/transformers."""
import dataclasses
import enum
import json
import sys
import types

import pytest

np = pytest.importorskip("numpy")
text_ui = pytest.importorskip("src.text_ui")


# ------------------------------------------------------------------ test doubles
class Color(enum.Enum):
    RED = "red"


@dataclasses.dataclass
class Box:
    name: str
    value: float


class Weird:
    def __repr__(self):
        return "<weird>"


GOOD_TEXT = "This is a perfectly normal sentence that is long enough to analyze."


def english_intel(word_count=150, **over):
    intel = {
        "language": {"language_code": "en", "language_name": "English", "script": "Latin", "confidence": 0.99},
        "stats": {"word_count": word_count, "sentence_count": 10, "paragraph_count": 2, "char_count": 900,
                  "avg_word_length": 4.5, "avg_sentence_length": 15.0, "vocabulary_size": 80,
                  "type_token_ratio": 0.5, "top_20_words": [["the", 9]], "ignored_key": "x"},
        "sentiment": {"doc_level": {"label": "positive", "score": 0.9}},
        "readability": {"combined_score": 62.5, "grade_level": 8, "indices": {"flesch": 62}},
        "keywords": {"keywords": [{"keyword": "data", "avg_score": 0.123456}]},
        "entities": {"entities": [], "by_type": {"ORG": ["Acme"]}},
        "topics": {},
        "style": {"style_label": "formal"},
        "coherence": {"coherence_score": 0.7},
    }
    intel.update(over)
    return intel


def foreign_intel(code="ar", name="Arabic", entities=None):
    return {
        "language": {"language_code": code, "language_name": name, "script": "Arabic", "confidence": 1.0},
        "stats": {"word_count": 40},
        "entities": {"entities": entities or [], "by_type": {}},
    }


def quality(breakdown=None, issues=None, score=88.1, verdict="Good"):
    return {
        "overall_score": score, "verdict": verdict,
        "score_info": {"breakdown": breakdown if breakdown is not None else
                       {"duplicate_score": 100, "noise_score": 90, "encoding_score": 100, "spelling_score": 70}},
        "all_issues": issues or [],
        "pii_info": {"pii_by_type": {"email": 1}, "score": 80},
        "language_result": {"spelling_error_rate": 0.02, "spelling_errors": ["teh"] * 25},
    }


def install(monkeypatch, intel, qual):
    """Replace the model loader with fakes. Returns a dict recording what they saw."""
    seen = {"texts": [], "check_calls": []}

    class FakeAnalyzer:
        def __init__(self, text):
            seen["texts"].append(text)

        def analyze(self):
            return intel

    class FakeChecker:
        def check(self, text, verbose=False):
            seen["check_calls"].append((text, verbose))
            return qual

    monkeypatch.setattr(text_ui, "_modules", lambda: {"analyzer": FakeAnalyzer, "checker": FakeChecker()})
    return seen


# ------------------------------------------------------------------------ _clean
def test_clean_enum_uses_value():
    assert text_ui._clean(Color.RED) == "red"


def test_clean_dataclass_becomes_dict():
    assert text_ui._clean(Box("a", 1.5)) == {"name": "a", "value": 1.5}


def test_clean_numpy_values():
    assert text_ui._clean(np.array([1, 2, 3])) == [1, 2, 3]
    assert text_ui._clean(np.int64(3)) == 3
    assert text_ui._clean(np.array([float("nan"), 1.0])) == [None, 1.0]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_clean_nan_and_inf_become_none(bad):
    assert text_ui._clean(bad) is None


def test_clean_keeps_plain_values():
    assert text_ui._clean(True) is True
    assert text_ui._clean(0) == 0
    assert text_ui._clean("x") == "x"
    assert text_ui._clean(None) is None
    assert text_ui._clean(1.25) == 1.25


def test_clean_dict_keys_become_strings():
    assert text_ui._clean({1: "a", (2, 3): "b"}) == {"1": "a", "(2, 3)": "b"}


def test_clean_tuple_and_set_become_lists():
    assert text_ui._clean((1, 2)) == [1, 2]
    assert text_ui._clean({5}) == [5]


def test_clean_unknown_object_becomes_string():
    assert text_ui._clean(Weird()) == "<weird>"


def test_clean_depth_limit_stops_recursion():
    o = 1
    for _ in range(12):
        o = [o]
    out = text_ui._clean(o)
    while isinstance(out, list):
        out = out[0]
    assert isinstance(out, str)


def test_clean_result_is_json_safe():
    messy = {"e": Color.RED, "b": Box("x", float("nan")), "a": np.array([1.5, float("inf")]), "w": Weird(), "t": (1, {2})}
    json.dumps(text_ui._clean(messy))


# --------------------------------------------------------------------- read_text
def test_read_text_utf8(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("hello world".encode("utf-8"))
    assert text_ui.read_text(str(p)) == ("hello world", "utf-8-sig")


def test_read_text_strips_bom(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"\xef\xbb\xbfhello")
    text, enc = text_ui.read_text(str(p))
    assert text == "hello" and enc == "utf-8-sig"


def test_read_text_arabic_cp1256_fallback(tmp_path):
    p = tmp_path / "ar.txt"
    arabic = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
    p.write_bytes(arabic.encode("cp1256"))
    text, enc = text_ui.read_text(str(p))
    assert enc == "cp1256"
    assert text == arabic


def test_read_text_utf8_arabic(tmp_path):
    p = tmp_path / "ar.txt"
    arabic = "\u0645\u0631\u062d\u0628\u0627"
    p.write_text(arabic, encoding="utf-8")
    assert text_ui.read_text(str(p)) == (arabic, "utf-8-sig")


@pytest.mark.parametrize("ext", [".md", ".text", ".csv", ".TXT"])
def test_read_text_plain_extensions(tmp_path, ext):
    p = tmp_path / ("f" + ext)
    p.write_text("some content", encoding="utf-8")
    assert text_ui.read_text(str(p)) == ("some content", "utf-8-sig")


def test_read_text_other_formats_use_document_reader(tmp_path, monkeypatch):
    called = {}
    fake_reader = types.ModuleType("src.vision_nlp.document_reader")

    def quick_read(path):
        called["path"] = path
        return "text from pdf"

    fake_reader.quick_read = quick_read
    fake_pkg = types.ModuleType("src.vision_nlp")
    fake_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "src.vision_nlp", fake_pkg)
    monkeypatch.setitem(sys.modules, "src.vision_nlp.document_reader", fake_reader)
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert text_ui.read_text(str(p)) == ("text from pdf", "document_reader")
    assert called["path"] == str(p)


# --------------------------------------------------------------- build: guards
@pytest.mark.parametrize("bad", [None, "", "   ", "short text", " " * 50 + "abc"])
def test_too_short_text_is_an_error(bad):
    rep = text_ui.build_text_report(bad)
    assert rep == {"status": "error", "message": "Text is too short to analyze."}


def test_short_text_never_loads_models(monkeypatch):
    def boom():
        raise AssertionError("models must not load for short text")
    monkeypatch.setattr(text_ui, "_modules", boom)
    assert text_ui.build_text_report("tiny")["status"] == "error"


def test_long_text_is_truncated(monkeypatch):
    seen = install(monkeypatch, english_intel(), quality())
    rep = text_ui.build_text_report("a" * (text_ui.MAX_CHARS + 500))
    assert rep["truncated"] is True
    assert len(seen["texts"][0]) == text_ui.MAX_CHARS


def test_normal_text_not_truncated_and_stripped(monkeypatch):
    seen = install(monkeypatch, english_intel(), quality())
    rep = text_ui.build_text_report("   " + GOOD_TEXT + "  \n")
    assert rep["truncated"] is False
    assert seen["texts"][0] == GOOD_TEXT
    assert seen["check_calls"] == [(GOOD_TEXT, False)]


# ----------------------------------------------------------------- build: English
def test_english_report_shape(monkeypatch):
    install(monkeypatch, english_intel(), quality())
    rep = text_ui.build_text_report(GOOD_TEXT, "my doc")
    assert rep["status"] == "success"
    assert rep["name"] == "my doc"
    assert rep["language"] == {"code": "en", "name": "English", "script": "Latin",
                               "confidence": 0.99, "fully_supported": True}
    assert rep["hidden"] == []
    assert rep["seconds"] >= 0
    json.dumps(rep)


def test_default_name(monkeypatch):
    install(monkeypatch, english_intel(), quality())
    assert text_ui.build_text_report(GOOD_TEXT)["name"] == "document"


def test_english_stats_keep_only_known_keys(monkeypatch):
    install(monkeypatch, english_intel(), quality())
    stats = text_ui.build_text_report(GOOD_TEXT)["stats"]
    assert stats["word_count"] == 150
    assert "ignored_key" not in stats
    assert stats["top_20_words"] == [["the", 9]]


def test_english_quality_comes_from_the_checker(monkeypatch):
    install(monkeypatch, english_intel(), quality(score=88.1, verdict="Good"))
    q = text_ui.build_text_report(GOOD_TEXT)["quality"]
    assert q["score"] == 88.1
    assert q["verdict"] == "Good"
    assert q["scored_on"] == ["duplicate_score", "encoding_score", "noise_score", "spelling_score"]


def test_english_issues_capped_at_30_but_counted_fully(monkeypatch):
    issues = [{"checker": "grammar", "issue_type": "t%d" % n} for n in range(45)]
    install(monkeypatch, english_intel(), quality(issues=issues))
    q = text_ui.build_text_report(GOOD_TEXT)["quality"]
    assert q["issue_count"] == 45
    assert len(q["issues"]) == 30


def test_english_intelligence_fields(monkeypatch):
    install(monkeypatch, english_intel(), quality())
    intel = text_ui.build_text_report(GOOD_TEXT)["intelligence"]
    assert intel["sentiment"] == {"label": "positive", "score": 0.9}
    assert intel["readability"] == {"score": 62.5, "grade": 8, "indices": {"flesch": 62}}
    assert intel["keywords"] == [{"keyword": "data", "score": 0.123}]
    assert intel["entities"] == {"ORG": ["Acme"]}
    assert intel["style"] == "formal"
    assert intel["pii"] == {"by_type": {"email": 1}, "score": 80}
    assert intel["spelling"]["error_rate"] == 0.02
    assert len(intel["spelling"]["errors"]) == 20


@pytest.mark.parametrize("words,reliable", [(99, False), (100, True), (500, True)])
def test_coherence_reliability_threshold(monkeypatch, words, reliable):
    install(monkeypatch, english_intel(word_count=words), quality())
    intel = text_ui.build_text_report(GOOD_TEXT)["intelligence"]
    assert intel["coherence"] == {"score": 0.7, "reliable": reliable}


def test_topics_error_becomes_note(monkeypatch):
    install(monkeypatch, english_intel(topics={"error": "need 50 words"}), quality())
    assert text_ui.build_text_report(GOOD_TEXT)["intelligence"]["topics"] == {"note": "need 50 words"}


def test_topics_pass_through_when_present(monkeypatch):
    topics = {"lda": [["a", "b"]]}
    install(monkeypatch, english_intel(topics=topics), quality())
    assert text_ui.build_text_report(GOOD_TEXT)["intelligence"]["topics"] == topics


def test_keywords_limited_to_12(monkeypatch):
    kws = [{"keyword": "k%d" % n, "avg_score": 0.5} for n in range(30)]
    install(monkeypatch, english_intel(keywords={"keywords": kws}), quality())
    assert len(text_ui.build_text_report(GOOD_TEXT)["intelligence"]["keywords"]) == 12


# -------------------------------------------------------------- build: non-English
def test_non_english_score_is_weighted_average(monkeypatch):
    bd = {"duplicate_score": 100, "noise_score": 50, "encoding_score": 100, "spelling_score": 5}
    install(monkeypatch, foreign_intel(), quality(breakdown=bd, score=3, verdict="Poor"))
    rep = text_ui.build_text_report(GOOD_TEXT)
    q = rep["quality"]
    assert q["score"] == 80.0  # (100*.4 + 50*.4 + 100*.2) / 1.0, spelling ignored
    assert q["verdict"] == "Good"
    assert q["scored_on"] == ["duplicate_score", "encoding_score", "noise_score"]
    assert rep["language"]["fully_supported"] is False


@pytest.mark.parametrize("value,verdict", [(80, "Good"), (79, "Needs Work"), (50, "Needs Work"), (49, "Poor")])
def test_non_english_verdict_bands(monkeypatch, value, verdict):
    install(monkeypatch, foreign_intel(), quality(breakdown={"duplicate_score": value}))
    q = text_ui.build_text_report(GOOD_TEXT)["quality"]
    assert q["score"] == value
    assert q["verdict"] == verdict


def test_non_english_weights_renormalize_over_present_parts(monkeypatch):
    install(monkeypatch, foreign_intel(), quality(breakdown={"duplicate_score": 100, "noise_score": 50}))
    assert text_ui.build_text_report(GOOD_TEXT)["quality"]["score"] == 75.0


def test_non_english_without_any_part_has_no_score(monkeypatch):
    install(monkeypatch, foreign_intel(), quality(breakdown={"spelling_score": 10}))
    q = text_ui.build_text_report(GOOD_TEXT)["quality"]
    assert q["score"] is None and q["verdict"] is None and q["scored_on"] == []


def test_non_english_non_numeric_part_is_ignored(monkeypatch):
    install(monkeypatch, foreign_intel(), quality(breakdown={"duplicate_score": "n/a", "noise_score": 60}))
    assert text_ui.build_text_report(GOOD_TEXT)["quality"]["score"] == 60.0


def test_non_english_filters_english_only_issues(monkeypatch):
    english_only = next(iter(text_ui._ENGLISH_ONLY_CHECKERS))
    issues = [
        {"checker": english_only, "issue_type": "grammar"},
        {"checker": "duplicate", "issue_type": "dup"},
        "not a dict",
    ]
    install(monkeypatch, foreign_intel(), quality(issues=issues))
    q = text_ui.build_text_report(GOOD_TEXT)["quality"]
    assert q["issue_count"] == 1
    assert q["issues"] == [{"checker": "duplicate", "issue_type": "dup"}]


def test_non_english_shows_only_contact_info(monkeypatch):
    label = next(iter(text_ui._CONTACT_LABELS))
    ents = [{"text": "a@b.com", "label": label}, {"text": "Ali", "label": "PERSON_NOT_CONTACT"}]
    install(monkeypatch, foreign_intel(entities=ents), quality())
    intel = text_ui.build_text_report(GOOD_TEXT)["intelligence"]
    assert list(intel.keys()) == ["contact_info"]
    assert intel["contact_info"] == [{"text": "a@b.com", "label": label}]


def test_non_english_hidden_sections_name_the_language(monkeypatch):
    install(monkeypatch, foreign_intel("fr", "French"), quality())
    hidden = text_ui.build_text_report(GOOD_TEXT)["hidden"]
    assert len(hidden) == 7
    assert {h["section"] for h in hidden} >= {"Sentiment", "Readability", "Keywords", "Coherence"}
    assert all("French" in h["reason"] for h in hidden)


def test_unknown_language_is_treated_as_non_english(monkeypatch):
    intel = foreign_intel()
    intel["language"] = {}
    install(monkeypatch, intel, quality())
    rep = text_ui.build_text_report(GOOD_TEXT)
    assert rep["language"]["code"] == "unknown"
    assert rep["language"]["fully_supported"] is False
    assert len(rep["hidden"]) == 7


def test_non_english_report_is_json_safe(monkeypatch):
    install(monkeypatch, foreign_intel(), quality(breakdown={"duplicate_score": float("nan"), "noise_score": 70}))
    json.dumps(text_ui.build_text_report(GOOD_TEXT))


# --------------------------------------------------------------- from_file wrapper
def test_from_file_adds_file_block(monkeypatch, tmp_path):
    install(monkeypatch, english_intel(), quality())
    p = tmp_path / "notes.md"
    p.write_text(GOOD_TEXT, encoding="utf-8")
    rep = text_ui.build_text_report_from_file(str(p))
    assert rep["status"] == "success"
    assert rep["name"] == "notes.md"
    assert rep["file"] == {"name": "notes.md", "type": "md", "encoding": "utf-8-sig"}


def test_from_file_custom_name(monkeypatch, tmp_path):
    install(monkeypatch, english_intel(), quality())
    p = tmp_path / "stored_123.txt"
    p.write_text(GOOD_TEXT, encoding="utf-8")
    rep = text_ui.build_text_report_from_file(str(p), "Original name.txt")
    assert rep["name"] == "Original name.txt"
    assert rep["file"]["name"] == "stored_123.txt"


def test_from_file_missing_file_never_raises(tmp_path):
    rep = text_ui.build_text_report_from_file(str(tmp_path / "nope.txt"))
    assert rep == {"status": "error", "message": "Could not read file: FileNotFoundError"}


def test_from_file_short_text_has_no_file_block(tmp_path):
    p = tmp_path / "tiny.txt"
    p.write_text("hi", encoding="utf-8")
    rep = text_ui.build_text_report_from_file(str(p))
    assert rep["status"] == "error"
    assert "file" not in rep