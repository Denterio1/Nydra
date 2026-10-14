"""Tests for api._save_cleaned_file (repaired CSV writer)."""
import sys

import pandas as pd
import pytest

import api


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPORT_DIR", tmp_path)
    return tmp_path


def _df():
    return pd.DataFrame({"name": ["Ali", "Sara", "Omar"], "score": [1.5, 2.5, 3.5]})


def _result(**extra):
    r = {
        "remove_duplicates": {"clean_df": _df(), "removed": 4},
        "impute": {"clean_df": _df().iloc[:2], "filled": 7, "method": "median"},
        "shape_original": [10, 2],
    }
    r.update(extra)
    return r


# ---------- which frame is used ----------
def test_prefers_remove_duplicates_over_impute(out_dir):
    s = api._save_cleaned_file(_result(), "job1")
    assert s["rows_after"] == 3


def test_falls_back_to_impute(out_dir):
    r = _result()
    del r["remove_duplicates"]
    s = api._save_cleaned_file(r, "job1")
    assert s["rows_after"] == 2


def test_skips_step_without_dataframe(out_dir):
    r = _result()
    r["remove_duplicates"] = {"clean_df": "not a frame", "removed": 1}
    s = api._save_cleaned_file(r, "job1")
    assert s["rows_after"] == 2          # impute frame used


@pytest.mark.parametrize("result", [
    {},
    {"impute": None},
    {"impute": {"filled": 3}},
    {"remove_duplicates": "oops", "impute": 5},
])
def test_no_usable_frame_returns_none(out_dir, result):
    assert api._save_cleaned_file(result, "job1") is None
    assert list(out_dir.iterdir()) == []


# ---------- file content ----------
def test_file_written_with_bom_and_no_index(out_dir):
    s = api._save_cleaned_file(_result(), "job1")
    p = out_dir / s["filename"]
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    back = pd.read_csv(p, encoding="utf-8-sig")
    assert list(back.columns) == ["name", "score"]
    assert len(back) == 3


def test_arabic_survives_roundtrip(out_dir):
    r = {"impute": {"clean_df": pd.DataFrame({"city": ["وهران", "الجزائر"], "n": [1, 2]})}}
    s = api._save_cleaned_file(r, "job1")
    back = pd.read_csv(out_dir / s["filename"], encoding="utf-8-sig")
    assert back["city"].tolist() == ["وهران", "الجزائر"]


def test_formula_cells_are_neutralized(out_dir):
    df = pd.DataFrame({"note": ["=SUM(A1:A9)", "fine"], "n": [1, 2]})
    s = api._save_cleaned_file({"impute": {"clean_df": df}}, "job1")
    assert s["formula_cells_neutralized"] >= 1
    back = pd.read_csv(out_dir / s["filename"], encoding="utf-8-sig")
    assert not str(back["note"].iloc[0]).startswith("=")


def test_no_formulas_means_zero(out_dir):
    s = api._save_cleaned_file(_result(), "job1")
    assert s["formula_cells_neutralized"] == 0


# ---------- summary ----------
def test_summary_keys_and_values(out_dir):
    s = api._save_cleaned_file(_result(), "job1")
    assert set(s) == {"filename", "rows_before", "rows_after", "columns", "missing_filled",
                      "impute_method", "duplicates_removed", "formula_cells_neutralized"}
    assert s["filename"] == "job1_cleaned.csv"
    assert s["rows_before"] == 10
    assert s["columns"] == 2
    assert s["missing_filled"] == 7
    assert s["impute_method"] == "median"
    assert s["duplicates_removed"] == 4


def test_missing_shape_original_gives_none(out_dir):
    r = _result()
    del r["shape_original"]
    assert api._save_cleaned_file(r, "job1")["rows_before"] is None


def test_none_counts_become_zero(out_dir):
    r = _result()
    r["impute"] = {"clean_df": _df(), "filled": None, "method": None}
    r["remove_duplicates"] = {"clean_df": _df(), "removed": None}
    s = api._save_cleaned_file(r, "job1")
    assert s["missing_filled"] == 0
    assert s["duplicates_removed"] == 0
    assert s["impute_method"] is None


def test_summary_is_json_safe(out_dir):
    import json
    json.dumps(api._save_cleaned_file(_result(), "job1"))


# ---------- safety ----------
def test_job_id_traversal_is_neutralized(out_dir):
    s = api._save_cleaned_file(_result(), "../../evil")
    assert s["filename"] == "evil_cleaned.csv"
    assert (out_dir / "evil_cleaned.csv").exists()
    assert not (out_dir.parent / "evil_cleaned.csv").exists()


def test_never_raises_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REPORT_DIR", tmp_path / "does" / "not" / "exist")
    assert api._save_cleaned_file(_result(), "job1") is None


def test_never_raises_when_sanitizer_fails(out_dir, monkeypatch):
    import src.input_sanitizer as san

    def boom(df):
        raise RuntimeError("x")

    monkeypatch.setattr(san, "neutralize_dataframe", boom)
    assert api._save_cleaned_file(_result(), "job1") is None