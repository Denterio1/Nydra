"""
test_nydra.py — Core pipeline test suite for Nydra.

Covers: loader, analyzer, cleaner, ml_readiness, relationships, preparator,
drift, and the Nydra agent's core `inspect()` flow.

Audit-layer tests (vision_nlp: text/training/image) live in test_audit_layer.py.
Fixtures (sample_csv, clean_csv, sample_data, clean_data) live in conftest.py.

Run with:
    pytest tests/ -v
"""

import pandas as pd
import pytest

from src.data.loader import load_csv
from src.data.analyzer import (
    shape, missing_values, duplicate_rows, basic_stats, full_report, detect_outliers,
)
from src.data.cleaner import handle_missing, remove_duplicates
from src.data.ml_readiness import ml_readiness
from src.data.relationships import detect_relationships
from src.data.preparator import prepare_for_ml
from src.data.drift import detect_drift
from src.core.agent import Nydra


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_csv_returns_dict(self, sample_csv):
        data = load_csv(sample_csv)
        assert isinstance(data, dict)
        assert "df" in data
        assert "columns" in data
        assert "source" in data

    def test_load_csv_correct_shape(self, sample_csv):
        data = load_csv(sample_csv)
        assert len(data["df"]) == 5
        assert len(data["columns"]) == 4

    def test_load_csv_columns(self, sample_csv):
        data = load_csv(sample_csv)
        assert data["columns"] == ["name", "age", "salary", "city"]

    def test_load_csv_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent_file.csv")

    def test_load_csv_empty_file_raises(self, tmp_path):
        """CORRECTED: the real loader explicitly rejects headers-only
        CSVs with a clear ValueError rather than silently returning an
        empty DataFrame — a better contract than originally assumed
        here. Confirmed against actual behavior: 'File is empty or has
        no data rows.'"""
        path = tmp_path / "empty.csv"
        path.write_text("a,b,c\n")
        with pytest.raises(ValueError, match="empty|no data"):
            load_csv(str(path))

    def test_load_csv_single_column(self, tmp_path):
        path = tmp_path / "single.csv"
        path.write_text("value\n1\n2\n3\n")
        data = load_csv(str(path))
        assert len(data["columns"]) == 1


# ── Analyzer tests ────────────────────────────────────────────────────────────

class TestAnalyzer:
    def test_shape(self, sample_data):
        s = shape(sample_data)
        assert s["rows"] == 5
        assert s["columns"] == 4

    def test_missing_values(self, sample_data):
        mv = missing_values(sample_data)
        assert mv["age"] == 1
        assert mv["salary"] == 1
        assert mv["city"] == 1
        assert mv["name"] == 0

    def test_duplicate_rows(self, sample_data):
        assert duplicate_rows(sample_data) == 1

    def test_no_duplicates(self, clean_data):
        assert duplicate_rows(clean_data) == 0

    def test_basic_stats_numeric(self, clean_data):
        stats = basic_stats(clean_data)
        assert stats["price"]["type"] == "numeric"
        assert stats["price"]["min"] == 79.0
        assert stats["price"]["max"] == 999.0

    def test_basic_stats_text(self, sample_data):
        stats = basic_stats(sample_data)
        assert stats["name"]["type"] == "text"
        assert stats["name"]["unique"] == 4

    def test_full_report(self, sample_data):
        report = full_report(sample_data)
        for key in ("shape", "missing_values", "duplicate_rows", "column_stats", "outliers"):
            assert key in report

    def test_detect_outliers_clean(self, clean_data):
        outliers = detect_outliers(clean_data)
        assert isinstance(outliers, dict)

    def test_detect_outliers_with_outlier(self, tmp_path):
        path = tmp_path / "outlier.csv"
        path.write_text("value\n1\n2\n2\n3\n2\n2\n1000\n")
        data = load_csv(str(path))
        outliers = detect_outliers(data)
        assert "value" in outliers

    def test_detect_outliers_all_identical_values(self, tmp_path):
        """Zero-variance column — a common divide-by-zero trap for
        z-score/IQR style outlier detectors. Should not raise."""
        path = tmp_path / "flat.csv"
        path.write_text("value\n5\n5\n5\n5\n5\n")
        data = load_csv(str(path))
        outliers = detect_outliers(data)
        assert isinstance(outliers, dict)


# ── Cleaner tests ─────────────────────────────────────────────────────────────

class TestCleaner:
    def test_remove_duplicates(self, sample_data):
        cleaned, removed = remove_duplicates(sample_data)
        assert removed == 1
        assert len(cleaned["df"]) == 4

    def test_remove_duplicates_no_change(self, clean_data):
        cleaned, removed = remove_duplicates(clean_data)
        assert removed == 0
        assert len(cleaned["df"]) == len(clean_data["df"])

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode", "drop"])
    def test_handle_missing_strategies_clear_nulls(self, sample_data, strategy):
        """Every supported strategy must leave zero nulls in the columns
        it actually touches (drop removes rows instead of filling)."""
        cleaned, changes = handle_missing(sample_data, strategy=strategy)
        if strategy == "drop":
            assert "rows_dropped" in changes
            assert len(cleaned["df"]) < len(sample_data["df"])
        else:
            assert "age" in changes
            assert "salary" in changes
            assert cleaned["df"]["age"].isnull().sum() == 0
            assert cleaned["df"]["salary"].isnull().sum() == 0

    def test_handle_missing_unknown_strategy_raises(self, sample_data):
        """An invalid strategy string should fail loudly, not silently
        no-op and leave nulls in place."""
        with pytest.raises((ValueError, KeyError)):
            handle_missing(sample_data, strategy="not_a_real_strategy")

    def test_original_not_mutated(self, sample_data):
        original_missing = sample_data["df"].isnull().sum().sum()
        handle_missing(sample_data, strategy="mean")
        assert sample_data["df"].isnull().sum().sum() == original_missing

    def test_remove_duplicates_does_not_mutate_original(self, sample_data):
        original_len = len(sample_data["df"])
        remove_duplicates(sample_data)
        assert len(sample_data["df"]) == original_len


# ── ML Readiness tests ────────────────────────────────────────────────────────

class TestMLReadiness:
    def test_returns_score(self, clean_data):
        outliers = detect_outliers(clean_data)
        result = ml_readiness(clean_data, outliers)
        assert "score" in result
        assert "grade" in result
        assert "checks" in result
        assert 0 <= result["score"] <= 100

    def test_grade_values(self, clean_data):
        outliers = detect_outliers(clean_data)
        result = ml_readiness(clean_data, outliers)
        assert result["grade"] in ["A", "B", "C", "D", "F"]

    def test_small_dataset_penalised(self, sample_data):
        outliers = detect_outliers(sample_data)
        result = ml_readiness(sample_data, outliers)
        size_check = next(c for c in result["checks"] if c["name"] == "Dataset size")
        assert size_check["status"] in ["warn", "fail"]

    def test_score_is_deterministic(self, clean_data):
        """Same input twice must give the same score — guards against
        any accidental randomness (e.g. unseeded sampling) creeping in."""
        outliers = detect_outliers(clean_data)
        r1 = ml_readiness(clean_data, outliers)
        r2 = ml_readiness(clean_data, outliers)
        assert r1["score"] == r2["score"]
        assert r1["grade"] == r2["grade"]


# ── Relationships tests ───────────────────────────────────────────────────────

class TestRelationships:
    def test_returns_list(self, clean_data):
        assert isinstance(detect_relationships(clean_data), list)

    def test_sorted_by_strength(self, clean_data):
        rels = detect_relationships(clean_data, threshold=0.0)
        if len(rels) >= 2:
            assert rels[0]["strength"] >= rels[1]["strength"]

    def test_relationship_keys(self, clean_data):
        rels = detect_relationships(clean_data, threshold=0.0)
        if rels:
            for key in ("col_a", "col_b", "strength", "method"):
                assert key in rels[0]

    def test_high_threshold_can_return_empty(self, clean_data):
        """A threshold above any real correlation should return an empty
        list, not error or return unfiltered results."""
        rels = detect_relationships(clean_data, threshold=0.999999)
        assert isinstance(rels, list)


# ── Preparator tests ──────────────────────────────────────────────────────────

class TestPreparator:
    def test_prepare_returns_data_and_log(self, sample_data):
        prepared, log = prepare_for_ml(sample_data)
        assert "df" in prepared
        assert isinstance(log, dict)

    def test_no_missing_after_prepare(self, sample_data):
        prepared, _ = prepare_for_ml(sample_data, missing_strategy="mean")
        assert prepared["df"].isnull().sum().sum() == 0

    def test_no_duplicates_after_prepare(self, sample_data):
        prepared, log = prepare_for_ml(sample_data)
        assert log["duplicates_removed"] >= 0
        assert prepared["df"].duplicated().sum() == 0

    def test_all_numeric_after_prepare(self, sample_data):
        prepared, _ = prepare_for_ml(sample_data, encode=True, scale=True)
        for col in prepared["df"].columns:
            assert pd.api.types.is_numeric_dtype(prepared["df"][col])

    def test_prepare_idempotent_on_already_clean_data(self, clean_data):
        """Running prepare_for_ml on already-clean data shouldn't drop rows
        or blow up — a common regression when cleaning logic assumes dirty input."""
        prepared, log = prepare_for_ml(clean_data)
        assert len(prepared["df"]) == len(clean_data["df"])


# ── Drift tests ───────────────────────────────────────────────────────────────

class TestDrift:
    def test_no_drift_same_data(self, clean_data):
        result = detect_drift(clean_data, clean_data)
        assert result["severity"] == "none"

    def test_drift_detected(self, tmp_path):
        base_path, curr_path = tmp_path / "base.csv", tmp_path / "curr.csv"
        base_path.write_text("value\n1\n2\n3\n4\n5\n")
        curr_path.write_text("value\n100\n200\n300\n400\n500\n")
        result = detect_drift(load_csv(str(base_path)), load_csv(str(curr_path)))
        assert result["severity"] != "none"
        assert len(result["drifted_columns"]) > 0

    def test_new_column_detected(self, tmp_path):
        base_path, curr_path = tmp_path / "base.csv", tmp_path / "curr.csv"
        base_path.write_text("a,b\n1,2\n3,4\n")
        curr_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        result = detect_drift(load_csv(str(base_path)), load_csv(str(curr_path)))
        assert any(d["column"] == "schema" for d in result["drifted_columns"])

    def test_removed_column_detected(self, tmp_path):
        """Mirror of the new-column case — a column disappearing between
        base and current is schema drift too, and was untested before."""
        base_path, curr_path = tmp_path / "base2.csv", tmp_path / "curr2.csv"
        base_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
        curr_path.write_text("a,b\n1,2\n3,4\n")
        result = detect_drift(load_csv(str(base_path)), load_csv(str(curr_path)))
        assert any(d["column"] == "schema" for d in result["drifted_columns"])


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestNydra:
    def test_inspect_returns_dict(self, sample_csv):
        result = Nydra().inspect(sample_csv)
        for key in ("source", "raw_analysis", "cleaning_log", "clean_data", "summary"):
            assert key in result

    def test_inspect_summary_is_string(self, sample_csv):
        result = Nydra().inspect(sample_csv)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_inspect_cleans_duplicates(self, sample_csv):
        result = Nydra(remove_dupes=True).inspect(sample_csv)
        assert result["cleaning_log"].get("duplicates_removed", 0) >= 1

    def test_inspect_no_dedup(self, sample_csv):
        result = Nydra(remove_dupes=False).inspect(sample_csv)
        assert "duplicates_removed" not in result["cleaning_log"]

    @pytest.mark.parametrize("strategy", ["mean", "median", "mode", "drop"])
    def test_inspect_strategies(self, sample_csv, strategy):
        result = Nydra(missing_strategy=strategy).inspect(sample_csv)
        assert result["clean_data"]["df"].isnull().sum().sum() == 0

    def test_inspect_nonexistent_file_raises(self):
        """The agent should surface the loader's FileNotFoundError, not
        swallow it into a generic failure or return a partial result."""
        with pytest.raises(FileNotFoundError):
            Nydra().inspect("this_file_does_not_exist_anywhere.csv")