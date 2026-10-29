"""Tests for src/data/cognitive_dna.py — they check real values, not just 'no crash'."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data import cognitive_dna as cd
from src.data.cognitive_dna import (
    CognitiveDNA,
    DataDNA,
    DNASimilarityEngine,
    compute_dna,
)


def _frame(n=300, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.normal(40 + shift, 10, n),
            "income": rng.lognormal(10 + shift / 10, 0.5, n),
            "city": rng.choice(["A", "B", "C"], n),
        }
    )


# ── identity and hashing ─────────────────────────────────────────────────────

def test_entropy_vector_has_one_value_per_column():
    df = _frame()
    dna = compute_dna(df)
    assert len(dna.statistical.entropy_vector) == df.shape[1]
    # city has 3 balanced levels -> about log2(3) bits; it is the third column
    assert dna.statistical.entropy_vector[2] == pytest.approx(np.log2(3), abs=0.1)


def test_same_data_same_hash_even_after_shuffle():
    df = _frame()
    a = compute_dna(df).dna_hash
    b = compute_dna(df.sample(frac=1, random_state=3).reset_index(drop=True)).dna_hash
    assert a == b


def test_same_shape_different_content_different_hash():
    a = compute_dna(_frame(seed=1))
    b = compute_dna(_frame(seed=2))
    assert a.structural.shape_signature == b.structural.shape_signature
    assert a.dna_hash != b.dna_hash


def test_schema_hash_follows_columns_not_values():
    df = _frame()
    same_schema = compute_dna(_frame(seed=9)).structural.schema_hash
    renamed = compute_dna(df.rename(columns={"age": "years"})).structural.schema_hash
    assert compute_dna(df).structural.schema_hash == same_schema
    assert renamed != same_schema


def test_short_hash_is_eight_chars():
    assert len(compute_dna(_frame()).short_hash) == 8


# ── structural ───────────────────────────────────────────────────────────────

def test_planted_outliers_raise_outlier_density():
    df = _frame(500)
    clean = compute_dna(df).structural.outlier_density
    df.loc[:9, "age"] = 5000.0
    dirty = compute_dna(df).structural.outlier_density
    assert dirty > clean
    assert dirty > 0


def test_duplicate_ratio_is_exact():
    base = _frame(100)
    df = pd.concat([base, base.head(10)], ignore_index=True)
    assert compute_dna(df).structural.duplicate_ratio == pytest.approx(10 / 110, abs=1e-4)


def test_missing_none_mcar_and_mar():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = rng.normal(size=n)

    none = pd.DataFrame({"x": x, "y": y})
    assert compute_dna(none).structural.missing_mechanism == "NONE"
    assert compute_dna(none).structural.missing_pattern_hash == "none"

    mcar = none.copy()
    mcar.loc[rng.random(n) < 0.2, "y"] = np.nan
    assert compute_dna(mcar).structural.missing_mechanism == "MCAR"

    mar = none.copy()
    mar.loc[(x > 0.3) & (rng.random(n) < 0.8), "y"] = np.nan  # depends on observed x
    assert compute_dna(mar).structural.missing_mechanism == "MAR"
    assert compute_dna(mar).structural.missing_pattern_hash != "none"


def test_cardinality_profile():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "x", "y", "y"]})
    assert compute_dna(df).structural.cardinality_profile == (4, 2)


# ── ML ───────────────────────────────────────────────────────────────────────

def test_no_target_is_unsupervised():
    dna = compute_dna(_frame())
    assert dna.ml.recommended_task == "unsupervised"
    assert dna.ml.target_signal == "none"


def test_classification_strong_vs_noise_signal():
    rng = np.random.default_rng(1)
    n = 600
    label = rng.integers(0, 2, n)
    signal = pd.DataFrame(
        {"f": label * 3.0 + rng.normal(0, 0.5, n), "noise": rng.normal(size=n), "y": label}
    )
    noise = pd.DataFrame(
        {"f": rng.normal(size=n), "noise": rng.normal(size=n), "y": label}
    )
    s = compute_dna(signal, target_col="y")
    z = compute_dna(noise, target_col="y")
    assert s.ml.recommended_task == "classification"
    assert s.ml.target_signal == "strong"
    assert z.ml.target_signal in ("none", "weak")
    assert s.ml.separability_score > z.ml.separability_score


def test_regression_detected_with_strong_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=500), "y": 2 * x + rng.normal(0, 0.1, 500)})
    dna = compute_dna(df, target_col="y")
    assert dna.ml.recommended_task == "regression"
    assert dna.ml.target_signal == "strong"


def test_categorical_feature_signal():
    rng = np.random.default_rng(3)
    cat = rng.choice(["a", "b", "c"], 600)
    y = (cat == "a").astype(int)
    df = pd.DataFrame({"cat": cat, "noise": rng.normal(size=600), "y": y})
    assert compute_dna(df, target_col="y").ml.target_signal == "strong"


def test_id_like_column_is_not_a_signal():
    rng = np.random.default_rng(4)
    n = 400
    y = np.sort(rng.integers(0, 2, n))  # sorted -> row id would correlate perfectly
    df = pd.DataFrame({"row_id": np.arange(n), "noise": rng.normal(size=n), "y": y})
    assert compute_dna(df, target_col="y").ml.target_signal in ("none", "weak")


def test_class_balance_score():
    n = 400
    balanced = pd.DataFrame({"f": np.arange(n, dtype=float), "y": [0, 1] * (n // 2)})
    skewed = pd.DataFrame({"f": np.arange(n, dtype=float), "y": [0] * 380 + [1] * 20})
    assert compute_dna(balanced, target_col="y").ml.class_balance_score == pytest.approx(1.0, abs=0.01)
    assert compute_dna(skewed, target_col="y").ml.class_balance_score < 0.5


def test_feature_redundancy_detects_duplicated_feature():
    rng = np.random.default_rng(5)
    a = rng.normal(size=300)
    redundant = pd.DataFrame({"a": a, "b": a * 2 + 1, "c": a + rng.normal(0, 0.01, 300)})
    independent = pd.DataFrame({k: rng.normal(size=300) for k in "abc"})
    assert compute_dna(redundant).ml.feature_redundancy > 0.9
    assert compute_dna(independent).ml.feature_redundancy < 0.2


# ── temporal ─────────────────────────────────────────────────────────────────

def test_temporal_trend_and_uniform_gaps():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.arange(120) + np.random.default_rng(0).normal(0, 2, 120)})
    t = compute_dna(df).temporal
    assert t.has_temporal and t.temporal_columns == ("date",)
    assert t.trend_detected
    assert t.time_gaps_uniform


def test_string_date_column_is_detected():
    df = pd.DataFrame(
        {"when": [f"2024-01-{d:02d}" for d in range(1, 29)], "v": np.arange(28, dtype=float)}
    )
    assert compute_dna(df).temporal.has_temporal


def test_seasonality_score_higher_for_periodic_series():
    n = 240
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    seasonal = pd.DataFrame({"d": dates, "v": np.sin(np.arange(n) * 2 * np.pi / 7) + rng.normal(0, 0.1, n)})
    noisy = pd.DataFrame({"d": dates, "v": rng.normal(size=n)})
    assert compute_dna(seasonal).temporal.seasonality_score > compute_dna(noisy).temporal.seasonality_score + 0.3


def test_non_temporal_frame():
    t = compute_dna(_frame()).temporal
    assert not t.has_temporal and t.temporal_columns == ()


# ── similarity ───────────────────────────────────────────────────────────────

def test_similarity_with_itself_is_one():
    dna = compute_dna(_frame())
    report = DNASimilarityEngine().compare(dna, dna)
    assert report.overall_similarity == 1.0
    assert report.key_differences == []


def test_similarity_orders_same_shifted_and_different_schema():
    base = compute_dna(_frame(seed=1))
    same_dist = compute_dna(_frame(seed=2))
    shifted = compute_dna(_frame(seed=2, shift=40))
    other = compute_dna(pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) ** 2}))
    eng = DNASimilarityEngine()
    s_same = eng.compare(base, same_dist).overall_similarity
    s_shift = eng.compare(base, shifted).overall_similarity
    s_other = eng.compare(base, other)
    assert s_same > s_shift > s_other.overall_similarity
    assert s_same > 0.85
    assert "Schema has changed" in s_other.key_differences


def test_similarity_report_shape():
    r = DNASimilarityEngine().compare(compute_dna(_frame(seed=1)), compute_dna(_frame(seed=2)))
    assert set(r.dimension_scores) == {"statistical", "structural", "ml", "temporal"}
    assert 0.0 <= r.overall_similarity <= 1.0
    assert isinstance(r.recommendation, str)


# ── wrapper, serialisation, robustness ───────────────────────────────────────

def test_wrapper_exposes_everything_the_cli_reads():
    df = _frame()
    dna = CognitiveDNA(df, target_col=None)
    assert isinstance(dna.data_dna, DataDNA)
    for attr in ("short_hash", "personality", "tags", "created_at", "dna_hash"):
        assert getattr(dna, attr)
    assert dna.statistical.correlation_hash
    assert len(dna.statistical.entropy_vector) == df.shape[1]
    assert dna.structural.schema_hash and dna.structural.missing_mechanism
    for attr in ("outlier_density", "duplicate_ratio"):
        assert isinstance(getattr(dna.structural, attr), float)
    for attr in ("recommended_task", "target_signal", "separability_score",
                 "feature_redundancy", "complexity_score"):
        assert hasattr(dna.ml, attr)
    for attr in ("has_temporal", "temporal_columns", "trend_detected", "seasonality_score"):
        assert hasattr(dna.temporal, attr)


def test_wrapper_similarity_method():
    a, b = CognitiveDNA(_frame(seed=1)), CognitiveDNA(_frame(seed=2))
    assert a.similarity(a).overall_similarity == 1.0
    assert 0 < a.similarity(b).overall_similarity < 1.0


def test_to_dict_is_json_safe_and_has_the_keys_dna_memory_reads():
    d = compute_dna(_frame(), source="x.csv").to_dict()
    json.dumps(d)
    assert d["source"] == "x.csv"
    for k in ("entropy_vector", "moment_signatures", "distribution_types",
              "correlation_hash", "normality_scores", "mutual_info_matrix"):
        assert k in d["statistical"]
    for k in ("schema_hash", "missing_mechanism", "missing_pattern_hash",
              "cardinality_profile", "outlier_density", "duplicate_ratio", "shape_signature"):
        assert k in d["structural"]
    for k in ("feature_redundancy", "target_signal", "separability_score",
              "class_balance_score", "recommended_task", "complexity_score"):
        assert k in d["ml"]
    for k in ("has_temporal", "temporal_columns", "trend_detected",
              "seasonality_score", "time_gaps_uniform"):
        assert k in d["temporal"]


def test_empty_frame_and_missing_target_raise_value_error():
    with pytest.raises(ValueError):
        compute_dna(pd.DataFrame())
    with pytest.raises(ValueError):
        compute_dna(_frame(), target_col="nope")
    with pytest.raises(TypeError):
        compute_dna([1, 2, 3])


def test_degenerate_columns_do_not_crash():
    df = pd.DataFrame(
        {
            "const": [1.0] * 30,
            "all_nan": [np.nan] * 30,
            "text": ["a"] * 30,
            "inf": [np.inf] + [1.0] * 29,
            "flag": [True, False] * 15,
        }
    )
    dna = compute_dna(df)
    assert len(dna.statistical.entropy_vector) == 5
    assert dna.statistical.entropy_vector[0] == 0.0


def test_duplicate_column_names_are_handled():
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "a"])
    assert len(compute_dna(df).statistical.entropy_vector) == 2


def test_large_frame_is_sampled_deterministically(monkeypatch):
    monkeypatch.setattr(cd, "SAMPLE_ROWS", 500)
    df = _frame(3000)
    a, b = compute_dna(df), compute_dna(df)
    assert a.dna_hash == b.dna_hash
    assert a.structural.shape_signature == (3000, 3)  # true size, not the sample


def test_dna_memory_roundtrip(tmp_path):
    memory = pytest.importorskip("src.data.dna_memory")
    store = memory.DNAStore(str(tmp_path / "dna.db"))
    dna = compute_dna(_frame(), source="demo.csv")
    store.save(dna, ml_score=70)
    back = store.get_by_hash(dna.dna_hash)
    assert back is not None
    restored = back.to_dna()
    assert restored.dna_hash == dna.dna_hash
    assert restored.statistical.entropy_vector == dna.statistical.entropy_vector
    assert restored.ml.recommended_task == dna.ml.recommended_task
    # different same-shaped data must now be stored separately (old hash was shape-only)
    store.save(compute_dna(_frame(seed=7), source="other.csv"))
    assert store.count() == 2