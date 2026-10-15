"""Tests for src/distributions_ui.py (fake advanced_stats, no slow fitting)."""
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import distributions_ui as du


# ---------- fakes ----------
def _install(monkeypatch, norm=None, fit=None, skew=None, trans=None, raises=()):
    """Inject a fake src.data.advanced_stats module."""
    def make(method, result):
        def fn(self, df):
            if method in raises:
                raise RuntimeError("boom")
            if callable(result):
                return result(df)
            return result if result is not None else {}
        return type(method + "_cls", (), {method: fn})

    mod = types.ModuleType("src.data.advanced_stats")
    mod.NormalityTester = make("test_all", norm)
    mod.DistributionFitter = make("fit_all", fit)
    mod.SkewnessKurtosisAnalyzer = make("analyze", skew)
    mod.TransformationAdvisor = make("advise", trans)
    monkeypatch.setitem(sys.modules, "src.data.advanced_stats", mod)


def _df(n=100, **extra):
    rng = np.random.default_rng(0)
    data = {"a": rng.normal(size=n), "b": rng.exponential(size=n)}
    data.update(extra)
    return pd.DataFrame(data)


# ---------- _f ----------
@pytest.mark.parametrize("x,expected", [(1, 1.0), ("2.5", 2.5), (0, 0.0)])
def test_f_valid(x, expected):
    assert du._f(x) == expected


@pytest.mark.parametrize("x", [None, float("nan"), float("inf"), float("-inf"), "abc", [1]])
def test_f_invalid(x):
    assert du._f(x) is None


# ---------- _histogram ----------
def test_histogram_shape_and_total():
    h = du._histogram(np.arange(200, dtype=float))
    assert len(h["counts"]) == du.BINS
    assert len(h["edges"]) == du.BINS + 1
    assert sum(h["counts"]) == 200
    assert all(isinstance(c, int) for c in h["counts"])


def test_histogram_constant_values_do_not_crash():
    h = du._histogram(np.ones(10))
    assert sum(h["counts"]) == 10


# ---------- column selection ----------
def test_no_numeric_columns(monkeypatch):
    _install(monkeypatch)
    out = du.build_distributions(pd.DataFrame({"t": ["x", "y", "z", "w"]}))
    assert out["columns"] == []
    assert "note" in out


@pytest.mark.parametrize("name", ["id", "ID", "user_id", "order id"])
def test_id_named_columns_skipped(monkeypatch, name):
    _install(monkeypatch)
    df = _df()
    df[name] = np.random.default_rng(1).normal(size=len(df))
    cols = [c["column"] for c in du.build_distributions(df)["columns"]]
    assert name not in cols
    assert "a" in cols


def test_unique_integer_column_skipped_but_float_kept(monkeypatch):
    _install(monkeypatch)
    df = _df()
    df["serial"] = np.arange(len(df))                      # unique ints -> id-like
    df["measure"] = np.random.default_rng(2).normal(size=len(df))  # unique floats -> kept
    cols = [c["column"] for c in du.build_distributions(df)["columns"]]
    assert "serial" not in cols
    assert "measure" in cols


def test_low_variation_columns_skipped(monkeypatch):
    _install(monkeypatch)
    df = _df()
    df["flag"] = [0, 1] * (len(df) // 2)
    df["three"] = [1, 2, 3, 1] * (len(df) // 4)
    cols = [c["column"] for c in du.build_distributions(df)["columns"]]
    assert "flag" not in cols and "three" not in cols


def test_all_nan_column_skipped(monkeypatch):
    _install(monkeypatch)
    df = _df()
    df["empty"] = np.nan
    cols = [c["column"] for c in du.build_distributions(df)["columns"]]
    assert "empty" not in cols


def test_max_columns_cap(monkeypatch):
    _install(monkeypatch)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({f"c{i}": rng.normal(size=50) for i in range(12)})
    out = du.build_distributions(df)
    assert len(out["columns"]) == du.MAX_COLS
    assert [c["column"] for c in out["columns"]] == [f"c{i}" for i in range(du.MAX_COLS)]


# ---------- sampling ----------
def test_large_frame_is_sampled(monkeypatch):
    _install(monkeypatch)
    df = _df(n=du.MAX_ROWS + 1000)
    out = du.build_distributions(df)
    assert out["sampled_rows"] == du.MAX_ROWS
    assert out["columns"][0]["n"] <= du.MAX_ROWS


def test_small_frame_not_sampled(monkeypatch):
    _install(monkeypatch)
    out = du.build_distributions(_df(n=80))
    assert out["sampled_rows"] == 80


def test_sampling_is_reproducible(monkeypatch):
    _install(monkeypatch)
    df = _df(n=du.MAX_ROWS + 500)
    a = du.build_distributions(df)["columns"][0]["histogram"]
    b = du.build_distributions(df)["columns"][0]["histogram"]
    assert a == b


# ---------- full item ----------
def _full_fakes():
    norm = {"a": SimpleNamespace(is_normal=True, confidence=0.9, verdict="Normal", shapiro_p=0.4)}
    fit = {"a": SimpleNamespace(best_distribution="norm", aic=12.5, ks_p=0.7, best_params=[0.1, 1.0])}
    skew = {"a": {"skewness": 0.2, "skewness_type": "symmetric", "kurtosis_excess": -0.1,
                  "kurtosis_type": "mesokurtic", "possibly_bimodal": False, "heavy_tail_ratio": 1.1}}
    trans = {"a": {"needs_transformation": True, "best_transform": "log1p",
                   "best_result": {"skewness_after": 0.05, "normality_p": 0.3}}}
    return norm, fit, skew, trans


def test_full_item_fields(monkeypatch):
    norm, fit, skew, trans = _full_fakes()
    _install(monkeypatch, norm, fit, skew, trans)
    item = du.build_distributions(_df())["columns"][0]
    assert item["column"] == "a"
    assert item["normality"] == {"is_normal": True, "confidence": 0.9,
                                 "verdict": "Normal", "shapiro_p": 0.4}
    assert item["best_fit"]["distribution"] == "norm"
    assert item["best_fit"]["params"] == [0.1, 1.0]
    assert item["shape"]["skewness_type"] == "symmetric"
    assert item["shape"]["possibly_bimodal"] is False
    assert item["transform"] == {"needed": True, "best": "log1p",
                                 "skewness_after": 0.05, "normality_p_after": 0.3}


def test_payload_is_json_safe(monkeypatch):
    norm, fit, skew, trans = _full_fakes()
    fit["a"].aic = float("nan")           # must become None, not crash json
    skew["a"]["skewness"] = float("inf")
    _install(monkeypatch, norm, fit, skew, trans)
    out = du.build_distributions(_df())
    json.dumps(out)                        # no NaN-breaking types, numpy ints
    item = out["columns"][0]
    assert item["best_fit"]["aic"] is None
    assert item["shape"]["skewness"] is None


def test_column_missing_from_analyzers_still_has_histogram(monkeypatch):
    _install(monkeypatch)                  # all analyzers return {}
    item = du.build_distributions(_df())["columns"][0]
    assert "histogram" in item and item["n"] > 0
    for key in ("normality", "best_fit", "shape", "transform"):
        assert key not in item


def test_string_results_are_ignored(monkeypatch):
    _install(monkeypatch, norm={"a": "error text"}, fit={"a": "error text"})
    item = du.build_distributions(_df())["columns"][0]
    assert "normality" not in item and "best_fit" not in item


def test_transform_without_best_result(monkeypatch):
    _install(monkeypatch, trans={"a": {"needs_transformation": False, "best_transform": None}})
    item = du.build_distributions(_df())["columns"][0]
    assert item["transform"]["needed"] is False
    assert item["transform"]["skewness_after"] is None


# ---------- resilience ----------
@pytest.mark.parametrize("failing", ["test_all", "fit_all", "analyze", "advise"])
def test_one_analyzer_failing_does_not_kill_payload(monkeypatch, failing):
    norm, fit, skew, trans = _full_fakes()
    _install(monkeypatch, norm, fit, skew, trans, raises=(failing,))
    out = du.build_distributions(_df())
    item = out["columns"][0]
    assert "histogram" in item
    key = {"test_all": "normality", "fit_all": "best_fit",
           "analyze": "shape", "advise": "transform"}[failing]
    assert key not in item
    others = {"normality", "best_fit", "shape", "transform"} - {key}
    assert others.issubset(item.keys())


def test_all_analyzers_failing_still_returns_columns(monkeypatch):
    _install(monkeypatch, raises=("test_all", "fit_all", "analyze", "advise"))
    out = du.build_distributions(_df())
    assert len(out["columns"]) == 2
    assert all("histogram" in c for c in out["columns"])