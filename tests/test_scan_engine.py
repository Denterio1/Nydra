"""Tests for src/scan_engine.py (bounded-memory scan + bounded sample)."""
import subprocess

import numpy as np
import pandas as pd
import pytest

from src import scan_engine
from src.scan_engine import ScanError, max_table_rows, needs_scan, scan_table


@pytest.fixture()
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "id": np.arange(n),
        "unit price": rng.normal(100, 20, n).round(2),
        "category": rng.choice(["a", "b", "c", "d"], n),
        "score": np.where(rng.random(n) < 0.2, np.nan, rng.random(n)),
    })
    df = pd.concat([df, df.iloc[:100]], ignore_index=True)  # 100 exact duplicate rows
    p = tmp_path / "big.csv"
    df.to_csv(p, index=False)
    return p


def test_exact_counts_and_bounded_sample(csv_path, tmp_path):
    r = scan_table(csv_path, tmp_path / "out", max_rows=1000)
    assert r["total_rows"] == 5100
    assert r["duplicate_rows"] == 100
    assert r["duplicate_method"] == "hash"
    assert r["sampled"] is True
    assert r["analyzed_rows"] == 1000
    s = pd.read_parquet(r["sample_path"])
    assert len(s) == 1000
    assert list(s.columns) == ["id", "unit price", "category", "score"]


def test_sample_rows_come_from_the_file(csv_path, tmp_path):
    r = scan_table(csv_path, tmp_path / "out", max_rows=500)
    s = pd.read_parquet(r["sample_path"])
    assert set(s["id"]).issubset(set(range(5000)))


def test_sample_is_reproducible(csv_path, tmp_path):
    a = pd.read_parquet(scan_table(csv_path, tmp_path / "o1", max_rows=300)["sample_path"])
    b = pd.read_parquet(scan_table(csv_path, tmp_path / "o2", max_rows=300)["sample_path"])
    assert sorted(a["id"]) == sorted(b["id"])


def test_under_cap_keeps_every_row(csv_path, tmp_path):
    r = scan_table(csv_path, tmp_path / "out", max_rows=10000)
    assert r["sampled"] is False
    assert r["analyzed_rows"] == 5100


def test_cap_zero_disables_sampling(csv_path, tmp_path):
    r = scan_table(csv_path, tmp_path / "out", max_rows=0)
    assert r["sampled"] is False
    assert r["analyzed_rows"] == 5100


def test_column_stats(csv_path, tmp_path):
    r = scan_table(csv_path, tmp_path / "out", max_rows=1000)
    cols = {c["name"]: c for c in r["columns"]}
    assert set(cols) == {"id", "unit price", "category", "score"}
    assert cols["id"]["min"] == "0" and cols["id"]["max"] == "4999"
    assert 15 < cols["score"]["null_percentage"] < 25
    assert cols["category"]["approx_unique"] in (3, 4, 5)
    assert "approx_unique" in r["approximate_fields"]


def test_tsv_is_supported(tmp_path):
    p = tmp_path / "t.tsv"
    pd.DataFrame({"a": range(300), "b": ["x"] * 300}).to_csv(p, sep="\t", index=False)
    r = scan_table(p, tmp_path / "out", max_rows=100)
    assert r["total_rows"] == 300 and r["analyzed_rows"] == 100


def test_parquet_is_supported(tmp_path):
    p = tmp_path / "t.parquet"
    pd.DataFrame({"a": range(400), "b": ["x"] * 400}).to_parquet(p)
    r = scan_table(p, tmp_path / "out", max_rows=100)
    assert r["total_rows"] == 400 and r["analyzed_rows"] == 100


def test_unsupported_type_returns_none(tmp_path):
    p = tmp_path / "t.xlsx"
    p.write_bytes(b"x")
    assert scan_table(p, tmp_path / "out") is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(ScanError, match="not found"):
        scan_table(tmp_path / "nope.csv", tmp_path / "out")


def test_corrupt_parquet_raises_clear_error(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"this is not a parquet file")
    with pytest.raises(ScanError, match="Scan failed"):
        scan_table(p, tmp_path / "out")


def test_timeout_is_a_clear_error(csv_path, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(scan_engine.subprocess, "run", boom)
    with pytest.raises(ScanError, match="timed out"):
        scan_table(csv_path, tmp_path / "out")


def test_out_of_memory_is_a_clear_error(csv_path, tmp_path, monkeypatch):
    def fake(*a, **k):
        return subprocess.CompletedProcess(a, 1, stdout="", stderr="_duckdb.OutOfMemoryException: Out of Memory Error: x")
    monkeypatch.setattr(scan_engine.subprocess, "run", fake)
    with pytest.raises(ScanError, match="out of memory"):
        scan_table(csv_path, tmp_path / "out")


def test_needs_scan_rules(tmp_path, monkeypatch):
    small = tmp_path / "s.csv"
    small.write_text("a\n1\n")
    assert needs_scan(small) is False                    # below SCAN_MIN_MB
    monkeypatch.setenv("SCAN_MIN_MB", "0")
    assert needs_scan(small) is True
    x = tmp_path / "s.xlsx"
    x.write_bytes(b"x")
    assert needs_scan(x) is False                        # unsupported type
    assert needs_scan(tmp_path / "missing.csv") is False


def test_env_cap(monkeypatch):
    monkeypatch.delenv("MAX_TABLE_ROWS", raising=False)
    assert max_table_rows() == 25000
    monkeypatch.setenv("MAX_TABLE_ROWS", "123")
    assert max_table_rows() == 123
    monkeypatch.setenv("MAX_TABLE_ROWS", "0")
    assert max_table_rows() == 0
    monkeypatch.setenv("MAX_TABLE_ROWS", "abc")
    assert max_table_rows() == 25000