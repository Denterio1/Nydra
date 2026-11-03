"""Wiring tests: big-table scan in api.py, retention clock, hard delete."""
import os
import time

import numpy as np
import pandas as pd
import pytest


def _write_csv(path, n=3000):
    pd.DataFrame({"a": np.arange(n), "b": np.random.default_rng(1).random(n)}).to_csv(path, index=False)


@pytest.fixture()
def api_mod(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path / "uploads")
    return api


def test_loading_scans_big_table_and_points_at_sample(api_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("SCAN_MIN_MB", "0")
    monkeypatch.setenv("MAX_TABLE_ROWS", "200")
    big = tmp_path / "big.csv"
    _write_csv(big)
    r = api_mod._dispatch_step(None, "Loading", "inspect", big, {}, {}, "job1")
    assert r["_df_path"].endswith(".sample.parquet")
    assert r["_orig_path"] == str(big)
    assert r["rows"] == 200
    assert r["scan"]["total_rows"] == 3000
    assert r["scan"]["analyzed_rows"] == 200
    assert r["scan"]["sampled"] is True
    assert "sample_path" not in r["scan"]
    assert (tmp_path / "uploads" / "_scan" / "job1").is_dir()


def test_later_step_runs_on_the_sample(api_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("SCAN_MIN_MB", "0")
    monkeypatch.setenv("MAX_TABLE_ROWS", "200")
    big = tmp_path / "big.csv"
    _write_csv(big)
    acc = api_mod._dispatch_step(None, "Loading", "inspect", big, {}, {}, "job1")
    out = api_mod._dispatch_step(None, "DNA", "dna", big, {}, dict(acc), "job1")
    assert "dna" in out


def test_small_file_keeps_old_behaviour(api_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("SCAN_MIN_MB", "2")
    small = tmp_path / "small.csv"
    _write_csv(small, n=50)
    r = api_mod._dispatch_step(None, "Loading", "inspect", small, {}, {}, "job2")
    assert r["_df_path"] == str(small)
    assert "scan" not in r and "_orig_path" not in r
    assert r["rows"] == 50


def test_retention_sample_follows_report_clock(tmp_path):
    from src.retention import sweep_files
    up, rep = tmp_path / "uploads", tmp_path / "reports"
    rep.mkdir(parents=True)
    old, mid, new = (up / "_scan" / n for n in ("old", "mid", "new"))
    for d in (old, mid, new):
        d.mkdir(parents=True)
        (d / "x.sample.parquet").write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 10 * 86400, now - 10 * 86400))   # past 7 days: removed
    os.utime(mid, (now - 2 * 86400, now - 2 * 86400))     # past 24 h work clock, inside 7 days: kept
    sweep_files(rep, up, now=now, report_days=7)
    assert not old.exists()
    assert mid.exists()
    assert new.exists()


def test_hard_delete_removes_scan_folder(tmp_path):
    from src.hard_delete import remove_job_artifacts
    up, rep = tmp_path / "uploads", tmp_path / "reports"
    rep.mkdir(parents=True)
    d = up / "_scan" / "jobX"
    d.mkdir(parents=True)
    (d / "big.sample.parquet").write_bytes(b"x")
    keep = up / "_scan" / "jobY"
    keep.mkdir(parents=True)
    n = remove_job_artifacts("jobX", rep, up)
    assert n >= 1
    assert not d.exists()
    assert keep.exists()