import os
import time

from src.retention import prune_state, retention_days, sweep_files

DAY = 86400


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))


def _setup(tmp_path):
    rep = tmp_path / "reports"
    up = tmp_path / "uploads"
    rep.mkdir()
    up.mkdir()
    return rep, up


def test_old_outputs_removed_new_kept(tmp_path):
    rep, up = _setup(tmp_path)
    old_csv = rep / "a_cleaned.csv"
    old_zip = rep / "b_cleaned_images.zip"
    old_dir = rep / "c_image_report"
    new_csv = rep / "d_cleaned.csv"
    new_dir = rep / "e_image_report"
    for f in (old_csv, old_zip, new_csv):
        f.write_bytes(b"x")
    for d in (old_dir, new_dir):
        d.mkdir()
        (d / "image_report.html").write_text("x")
    for p in (old_csv, old_zip, old_dir):
        _age(p, 8 * DAY)
    res = sweep_files(rep, up, report_days=7)
    assert not old_csv.exists() and not old_zip.exists() and not old_dir.exists()
    assert new_csv.exists() and new_dir.exists()
    assert res["files"] == 2 and res["dirs"] == 1 and res["failed"] == 0


def test_dashboard_reports_never_touched(tmp_path):
    rep, up = _setup(tmp_path)
    pdf = rep / "00c2b45c-707.pdf"
    other = rep / "notes.txt"
    pdf.write_bytes(b"x")
    other.write_bytes(b"x")
    _age(pdf, 400 * DAY)
    _age(other, 400 * DAY)
    sweep_files(rep, up, report_days=7)
    assert pdf.exists() and other.exists()


def test_stale_partial_folders_removed(tmp_path):
    rep, up = _setup(tmp_path)
    old = rep / "x_image_report.partial"
    new = rep / "y_image_report.partial"
    old.mkdir()
    new.mkdir()
    _age(old, 2 * DAY)
    sweep_files(rep, up, report_days=7)
    assert not old.exists() and new.exists()


def test_work_folders_swept_by_age(tmp_path):
    rep, up = _setup(tmp_path)
    old = up / "_clean" / "job-old"
    new = up / "_report" / "job-new"
    zold = up / "_zip" / "stem-old"
    for d in (old, new, zold):
        d.mkdir(parents=True)
        (d / "f.jpg").write_bytes(b"x")
    _age(old, 2 * DAY)
    _age(zold, 2 * DAY)
    sweep_files(rep, up, report_days=7, work_hours=24)
    assert not old.exists() and not zold.exists()
    assert new.exists()


def test_other_upload_folders_untouched(tmp_path):
    rep, up = _setup(tmp_path)
    keep = up / "1_abc.zip"
    chunks = up / "_chunks" / "sess"
    keep.write_bytes(b"x")
    chunks.mkdir(parents=True)
    _age(keep, 100 * DAY)
    _age(chunks, 100 * DAY)
    sweep_files(rep, up)
    assert keep.exists() and chunks.exists()


def test_missing_folders_do_not_raise(tmp_path):
    res = sweep_files(tmp_path / "nope1", tmp_path / "nope2")
    assert res == {"files": 0, "dirs": 0, "failed": 0}


def test_prune_state_keeps_running_and_recent(tmp_path):
    now = time.time()
    clean = {
        "old_done": {"status": "done", "started": now - 9 * DAY},
        "old_err": {"status": "error", "started": now - 9 * DAY},
        "old_running": {"status": "running", "started": now - 9 * DAY},
        "recent_done": {"status": "done", "started": now - 1 * DAY},
    }
    report = {"old_done2": {"status": "done", "started": now - 30 * DAY}}
    n = prune_state([clean, report], now=now, max_age_days=7)
    assert n == 3
    assert set(clean) == {"old_running", "recent_done"}
    assert report == {}


def test_prune_state_ignores_malformed_entries():
    d = {"a": "not a dict", "b": {"status": "done", "started": "bad"}}
    assert prune_state([d], max_age_days=7) == 0
    assert len(d) == 2


def test_retention_days_env(monkeypatch):
    monkeypatch.setenv("REPORT_RETENTION_DAYS", "3")
    assert retention_days() == 3
    monkeypatch.setenv("REPORT_RETENTION_DAYS", "junk")
    assert retention_days() == 7