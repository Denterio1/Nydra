"""Retention sweeps for derived outputs and leftover work folders.

Deletes only direct children of known folders, never follows symlinks, never raises.
Old dashboard reports (*.pdf etc.) in REPORT_DIR are NOT touched.

Env: REPORT_RETENTION_DAYS (default 7).
"""
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

DEFAULT_RETENTION_DAYS = 7
DEFAULT_WORK_HOURS = 24

_REPORT_FILE_SUFFIXES = ("_cleaned.csv", "_cleaned_images.zip")
_REPORT_DIR_SUFFIXES = ("_image_report",)
_WORK_SUBDIRS = ("_clean", "_report", "_zip")


def retention_days() -> float:
    try:
        return max(0.01, float(os.getenv("REPORT_RETENTION_DAYS", DEFAULT_RETENTION_DAYS)))
    except ValueError:
        return float(DEFAULT_RETENTION_DAYS)


def _age_s(p: Path, now: float) -> Optional[float]:
    try:
        return now - p.stat().st_mtime
    except OSError:
        return None


def _remove(p: Path, counts: Dict[str, int]) -> None:
    try:
        if p.is_symlink():
            return
        if p.is_dir():
            shutil.rmtree(p)
            counts["dirs"] += 1
        else:
            p.unlink()
            counts["files"] += 1
    except OSError:
        counts["failed"] += 1


def sweep_files(report_dir, upload_dir, now: Optional[float] = None,
                report_days: Optional[float] = None,
                work_hours: float = DEFAULT_WORK_HOURS) -> Dict[str, int]:
    """Remove old derived outputs and stale work folders. Returns counts."""
    now = time.time() if now is None else now
    keep_s = (retention_days() if report_days is None else report_days) * 86400
    work_s = work_hours * 3600
    counts = {"files": 0, "dirs": 0, "failed": 0}

    rd = Path(report_dir)
    if rd.is_dir():
        for p in list(rd.iterdir()):
            age = _age_s(p, now)
            if age is None:
                continue
            name = p.name
            if name.endswith(".partial"):
                if age > work_s:
                    _remove(p, counts)
            elif p.is_dir() and name.endswith(_REPORT_DIR_SUFFIXES):
                if age > keep_s:
                    _remove(p, counts)
            elif p.is_file() and name.endswith(_REPORT_FILE_SUFFIXES):
                if age > keep_s:
                    _remove(p, counts)

    ud = Path(upload_dir)
    for sub in _WORK_SUBDIRS:
        base = ud / sub
        if not base.is_dir():
            continue
        for p in list(base.iterdir()):
            age = _age_s(p, now)
            if age is not None and age > work_s:
                _remove(p, counts)
    return counts


def prune_state(dicts: Iterable[dict], now: Optional[float] = None,
                max_age_days: Optional[float] = None) -> int:
    """Drop finished (not running) job-state entries older than the retention period.

    Call from the event-loop thread (the dicts are read by request handlers).
    """
    now = time.time() if now is None else now
    max_s = (retention_days() if max_age_days is None else max_age_days) * 86400
    removed = 0
    for d in dicts:
        for key, st in list(d.items()):
            try:
                if st.get("status") != "running" and now - float(st.get("started", now)) > max_s:
                    d.pop(key, None)
                    removed += 1
            except (AttributeError, TypeError, ValueError):
                continue
    return removed