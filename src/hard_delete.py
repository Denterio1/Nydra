"""Real deletion helpers (files and job leftovers). Pure functions, easy to test."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("nydra.hard_delete")


def _inside(path: Path, root: Path) -> bool:
    try:
        return root.resolve() in path.resolve().parents
    except Exception:
        return False


def safe_unlink(path, root) -> bool:
    """Delete one file, only if it lives inside `root`. Never raises."""
    try:
        p = Path(path)
        if not _inside(p, Path(root)) or not p.is_file():
            return False
        p.unlink()
        return True
    except Exception as exc:
        log.warning("safe_unlink failed | %s: %s", type(exc).__name__, exc)
        return False


def remove_job_artifacts(job_id: str, report_dir, upload_dir) -> int:
    """Remove report files and work folders of one job. Returns items removed."""
    if not job_id or any(c in job_id for c in "/\\*?[]"):
        return 0
    report_dir, upload_dir = Path(report_dir), Path(upload_dir)
    targets = list(report_dir.glob(f"{job_id}_*"))
    targets += [upload_dir / "_clean" / job_id, upload_dir / "_report" / job_id]
    removed = 0
    for t in targets:
        try:
            root = report_dir if t.parent == report_dir else upload_dir
            if not t.exists() or not _inside(t, root):
                continue
            if t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
            else:
                t.unlink()
            removed += 1
        except Exception as exc:
            log.warning("remove_job_artifacts failed | %s: %s", type(exc).__name__, exc)
    return removed