"""
memory.py — Data Memory: persistent storage of inspection history.

Uses SQLite to store inspection snapshots per file.
Enables comparison between runs and trend tracking.

Database: nydra_memory.db (created in project root)
"""

from __future__ import annotations
import sqlite3
import json
import os
import hashlib
from datetime import datetime
from typing import Any


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "nydra_memory.db")


# ── Database setup ────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash   TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            file_name   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            rows        INTEGER,
            columns     INTEGER,
            missing     INTEGER,
            duplicates  INTEGER,
            ml_score    INTEGER,
            outlier_cols INTEGER,
            analysis    TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON snapshots(file_hash)")
    conn.commit()
    return conn


def _file_hash(filepath: str) -> str:
    """Generate a stable hash for a file path (not content — path-based ID)."""
    return hashlib.md5(os.path.abspath(filepath).encode()).hexdigest()


# ── Public API ────────────────────────────────────────────────────────────────

def save_snapshot(
    filepath: str,
    analysis: dict[str, Any],
    ml_score: dict[str, Any],
    outliers: dict[str, Any],
) -> None:
    """Save an inspection snapshot to memory."""
    conn    = _get_conn()
    fhash   = _file_hash(filepath)
    missing = sum(analysis["missing_values"].values())

    conn.execute("""
        INSERT INTO snapshots
            (file_hash, file_path, file_name, timestamp, rows, columns,
             missing, duplicates, ml_score, outlier_cols, analysis)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fhash,
        os.path.abspath(filepath),
        os.path.basename(filepath),
        datetime.now().isoformat(),
        analysis.get("shape", {}).get("rows", 0),
        analysis.get("shape", {}).get("columns", 0),
        missing,
        analysis["duplicate_rows"],
        ml_score.get("score", 0),
        len(outliers),
        json.dumps({
            "shape":        analysis["shape"],
            "missing":      analysis["missing_values"],
            "duplicates":   analysis["duplicate_rows"],
            "column_stats": {
                col: {k: v for k, v in s.items() if k != "type"}
                for col, s in analysis["column_stats"].items()
            },
            "ml_score":     ml_score,
            "outliers":     list(outliers.keys()),
        }),
    ))
    conn.commit()
    conn.close()


def get_history(filepath: str, limit: int = 10) -> list[dict]:
    """Return past snapshots for a file, newest first."""
    conn  = _get_conn()
    fhash = _file_hash(filepath)
    rows  = conn.execute("""
        SELECT * FROM snapshots
        WHERE file_hash = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (fhash, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def compare_last_two(filepath: str) -> dict[str, Any] | None:
    """
    Compare the last two snapshots for a file.

    Returns a dict of changes, or None if fewer than 2 snapshots exist.
    """
    history = get_history(filepath, limit=2)
    if len(history) < 2:
        return None

    new = history[0]
    old = history[1]

    def _diff(key: str) -> dict:
        n, o = new[key], old[key]
        delta = n - o
        return {"old": o, "new": n, "delta": delta, "improved": delta < 0 if key != "ml_score" else delta > 0}

    changes = {
        "file":      new["file_name"],
        "old_time":  old["timestamp"][:19].replace("T", " "),
        "new_time":  new["timestamp"][:19].replace("T", " "),
        "rows":      _diff("rows"),
        "columns":   _diff("columns"),
        "missing":   _diff("missing"),
        "duplicates":_diff("duplicates"),
        "ml_score":  _diff("ml_score"),
        "outliers":  _diff("outlier_cols"),
    }
    return changes


def get_all_files() -> list[dict]:
    """Return a summary of all files in memory."""
    conn  = _get_conn()
    rows  = conn.execute("""
        SELECT file_name, file_path, COUNT(*) as inspections,
               MAX(timestamp) as last_seen, MAX(ml_score) as best_score
        FROM snapshots
        GROUP BY file_hash
        ORDER BY last_seen DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history(filepath: str) -> int:
    """Delete all snapshots for a file. Returns rows deleted."""
    conn  = _get_conn()
    fhash = _file_hash(filepath)
    cur   = conn.execute("DELETE FROM snapshots WHERE file_hash = ?", (fhash,))
    conn.commit()
    conn.close()
    return cur.rowcount


def clear_all() -> int:
    """Delete all snapshots. Returns rows deleted."""
    conn = _get_conn()
    cur  = conn.execute("DELETE FROM snapshots")
    conn.commit()
    conn.close()
    return cur.rowcount
