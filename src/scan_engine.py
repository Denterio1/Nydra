"""
src/scan_engine.py - bounded-memory scan of large table files.

Layer 1: every row is scanned by DuckDB (exact row count, exact duplicate
         count via 64-bit row hashes, per-column statistics).
Layer 2: a bounded random sample is written to a parquet file; the heavy
         pandas steps (impute, outliers, DNA, distributions) work on that.

The scan runs in a SEPARATE process with an explicit memory limit and a
timeout, so a failure kills only that job and raises ScanError with a clear
message. No new algorithms: DuckDB SUMMARIZE, reservoir sampling and hash().

Exact:       total_rows, duplicate_rows (hash based: collision risk is
             negligible but non-zero), null counts, min, max.
Approximate: approx_unique, q25, q50, q75, avg, std (see APPROXIMATE_FIELDS).

Environment (read at call time):
    MAX_TABLE_ROWS   sample cap for the deep steps (default 25000, 0 = no cap)
    SCAN_MIN_MB      files smaller than this skip the scan (default 2)
    SCAN_MEM_MB      DuckDB memory_limit (default 512)
    SCAN_THREADS     DuckDB threads (default 2)
    SCAN_TIMEOUT_S   hard timeout for the scan process (default 600)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SUPPORTED_EXTS = {"csv", "tsv", "parquet"}  # xlsx / xls / json stay on pandas
APPROXIMATE_FIELDS = ["approx_unique", "q25", "q50", "q75", "avg", "std"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ScanError(RuntimeError):
    """The scan failed (out of memory, timeout, unreadable file). Message is user-safe."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def max_table_rows() -> int:
    return _env_int("MAX_TABLE_ROWS", 25000)


def needs_scan(path: Path) -> bool:
    """True only for supported, large-enough files. Small files keep the pandas path."""
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        return False
    try:
        size_mb = p.stat().st_size / (1024 * 1024)
    except OSError:
        return False
    return size_mb >= _env_int("SCAN_MIN_MB", 2)


def scan_table(path: Path, out_dir: Path, *, max_rows: Optional[int] = None,
               mem_mb: Optional[int] = None, threads: Optional[int] = None,
               timeout_s: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Scan `path` and write <out_dir>/<name>.sample.parquet.

    Returns None for unsupported types (caller falls back to pandas), else a dict:
      total_rows, analyzed_rows, sampled, sample_path, duplicate_rows,
      duplicate_method, columns (list of per-column stats), approximate_fields.
    Raises ScanError on any failure.
    """
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        return None
    if not p.exists():
        raise ScanError("Source file not found.")

    cap = max_table_rows() if max_rows is None else int(max_rows)
    args = {
        "path": str(p),
        "ext": ext,
        "out_dir": str(out_dir),
        "max_rows": cap,
        "mem_mb": mem_mb or _env_int("SCAN_MEM_MB", 512),
        "threads": threads or _env_int("SCAN_THREADS", 2),
    }
    timeout = timeout_s or _env_int("SCAN_TIMEOUT_S", 600)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "src.scan_engine", "--child", json.dumps(args)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(_PROJECT_ROOT))
    except subprocess.TimeoutExpired:
        raise ScanError(f"Scan timed out after {timeout} s. The file is too large for this server.")

    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            out = json.loads(line[len("RESULT_JSON:"):])
            out["approximate_fields"] = APPROXIMATE_FIELDS
            return out

    err = (proc.stderr or "").strip().splitlines()
    last = err[-1] if err else "unknown error"
    if "Out of Memory" in (proc.stderr or ""):
        raise ScanError("Scan ran out of memory. The file is too large or too wide for this server.")
    raise ScanError("Scan failed: " + last[:200])


# --------------------------------------------------------------------------
# child process
# --------------------------------------------------------------------------
def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _reader(ext: str, p: str, sample_size: int) -> str:
    p = p.replace("\\", "/").replace("'", "''")
    if ext == "parquet":
        return f"read_parquet('{p}')"
    delim = ", delim='\\t'" if ext == "tsv" else ""
    return f"read_csv_auto('{p}', sample_size={sample_size}{delim})"


def _child(args: Dict[str, Any]) -> None:
    import duckdb

    out_dir = Path(args["out_dir"])
    tmp_dir = (out_dir / "_duck_tmp").as_posix()
    last_exc: Optional[Exception] = None
    # CSV type detection looks at a sample; retry once with a full-file pass on a type error
    for sample_size in (100000, -1):
        con = duckdb.connect()
        try:
            con.execute(f"SET memory_limit='{int(args['mem_mb'])}MB'")
            con.execute(f"SET threads={int(args['threads'])}")
            con.execute("SET preserve_insertion_order=false")
            con.execute(f"SET temp_directory='{tmp_dir}'")
            con.execute(f"CREATE VIEW t AS SELECT * FROM {_reader(args['ext'], args['path'], sample_size)}")

            total = con.execute("SELECT count(*) FROM t").fetchone()[0]

            cols = [r[0] for r in con.execute("DESCRIBE t").fetchall()]
            hash_args = ", ".join(_q(c) for c in cols)
            distinct = con.execute(f"SELECT count(DISTINCT hash({hash_args})) FROM t").fetchone()[0]

            summ = con.execute("SUMMARIZE t").fetchall()
            names = [d[0] for d in con.description]
            columns = []
            for row in summ:
                d = dict(zip(names, row))
                columns.append({
                    "name": d.get("column_name"),
                    "type": d.get("column_type"),
                    "min": None if d.get("min") is None else str(d["min"]),
                    "max": None if d.get("max") is None else str(d["max"]),
                    "approx_unique": d.get("approx_unique"),
                    "avg": None if d.get("avg") is None else str(d["avg"]),
                    "std": None if d.get("std") is None else str(d["std"]),
                    "q25": None if d.get("q25") is None else str(d["q25"]),
                    "q50": None if d.get("q50") is None else str(d["q50"]),
                    "q75": None if d.get("q75") is None else str(d["q75"]),
                    "null_percentage": float(d["null_percentage"]) if d.get("null_percentage") is not None else None,
                })

            cap = int(args["max_rows"])
            sampled = bool(cap and total > cap)
            name = Path(args["path"]).stem
            sample_path = (out_dir / f"{name}.sample.parquet").as_posix()
            if sampled:
                q = f"SELECT * FROM t USING SAMPLE reservoir({cap} ROWS) REPEATABLE (42)"
            else:
                q = "SELECT * FROM t"
            con.execute(f"COPY ({q}) TO '{sample_path}' (FORMAT PARQUET)")
            analyzed = con.execute(f"SELECT count(*) FROM read_parquet('{sample_path}')").fetchone()[0]

            print("RESULT_JSON:" + json.dumps({
                "total_rows": int(total),
                "analyzed_rows": int(analyzed),
                "sampled": sampled,
                "sample_path": sample_path,
                "duplicate_rows": int(total - distinct),
                "duplicate_method": "hash",
                "columns": columns,
            }))
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            if sample_size != -1 and args["ext"] != "parquet" and ("Conversion Error" in msg or "type" in msg.lower()):
                continue
            raise
        finally:
            con.close()
    if last_exc:
        raise last_exc


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        _child(json.loads(sys.argv[2]))