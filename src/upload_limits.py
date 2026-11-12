"""Per-file-type upload size limits, configurable through environment variables.

Tables (csv, tsv, parquet) are read by the bounded-memory DuckDB scan, so they
can be very large. Every other type is still loaded fully into memory by
pandas or a parser, so it gets a much smaller cap.

Environment (read at call time, so a restart is enough to change them):
    MAX_TABLE_GB   limit for csv/tsv/parquet in GB (falls back to MAX_CHUNKED_GB, then 10)
    MAX_OTHER_MB   limit for every other type in MB (default 500)
Invalid, zero or negative values are ignored and the default is used.
"""
from __future__ import annotations

import os

TABLE_EXTENSIONS = frozenset({"csv", "tsv", "parquet"})
_GB = 1024 ** 3
_MB = 1024 ** 2


def _env_positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value != value or value <= 0 or value == float("inf"):
        return default
    return value


def ext_of(filename: str) -> str:
    """Lower-case extension without the dot ('' when there is none)."""
    name = str(filename or "")
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def is_table(ext: str) -> bool:
    return str(ext or "").lower().lstrip(".") in TABLE_EXTENSIONS


def limit_bytes_for(ext: str) -> int:
    """Maximum accepted size in bytes for a file with this extension."""
    if is_table(ext):
        gb = _env_positive_float("MAX_TABLE_GB", _env_positive_float("MAX_CHUNKED_GB", 10.0))
        return int(gb * _GB)
    return int(_env_positive_float("MAX_OTHER_MB", 500.0) * _MB)


def max_any_bytes() -> int:
    """Largest limit across all types (the hard ceiling for the chunk store)."""
    return max(limit_bytes_for("csv"), limit_bytes_for("txt"))


def too_large_message(ext: str) -> str:
    mb = limit_bytes_for(ext) // _MB
    kind = "tables (CSV, TSV, Parquet)" if is_table(ext) else "this file type"
    return f"File too large. Maximum for {kind} is {mb} MB."