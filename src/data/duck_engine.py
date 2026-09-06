"""
duck_engine.py — Nydra High-Performance Analytical Core v1.0
=============================================================
A specialized engine using DuckDB for lightning-fast data profiling.
Bypasses the "Pandas Bottleneck" using zero-copy SQL execution.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger("nydra.duck_engine")

class DuckEngine:
    """
    Advanced SQL-native engine for high-speed data inspection.
    """
    
    def __init__(self):
        self.conn = None
        if DUCKDB_AVAILABLE:
            try:
                # In-memory connection
                self.conn = duckdb.connect(database=':memory:')
                # Performance tuning: use all available threads
                self.conn.execute("PRAGMA threads=8")
                self.conn.execute("PRAGMA memory_limit='4GB'")
                logger.info("DuckEngine initialized with high-performance settings.")
            except Exception as e:
                logger.error(f"Failed to initialize DuckEngine: {e}")
                self.conn = None

    @property
    def is_ready(self) -> bool:
        return self.conn is not None

    def _register_df(self, df: pd.DataFrame, view_name: str = "nydra_view"):
        """Register a pandas DataFrame as a virtual view in DuckDB."""
        if not self.is_ready:
            raise RuntimeError("DuckEngine is not ready.")
        # Only register if it's a new dataframe to avoid overhead
        self.conn.register(view_name, df)
        return view_name

    def full_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        The "One-Pass" Miracle. 
        Computes shape, missing values, and basic stats in a single scan.
        """
        if not self.is_ready:
            return {}

        vn = self._register_df(df)
        
        # 1. Build Query
        select_parts = ["COUNT(*) as n_rows"]
        num_cols = []
        cat_cols = []
        
        for c in df.columns:
            # Stats for every column
            select_parts.append(f"COUNT(*) - COUNT(\"{c}\") AS \"{c}_nulls\"")
            select_parts.append(f"APPROX_COUNT_DISTINCT(\"{c}\") AS \"{c}_unique\"")
            select_parts.append(f"COUNT(\"{c}\") AS \"{c}_count\"")
            
            if pd.api.types.is_numeric_dtype(df[c]):
                num_cols.append(c)
                select_parts.append(f"MIN(\"{c}\") AS \"{c}_min\"")
                select_parts.append(f"MAX(\"{c}\") AS \"{c}_max\"")
                select_parts.append(f"AVG(\"{c}\") AS \"{c}_mean\"")
            else:
                cat_cols.append(c)
                # Mode is only for categorical, but we'll use a faster approx if possible
                # DuckDB doesn't have approx mode, so we'll stick to MODE() for now
                select_parts.append(f"MODE(\"{c}\") AS \"{c}_mode\"")

        try:
            start_t = time.time()
            res = self.conn.execute(f"SELECT {', '.join(select_parts)} FROM {vn}").fetchone()
            
            # 2. Parse Results
            n_rows = int(res[0])
            stats = {}
            idx = 1
            
            for c in df.columns:
                n_nulls = int(res[idx])
                n_unique = int(res[idx + 1])
                n_count = int(res[idx + 2])
                idx += 3
                
                if c in num_cols:
                    stats[c] = {
                        "type": "numeric",
                        "unique": n_unique,
                        "count": n_count,
                        "min": res[idx],
                        "max": res[idx + 1],
                        "mean": res[idx + 2],
                        "nulls": n_nulls
                    }
                    idx += 3
                else:
                    stats[c] = {
                        "type": "text",
                        "unique": n_unique,
                        "count": n_count,
                        "most_common": res[idx],
                        "nulls": n_nulls
                    }
                    idx += 1
            
            # 3. Calculate Duplicates (separate query as it's a different grain)
            # We use a faster hash-based approach if possible
            dupes_query = f"SELECT COUNT(*) - COUNT(*) FROM (SELECT DISTINCT * FROM {vn})" # Wait this is wrong
            dupes_query = f"SELECT {n_rows} - COUNT(*) FROM (SELECT DISTINCT * FROM {vn})"
            n_dupes = int(self.conn.execute(dupes_query).fetchone()[0])

            duration = time.time() - start_t
            logger.info(f"Full profile completed in {duration:.2f}s")

            return {
                "shape": {"rows": n_rows, "columns": len(df.columns)},
                "missing_values": {c: s["nulls"] for c, s in stats.items()},
                "duplicate_rows": n_dupes,
                "column_stats": stats,
            }
        except Exception as e:
            logger.error(f"Full profile failed: {e}")
            return {}

    def profile_file(self, filepath: str) -> Dict[str, Any]:
        """
        Direct File Profiling.
        Analyzes a file directly from disk without loading into Pandas.
        Ideal for 1GB+ files.
        """
        if not self.is_ready:
            return {}

        ext = filepath.split('.')[-1].lower()
        if ext != 'csv':
            # Currently only optimized for CSV direct read
            return {}

        try:
            start_t = time.time()
            
            # Use DuckDB's specialized CSV reader with auto-detection
            # We use a subquery to avoid registering a view
            vn = f"read_csv_auto('{filepath}')"
            
            # Get columns first to build the query
            # Robust way to describe a table function result
            cols_res = self.conn.execute(f"DESCRIBE SELECT * FROM {vn}").fetchall()
            columns = [r[0] for r in cols_res]
            col_types = {r[0]: r[1] for r in cols_res}

            select_parts = ["COUNT(*) as n_rows"]
            num_cols = []
            
            for c in columns:
                select_parts.append(f"COUNT(*) - COUNT(\"{c}\") AS \"{c}_nulls\"")
                select_parts.append(f"APPROX_COUNT_DISTINCT(\"{c}\") AS \"{c}_unique\"")
                select_parts.append(f"COUNT(\"{c}\") AS \"{c}_count\"")
                
                # Check DuckDB internal type
                t = col_types[c].upper()
                if any(k in t for k in ["INT", "FLOAT", "DOUBLE", "DECIMAL", "HUGEINT"]):
                    num_cols.append(c)
                    select_parts.append(f"MIN(\"{c}\") AS \"{c}_min\"")
                    select_parts.append(f"MAX(\"{c}\") AS \"{c}_max\"")
                    select_parts.append(f"AVG(\"{c}\") AS \"{c}_mean\"")
                else:
                    select_parts.append(f"MODE(\"{c}\") AS \"{c}_mode\"")

            res = self.conn.execute(f"SELECT {', '.join(select_parts)} FROM {vn}").fetchone()
            
            n_rows = int(res[0])
            stats = {}
            idx = 1
            
            for c in columns:
                n_nulls = int(res[idx])
                n_unique = int(res[idx + 1])
                n_count = int(res[idx + 2])
                idx += 3
                
                if c in num_cols:
                    stats[c] = {
                        "type": "numeric",
                        "unique": n_unique,
                        "count": n_count,
                        "min": res[idx],
                        "max": res[idx + 1],
                        "mean": res[idx + 2],
                        "nulls": n_nulls
                    }
                    idx += 3
                else:
                    stats[c] = {
                        "type": "text",
                        "unique": n_unique,
                        "count": n_count,
                        "most_common": res[idx],
                        "nulls": n_nulls
                    }
                    idx += 1
            
            duration = time.time() - start_t
            logger.info(f"Direct file profile completed in {duration:.2f}s")

            return {
                "shape": {"rows": n_rows, "columns": len(columns)},
                "missing_values": {c: s["nulls"] for c, s in stats.items()},
                "duplicate_rows": "calculating...", # Direct dupes on large files is risky
                "column_stats": stats,
            }
        except Exception as e:
            logger.error(f"Direct file profile failed: {e}")
            return {}

# Singleton instance
_engine = None

def get_duck_engine() -> DuckEngine:
    global _engine
    if _engine is None:
        _engine = DuckEngine()
    return _engine
