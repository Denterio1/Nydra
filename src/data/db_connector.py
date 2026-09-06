"""
db_connector.py — Advanced Database Connection Manager
Nydra v0.5.0

Supports:
  - PostgreSQL, MySQL, SQLite, MariaDB
  - Smart Sampling for large tables
  - SQL Injection protection
  - Schema Analysis (tables, columns, foreign keys, indexes)
  - Connection Pooling
  - Query History
  - Query Optimizer suggestions
  - Auto Preview
  - Full compatibility with Nydra data dict
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

logger = logging.getLogger("nydra.db")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_DBS = {
    "postgresql": {"default_port": 5432, "driver": "psycopg2"},
    "mysql":      {"default_port": 3306, "driver": "pymysql"},
    "mariadb":    {"default_port": 3306, "driver": "pymysql"},
    "sqlite":     {"default_port": None, "driver": None},
}


@dataclass
class DBConfig:
    db_type  : str
    host     : str  = "localhost"
    port     : int  = 0
    database : str  = ""
    username : str  = ""
    password : str  = ""
    filepath : str  = ""          # SQLite only
    pool_size: int  = 5
    timeout  : int  = 10

    def __post_init__(self):
        self.db_type = self.db_type.lower()
        if self.db_type not in SUPPORTED_DBS:
            raise ValueError(
                f"Unsupported db_type '{self.db_type}'. "
                f"Choose from: {list(SUPPORTED_DBS)}"
            )
        if self.port == 0:
            self.port = SUPPORTED_DBS[self.db_type]["default_port"] or 0

    def connection_string(self) -> str:
        if self.db_type == "sqlite":
            path = self.filepath or ":memory:"
            return f"sqlite:///{path}"
        driver = SUPPORTED_DBS[self.db_type]["driver"]
        if self.db_type in ("postgresql",):
            return (
                f"postgresql+{driver}://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
        if self.db_type in ("mysql", "mariadb"):
            return (
                f"mysql+{driver}://{self.username}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
                f"?charset=utf8mb4"
            )
        raise ValueError(f"Cannot build connection string for {self.db_type}")

    def masked(self) -> dict:
        """Return config with password hidden — safe to display."""
        return {
            "db_type":  self.db_type,
            "host":     self.host,
            "port":     self.port,
            "database": self.database,
            "username": self.username,
            "password": "****" if self.password else "",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. SECURITY — SQL Injection Protection
# ══════════════════════════════════════════════════════════════════════════════

_DANGEROUS_PATTERNS = re.compile(
    r"""
    (--|;|\bDROP\b|\bDELETE\b|\bTRUNCATE\b|\bALTER\b|
     \bINSERT\b|\bUPDATE\b|\bCREATE\b|\bGRANT\b|
     \bREVOKE\b|\bEXEC\b|\bEXECUTE\b|\bXP_\w+|
     \bINTO\s+OUTFILE\b|\bLOAD_FILE\b)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Allow only SELECT statements.
    Returns (is_safe, reason).
    """
    sql_clean = sql.strip()
    if not sql_clean.upper().startswith("SELECT"):
        return False, "Only SELECT statements are allowed."
    if _DANGEROUS_PATTERNS.search(sql_clean):
        return False, "Potentially dangerous SQL detected."
    if len(sql_clean) > 10_000:
        return False, "Query too long (max 10,000 characters)."
    return True, "OK"


def sanitize_identifier(name: str) -> str:
    """Strip non-alphanumeric chars from table/column names."""
    return re.sub(r"[^\w]", "", name)


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUERY HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryRecord:
    sql        : str
    rows       : int
    duration_ms: float
    success    : bool
    error      : str        = ""
    timestamp  : str        = field(default_factory=lambda: datetime.now().isoformat())
    query_hash : str        = field(init=False)

    def __post_init__(self):
        self.query_hash = hashlib.md5(self.sql.encode()).hexdigest()[:8]


class QueryHistory:
    def __init__(self, max_records: int = 100):
        self._records: list[QueryRecord] = []
        self._max     = max_records

    def add(self, record: QueryRecord):
        self._records.append(record)
        if len(self._records) > self._max:
            self._records.pop(0)

    def recent(self, n: int = 10) -> list[QueryRecord]:
        return self._records[-n:]

    def successful(self) -> list[QueryRecord]:
        return [r for r in self._records if r.success]

    def as_df(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "timestamp":   r.timestamp[:19],
                "sql":         r.sql[:80] + ("..." if len(r.sql) > 80 else ""),
                "rows":        r.rows,
                "duration_ms": r.duration_ms,
                "success":     r.success,
                "error":       r.error,
            }
            for r in self._records
        ])

    def clear(self):
        self._records.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 4. SCHEMA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnInfo:
    name      : str
    dtype     : str
    nullable  : bool
    primary_key: bool = False
    foreign_key: str  = ""     # "other_table.other_col"


@dataclass
class TableInfo:
    name        : str
    row_count   : int
    columns     : list[ColumnInfo]
    indexes     : list[str]
    size_bytes  : int  = 0

    @property
    def col_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_cols(self) -> list[str]:
        numeric_types = {"int", "float", "double", "decimal", "numeric",
                         "bigint", "smallint", "real", "money"}
        return [
            c.name for c in self.columns
            if any(t in c.dtype.lower() for t in numeric_types)
        ]

    @property
    def text_cols(self) -> list[str]:
        text_types = {"char", "text", "varchar", "string", "clob"}
        return [
            c.name for c in self.columns
            if any(t in c.dtype.lower() for t in text_types)
        ]


class SchemaAnalyzer:
    def __init__(self, engine):
        self._engine    = engine
        self._inspector = inspect(engine)
        self._cache: dict[str, TableInfo] = {}

    def get_tables(self) -> list[str]:
        return self._inspector.get_table_names()

    def get_views(self) -> list[str]:
        try:
            return self._inspector.get_view_names()
        except Exception:
            return []

    def analyse_table(self, table: str) -> TableInfo:
        table = sanitize_identifier(table)
        if table in self._cache:
            return self._cache[table]

        # Columns
        raw_cols = self._inspector.get_columns(table)
        pk_cols  = set(
            c for c in self._inspector.get_pk_constraint(table).get("constrained_columns", [])
        )
        fk_map: dict[str, str] = {}
        for fk in self._inspector.get_foreign_keys(table):
            for local, remote_col in zip(
                fk["constrained_columns"],
                fk["referred_columns"],
            ):
                fk_map[local] = f"{fk['referred_table']}.{remote_col}"

        columns = [
            ColumnInfo(
                name        = c["name"],
                dtype       = str(c["type"]),
                nullable    = c.get("nullable", True),
                primary_key = c["name"] in pk_cols,
                foreign_key = fk_map.get(c["name"], ""),
            )
            for c in raw_cols
        ]

        # Indexes
        indexes = [
            idx["name"] or "unnamed"
            for idx in self._inspector.get_indexes(table)
        ]

        # Row count
        try:
            with self._engine.connect() as conn:
                row_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar() or 0
        except Exception:
            row_count = -1

        info = TableInfo(
            name      = table,
            row_count = row_count,
            columns   = columns,
            indexes   = indexes,
        )
        self._cache[table] = info
        return info

    def get_all_tables_info(self) -> list[TableInfo]:
        return [self.analyse_table(t) for t in self.get_tables()]

    def find_relationships(self) -> list[dict]:
        """Return all FK relationships across the schema."""
        rels = []
        for table in self.get_tables():
            for fk in self._inspector.get_foreign_keys(table):
                rels.append({
                    "from_table":  table,
                    "from_cols":   fk["constrained_columns"],
                    "to_table":    fk["referred_table"],
                    "to_cols":     fk["referred_columns"],
                })
        return rels

    def clear_cache(self):
        self._cache.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMART SAMPLER — handles million-row tables
# ══════════════════════════════════════════════════════════════════════════════

class SmartSampler:
    """
    Intelligently samples large tables without loading everything into RAM.
    Strategy:
      - < 50k rows   → load all
      - 50k–500k     → random sample
      - > 500k       → stratified sample if categorical target exists
    """

    SMALL  =  50_000
    MEDIUM = 500_000

    def sample(
        self,
        engine,
        table     : str,
        limit     : int         = 10_000,
        target_col: str | None  = None,
    ) -> pd.DataFrame:
        table = sanitize_identifier(table)

        with engine.connect() as conn:
            total = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar() or 0

        if total <= self.SMALL or total <= limit:
            with engine.connect() as conn:
                return pd.read_sql(text(f"SELECT * FROM {table}"), conn)

        if total <= self.MEDIUM or target_col is None:
            return self._random_sample(engine, table, limit)

        return self._stratified_sample(engine, table, limit, target_col)

    def _random_sample(self, engine, table: str, limit: int) -> pd.DataFrame:
        db_type = engine.dialect.name
        if db_type in ("postgresql", "sqlite"):
            sql = f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT {limit}"
        elif db_type in ("mysql", "mariadb"):
            sql = f"SELECT * FROM {table} ORDER BY RAND() LIMIT {limit}"
        else:
            sql = f"SELECT * FROM {table} LIMIT {limit}"

        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def _stratified_sample(
        self, engine, table: str, limit: int, target_col: str
    ) -> pd.DataFrame:
        target_col = sanitize_identifier(target_col)
        db_type    = engine.dialect.name
        rand_fn    = "RANDOM()" if db_type in ("postgresql", "sqlite") else "RAND()"

        with engine.connect() as conn:
            classes_df = pd.read_sql(
                text(f"SELECT DISTINCT {target_col} FROM {table}"), conn
            )

        classes   = classes_df[target_col].tolist()
        per_class = max(1, limit // len(classes))
        frames    = []

        for cls in classes:
            sql = (
                f"SELECT * FROM {table} "
                f"WHERE {target_col} = :cls "
                f"ORDER BY {rand_fn} LIMIT {per_class}"
            )
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn, params={"cls": cls})
            frames.append(df)

        return pd.concat(frames, ignore_index=True).sample(
            frac=1, random_state=42
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUERY OPTIMIZER — suggests improvements
# ══════════════════════════════════════════════════════════════════════════════

class QueryOptimizer:
    def suggest(self, sql: str, table_info: TableInfo | None = None) -> list[str]:
        suggestions = []
        sql_up = sql.upper()

        if "SELECT *" in sql_up:
            suggestions.append(
                "Avoid SELECT * — specify only the columns you need for faster queries."
            )
        if "WHERE" not in sql_up:
            suggestions.append(
                "No WHERE clause — consider filtering rows to reduce data transfer."
            )
        if "ORDER BY" in sql_up and "LIMIT" not in sql_up:
            suggestions.append(
                "ORDER BY without LIMIT can be very slow on large tables."
            )
        if table_info:
            indexed = {idx for idx in table_info.indexes}
            if not indexed:
                suggestions.append(
                    f"Table '{table_info.name}' has no indexes — queries may be slow."
                )
            if table_info.row_count > 100_000 and "LIMIT" not in sql_up:
                suggestions.append(
                    f"Table has {table_info.row_count:,} rows — add LIMIT to avoid slow loads."
                )

        return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN CONNECTOR
# ══════════════════════════════════════════════════════════════════════════════

class DBConnector:
    """
    Main entry point for all database operations in nydra.
    """

    def __init__(self, config: DBConfig):
        self.config    = config
        self.engine    = None
        self.schema    = None
        self.history   = QueryHistory()
        self.sampler   = SmartSampler()
        self.optimizer = QueryOptimizer()
        self._connected_at: str | None = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> tuple[bool, str]:
        try:
            pool_args = {}
            if self.config.db_type != "sqlite":
                pool_args = {
                    "poolclass":        QueuePool,
                    "pool_size":        self.config.pool_size,
                    "max_overflow":     10,
                    "pool_timeout":     30,
                    "pool_pre_ping":    True,
                    "connect_args":     {"connect_timeout": self.config.timeout},
                }

            self.engine = create_engine(
                self.config.connection_string(),
                **pool_args,
            )

            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            self.schema         = SchemaAnalyzer(self.engine)
            self._connected_at  = datetime.now().isoformat()
            logger.info("Connected to %s", self.config.db_type)
            return True, f"Connected to {self.config.db_type} successfully."

        except SQLAlchemyError as e:
            self.engine = None
            return False, str(e)
        except Exception as e:
            self.engine = None
            return False, f"Unexpected error: {e}"

    def disconnect(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.schema = None
        logger.info("Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self.engine is not None

    def ping(self) -> bool:
        if not self.engine:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ── Schema ────────────────────────────────────────────────────────────────

    def get_tables(self) -> list[dict]:
        if not self.schema:
            return []
        infos = self.schema.get_all_tables_info()
        return [
            {
                "table":   t.name,
                "rows":    t.row_count,
                "columns": len(t.columns),
                "numeric": len(t.numeric_cols),
                "text":    len(t.text_cols),
            }
            for t in infos
        ]

    def get_table_info(self, table: str) -> TableInfo | None:
        if not self.schema:
            return None
        return self.schema.analyse_table(table)

    def get_relationships(self) -> list[dict]:
        if not self.schema:
            return []
        return self.schema.find_relationships()

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        sql        : str,
        limit      : int = 10_000,
        auto_limit : bool = True,
    ) -> dict[str, Any]:
        """
        Run a validated SQL SELECT and return a Nydra-compatible dict.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected. Call connect() first.")

        safe, reason = validate_sql(sql)
        if not safe:
            raise ValueError(reason)

        # Auto-add LIMIT if missing
        if auto_limit and "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

        suggestions = self.optimizer.suggest(sql)

        start = time.time()
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)

            duration = (time.time() - start) * 1000
            self.history.add(QueryRecord(
                sql=sql, rows=len(df),
                duration_ms=round(duration, 2), success=True,
            ))

            return {
                "columns":     list(df.columns),
                "df":          df,
                "source":      f"db://{self.config.database}",
                "row_count":   len(df),
                "duration_ms": round(duration, 2),
                "suggestions": suggestions,
            }

        except Exception as e:
            duration = (time.time() - start) * 1000
            self.history.add(QueryRecord(
                sql=sql, rows=0,
                duration_ms=round(duration, 2),
                success=False, error=str(e),
            ))
            raise RuntimeError(f"Query failed: {e}")

    def load_table(
        self,
        table      : str,
        limit      : int        = 10_000,
        target_col : str | None = None,
        smart      : bool       = True,
    ) -> dict[str, Any]:
        """
        Load a table with smart sampling for large datasets.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected.")

        table = sanitize_identifier(table)
        start = time.time()

        if smart:
            df = self.sampler.sample(
                self.engine, table, limit=limit, target_col=target_col
            )
        else:
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(f"SELECT * FROM {table} LIMIT {limit}"), conn
                )

        duration = (time.time() - start) * 1000
        self.history.add(QueryRecord(
            sql=f"SELECT * FROM {table}",
            rows=len(df),
            duration_ms=round(duration, 2),
            success=True,
        ))

        table_info = self.get_table_info(table)
        suggestions = self.optimizer.suggest(
            f"SELECT * FROM {table}", table_info
        )

        return {
            "columns":     list(df.columns),
            "df":          df,
            "source":      f"db://{self.config.database}/{table}",
            "row_count":   len(df),
            "total_rows":  table_info.row_count if table_info else len(df),
            "duration_ms": round(duration, 2),
            "suggestions": suggestions,
            "sampled":     table_info is not None and table_info.row_count > limit,
        }

    def preview(self, table: str, n: int = 5) -> pd.DataFrame:
        """Quick preview of first N rows."""
        if not self.is_connected:
            return pd.DataFrame()
        table = sanitize_identifier(table)
        with self.engine.connect() as conn:
            return pd.read_sql(
                text(f"SELECT * FROM {table} LIMIT {n}"), conn
            )

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "connected":      self.is_connected,
            "db_type":        self.config.db_type,
            "database":       self.config.database,
            "connected_at":   self._connected_at,
            "ping":           self.ping(),
            "query_count":    len(self.history._records),
            "tables":         len(self.get_tables()) if self.is_connected else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8. FACTORY — easy creation
# ══════════════════════════════════════════════════════════════════════════════

def connect_postgresql(
    host: str, database: str, username: str, password: str, port: int = 5432
) -> DBConnector:
    cfg = DBConfig("postgresql", host, port, database, username, password)
    c   = DBConnector(cfg)
    ok, msg = c.connect()
    if not ok:
        raise ConnectionError(msg)
    return c


def connect_mysql(
    host: str, database: str, username: str, password: str, port: int = 3306
) -> DBConnector:
    cfg = DBConfig("mysql", host, port, database, username, password)
    c   = DBConnector(cfg)
    ok, msg = c.connect()
    if not ok:
        raise ConnectionError(msg)
    return c


def connect_sqlite(filepath: str) -> DBConnector:
    cfg = DBConfig("sqlite", filepath=filepath)
    c   = DBConnector(cfg)
    ok, msg = c.connect()
    if not ok:
        raise ConnectionError(msg)
    return c


def build_connector(
    db_type : str,
    host    : str = "localhost",
    port    : int = 0,
    database: str = "",
    username: str = "",
    password: str = "",
    filepath: str = "",
) -> DBConnector:
    cfg = DBConfig(db_type, host, port, database, username, password, filepath)
    return DBConnector(cfg)
