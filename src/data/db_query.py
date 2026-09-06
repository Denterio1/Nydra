"""
db_query.py — Query Interface for Nydra v0.5.0
Sits between db_connector.py and the UI.

Features:
  - Table browser with stats
  - SQL editor with validation
  - Saved/favourite queries
  - Auto-suggestions
  - Result → Nydra data dict
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from src.data.db_connector import DBConnector, validate_sql


# ══════════════════════════════════════════════════════════════════════════════
# 1. SAVED QUERIES
# ══════════════════════════════════════════════════════════════════════════════

SAVED_QUERIES_FILE = "nydra_saved_queries.json"


@dataclass
class SavedQuery:
    name       : str
    sql        : str
    db_type    : str
    database   : str
    description: str = ""
    created_at : str = field(default_factory=lambda: datetime.now().isoformat())
    used_count : int = 0


class QueryLibrary:
    """Persist favourite queries to a local JSON file."""

    def __init__(self, filepath: str = SAVED_QUERIES_FILE):
        self._path    = filepath
        self._queries : list[SavedQuery] = []
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._queries = [SavedQuery(**q) for q in raw]
            except Exception:
                self._queries = []

    def _save(self):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(
                [q.__dict__ for q in self._queries],
                f, indent=2, ensure_ascii=False,
            )

    def add(self, query: SavedQuery):
        # avoid duplicates by name
        self._queries = [q for q in self._queries if q.name != query.name]
        self._queries.append(query)
        self._save()

    def remove(self, name: str):
        self._queries = [q for q in self._queries if q.name != name]
        self._save()

    def get(self, name: str) -> SavedQuery | None:
        for q in self._queries:
            if q.name == name:
                return q
        return None

    def all(self) -> list[SavedQuery]:
        return sorted(self._queries, key=lambda q: q.used_count, reverse=True)

    def mark_used(self, name: str):
        q = self.get(name)
        if q:
            q.used_count += 1
            self._save()


# ══════════════════════════════════════════════════════════════════════════════
# 2. TABLE BROWSER
# ══════════════════════════════════════════════════════════════════════════════

class TableBrowser:
    """Browse and explore tables in a connected database."""

    def __init__(self, connector: DBConnector):
        self._conn = connector

    def list_tables(self) -> pd.DataFrame:
        """Return a DataFrame of all tables with stats."""
        tables = self._conn.get_tables()
        if not tables:
            return pd.DataFrame()
        df = pd.DataFrame(tables)
        df = df.rename(columns={
            "table":   "Table",
            "rows":    "Rows",
            "columns": "Columns",
            "numeric": "Numeric cols",
            "text":    "Text cols",
        })
        return df

    def table_summary(self, table: str) -> dict:
        """Return detailed info about a single table."""
        info = self._conn.get_table_info(table)
        if not info:
            return {}
        return {
            "name":         info.name,
            "row_count":    info.row_count,
            "column_count": len(info.columns),
            "numeric_cols": info.numeric_cols,
            "text_cols":    info.text_cols,
            "indexes":      info.indexes,
            "columns": [
                {
                    "name":        c.name,
                    "type":        c.dtype,
                    "nullable":    c.nullable,
                    "primary_key": c.primary_key,
                    "foreign_key": c.foreign_key,
                }
                for c in info.columns
            ],
        }

    def preview(self, table: str, n: int = 5) -> pd.DataFrame:
        """Return first N rows of a table."""
        return self._conn.preview(table, n=n)

    def quick_stats(self, table: str) -> dict:
        """Return quick numeric stats for a table."""
        data = self._conn.load_table(table, limit=5_000, smart=True)
        df   = data["df"]
        stats = {}
        for col in df.select_dtypes(include="number").columns:
            stats[col] = {
                "min":  round(float(df[col].min()), 4),
                "max":  round(float(df[col].max()), 4),
                "mean": round(float(df[col].mean()), 4),
                "null": int(df[col].isnull().sum()),
            }
        return stats


# ══════════════════════════════════════════════════════════════════════════════
# 3. SQL EDITOR
# ══════════════════════════════════════════════════════════════════════════════

class SQLEditor:
    """
    Validates, runs, and explains SQL queries.
    Returns Nydra-compatible data dicts.
    """

    def __init__(self, connector: DBConnector):
        self._conn    = connector
        self._library = QueryLibrary()

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        sql       : str,
        limit     : int  = 10_000,
        auto_limit: bool = True,
    ) -> dict[str, Any]:
        """Validate and execute SQL, return Nydra data dict."""
        safe, reason = validate_sql(sql)
        if not safe:
            raise ValueError(reason)
        return self._conn.query(sql, limit=limit, auto_limit=auto_limit)

    def run_table(
        self,
        table      : str,
        limit      : int        = 10_000,
        target_col : str | None = None,
    ) -> dict[str, Any]:
        """Load a full table with smart sampling."""
        return self._conn.load_table(table, limit=limit, target_col=target_col)

    # ── Suggestions ───────────────────────────────────────────────────────────

    def autocomplete(self, partial: str) -> list[str]:
        """
        Suggest table and column names based on partial input.
        Useful for UI autocomplete dropdowns.
        """
        partial = partial.lower().strip()
        suggestions = []

        for t in self._conn.get_tables():
            tname = t["table"]
            if partial in tname.lower():
                suggestions.append(tname)
            info = self._conn.get_table_info(tname)
            if info:
                for col in info.col_names:
                    if partial in col.lower():
                        suggestions.append(f"{tname}.{col}")

        return sorted(set(suggestions))[:20]

    def template_queries(self, table: str) -> dict[str, str]:
        """Return ready-to-use SQL templates for a table."""
        info = self._conn.get_table_info(table)
        cols = ", ".join(info.col_names[:6]) if info else "*"
        num  = info.numeric_cols[0] if info and info.numeric_cols else None

        templates: dict[str, str] = {
            "Preview (10 rows)":
                f"SELECT * FROM {table} LIMIT 10",

            "Count rows":
                f"SELECT COUNT(*) AS total FROM {table}",

            "Select columns":
                f"SELECT {cols} FROM {table} LIMIT 1000",

            "Null check":
                f"SELECT * FROM {table} WHERE "
                + " OR ".join(
                    f"{c} IS NULL"
                    for c in (info.col_names[:4] if info else ["id"])
                )
                + f" LIMIT 100",

            "Distinct values":
                f"SELECT DISTINCT {info.col_names[0] if info else 'id'} "
                f"FROM {table} LIMIT 100",
        }

        if num:
            templates["Numeric stats"] = (
                f"SELECT "
                f"MIN({num}) AS min_val, "
                f"MAX({num}) AS max_val, "
                f"AVG({num}) AS avg_val, "
                f"COUNT({num}) AS count_val "
                f"FROM {table}"
            )

        return templates

    def explain(self, sql: str) -> str:
        """
        Return a human-readable explanation of what the SQL does.
        (Simple rule-based — no LLM needed.)
        """
        sql_up = sql.upper().strip()
        parts  = []

        if "SELECT *" in sql_up:
            parts.append("Selects ALL columns")
        else:
            parts.append("Selects specific columns")

        from_match = __import__("re").search(r"FROM\s+(\w+)", sql_up)
        if from_match:
            parts.append(f"from table '{from_match.group(1).lower()}'")

        if "WHERE" in sql_up:
            parts.append("with a filter condition")
        if "ORDER BY" in sql_up:
            parts.append("sorted by a column")
        if "LIMIT" in sql_up:
            limit_match = __import__("re").search(r"LIMIT\s+(\d+)", sql_up)
            if limit_match:
                parts.append(f"limited to {limit_match.group(1)} rows")
        if "JOIN" in sql_up:
            parts.append("joining multiple tables")
        if "GROUP BY" in sql_up:
            parts.append("grouped by a column")

        return " ".join(parts) + "."

    # ── Saved Queries ─────────────────────────────────────────────────────────

    def save_query(
        self,
        name       : str,
        sql        : str,
        description: str = "",
    ):
        q = SavedQuery(
            name        = name,
            sql         = sql,
            db_type     = self._conn.config.db_type,
            database    = self._conn.config.database,
            description = description,
        )
        self._library.add(q)

    def load_saved(self, name: str) -> str | None:
        q = self._library.get(name)
        if q:
            self._library.mark_used(name)
            return q.sql
        return None

    def list_saved(self) -> list[dict]:
        return [
            {
                "name":        q.name,
                "sql":         q.sql[:60] + ("..." if len(q.sql) > 60 else ""),
                "db_type":     q.db_type,
                "description": q.description,
                "used":        q.used_count,
                "saved_at":    q.created_at[:10],
            }
            for q in self._library.all()
        ]

    def delete_saved(self, name: str):
        self._library.remove(name)

    # ── History ───────────────────────────────────────────────────────────────

    def history_df(self) -> pd.DataFrame:
        return self._conn.history.as_df()

    def clear_history(self):
        self._conn.history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 4. RELATIONSHIP EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

class RelationshipExplorer:
    """Explore foreign key relationships and suggest JOIN queries."""

    def __init__(self, connector: DBConnector):
        self._conn = connector

    def get_relationships(self) -> list[dict]:
        return self._conn.get_relationships()

    def suggest_joins(self, table: str) -> list[str]:
        """Suggest JOIN queries based on FK relationships."""
        rels  = self.get_relationships()
        joins = []

        for r in rels:
            if r["from_table"] == table:
                from_col = r["from_cols"][0]
                to_table = r["to_table"]
                to_col   = r["to_cols"][0]
                joins.append(
                    f"SELECT * FROM {table} "
                    f"JOIN {to_table} ON {table}.{from_col} = {to_table}.{to_col} "
                    f"LIMIT 1000"
                )
            elif r["to_table"] == table:
                from_table = r["from_table"]
                from_col   = r["from_cols"][0]
                to_col     = r["to_cols"][0]
                joins.append(
                    f"SELECT * FROM {from_table} "
                    f"JOIN {table} ON {from_table}.{from_col} = {table}.{to_col} "
                    f"LIMIT 1000"
                )

        return joins

    def relationship_summary(self) -> pd.DataFrame:
        rels = self.get_relationships()
        if not rels:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "From table": r["from_table"],
                "From col":   ", ".join(r["from_cols"]),
                "To table":   r["to_table"],
                "To col":     ", ".join(r["to_cols"]),
            }
            for r in rels
        ])


# ══════════════════════════════════════════════════════════════════════════════
# 5. SESSION — ties everything together
# ══════════════════════════════════════════════════════════════════════════════

class DBSession:
    """
    Single entry point for the UI.
    Combines connector + browser + editor + relationships.
    """

    def __init__(self, connector: DBConnector):
        self.connector     = connector
        self.browser       = TableBrowser(connector)
        self.editor        = SQLEditor(connector)
        self.relationships = RelationshipExplorer(connector)

    @property
    def is_connected(self) -> bool:
        return self.connector.is_connected

    def status(self) -> dict:
        return self.connector.status()

    def disconnect(self):
        self.connector.disconnect()
