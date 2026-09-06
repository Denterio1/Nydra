"""
dna_memory.py — DNA Strategy Memory & Similarity Engine (v0.4.0)

Stores DataDNA fingerprints persistently and enables:
    - Dataset similarity search
    - Strategy recommendation based on past successes
    - DNA evolution tracking over time
    - Auto strategy selection

Architecture:
    DNAStore          — SQLite-based persistent DNA storage
    SimilaritySearch  — finds similar past datasets
    StrategyMemory    — remembers what worked for similar data
    AutoStrategySelector — recommends best strategy automatically
    DNAEvolutionTracker  — tracks how a dataset changes over time
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from src.data.cognitive_dna import DataDNA, DNASimilarityEngine, SimilarityReport

# ── Database path (Move to root for consistency) ──────────────────────────────
DNA_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "nydra_dna.db"
)


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StoredDNA:
    """A DataDNA stored in the database with metadata."""
    id:           int
    source:       str
    dna_hash:     str
    personality:  str
    tags:         list[str]
    ml_score:     int
    task_type:    str
    target_signal:str
    rows:         int
    columns:      int
    missing_mech: str
    created_at:   str
    dna_json:     str

    def to_dna(self) -> DataDNA:
        """Deserialize back to DataDNA."""
        from src.data.cognitive_dna import (
            DataDNA, StatisticalDNA, StructuralDNA, MLDNA, TemporalDNA
        )
        d = json.loads(self.dna_json)

        statistical = StatisticalDNA(
            entropy_vector    = tuple(d["statistical"]["entropy_vector"]),
            moment_signatures = tuple(tuple(m) for m in d["statistical"]["moment_signatures"]),
            distribution_types= tuple(d["statistical"]["distribution_types"]),
            correlation_hash  = d["statistical"]["correlation_hash"],
            normality_scores  = tuple(d["statistical"]["normality_scores"]),
            mutual_info_matrix= tuple(tuple(r) for r in d["statistical"]["mutual_info_matrix"]),
        )
        structural = StructuralDNA(
            schema_hash          = d["structural"]["schema_hash"],
            missing_mechanism    = d["structural"]["missing_mechanism"],
            missing_pattern_hash = d["structural"]["missing_pattern_hash"],
            cardinality_profile  = tuple(d["structural"]["cardinality_profile"]),
            outlier_density      = d["structural"]["outlier_density"],
            duplicate_ratio      = d["structural"]["duplicate_ratio"],
            shape_signature      = tuple(d["structural"]["shape_signature"]),
        )
        ml = MLDNA(
            feature_redundancy  = d["ml"]["feature_redundancy"],
            target_signal       = d["ml"]["target_signal"],
            separability_score  = d["ml"]["separability_score"],
            class_balance_score = d["ml"]["class_balance_score"],
            recommended_task    = d["ml"]["recommended_task"],
            complexity_score    = d["ml"]["complexity_score"],
        )
        temporal = TemporalDNA(
            has_temporal      = d["temporal"]["has_temporal"],
            temporal_columns  = tuple(d["temporal"]["temporal_columns"]),
            trend_detected    = d["temporal"]["trend_detected"],
            seasonality_score = d["temporal"]["seasonality_score"],
            time_gaps_uniform = d["temporal"]["time_gaps_uniform"],
        )
        return DataDNA(
            statistical = statistical,
            structural  = structural,
            ml          = ml,
            temporal    = temporal,
            dna_hash    = d["dna_hash"],
            created_at  = d["created_at"],
            source      = d["source"],
            version     = d.get("version", "0.4.0"),
            personality = d.get("personality", ""),
            tags        = d.get("tags", []),
        )


@dataclass
class StrategyRecord:
    """A strategy that was applied to a dataset and its outcome."""
    dna_hash:    str
    strategy:    str
    parameters:  dict[str, Any]
    ml_score:    int
    outcome:     str        # "excellent" | "good" | "poor"
    notes:       str
    created_at:  str


@dataclass
class SimilarDataset:
    """Result of similarity search."""
    stored_dna:      StoredDNA
    similarity:      float
    report:          SimilarityReport
    recommended_strategy: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# DNA Store
# ══════════════════════════════════════════════════════════════════════════════

class DNAStore:
    """Persistent SQLite store for DataDNA fingerprints."""

    def __init__(self, db_path: str = DNA_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS dna_store (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source        TEXT NOT NULL,
                dna_hash      TEXT NOT NULL,
                personality   TEXT,
                tags          TEXT,
                ml_score      INTEGER DEFAULT 0,
                task_type     TEXT,
                target_signal TEXT,
                rows          INTEGER,
                columns       INTEGER,
                missing_mech  TEXT,
                created_at    TEXT,
                dna_json      TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_dna_hash ON dna_store(dna_hash);
            CREATE INDEX IF NOT EXISTS idx_task_type ON dna_store(task_type);

            CREATE TABLE IF NOT EXISTS strategy_memory (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dna_hash    TEXT NOT NULL,
                strategy    TEXT NOT NULL,
                parameters  TEXT,
                ml_score    INTEGER DEFAULT 0,
                outcome     TEXT,
                notes       TEXT,
                created_at  TEXT
            );

            CREATE TABLE IF NOT EXISTS dna_evolution (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key  TEXT NOT NULL,
                dna_hash    TEXT NOT NULL,
                version     INTEGER DEFAULT 1,
                changes     TEXT,
                created_at  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_source_key ON dna_evolution(source_key);
        """)
        conn.commit()
        conn.close()

    def save(self, dna: DataDNA, ml_score: int = 0) -> int:
        """Save a DataDNA to the store. Returns the new record ID."""
        conn = self._get_conn()

        # Check if exact same DNA already exists
        existing = conn.execute(
            "SELECT id FROM dna_store WHERE dna_hash = ?", (dna.dna_hash,)
        ).fetchone()

        if existing:
            # Update score if it's better
            conn.execute(
                "UPDATE dna_store SET ml_score = MAX(ml_score, ?) WHERE id = ?",
                (ml_score, existing["id"])
            )
            conn.commit()
            conn.close()
            return existing["id"]

        dna_dict = asdict(dna)
        row_id = conn.execute("""
            INSERT INTO dna_store
                (source, dna_hash, personality, tags, ml_score, task_type,
                 target_signal, rows, columns, missing_mech, created_at, dna_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dna.source,
            dna.dna_hash,
            dna.personality,
            json.dumps(dna.tags),
            ml_score,
            dna.ml.recommended_task,
            dna.ml.target_signal,
            dna.structural.shape_signature[0],
            dna.structural.shape_signature[1],
            dna.structural.missing_mechanism,
            dna.created_at,
            json.dumps(dna_dict, default=str),
        )).lastrowid

        conn.commit()
        conn.close()
        return row_id

    def get_all(self, limit: int = 100) -> list[StoredDNA]:
        """Get all stored DNAs."""
        conn  = self._get_conn()
        rows  = conn.execute(
            "SELECT * FROM dna_store ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [self._row_to_stored(r) for r in rows]

    def get_by_task(self, task_type: str) -> list[StoredDNA]:
        """Get DNAs filtered by task type."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM dna_store WHERE task_type = ? ORDER BY ml_score DESC",
            (task_type,)
        ).fetchall()
        conn.close()
        return [self._row_to_stored(r) for r in rows]

    def get_by_hash(self, dna_hash: str) -> StoredDNA | None:
        conn = self._get_conn()
        row  = conn.execute(
            "SELECT * FROM dna_store WHERE dna_hash = ?", (dna_hash,)
        ).fetchone()
        conn.close()
        return self._row_to_stored(row) if row else None

    def count(self) -> int:
        conn = self._get_conn()
        n    = conn.execute("SELECT COUNT(*) FROM dna_store").fetchone()[0]
        conn.close()
        return n

    def save_strategy(self, record: StrategyRecord) -> None:
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO strategy_memory
                (dna_hash, strategy, parameters, ml_score, outcome, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.dna_hash,
            record.strategy,
            json.dumps(record.parameters),
            record.ml_score,
            record.outcome,
            record.notes,
            record.created_at,
        ))
        conn.commit()
        conn.close()

    def get_strategies_for_hash(self, dna_hash: str) -> list[StrategyRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM strategy_memory WHERE dna_hash = ? ORDER BY ml_score DESC",
            (dna_hash,)
        ).fetchall()
        conn.close()
        return [
            StrategyRecord(
                dna_hash   = r["dna_hash"],
                strategy   = r["strategy"],
                parameters = json.loads(r["parameters"] or "{}"),
                ml_score   = r["ml_score"],
                outcome    = r["outcome"] or "",
                notes      = r["notes"] or "",
                created_at = r["created_at"] or "",
            )
            for r in rows
        ]

    def save_evolution(self, source_key: str, new_dna: DataDNA, changes: list[str]) -> None:
        conn    = self._get_conn()
        # Find current version for this source_key (e.g. filename)
        version = conn.execute(
            "SELECT COUNT(*) FROM dna_evolution WHERE source_key = ?", (source_key,)
        ).fetchone()[0] + 1
        
        conn.execute("""
            INSERT INTO dna_evolution (source_key, dna_hash, version, changes, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            source_key,
            new_dna.dna_hash,
            version,
            json.dumps(changes),
            datetime.now().isoformat(),
        ))
        conn.commit()
        conn.close()

    def get_evolution(self, source_key: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM dna_evolution WHERE source_key = ? ORDER BY version",
            (source_key,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _row_to_stored(self, row: sqlite3.Row) -> StoredDNA:
        return StoredDNA(
            id           = row["id"],
            source       = row["source"],
            dna_hash     = row["dna_hash"],
            personality  = row["personality"] or "",
            tags         = json.loads(row["tags"] or "[]"),
            ml_score     = row["ml_score"] or 0,
            task_type    = row["task_type"] or "unknown",
            target_signal= row["target_signal"] or "none",
            rows         = row["rows"] or 0,
            columns      = row["columns"] or 0,
            missing_mech = row["missing_mech"] or "NONE",
            created_at   = row["created_at"] or "",
            dna_json     = row["dna_json"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# Similarity Search
# ══════════════════════════════════════════════════════════════════════════════

class SimilaritySearch:
    """
    Find similar datasets from the DNA store.

    Uses DNASimilarityEngine to compare against all stored DNAs
    and returns the top-k most similar.
    """

    def __init__(self, store: DNAStore):
        self.store  = store
        self.engine = DNASimilarityEngine()

    def find_similar(
        self,
        query_dna:  DataDNA,
        top_k:      int   = 5,
        min_sim:    float = 0.4,
        task_filter:str | None = None,
    ) -> list[SimilarDataset]:
        """
        Find the most similar stored datasets.

        Args:
            query_dna:   The DNA to search for.
            top_k:       Number of results to return.
            min_sim:     Minimum similarity threshold.
            task_filter: Filter by task type ("classification"/"regression").

        Returns:
            List of SimilarDataset sorted by similarity descending.
        """
        if task_filter:
            candidates = self.store.get_by_task(task_filter)
        else:
            candidates = self.store.get_all(limit=1000)

        # Exclude exact same DNA
        candidates = [c for c in candidates if c.dna_hash != query_dna.dna_hash]

        if not candidates:
            return []

        results: list[SimilarDataset] = []

        for stored in candidates:
            try:
                stored_dna = stored.to_dna()
                report     = self.engine.compare(query_dna, stored_dna)

                if report.overall_similarity >= min_sim:
                    results.append(SimilarDataset(
                        stored_dna = stored,
                        similarity = report.overall_similarity,
                        report     = report,
                    ))
            except (json.JSONDecodeError, ImportError, KeyError, AttributeError):
                # Skip corrupt or incompatible records instead of failing entire search
                continue
            except Exception:
                continue

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Memory
# ══════════════════════════════════════════════════════════════════════════════

class StrategyMemory:
    """
    Remember what strategies worked best for similar datasets.

    Learns from past successes and failures to recommend
    the optimal cleaning and ML strategy for new datasets.
    """

    def __init__(self, store: DNAStore, search: SimilaritySearch):
        self.store  = store
        self.search = search

    def record_success(
        self,
        dna:        DataDNA,
        strategy:   str,
        parameters: dict,
        ml_score:   int,
        notes:      str = "",
    ) -> None:
        """Record a successful strategy application."""
        outcome = "excellent" if ml_score >= 85 else "good" if ml_score >= 65 else "poor"
        record  = StrategyRecord(
            dna_hash   = dna.dna_hash,
            strategy   = strategy,
            parameters = parameters,
            ml_score   = ml_score,
            outcome    = outcome,
            notes      = notes,
            created_at = datetime.now().isoformat(),
        )
        self.store.save_strategy(record)

    def get_recommendation(
        self,
        query_dna: DataDNA,
        min_sim:   float = 0.5,
    ) -> dict[str, Any]:
        """
        Get strategy recommendation based on similar past datasets.

        Returns:
            {
                "has_recommendation": bool,
                "strategy":           str,
                "parameters":         dict,
                "confidence":         float,
                "based_on":           list of similar dataset names,
                "explanation":        str,
            }
        """
        similar = self.search.find_similar(query_dna, top_k=10, min_sim=min_sim)

        if not similar:
            return {
                "has_recommendation": False,
                "strategy":    "mean",
                "parameters":  {"missing_strategy": "mean", "remove_dupes": True},
                "confidence":  0.0,
                "based_on":    [],
                "explanation": "No similar datasets found. Using default strategy.",
            }

        # Collect strategies from similar datasets
        all_strategies: list[tuple[str, dict, int, float]] = []

        for sim_dataset in similar:
            records = self.store.get_strategies_for_hash(sim_dataset.stored_dna.dna_hash)
            for r in records:
                if r.outcome in ("excellent", "good"):
                    all_strategies.append((
                        r.strategy,
                        r.parameters,
                        r.ml_score,
                        sim_dataset.similarity,
                    ))

        if not all_strategies:
            return {
                "has_recommendation": False,
                "strategy":    "mean",
                "parameters":  {"missing_strategy": "mean", "remove_dupes": True},
                "confidence":  0.0,
                "based_on":    [s.stored_dna.source for s in similar[:3]],
                "explanation": f"Found {len(similar)} similar datasets but no recorded strategies yet.",
            }

        # Weight strategies by similarity × ml_score
        strategy_scores: dict[str, float] = {}
        strategy_params: dict[str, dict]  = {}

        for strategy, params, ml_score, sim in all_strategies:
            weight = (ml_score / 100) * sim
            strategy_scores[strategy] = strategy_scores.get(strategy, 0) + weight
            if strategy not in strategy_params or ml_score > strategy_params.get(f"{strategy}_score", 0):
                strategy_params[strategy] = params
                strategy_params[f"{strategy}_score"] = ml_score

        best_strategy = max(strategy_scores, key=strategy_scores.get)
        total_weight  = sum(strategy_scores.values())
        confidence    = strategy_scores[best_strategy] / total_weight if total_weight > 0 else 0

        return {
            "has_recommendation": True,
            "strategy":    best_strategy,
            "parameters":  strategy_params.get(best_strategy, {}),
            "confidence":  round(confidence, 3),
            "based_on":    [s.stored_dna.source for s in similar[:3]],
            "explanation": (
                f"Based on {len(similar)} similar datasets. "
                f"'{best_strategy}' succeeded {len([s for s in all_strategies if s[0] == best_strategy])} "
                f"times with avg ML score "
                f"{sum(s[2] for s in all_strategies if s[0] == best_strategy) // max(1, len([s for s in all_strategies if s[0] == best_strategy]))}."
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Auto Strategy Selector
# ══════════════════════════════════════════════════════════════════════════════

class AutoStrategySelector:
    """
    Automatically select the best strategy for a dataset.

    Combines rule-based heuristics with learned strategies
    from StrategyMemory to recommend optimal parameters.
    """

    def __init__(self, strategy_memory: StrategyMemory):
        self.memory = strategy_memory

    def select(self, dna: DataDNA) -> dict[str, Any]:
        """
        Select optimal strategy based on DNA properties + past learning.

        Returns complete strategy configuration.
        """
        # Get learned recommendation
        learned = self.memory.get_recommendation(dna, min_sim=0.5)

        # Rule-based heuristics
        rules = self._apply_rules(dna)

        # Merge: learned takes priority if high confidence
        if learned["has_recommendation"] and learned["confidence"] >= 0.6:
            strategy = {**rules, **learned["parameters"]}
            source   = "learned"
            confidence = learned["confidence"]
            explanation = learned["explanation"]
        else:
            strategy   = rules
            source     = "rules"
            confidence = 0.5
            explanation = self._explain_rules(dna, rules)

        return {
            "strategy":    strategy,
            "source":      source,
            "confidence":  confidence,
            "explanation": explanation,
            "based_on":    learned.get("based_on", []),
            "rules_used":  self._list_rules(dna),
        }

    def _apply_rules(self, dna: DataDNA) -> dict[str, Any]:
        """Apply rule-based strategy selection."""
        config: dict[str, Any] = {}

        # Missing value strategy
        mech = dna.structural.missing_mechanism
        if mech == "NONE":
            config["missing_strategy"] = "mean"
        elif mech == "MCAR":
            config["missing_strategy"] = "mean"
        elif mech == "MAR":
            config["missing_strategy"] = "median"
        elif mech == "MNAR":
            config["missing_strategy"] = "mode"

        # Remove duplicates
        config["remove_dupes"] = dna.structural.duplicate_ratio > 0

        # Scaling
        config["scale"] = dna.ml.feature_redundancy < 0.8

        # Encoding
        config["encode"] = True

        # Handle imbalance
        config["handle_imbalance"] = (
            dna.ml.recommended_task == "classification" and
            dna.ml.class_balance_score < 0.7
        )

        # Outlier treatment
        config["treat_outliers"] = dna.structural.outlier_density > 0.05

        # Feature engineering
        config["feature_engineering"] = (
            dna.temporal.has_temporal or
            dna.structural.shape_signature[1] < 10
        )

        return config

    def _explain_rules(self, dna: DataDNA, rules: dict) -> str:
        parts = []
        if rules.get("missing_strategy") == "median":
            parts.append("Using median imputation (MAR pattern detected)")
        elif rules.get("missing_strategy") == "mode":
            parts.append("Using mode imputation (MNAR pattern detected)")
        if rules.get("handle_imbalance"):
            parts.append("SMOTE recommended (class imbalance detected)")
        if rules.get("feature_engineering"):
            parts.append("Feature engineering recommended (temporal data or few features)")
        if rules.get("treat_outliers"):
            parts.append(f"Outlier treatment needed ({dna.structural.outlier_density:.1%} density)")
        return " | ".join(parts) if parts else "Default strategy applied."

    def _list_rules(self, dna: DataDNA) -> list[str]:
        rules = []
        rules.append(f"missing_mechanism={dna.structural.missing_mechanism}")
        rules.append(f"task={dna.ml.recommended_task}")
        rules.append(f"balance={dna.ml.class_balance_score:.2f}")
        rules.append(f"outliers={dna.structural.outlier_density:.3f}")
        rules.append(f"temporal={dna.temporal.has_temporal}")
        return rules


# ══════════════════════════════════════════════════════════════════════════════
# DNA Evolution Tracker
# ══════════════════════════════════════════════════════════════════════════════

class DNAEvolutionTracker:
    """
    Track how a dataset's DNA changes over time.

    Detects if data has drifted, improved, or degraded
    by comparing DNA snapshots across versions.
    """

    def __init__(self, store: DNAStore):
        self.store  = store
        self.engine = DNASimilarityEngine()

    def track(self, source_key: str, new_dna: DataDNA) -> dict[str, Any]:
        """
        Track a new version of a dataset.

        Args:
            source_key: Stable identifier for the dataset (e.g. filename).
            new_dna:    The newly computed DNA.

        Returns:
            Evolution report with changes and trends.
        """
        # Get previous DNA versions for this source
        history = self.store.get_evolution(source_key)

        if not history:
            # First time seeing this dataset
            self.store.save(new_dna)
            self.store.save_evolution(source_key, new_dna, ["Initial snapshot"])
            return {
                "is_new":      True,
                "version":     1,
                "changes":     [],
                "trend":       "new",
                "explanation": "First snapshot recorded.",
            }

        # Get most recent previous DNA from store
        prev_hash = history[-1]["dna_hash"]
        prev_stored = self.store.get_by_hash(prev_hash)

        if not prev_stored:
             self.store.save(new_dna)
             self.store.save_evolution(source_key, new_dna, ["Baseline lost - restarted history"])
             return {"is_new": True, "version": 1, "trend": "restarted"}

        try:
            prev_dna = prev_stored.to_dna()
            if prev_dna.dna_hash == new_dna.dna_hash:
                return {
                    "is_new": False,
                    "version": len(history),
                    "trend": "stable",
                    "explanation": "No changes detected."
                }
            
            report = self.engine.compare(prev_dna, new_dna)
        except Exception as e:
            return {"error": str(e), "is_new": False}

        # Detect changes
        changes = report.key_differences

        # Detect trend
        if report.overall_similarity >= 0.95:
            trend = "stable"
        elif report.overall_similarity >= 0.75:
            trend = "minor_drift"
        elif report.overall_similarity >= 0.5:
            trend = "moderate_drift"
        else:
            trend = "major_drift"

        # ML score trend
        ml_change = new_dna.ml.separability_score - prev_dna.ml.separability_score
        if ml_change > 0.1:
            ml_trend = "improving"
        elif ml_change < -0.1:
            ml_trend = "degrading"
        else:
            ml_trend = "stable"

        # Save evolution record
        self.store.save(new_dna)
        self.store.save_evolution(source_key, new_dna, changes)

        return {
            "is_new":        False,
            "version":       len(history) + 1,
            "similarity":    report.overall_similarity,
            "trend":         trend,
            "ml_trend":      ml_trend,
            "changes":       changes,
            "prev_hash":     prev_dna.dna_hash,
            "new_hash":      new_dna.dna_hash,
            "prev_date":     prev_stored.created_at[:19],
            "explanation":   report.recommendation,
            "history_count": len(history) + 1,
        }


# ══════════════════════════════════════════════════════════════════════════════
# DNA Manager — Main Interface
# ══════════════════════════════════════════════════════════════════════════════

class DNAManager:
    """
    Main interface for all DNA v0.4.0 features.

    Usage:
        manager = DNAManager()
        dna     = compute_dna(data)

        # Save and find similar
        manager.store.save(dna, ml_score=82)
        similar = manager.search.find_similar(dna)

        # Get strategy recommendation
        rec = manager.auto_selector.select(dna)

        # Track evolution
        evo = manager.evolution.track("mydata.csv", dna)
    """

    def __init__(self, db_path: str = DNA_DB_PATH):
        self.store         = DNAStore(db_path)
        self.search        = SimilaritySearch(self.store)
        self.strategy_mem  = StrategyMemory(self.store, self.search)
        self.auto_selector = AutoStrategySelector(self.strategy_mem)
        self.evolution     = DNAEvolutionTracker(self.store)

    def full_analysis(
        self,
        dna:        DataDNA,
        source_key: str,
        ml_score:   int = 0,
    ) -> dict[str, Any]:
        """
        Run complete DNA analysis: save, search, recommend, track.

        Returns comprehensive report.
        """
        # Save DNA
        record_id = self.store.save(dna, ml_score)

        # Find similar datasets
        similar = self.search.find_similar(dna, top_k=5, min_sim=0.4)

        # Get strategy recommendation
        recommendation = self.auto_selector.select(dna)

        # Track evolution
        evolution = self.evolution.track(source_key, dna)

        return {
            "record_id":      record_id,
            "dna_hash":       dna.short_hash,
            "personality":    dna.personality,
            "tags":           dna.tags,
            "similar":        [
                {
                    "source":     s.stored_dna.source,
                    "similarity": s.similarity,
                    "ml_score":   s.stored_dna.ml_score,
                    "personality":s.stored_dna.personality,
                }
                for s in similar
            ],
            "recommendation": recommendation,
            "evolution":      evolution,
            "total_in_db":    self.store.count(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager: DNAManager | None = None


def get_dna_manager() -> DNAManager:
    global _manager
    if _manager is None:
        _manager = DNAManager()
    return _manager
