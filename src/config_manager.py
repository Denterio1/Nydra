"""
config_manager.py — dataDoctor Configuration Manager v1.0
==========================================================
Production-grade user settings management backed by SQLite.
Integrates with SecurityVault for transparent encryption of
sensitive values (API keys, tokens, passwords).

Features:
     Thread-safe SQLite with WAL journal mode
     Schema versioning + auto-migrations
     In-memory LRU cache with TTL (avoid DB hit every request)
     Full config history (every change tracked with old value)
     Auto-encrypt sensitive fields via SecurityVault
     Config validation (type, allowed values, length limits)
     Default value templates per user profile
     Bulk save / bulk load
     Config diff (what changed between two snapshots)
     Export to JSON / Import from JSON
     Database backup
     Admin utilities (list all users, purge old data)

Database Schema:
    user_configs    — main settings table
    config_history  — immutable change log
    schema_version  — migration tracking

Architecture:
    ConfigException         — typed exception hierarchy
    ConfigEntry             — single setting with full metadata
    ConfigDiff              — diff between two config snapshots
    DatabaseConnection      — thread-safe SQLite pool + WAL
    SchemaManager           — versioned migrations
    LRUCache                — in-memory cache with TTL
    ConfigValidator         — type + value validation
    ConfigRepository        — raw DB CRUD operations
    ConfigHistoryRepo       — change history read/write
    ConfigManager           — master class (use this in your app)

"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import sqlite3
import logging
import hashlib
import threading
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any
from contextlib import contextmanager

# ── Logger ────────────────────────────────────────────────────────────────────
logger = logging.getLogger("dataDoctor.config")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & DEFAULT CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Default database path
DEFAULT_DB_PATH: str = "datadoctor.db"

# Schema version — bump this when you add migrations
CURRENT_SCHEMA_VERSION: int = 3

# Cache TTL in seconds (5 minutes default)
DEFAULT_CACHE_TTL: int = 300

# Maximum LRU cache entries
MAX_CACHE_SIZE: int = 500

# Maximum config value length (bytes)
MAX_VALUE_LENGTH: int = 10_000

# Maximum config key length
MAX_KEY_LENGTH: int = 128

# History retention: how many change records to keep per (email, key)
MAX_HISTORY_PER_KEY: int = 50

# Sensitive key patterns — these are auto-encrypted by ConfigManager
SENSITIVE_CONFIG_KEYS: set[str] = {
    "api_key", "apikey", "secret", "password", "passwd", "token",
    "auth_token", "access_token", "refresh_token", "private_key",
    "client_secret", "api_secret", "bearer_token",
}

# Default config values — used when a key has never been set
DEFAULT_CONFIG: dict[str, Any] = {
    "llm_provider"      : "groq",
    "llm_model"         : "llama-3.3-70b-versatile",
    "llm_temperature"   : 0.2,
    "llm_max_tokens"    : 1500,
    "theme"             : "light",
    "language"          : "en",
    "auto_save"         : True,
    "chart_type"        : "bar",
    "max_rows_display"  : 100,
    "enable_audit_log"  : True,
    "session_timeout"   : 3600,
}

# Allowed values for specific config keys (validation)
ALLOWED_VALUES: dict[str, list] = {
    "theme"         : ["light", "dark", "system"],
    "language"      : ["en", "ar", "fr", "es", "de"],
    "chart_type"    : ["bar", "line", "pie", "scatter", "area"],
    "llm_provider"  : [
        "openai", "anthropic", "google", "groq", "mistral",
        "together", "deepseek", "xai", "perplexity", "cohere",
        "openrouter", "ollama",
    ],
}

# Type enforcement for specific config keys
CONFIG_TYPES: dict[str, type] = {
    "llm_temperature"   : float,
    "llm_max_tokens"    : int,
    "max_rows_display"  : int,
    "session_timeout"   : int,
    "auto_save"         : bool,
    "enable_audit_log"  : bool,
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXCEPTION HIERARCHY
# ══════════════════════════════════════════════════════════════════════════════

class ConfigError(Exception):
    """Base exception for all config manager errors."""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when a requested config key does not exist."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when a config value fails validation."""
    pass

class ConfigDatabaseError(ConfigError):
    """Raised for database-level errors."""
    pass

class ConfigMigrationError(ConfigError):
    """Raised when a schema migration fails."""
    pass

class ConfigImportError(ConfigError):
    """Raised when importing a config file fails."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigEntry:
    """
    A single configuration entry with full metadata.
    This is what gets stored in the database and returned to callers.
    """
    email        : str
    config_key   : str
    config_value : str              # Always stored as string; typed on load
    is_encrypted : bool  = False
    is_sensitive : bool  = False
    value_type   : str   = "str"    # "str" | "int" | "float" | "bool" | "json"
    created_at   : float = field(default_factory=time.time)
    updated_at   : float = field(default_factory=time.time)
    description  : str   = ""

    @property
    def updated_iso(self) -> str:
        return datetime.fromtimestamp(self.updated_at, tz=timezone.utc).isoformat()

    @property
    def created_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat()

    def typed_value(self) -> Any:
        """Return the value cast to its proper Python type."""
        try:
            if self.value_type == "int":
                return int(self.config_value)
            if self.value_type == "float":
                return float(self.config_value)
            if self.value_type == "bool":
                return self.config_value.lower() in ("true", "1", "yes")
            if self.value_type == "json":
                return json.loads(self.config_value)
            return self.config_value
        except (ValueError, json.JSONDecodeError):
            return self.config_value

    def to_dict(self, mask_sensitive: bool = True) -> dict:
        value = self.config_value
        if mask_sensitive and self.is_sensitive:
            value = "***" + value[-4:] if len(value) > 4 else "****"
        return {
            "email"        : self.email,
            "config_key"   : self.config_key,
            "config_value" : value,
            "is_encrypted" : self.is_encrypted,
            "is_sensitive" : self.is_sensitive,
            "value_type"   : self.value_type,
            "updated_at"   : self.updated_iso,
        }


@dataclass
class ConfigHistoryEntry:
    """A single entry in the configuration change history."""
    id           : int
    email        : str
    config_key   : str
    old_value    : str | None
    new_value    : str
    changed_at   : float
    changed_by   : str   = "user"   # "user" | "system" | "import" | "reset"
    change_note  : str   = ""

    @property
    def changed_iso(self) -> str:
        return datetime.fromtimestamp(self.changed_at, tz=timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "id"          : self.id,
            "email"       : self.email,
            "config_key"  : self.config_key,
            "old_value"   : "***" if self.old_value and len(self.old_value) > 20 else self.old_value,
            "new_value"   : "***" if len(self.new_value) > 20 else self.new_value,
            "changed_at"  : self.changed_iso,
            "changed_by"  : self.changed_by,
            "change_note" : self.change_note,
        }


@dataclass
class ConfigDiff:
    """
    Represents the difference between two config snapshots.
    Used to show what changed between saves or imports.
    """
    added   : dict[str, Any] = field(default_factory=dict)
    removed : dict[str, Any] = field(default_factory=dict)
    changed : dict[str, dict] = field(default_factory=dict)   # key → {old, new}

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        return ", ".join(parts) if parts else "no changes"

    def to_dict(self) -> dict:
        return {
            "added"      : self.added,
            "removed"    : self.removed,
            "changed"    : self.changed,
            "has_changes": self.has_changes,
            "summary"    : self.summary(),
        }


@dataclass
class ValidationResult:
    """Result of validating a config value."""
    valid   : bool
    error   : str = ""
    warning : str = ""
    coerced : Any = None   # The value after type coercion (if applicable)


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATABASE CONNECTION — Thread-safe SQLite with WAL mode
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseConnection:
    """
    Thread-safe SQLite connection manager.

    Uses:
    - WAL (Write-Ahead Logging) journal mode for better concurrency
    - Thread-local connections (each thread gets its own connection)
    - Proper PRAGMA settings for production use
    - Context manager for automatic commit/rollback
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self._db_path    = db_path
        self._local      = threading.local()
        self._lock       = threading.RLock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create the database directory if it doesn't exist."""
        db_dir = os.path.dirname(os.path.abspath(self._db_path))
        os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread = False,
                timeout           = 30,
            )
            conn.row_factory = sqlite3.Row

            # Production SQLite settings
            conn.execute("PRAGMA journal_mode=WAL")        # Better concurrency
            conn.execute("PRAGMA synchronous=NORMAL")      # Balance safety/speed
            conn.execute("PRAGMA foreign_keys=ON")         # Enforce FK constraints
            conn.execute("PRAGMA cache_size=-32000")       # 32MB page cache
            conn.execute("PRAGMA temp_store=MEMORY")       # Temp tables in memory
            conn.execute("PRAGMA mmap_size=268435456")     # 256MB memory map

            self._local.conn = conn
            logger.debug(f"New SQLite connection created for thread {threading.get_ident()}")

        return self._local.conn

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        Automatically commits on success, rolls back on exception.

        Usage:
            with db.transaction() as conn:
                conn.execute("INSERT INTO ...")
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise ConfigDatabaseError(f"Database transaction failed: {e}") from e

    @contextmanager
    def cursor(self):
        """Context manager for read-only queries (no commit needed)."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            raise ConfigDatabaseError(f"Database query failed: {e}") from e

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a single SQL statement."""
        conn = self._get_connection()
        return conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        conn = self._get_connection()
        conn.executemany(sql, params_list)
        conn.commit()

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    @property
    def db_path(self) -> str:
        return self._db_path

    def db_size_mb(self) -> float:
        """Return database file size in MB."""
        try:
            return os.path.getsize(self._db_path) / 1_048_576
        except OSError:
            return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCHEMA MANAGER — Versioned migrations
# ══════════════════════════════════════════════════════════════════════════════

class SchemaManager:
    """
    Manages database schema creation and versioned migrations.

    Each migration is a (version, description, sql) tuple.
    Migrations run in order and are tracked in schema_version table.
    Safe to call multiple times — already-applied migrations are skipped.
    """

    MIGRATIONS: list[tuple[int, str, str]] = [

        (1, "Initial schema — user_configs table", """
            CREATE TABLE IF NOT EXISTS user_configs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL,
                config_key   TEXT    NOT NULL,
                config_value TEXT    NOT NULL DEFAULT '',
                is_encrypted INTEGER NOT NULL DEFAULT 0,
                is_sensitive INTEGER NOT NULL DEFAULT 0,
                value_type   TEXT    NOT NULL DEFAULT 'str',
                description  TEXT    NOT NULL DEFAULT '',
                created_at   REAL    NOT NULL DEFAULT (unixepoch('now')),
                updated_at   REAL    NOT NULL DEFAULT (unixepoch('now')),
                UNIQUE(email, config_key)
            );
            CREATE INDEX IF NOT EXISTS idx_configs_email
                ON user_configs(email);
            CREATE INDEX IF NOT EXISTS idx_configs_email_key
                ON user_configs(email, config_key);
        """),

        (2, "Config history table", """
            CREATE TABLE IF NOT EXISTS config_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL,
                config_key   TEXT    NOT NULL,
                old_value    TEXT,
                new_value    TEXT    NOT NULL DEFAULT '',
                changed_at   REAL    NOT NULL DEFAULT (unixepoch('now')),
                changed_by   TEXT    NOT NULL DEFAULT 'user',
                change_note  TEXT    NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_history_email
                ON config_history(email);
            CREATE INDEX IF NOT EXISTS idx_history_email_key
                ON config_history(email, config_key);
            CREATE INDEX IF NOT EXISTS idx_history_changed_at
                ON config_history(changed_at);
        """),

        (3, "Add config metadata columns", """
            -- SQLite doesn't support multiple ADD COLUMN in one statement
            -- Each column added separately for compatibility
            SELECT 1; -- placeholder: columns added via Python migration logic
        """),
    ]

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def initialize(self) -> None:
        """
        Create schema_version table and run all pending migrations.
        Safe to call on every app startup.
        """
        self._create_version_table()
        current = self._get_current_version()
        pending = [m for m in self.MIGRATIONS if m[0] > current]

        if not pending:
            logger.debug(f"Schema up to date at version {current}.")
            return

        for version, description, sql in pending:
            self._apply_migration(version, description, sql)

        final = self._get_current_version()
        logger.info(f"Schema migrated from v{current} to v{final}.")

    def _create_version_table(self) -> None:
        with self._db.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version     INTEGER PRIMARY KEY,
                    description TEXT    NOT NULL DEFAULT '',
                    applied_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
                )
            """)

    def _get_current_version(self) -> int:
        try:
            row = self._db.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0

    def _apply_migration(self, version: int, description: str, sql: str) -> None:
        logger.info(f"Applying migration v{version}: {description}")
        try:
            with self._db.transaction() as conn:
                # Execute migration SQL (may be multiple statements)
                conn.executescript(sql)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version (version, description) VALUES (?, ?)",
                    (version, description),
                )
            logger.info(f"Migration v{version} applied successfully.")
        except Exception as e:
            raise ConfigMigrationError(
                f"Migration v{version} failed: {e}"
            ) from e

    def get_version_history(self) -> list[dict]:
        """Return all applied migrations."""
        try:
            rows = self._db.execute(
                "SELECT version, description, applied_at FROM schema_version ORDER BY version"
            ).fetchall()
            return [
                {
                    "version"    : r["version"],
                    "description": r["description"],
                    "applied_at" : datetime.fromtimestamp(
                        r["applied_at"], tz=timezone.utc
                    ).isoformat(),
                }
                for r in rows
            ]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════════
# 6. LRU CACHE — In-memory cache with TTL
# ══════════════════════════════════════════════════════════════════════════════

class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) in-memory cache with TTL.

    Prevents hitting the database on every single request.
    Cache is invalidated on save/delete operations.

    Storage format: {cache_key: (value, expire_at_timestamp)}
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE, ttl: int = DEFAULT_CACHE_TTL) -> None:
        self._max_size  = max_size
        self._ttl       = ttl
        self._store     : dict[str, tuple[Any, float]] = {}
        self._order     : list[str] = []   # Tracks access order (oldest first)
        self._lock      = threading.RLock()
        self._hits      = 0
        self._misses    = 0

    def _make_key(self, email: str, config_key: str | None = None) -> str:
        if config_key:
            return f"{email}::{config_key}"
        return f"{email}::__all__"

    def get(self, email: str, config_key: str | None = None) -> Any | None:
        """Get a cached value. Returns None if not found or expired."""
        key = self._make_key(email, config_key)
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, expire_at = self._store[key]
            if time.time() > expire_at:
                # Expired — remove it
                del self._store[key]
                if key in self._order:
                    self._order.remove(key)
                self._misses += 1
                return None
            # Move to end (most recently used)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)
            self._hits += 1
            return value

    def set(self, email: str, value: Any, config_key: str | None = None) -> None:
        """Store a value in cache."""
        key = self._make_key(email, config_key)
        with self._lock:
            # Evict if at max size
            while len(self._store) >= self._max_size and self._order:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            expire_at = time.time() + self._ttl
            self._store[key] = (value, expire_at)
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def invalidate(self, email: str, config_key: str | None = None) -> None:
        """Invalidate cache for an email (and optionally a specific key)."""
        with self._lock:
            if config_key:
                # Invalidate specific key AND the "all configs" cache
                for k in [
                    self._make_key(email, config_key),
                    self._make_key(email),
                ]:
                    self._store.pop(k, None)
                    if k in self._order:
                        self._order.remove(k)
            else:
                # Invalidate everything for this email
                to_del = [k for k in self._store if k.startswith(f"{email}::")]
                for k in to_del:
                    del self._store[k]
                    if k in self._order:
                        self._order.remove(k)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._store.clear()
            self._order.clear()

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size"       : len(self._store),
                "max_size"   : self._max_size,
                "ttl_seconds": self._ttl,
                "hits"       : self._hits,
                "misses"     : self._misses,
                "hit_rate"   : round(self._hits / total * 100, 1) if total > 0 else 0,
            }


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFIG VALIDATOR — Type and value validation
# ══════════════════════════════════════════════════════════════════════════════

class ConfigValidator:
    """
    Validates config keys and values before saving.

    Checks:
    - Key length and format
    - Value length
    - Type enforcement (int, float, bool, json)
    - Allowed value lists (e.g. theme must be light/dark/system)
    - Email format (basic check)
    """

    MAX_KEY_LEN   : int = MAX_KEY_LENGTH
    MAX_VALUE_LEN : int = MAX_VALUE_LENGTH
    KEY_PATTERN   : re.Pattern = re.compile(r'^[a-zA-Z0-9_\-\.]{1,128}$')

    def validate_email(self, email: str) -> ValidationResult:
        """Basic email format validation."""
        if not email or not isinstance(email, str):
            return ValidationResult(False, "Email cannot be empty.")
        email = email.strip().lower()
        if len(email) > 254:
            return ValidationResult(False, "Email too long (max 254 chars).")
        if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
            return ValidationResult(False, f"Invalid email format: '{email}'")
        return ValidationResult(True, coerced=email)

    def validate_key(self, key: str) -> ValidationResult:
        """Validate a config key name."""
        if not key or not isinstance(key, str):
            return ValidationResult(False, "Config key cannot be empty.")
        if len(key) > self.MAX_KEY_LEN:
            return ValidationResult(False, f"Key too long (max {self.MAX_KEY_LEN} chars).")
        if not self.KEY_PATTERN.match(key):
            return ValidationResult(
                False,
                f"Invalid key format '{key}'. "
                "Use only letters, numbers, underscores, hyphens, and dots."
            )
        return ValidationResult(True, coerced=key.lower())

    def validate_value(self, key: str, value: Any) -> ValidationResult:
        """Validate a config value."""
        if value is None:
            return ValidationResult(False, "Value cannot be None. Use delete() to remove.")

        # Convert to string for storage
        str_value = self._to_string(value)

        if len(str_value) > self.MAX_VALUE_LEN:
            return ValidationResult(
                False,
                f"Value too long ({len(str_value)} chars, max {self.MAX_VALUE_LEN})."
            )

        # Check allowed values if defined
        if key in ALLOWED_VALUES:
            allowed = ALLOWED_VALUES[key]
            if str(value).lower() not in [str(a).lower() for a in allowed]:
                return ValidationResult(
                    False,
                    f"Invalid value '{value}' for '{key}'. "
                    f"Allowed: {allowed}"
                )

        # Type coercion check
        expected_type = CONFIG_TYPES.get(key)
        if expected_type:
            coerced, err = self._coerce_type(value, expected_type)
            if err:
                return ValidationResult(False, err)
            return ValidationResult(True, coerced=coerced)

        return ValidationResult(True, coerced=str_value)

    def _to_string(self, value: Any) -> str:
        """Convert any value to its string representation for storage."""
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    def _coerce_type(self, value: Any, target: type) -> tuple[Any, str]:
        """Attempt to coerce a value to the target type."""
        try:
            if target == bool:
                if isinstance(value, bool):
                    return value, ""
                if str(value).lower() in ("true", "1", "yes"):
                    return True, ""
                if str(value).lower() in ("false", "0", "no"):
                    return False, ""
                return None, f"Cannot convert '{value}' to bool."
            return target(value), ""
        except (ValueError, TypeError) as e:
            return None, f"Type error: expected {target.__name__}, got '{value}': {e}"

    def detect_value_type(self, value: Any) -> str:
        """Detect the Python type of a value for storage metadata."""
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, (dict, list)):
            return "json"
        return "str"

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a config key should be treated as sensitive."""
        key_lower = key.lower()
        return any(sk in key_lower for sk in SENSITIVE_CONFIG_KEYS)


# ══════════════════════════════════════════════════════════════════════════════
# 8. CONFIG REPOSITORY — Raw database CRUD
# ══════════════════════════════════════════════════════════════════════════════

class ConfigRepository:
    """
    Raw database operations for the user_configs table.
    Does NOT handle encryption or caching — that's ConfigManager's job.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def upsert(self, entry: ConfigEntry) -> None:
        """Insert or update a config entry."""
        with self._db.transaction() as conn:
            conn.execute("""
                INSERT INTO user_configs
                    (email, config_key, config_value, is_encrypted,
                     is_sensitive, value_type, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email, config_key) DO UPDATE SET
                    config_value = excluded.config_value,
                    is_encrypted = excluded.is_encrypted,
                    is_sensitive = excluded.is_sensitive,
                    value_type   = excluded.value_type,
                    description  = excluded.description,
                    updated_at   = excluded.updated_at
            """, (
                entry.email,
                entry.config_key,
                entry.config_value,
                int(entry.is_encrypted),
                int(entry.is_sensitive),
                entry.value_type,
                entry.description,
                entry.created_at,
                entry.updated_at,
            ))

    def fetch_one(self, email: str, config_key: str) -> ConfigEntry | None:
        """Fetch a single config entry by email + key."""
        row = self._db.execute(
            """SELECT email, config_key, config_value, is_encrypted,
                      is_sensitive, value_type, description, created_at, updated_at
               FROM user_configs
               WHERE email = ? AND config_key = ?""",
            (email, config_key),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def fetch_all(self, email: str) -> list[ConfigEntry]:
        """Fetch all config entries for a user."""
        rows = self._db.execute(
            """SELECT email, config_key, config_value, is_encrypted,
                      is_sensitive, value_type, description, created_at, updated_at
               FROM user_configs
               WHERE email = ?
               ORDER BY config_key ASC""",
            (email,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def fetch_by_prefix(self, email: str, prefix: str) -> list[ConfigEntry]:
        """Fetch all config entries whose key starts with prefix."""
        rows = self._db.execute(
            """SELECT email, config_key, config_value, is_encrypted,
                      is_sensitive, value_type, description, created_at, updated_at
               FROM user_configs
               WHERE email = ? AND config_key LIKE ?
               ORDER BY config_key ASC""",
            (email, f"{prefix}%"),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete_one(self, email: str, config_key: str) -> bool:
        """Delete a single config entry. Returns True if it existed."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM user_configs WHERE email = ? AND config_key = ?",
                (email, config_key),
            )
        return cursor.rowcount > 0

    def delete_all(self, email: str) -> int:
        """Delete all config entries for a user. Returns count deleted."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM user_configs WHERE email = ?",
                (email,),
            )
        return cursor.rowcount

    def exists(self, email: str, config_key: str) -> bool:
        """Check if a config entry exists."""
        row = self._db.execute(
            "SELECT 1 FROM user_configs WHERE email = ? AND config_key = ? LIMIT 1",
            (email, config_key),
        ).fetchone()
        return row is not None

    def list_users(self) -> list[str]:
        """Return list of all unique emails in the config table."""
        rows = self._db.execute(
            "SELECT DISTINCT email FROM user_configs ORDER BY email"
        ).fetchall()
        return [r["email"] for r in rows]

    def count_for_user(self, email: str) -> int:
        """Count config entries for a user."""
        row = self._db.execute(
            "SELECT COUNT(*) FROM user_configs WHERE email = ?",
            (email,),
        ).fetchone()
        return row[0] if row else 0

    def _row_to_entry(self, row: sqlite3.Row) -> ConfigEntry:
        return ConfigEntry(
            email        = row["email"],
            config_key   = row["config_key"],
            config_value = row["config_value"],
            is_encrypted = bool(row["is_encrypted"]),
            is_sensitive = bool(row["is_sensitive"]),
            value_type   = row["value_type"],
            description  = row["description"] or "",
            created_at   = row["created_at"],
            updated_at   = row["updated_at"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9. CONFIG HISTORY REPOSITORY — Change tracking
# ══════════════════════════════════════════════════════════════════════════════

class ConfigHistoryRepo:
    """
    Reads and writes to the config_history table.
    Every change to a config value is recorded here.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    def record(
        self,
        email      : str,
        config_key : str,
        old_value  : str | None,
        new_value  : str,
        changed_by : str = "user",
        change_note: str = "",
    ) -> None:
        """Record a config change in history."""
        with self._db.transaction() as conn:
            conn.execute("""
                INSERT INTO config_history
                    (email, config_key, old_value, new_value,
                     changed_at, changed_by, change_note)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                email, config_key, old_value, new_value,
                time.time(), changed_by, change_note,
            ))

        # Prune old history for this (email, key) pair
        self._prune(email, config_key)

    def fetch(
        self,
        email      : str,
        config_key : str | None = None,
        limit      : int = 20,
    ) -> list[ConfigHistoryEntry]:
        """Fetch history for a user, optionally filtered by key."""
        if config_key:
            rows = self._db.execute(
                """SELECT id, email, config_key, old_value, new_value,
                          changed_at, changed_by, change_note
                   FROM config_history
                   WHERE email = ? AND config_key = ?
                   ORDER BY changed_at DESC LIMIT ?""",
                (email, config_key, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                """SELECT id, email, config_key, old_value, new_value,
                          changed_at, changed_by, change_note
                   FROM config_history
                   WHERE email = ?
                   ORDER BY changed_at DESC LIMIT ?""",
                (email, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def _prune(self, email: str, config_key: str) -> None:
        """Keep only the latest N history entries per (email, key)."""
        try:
            with self._db.transaction() as conn:
                conn.execute("""
                    DELETE FROM config_history
                    WHERE email = ? AND config_key = ?
                      AND id NOT IN (
                          SELECT id FROM config_history
                          WHERE email = ? AND config_key = ?
                          ORDER BY changed_at DESC
                          LIMIT ?
                      )
                """, (email, config_key, email, config_key, MAX_HISTORY_PER_KEY))
        except Exception as e:
            logger.warning(f"History prune failed: {e}")

    def clear_history(self, email: str, config_key: str | None = None) -> int:
        """Delete history for a user (or a specific key). Returns count deleted."""
        if config_key:
            with self._db.transaction() as conn:
                c = conn.execute(
                    "DELETE FROM config_history WHERE email = ? AND config_key = ?",
                    (email, config_key),
                )
            return c.rowcount
        else:
            with self._db.transaction() as conn:
                c = conn.execute(
                    "DELETE FROM config_history WHERE email = ?",
                    (email,),
                )
            return c.rowcount

    def _row_to_entry(self, row: sqlite3.Row) -> ConfigHistoryEntry:
        return ConfigHistoryEntry(
            id          = row["id"],
            email       = row["email"],
            config_key  = row["config_key"],
            old_value   = row["old_value"],
            new_value   = row["new_value"],
            changed_at  = row["changed_at"],
            changed_by  = row["changed_by"],
            change_note = row["change_note"] or "",
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. CONFIG MANAGER — Master Class
# ══════════════════════════════════════════════════════════════════════════════

class ConfigManager:
    """
    Master configuration manager for dataDoctor.

    This is the ONLY class you import in app.py and cli.py.

    Features:
    - save_config()       → Save one setting (with auto-encryption)
    - load_config()       → Load all settings as dict (with auto-decryption)
    - get_config()        → Get a single value with optional default
    - delete_config()     → Remove one setting
    - bulk_save()         → Save an entire dict at once
    - reset_to_defaults() → Reset user settings to factory defaults
    - get_history()       → See all changes for a user/key
    - export_json()       → Export all settings as JSON
    - import_json()       → Restore settings from JSON
    - diff()              → Compare current settings to a snapshot
    - backup_db()         → Copy database file to backup path
    - session_state_dict()→ Ready-to-inject dict for Streamlit session_state

    Quick Usage:
        cm = ConfigManager()

        # Save API key (auto-encrypted)
        cm.save_config("alice@example.com", "api_key", "sk-abc123")

        # Load all settings (auto-decrypted)
        settings = cm.load_config("alice@example.com")
        print(settings["api_key"])   # "sk-abc123" (decrypted)

        # Inject into Streamlit session state
        st.session_state.update(cm.session_state_dict("alice@example.com"))
    """

    def __init__(
        self,
        db_path    : str  = DEFAULT_DB_PATH,
        vault      : Any  = None,    # SecurityVault instance (optional)
        cache_ttl  : int  = DEFAULT_CACHE_TTL,
        enable_history: bool = True,
    ) -> None:
        self._db          = DatabaseConnection(db_path)
        self._schema      = SchemaManager(self._db)
        self._repo        = ConfigRepository(self._db)
        self._history_repo= ConfigHistoryRepo(self._db)
        self._cache       = LRUCache(ttl=cache_ttl)
        self._validator   = ConfigValidator()
        self._vault       = vault
        self._enable_hist = enable_history

        # Run migrations on startup
        self._schema.initialize()
        logger.info(f"ConfigManager initialized. DB: {db_path}")

    # ── Core CRUD ─────────────────────────────────────────────────────────────

    def save_config(
        self,
        email      : str,
        config_key : str,
        value      : Any,
        sensitive  : bool | None = None,   # None = auto-detect
        description: str  = "",
        changed_by : str  = "user",
        change_note: str  = "",
    ) -> ConfigEntry:
        """
        Save a single config value.

        Args:
            email       : User email (unique identifier)
            config_key  : Setting name (e.g. "api_key", "llm_model")
            value       : The value to save (any type)
            sensitive   : Override sensitivity detection (None = auto)
            description : Optional description of this setting
            changed_by  : Who made the change ("user", "system", "import")
            change_note : Optional note about the change

        Returns:
            The saved ConfigEntry (with encrypted value if sensitive)

        Raises:
            ConfigValidationError : If key or value fails validation
        """
        # Validate inputs
        email_result = self._validator.validate_email(email)
        if not email_result.valid:
            raise ConfigValidationError(email_result.error)
        email = email_result.coerced

        key_result = self._validator.validate_key(config_key)
        if not key_result.valid:
            raise ConfigValidationError(key_result.error)
        config_key = key_result.coerced

        val_result = self._validator.validate_value(config_key, value)
        if not val_result.valid:
            raise ConfigValidationError(val_result.error)

        # Determine sensitivity
        is_sensitive = sensitive if sensitive is not None else self._validator.is_sensitive_key(config_key)

        # Detect value type
        value_type   = self._validator.detect_value_type(value)
        str_value    = self._validator._to_string(val_result.coerced or value)

        # Get old value for history
        old_entry    = self._repo.fetch_one(email, config_key)
        old_raw      = old_entry.config_value if old_entry else None

        # Encrypt if sensitive and vault is available
        is_encrypted = False
        if is_sensitive and self._vault:
            str_value    = self._vault.encrypt(str_value, field=config_key)
            is_encrypted = True

        # Build and save entry
        now   = time.time()
        entry = ConfigEntry(
            email        = email,
            config_key   = config_key,
            config_value = str_value,
            is_encrypted = is_encrypted,
            is_sensitive = is_sensitive,
            value_type   = value_type,
            description  = description,
            created_at   = old_entry.created_at if old_entry else now,
            updated_at   = now,
        )
        self._repo.upsert(entry)

        # Record history
        if self._enable_hist:
            self._history_repo.record(
                email      = email,
                config_key = config_key,
                old_value  = old_raw,
                new_value  = str_value,
                changed_by = changed_by,
                change_note= change_note,
            )

        # Invalidate cache
        self._cache.invalidate(email, config_key)

        logger.debug(f"Config saved: {email} / {config_key} (encrypted={is_encrypted})")
        return entry

    def get_config(
        self,
        email      : str,
        config_key : str,
        default    : Any = None,
        decrypt    : bool = True,
    ) -> Any:
        """
        Get a single config value with optional default.

        Args:
            email      : User email
            config_key : Setting name
            default    : Return this if the key doesn't exist
            decrypt    : Auto-decrypt encrypted values (default True)

        Returns:
            The config value (decrypted, typed) or default if not found
        """
        # Check cache first (stores decrypted values)
        cached = self._cache.get(email, config_key)
        if cached is not None:
            return cached

        entry = self._repo.fetch_one(email.strip().lower(), config_key.lower())
        if entry is None:
            return DEFAULT_CONFIG.get(config_key, default)

        value = self._decrypt_entry(entry) if decrypt else entry.config_value
        typed = self._type_cast(value, entry.value_type)

        # Cache the result
        self._cache.set(email, typed, config_key)
        return typed

    def load_config(self, email: str, decrypt: bool = True) -> dict[str, Any]:
        """
        Load ALL config settings for a user as a plain dict.
        Missing keys get their DEFAULT_CONFIG values.

        This is the main method used to inject into Streamlit session_state.

        Args:
            email  : User email
            decrypt: Auto-decrypt sensitive values (default True)

        Returns:
            dict of {config_key: typed_value}
        """
        # Check whole-user cache
        cached = self._cache.get(email)
        if cached is not None:
            return deepcopy(cached)

        email = email.strip().lower()
        entries = self._repo.fetch_all(email)

        # Start with defaults
        result = dict(DEFAULT_CONFIG)

        # Overlay with user's actual settings
        for entry in entries:
            value = self._decrypt_entry(entry) if decrypt else entry.config_value
            result[entry.config_key] = self._type_cast(value, entry.value_type)

        # Cache the result
        self._cache.set(email, result)
        return result

    def delete_config(
        self,
        email      : str,
        config_key : str,
        changed_by : str = "user",
    ) -> bool:
        """
        Delete a single config entry.

        Returns:
            True if the entry existed and was deleted, False if it didn't exist.
        """
        email = email.strip().lower()
        old   = self._repo.fetch_one(email, config_key)

        deleted = self._repo.delete_one(email, config_key)

        if deleted and self._enable_hist and old:
            self._history_repo.record(
                email      = email,
                config_key = config_key,
                old_value  = old.config_value,
                new_value  = "",
                changed_by = changed_by,
                change_note= "deleted",
            )

        self._cache.invalidate(email, config_key)
        logger.debug(f"Config deleted: {email} / {config_key}")
        return deleted

    def delete_all_configs(self, email: str) -> int:
        """Delete ALL config entries for a user. Returns count deleted."""
        email = email.strip().lower()
        count = self._repo.delete_all(email)
        self._history_repo.clear_history(email)
        self._cache.invalidate(email)
        logger.info(f"All configs deleted for {email}: {count} entries")
        return count

    # ── Bulk Operations ───────────────────────────────────────────────────────

    def bulk_save(
        self,
        email      : str,
        configs    : dict[str, Any],
        changed_by : str = "user",
        change_note: str = "bulk_save",
    ) -> dict[str, bool]:
        """
        Save an entire dict of config settings at once.

        Args:
            email   : User email
            configs : dict of {config_key: value}

        Returns:
            dict of {config_key: success_bool}
        """
        results = {}
        for key, value in configs.items():
            try:
                self.save_config(
                    email, key, value,
                    changed_by=changed_by,
                    change_note=change_note,
                )
                results[key] = True
            except (ConfigValidationError, ConfigDatabaseError) as e:
                logger.warning(f"bulk_save failed for {key}: {e}")
                results[key] = False

        # Invalidate the whole-user cache after bulk save
        self._cache.invalidate(email)
        return results

    def reset_to_defaults(
        self,
        email      : str,
        keys       : list[str] | None = None,
        changed_by : str = "system",
    ) -> int:
        """
        Reset user settings to DEFAULT_CONFIG values.

        Args:
            email : User email
            keys  : Specific keys to reset (None = reset all defaults)

        Returns:
            Number of settings reset
        """
        email      = email.strip().lower()
        target_keys= keys or list(DEFAULT_CONFIG.keys())
        count      = 0

        for key in target_keys:
            if key in DEFAULT_CONFIG:
                try:
                    self.save_config(
                        email, key, DEFAULT_CONFIG[key],
                        changed_by=changed_by,
                        change_note="reset_to_default",
                    )
                    count += 1
                except ConfigValidationError as e:
                    logger.warning(f"reset_to_defaults skipped {key}: {e}")

        self._cache.invalidate(email)
        logger.info(f"Reset {count} settings to defaults for {email}")
        return count

    # ── Config Diff ───────────────────────────────────────────────────────────

    def diff(self, email: str, snapshot: dict[str, Any]) -> ConfigDiff:
        """
        Compare current user config to a previous snapshot.

        Args:
            email    : User email
            snapshot : Previous config dict (e.g. from a previous load_config call)

        Returns:
            ConfigDiff showing what was added, removed, or changed
        """
        current = self.load_config(email)
        diff    = ConfigDiff()

        all_keys = set(current.keys()) | set(snapshot.keys())
        for key in all_keys:
            in_current  = key in current
            in_snapshot = key in snapshot
            if in_current and not in_snapshot:
                diff.added[key] = current[key]
            elif in_snapshot and not in_current:
                diff.removed[key] = snapshot[key]
            elif current.get(key) != snapshot.get(key):
                diff.changed[key] = {
                    "old": snapshot[key],
                    "new": current[key],
                }
        return diff

    # ── History ───────────────────────────────────────────────────────────────

    def get_history(
        self,
        email      : str,
        config_key : str | None = None,
        limit      : int = 20,
    ) -> list[dict]:
        """
        Return the change history for a user (or specific key).

        Args:
            email      : User email
            config_key : Filter to specific key (None = all keys)
            limit      : Max entries to return

        Returns:
            List of change dicts (newest first)
        """
        entries = self._history_repo.fetch(
            email.strip().lower(), config_key, limit
        )
        return [e.to_dict() for e in entries]

    # ── Import / Export ───────────────────────────────────────────────────────

    def export_json(
        self,
        email          : str,
        include_sensitive: bool = False,
        decrypt        : bool = True,
    ) -> str:
        """
        Export all config settings as a JSON string.

        Args:
            email            : User email
            include_sensitive: Include sensitive/encrypted values (default False)
            decrypt          : Decrypt encrypted values before export

        Returns:
            JSON string with all settings
        """
        email   = email.strip().lower()
        entries = self._repo.fetch_all(email)

        export = {
            "email"      : email,
            "exported_at": datetime.now(tz=timezone.utc).isoformat(),
            "version"    : CURRENT_SCHEMA_VERSION,
            "configs"    : {},
        }

        for entry in entries:
            if entry.is_sensitive and not include_sensitive:
                export["configs"][entry.config_key] = {
                    "value"      : "[REDACTED]",
                    "is_sensitive": True,
                    "value_type" : entry.value_type,
                }
                continue

            value = entry.config_value
            if decrypt and entry.is_encrypted and self._vault:
                value = self._vault.decrypt_safe(value, default="[DECRYPTION_FAILED]")

            export["configs"][entry.config_key] = {
                "value"      : value,
                "is_sensitive": entry.is_sensitive,
                "value_type" : entry.value_type,
                "description": entry.description,
                "updated_at" : entry.updated_iso,
            }

        return json.dumps(export, indent=2, default=str)

    def import_json(
        self,
        email    : str,
        json_str : str,
        overwrite: bool = True,
        changed_by: str = "import",
    ) -> dict:
        """
        Restore config settings from a JSON export string.

        Args:
            email     : User email to import into
            json_str  : JSON string from export_json()
            overwrite : If True, overwrite existing values (default True)

        Returns:
            Import report dict
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ConfigImportError(f"Invalid JSON: {e}") from e

        if "configs" not in data:
            raise ConfigImportError("Invalid export format: missing 'configs' key.")

        imported, skipped, failed = 0, 0, 0

        for key, info in data["configs"].items():
            value = info.get("value")
            if value == "[REDACTED]" or value == "[DECRYPTION_FAILED]":
                skipped += 1
                continue

            if not overwrite and self._repo.exists(email, key):
                skipped += 1
                continue

            try:
                self.save_config(
                    email, key, value,
                    description= info.get("description", ""),
                    changed_by = changed_by,
                    change_note= "imported",
                )
                imported += 1
            except (ConfigValidationError, ConfigDatabaseError) as e:
                logger.warning(f"Import failed for key '{key}': {e}")
                failed += 1

        self._cache.invalidate(email)
        report = {
            "imported": imported,
            "skipped" : skipped,
            "failed"  : failed,
            "total"   : imported + skipped + failed,
        }
        logger.info(f"Import for {email}: {report}")
        return report

    # ── Session State Integration (Streamlit) ────────────────────────────────

    def session_state_dict(self, email: str) -> dict[str, Any]:
        """
        Return a dict ready to inject into st.session_state.

        Usage in app.py:
            st.session_state.update(cm.session_state_dict(user_email))

        Returns:
            Plain dict of all user settings (decrypted, typed)
        """
        return self.load_config(email)

    # ── Database Utilities ────────────────────────────────────────────────────

    def backup_db(self, backup_path: str | None = None) -> str:
        """
        Create a backup copy of the database file.

        Args:
            backup_path : Path for backup file. If None, auto-generates name.

        Returns:
            Path to the backup file created.
        """
        if not os.path.exists(self._db.db_path):
            raise ConfigDatabaseError(f"Database file not found: {self._db.db_path}")

        if backup_path is None:
            ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
            base        = os.path.splitext(self._db.db_path)[0]
            backup_path = f"{base}_backup_{ts}.db"

        shutil.copy2(self._db.db_path, backup_path)
        size_mb = os.path.getsize(backup_path) / 1_048_576
        logger.info(f"Database backed up to: {backup_path} ({size_mb:.2f} MB)")
        return backup_path

    def vacuum_db(self) -> None:
        """
        Run VACUUM to reclaim unused space and defragment the database.
        Useful after many deletes.
        """
        self._db.execute("VACUUM")
        logger.info("Database VACUUM completed.")

    # ── Admin Utilities ───────────────────────────────────────────────────────

    def list_users(self) -> list[str]:
        """Return list of all users who have stored configs."""
        return self._repo.list_users()

    def user_stats(self, email: str) -> dict:
        """Return statistics for a specific user's config."""
        email   = email.strip().lower()
        entries = self._repo.fetch_all(email)
        return {
            "email"          : email,
            "total_settings" : len(entries),
            "encrypted_count": sum(1 for e in entries if e.is_encrypted),
            "sensitive_count": sum(1 for e in entries if e.is_sensitive),
            "keys"           : sorted(e.config_key for e in entries),
        }

    def db_stats(self) -> dict:
        """Return overall database statistics."""
        users      = self._repo.list_users()
        total_cfgs = 0
        for u in users:
            total_cfgs += self._repo.count_for_user(u)

        return {
            "db_path"       : self._db.db_path,
            "db_size_mb"    : round(self._db.db_size_mb(), 3),
            "total_users"   : len(users),
            "total_configs" : total_cfgs,
            "schema_version": CURRENT_SCHEMA_VERSION,
            "cache_stats"   : self._cache.stats(),
        }

    def migrate_user(self, old_email: str, new_email: str) -> int:
        """
        Move all configs from old_email to new_email.
        Used when a user changes their email address.

        Returns:
            Number of entries migrated.
        """
        old_email = old_email.strip().lower()
        new_email = new_email.strip().lower()

        # Validate new email
        result = self._validator.validate_email(new_email)
        if not result.valid:
            raise ConfigValidationError(result.error)

        entries = self._repo.fetch_all(old_email)
        count   = 0

        for entry in entries:
            entry.email = new_email
            self._repo.upsert(entry)
            count += 1

        # Remove old entries
        self._repo.delete_all(old_email)

        # Invalidate both caches
        self._cache.invalidate(old_email)
        self._cache.invalidate(new_email)

        logger.info(f"Migrated {count} configs from {old_email} → {new_email}")
        return count

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _decrypt_entry(self, entry: ConfigEntry) -> str:
        """Decrypt an entry's value if it's encrypted and vault is available."""
        if entry.is_encrypted and self._vault:
            return self._vault.decrypt_safe(
                entry.config_value,
                default=entry.config_value,
            )
        return entry.config_value

    def _type_cast(self, value: str, value_type: str) -> Any:
        """Cast a string value to its Python type."""
        try:
            if value_type == "int":
                return int(value)
            if value_type == "float":
                return float(value)
            if value_type == "bool":
                return value.lower() in ("true", "1", "yes")
            if value_type == "json":
                return json.loads(value)
            return value
        except (ValueError, json.JSONDecodeError):
            return value

    def close(self) -> None:
        """Close database connections gracefully."""
        self._db.close()
        self._cache.clear()
        logger.info("ConfigManager closed.")


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONVENIENCE FUNCTIONS — Quick access for app.py
# ══════════════════════════════════════════════════════════════════════════════

# Global singleton
_manager: ConfigManager | None = None
_manager_lock = threading.Lock()


def get_manager(
    db_path  : str = DEFAULT_DB_PATH,
    vault    : Any = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
) -> ConfigManager:
    """
    Get or create the global ConfigManager singleton.
    Thread-safe. Call this anywhere in the app.

    Example:
        cm = get_manager()
        cm.save_config("alice@example.com", "api_key", "sk-...")
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ConfigManager(
                db_path   = db_path,
                vault     = vault,
                cache_ttl = cache_ttl,
            )
    return _manager


def save(email: str, key: str, value: Any, **kwargs) -> bool:
    """Quick save using the global manager."""
    try:
        get_manager().save_config(email, key, value, **kwargs)
        return True
    except ConfigError as e:
        logger.error(f"Quick save failed: {e}")
        return False


def load(email: str) -> dict[str, Any]:
    """Quick load using the global manager."""
    return get_manager().load_config(email)


def get(email: str, key: str, default: Any = None) -> Any:
    """Quick get a single value using the global manager."""
    return get_manager().get_config(email, key, default=default)


def delete(email: str, key: str) -> bool:
    """Quick delete using the global manager."""
    return get_manager().delete_config(email, key)