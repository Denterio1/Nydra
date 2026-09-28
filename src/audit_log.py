"""
src/audit_log.py
================

Immutable, tamper-evident security audit log for Nydra (formerly dataDoctor).

Design
------
Every row in ``audit_log`` carries an ``entry_hash`` which is the SHA-256 of::

    prev_hash | timestamp | event_type | user_id | detail

where ``prev_hash`` is the ``entry_hash`` of the previous row (the chain is
ordered by ``timestamp`` ascending).  The first row in an un-archived table
uses the fixed :data:`GENESIS_HASH` seed.

Because every row commits the hash of its predecessor, any of the following
becomes detectable by :class:`IntegrityChecker`:

* editing a historical row's fields  -> its own recomputed hash no longer matches
* editing a row *and* fixing its hash -> the next row's ``prev_hash`` no longer matches
* deleting a row                      -> the next row's ``prev_hash`` no longer matches

Canonicalisation
----------------
Field values are joined with the ASCII unit-separator (``\\x1f``) so that
``("ab", "c")`` and ``("a", "bc")`` cannot collide.  Timestamps are formatted
with ``%.6f`` and SQLite stores REAL as IEEE-754 double, so the float round-trip
is bit-exact and re-hashing is deterministic.

Retention
---------
Rows may only ever be deleted through :meth:`AuditRetention.archive_older_than`.
That path writes a JSON export first and records a :class:`DBAuditChainAnchor`
so that :class:`IntegrityChecker` can still verify the *live* portion of the
chain after the head has been archived away.

Failure policy
--------------
:class:`AuditLogger` **never raises**.  A failure to write an audit row must
never break the request that triggered it, so every failure is swallowed and
reported through the stdlib ``logging`` logger ``nydra.audit``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import (
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "GENESIS_HASH",
    "AuditSeverity",
    "AuditEventType",
    "DBAuditLog",
    "DBAuditChainAnchor",
    "AUDIT_METADATA",
    "IntegrityResult",
    "AuditLogger",
    "AuditQuery",
    "IntegrityChecker",
    "AuditRetention",
    "AuditReporter",
    "compute_entry_hash",
    "coerce_event_type",
    "coerce_severity",
    "create_audit_tables",
    "log_event",
    "get_recent_events",
    "verify_integrity",
    "audit_summary",
]

logger = logging.getLogger("nydra.audit")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed seed hash used as the ``prev_hash`` of the very first row.
#: Derived from a versioned constant so it is obviously not a real row hash.
GENESIS_HASH: str = hashlib.sha256(b"nydra::audit::genesis::v1").hexdigest()

#: Separator used when building the canonical hash payload.
_SEP = "\x1f"

#: Default directory for retention exports, relative to CWD.
DEFAULT_ARCHIVE_DIR = "logs"

#: Safety cap so a corrupt DB can't make verify_chain allocate unboundedly.
_MAX_BROKEN_REPORTED = 25


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AuditSeverity(str, Enum):
    """Severity bucket for an audit event."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class AuditEventType(str, Enum):
    """Enumerated security-relevant event types.

    Values are the wire/DB representation (lower_snake_case) so that they are
    stable across refactors and pleasant to grep for in exports.
    """

    # --- authentication ---------------------------------------------------
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_REFRESHED = "token_refreshed"
    PASSWORD_CHANGED = "password_changed"
    OAUTH_LOGIN = "oauth_login"

    # --- registration -----------------------------------------------------
    REGISTER_SUCCESS = "register_success"
    REGISTER_FAILED = "register_failed"

    # --- account / settings ----------------------------------------------
    SETTINGS_CHANGED = "settings_changed"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"

    # --- jobs / files -----------------------------------------------------
    JOB_CREATED = "job_created"
    JOB_FAILED = "job_failed"
    FILE_UPLOADED = "file_uploaded"

    # --- defensive --------------------------------------------------------
    RATE_LIMIT_HIT = "rate_limit_hit"
    SUSPICIOUS_INPUT_BLOCKED = "suspicious_input_blocked"
    UNAUTHORIZED_ACCESS_ATTEMPT = "unauthorized_access_attempt"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------
#
# This module deliberately owns its own ``DeclarativeBase`` so that it can be
# imported without creating a circular dependency on api.py.  It does NOT own
# an engine -- api.py's existing engine is reused, and api.py's lifespan()
# startup calls :func:`create_audit_tables` alongside its existing
# ``Base.metadata.create_all``.
#


class AuditBase(DeclarativeBase):
    """Declarative base for the audit-log tables."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DBAuditLog(AuditBase):
    """One immutable audit event.

    Rows are append-only.  The only code path permitted to delete from this
    table is :meth:`AuditRetention.archive_older_than`.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    timestamp: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AuditSeverity.INFO.value, index=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True
    )
    ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    request_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    __table_args__ = (
        Index("ix_audit_log_ts_event", "timestamp", "event_type"),
        Index("ix_audit_log_ts_severity", "timestamp", "severity"),
        Index("ix_audit_log_event_success", "event_type", "success"),
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (used by retention exports and reporters)."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "timestamp_iso": _iso(self.timestamp),
            "event_type": self.event_type,
            "severity": self.severity,
            "user_id": self.user_id,
            "ip": self.ip,
            "session_id": self.session_id,
            "request_path": self.request_path,
            "success": bool(self.success),
            "detail": self.detail,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<DBAuditLog {self.event_type} sev={self.severity} "
            f"user={self.user_id!r} ts={self.timestamp:.6f} ok={self.success}>"
        )


class DBAuditChainAnchor(AuditBase):
    """Records the tail of an archived chain segment.

    After :meth:`AuditRetention.archive_older_than` removes rows from the head
    of the live table, the first remaining row's ``prev_hash`` points at a row
    that no longer exists.  This anchor table lets :class:`IntegrityChecker`
    confirm that dangling ``prev_hash`` really is the hash of the last
    archived row rather than an attacker's fabrication.
    """

    __tablename__ = "audit_chain_anchor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    cutoff_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    last_archived_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    archive_file: Mapped[str] = mapped_column(String(512), nullable=False)
    rows_archived: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


#: Exposed so api.py can do ``await conn.run_sync(AUDIT_METADATA.create_all)``.
AUDIT_METADATA = AuditBase.metadata


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _iso(ts: Optional[float]) -> Optional[str]:
    """Best-effort ISO-8601 (UTC) rendering of a unix timestamp."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):  # pragma: no cover - defensive
        return None


def _stringify_detail(detail: Any) -> Optional[str]:
    """Canonicalise ``detail`` into the exact string that gets hashed.

    Determinism matters here: whatever we hash at write time must be exactly
    what :class:`IntegrityChecker` reads back out of the DB at verify time.
    """
    if detail is None:
        return None
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        return str(detail)


def coerce_event_type(value: Any) -> str:
    """Accept an :class:`AuditEventType`, its name, or its value; return the value."""
    if isinstance(value, AuditEventType):
        return value.value
    if isinstance(value, Enum):
        return str(value.value)
    text = str(value)
    try:
        return AuditEventType(text).value
    except ValueError:
        pass
    try:
        return AuditEventType[text.upper()].value
    except KeyError:
        return text


def coerce_severity(value: Any) -> str:
    """Accept an :class:`AuditSeverity`, its name, or its value; default to info."""
    if isinstance(value, AuditSeverity):
        return value.value
    if isinstance(value, Enum):
        value = value.value
    text = str(value or "").strip().lower()
    try:
        return AuditSeverity(text).value
    except ValueError:
        return AuditSeverity.INFO.value
    
_FAILURE_EVENT_TYPES = {
    AuditEventType.LOGIN_FAILED.value,
    AuditEventType.REGISTER_FAILED.value,
    AuditEventType.JOB_FAILED.value,
}
_DEFENSIVE_EVENT_TYPES = {
    AuditEventType.RATE_LIMIT_HIT.value,
    AuditEventType.SUSPICIOUS_INPUT_BLOCKED.value,
    AuditEventType.UNAUTHORIZED_ACCESS_ATTEMPT.value,
}

def _infer_success(event_type: str) -> bool:
    """Default success=False for known-failure event types, True otherwise."""
    return event_type not in _FAILURE_EVENT_TYPES    


def _coerce_since(since: Any) -> Optional[float]:
    """Normalise a ``since`` filter to a unix timestamp."""
    if since is None:
        return None
    if isinstance(since, (int, float)):
        return float(since)
    if isinstance(since, datetime):
        dt = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(since, timedelta):
        return time.time() - since.total_seconds()
    if isinstance(since, str):
        text = since.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def compute_entry_hash(
    prev_hash: Optional[str],
    timestamp: float,
    event_type: Any,
    user_id: Optional[str],
    detail: Optional[str],
) -> str:
    """Recompute the chain hash for one row.

    This is the single source of truth for the chain and is used both when
    appending and when verifying, so the two can never drift apart.
    """
    prev = prev_hash or GENESIS_HASH
    evt = coerce_event_type(event_type)
    try:
        ts_repr = f"{float(timestamp):.6f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        ts_repr = str(timestamp)
    payload = _SEP.join([prev, ts_repr, evt, user_id or "", detail or ""])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Chain-serialisation lock
# ---------------------------------------------------------------------------
#
# Appending to a hash chain is a read-modify-write on the tail.  Two concurrent
# requests could otherwise read the same prev_hash and fork the chain.  The
# lock serialises appends within a process; SQLite's own write lock covers
# cross-process races for the commit itself.
#
# The lock is (re)created whenever the running event loop changes, so it is
# safe under pytest-asyncio's per-test loops as well as under uvicorn.
#

_CHAIN_LOCK: Optional[asyncio.Lock] = None
_CHAIN_LOCK_LOOP: Optional[asyncio.AbstractEventLoop] = None


def _get_chain_lock() -> asyncio.Lock:
    global _CHAIN_LOCK, _CHAIN_LOCK_LOOP
    try:
        loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - no loop, still return something
        loop = None
    if _CHAIN_LOCK is None or _CHAIN_LOCK_LOOP is not loop:
        _CHAIN_LOCK = asyncio.Lock()
        _CHAIN_LOCK_LOOP = loop
    return _CHAIN_LOCK


async def _fetch_tail(db: AsyncSession) -> tuple[Optional[float], str]:
    """Return ``(last_timestamp, last_entry_hash)`` or ``(None, GENESIS_HASH)``."""
    stmt = (
        select(DBAuditLog.timestamp, DBAuditLog.entry_hash)
        .order_by(DBAuditLog.timestamp.desc())
        .limit(1)
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return None, GENESIS_HASH
    return float(row[0]), str(row[1])


# ---------------------------------------------------------------------------
# AuditLogger (write side)
# ---------------------------------------------------------------------------


class AuditLogger:
    """Append-only writer for the audit chain.

    All public methods are classmethods and **never raise**.
    """

    @classmethod
    async def log(
        cls,
        db: AsyncSession,
        event_type: Any,
        user_id: Optional[str] = None,
        ip: Optional[str] = None,
        detail: Any = None,
        severity: Any = AuditSeverity.INFO,
        success: Optional[bool] = None,
        session_id: Optional[str] = None,
        request_path: Optional[str] = None,
        *,
        commit: bool = True,
        timestamp: Optional[float] = None,
    ) -> Optional[DBAuditLog]:
        """Append one audit event to the chain.

        Parameters
        ----------
        db:
            The request-scoped :class:`AsyncSession`.  Reused, never replaced.
        event_type:
            An :class:`AuditEventType` (or its name/value as a string).
        detail:
            Free-form context.  Non-string values are JSON-encoded with sorted
            keys so the hashed representation is deterministic.
        commit:
            When ``True`` (default) the row is committed immediately.  Pass
            ``False`` if you are inside a larger unit of work and will commit
            yourself.

        Returns
        -------
        The persisted :class:`DBAuditLog`, or ``None`` if the write failed.
        A ``None`` return is the *only* signal of failure; nothing is raised.
        """
        try:
            return await cls._log_inner(
                db=db,
                event_type=event_type,
                user_id=user_id,
                ip=ip,
                detail=detail,
                severity=severity,
                success=success,
                session_id=session_id,
                request_path=request_path,
                commit=commit,
                timestamp=timestamp,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate catch-all
            # The whole point of this module is that logging must never break
            # the caller.  Swallow everything, record it, move on.
            logger.error(
                "audit_log write failed (event_type=%r user_id=%r ip=%r): %s",
                event_type,
                user_id,
                ip,
                exc,
                exc_info=True,
            )
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001 - session may be unusable
                pass
            return None

    # -- internals ---------------------------------------------------------

    @classmethod
    async def _log_inner(
        cls,
        db: AsyncSession,
        event_type: Any,
        user_id: Optional[str],
        ip: Optional[str],
        detail: Any,
        severity: Any,
        success: Optional[bool],
        session_id: Optional[str],
        request_path: Optional[str],
        commit: bool,
        timestamp: Optional[float],
    ) -> DBAuditLog:
        evt = coerce_event_type(event_type)
        if success is None:
            success = _infer_success(evt)   
        sev = coerce_severity(severity)
        detail_str = _stringify_detail(detail)
        uid = str(user_id) if user_id is not None else None

        lock = _get_chain_lock()
        async with lock:
            last_ts, prev_hash = await _fetch_tail(db)

            ts = float(timestamp) if timestamp is not None else time.time()
            # Guarantee strictly increasing timestamps so chain order is
            # unambiguous even under a burst of same-microsecond events.
            if last_ts is not None and ts <= last_ts:
                ts = last_ts + 1e-6

            entry_hash = compute_entry_hash(prev_hash, ts, evt, uid, detail_str)

            row = DBAuditLog(
                id=str(uuid.uuid4()),
                timestamp=ts,
                event_type=evt,
                severity=sev,
                user_id=uid,
                ip=str(ip) if ip is not None else None,
                session_id=str(session_id) if session_id is not None else None,
                request_path=str(request_path) if request_path is not None else None,
                success=bool(success),
                detail=detail_str,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )

            # A SAVEPOINT means a failed audit insert cannot poison any
            # transaction the caller already has open.
            async with db.begin_nested():
                db.add(row)
                await db.flush()

            if commit:
                await db.commit()

        return row

    @classmethod
    async def log_many(
        cls,
        db: AsyncSession,
        events: Sequence[dict],
        *,
        commit: bool = True,
    ) -> list[DBAuditLog]:
        """Append several events, still chained correctly.

        Convenience for batch paths (e.g. a bulk job import).  Each event is a
        dict of kwargs accepted by :meth:`log`.
        """
        written: list[DBAuditLog] = []
        for spec in events:
            row = await cls.log(db, commit=False, **spec)
            if row is not None:
                written.append(row)
        if commit and written:
            try:
                await db.commit()
            except Exception as exc:  # noqa: BLE001
                logger.error("audit_log batch commit failed: %s", exc, exc_info=True)
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    pass
        return written


# ---------------------------------------------------------------------------
# AuditQuery (read side)
# ---------------------------------------------------------------------------


class AuditQuery:
    """Read-only helpers over the audit chain.

    Every method is async, ordered by ``timestamp`` descending, and paginated
    via ``limit`` / ``offset``.
    """

    @classmethod
    async def by_user(
        cls,
        db: AsyncSession,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        stmt = (
            select(DBAuditLog)
            .where(DBAuditLog.user_id == str(user_id))
            .order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def by_event_type(
        cls,
        db: AsyncSession,
        event_type: Any,
        since: Any = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        stmt = select(DBAuditLog).where(
            DBAuditLog.event_type == coerce_event_type(event_type)
        )
        since_ts = _coerce_since(since)
        if since_ts is not None:
            stmt = stmt.where(DBAuditLog.timestamp >= since_ts)
        stmt = (
            stmt.order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def by_severity(
        cls,
        db: AsyncSession,
        severity: Any,
        since: Any = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        stmt = select(DBAuditLog).where(
            DBAuditLog.severity == coerce_severity(severity)
        )
        since_ts = _coerce_since(since)
        if since_ts is not None:
            stmt = stmt.where(DBAuditLog.timestamp >= since_ts)
        stmt = (
            stmt.order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def failed_logins(
        cls,
        db: AsyncSession,
        user_id_or_ip: str,
        window_hours: int = 24,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        """Failed logins for a user id *or* an IP within the trailing window."""
        since_ts = time.time() - (float(window_hours) * 3600.0)
        needle = str(user_id_or_ip)
        stmt = (
            select(DBAuditLog)
            .where(
                DBAuditLog.event_type == AuditEventType.LOGIN_FAILED.value,
                DBAuditLog.timestamp >= since_ts,
            )
            .where(
                (DBAuditLog.user_id == needle) | (DBAuditLog.ip == needle)
            )
            .order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def recent(
        cls,
        db: AsyncSession,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        stmt = (
            select(DBAuditLog)
            .order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def in_range(
        cls,
        db: AsyncSession,
        start: Any,
        end: Any = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[DBAuditLog]:
        start_ts = _coerce_since(start)
        end_ts = _coerce_since(end) if end is not None else time.time()
        stmt = select(DBAuditLog)
        if start_ts is not None:
            stmt = stmt.where(DBAuditLog.timestamp >= start_ts)
        if end_ts is not None:
            stmt = stmt.where(DBAuditLog.timestamp <= end_ts)
        stmt = (
            stmt.order_by(DBAuditLog.timestamp.desc())
            .limit(_clamp_limit(limit))
            .offset(max(0, int(offset)))
        )
        return list((await db.execute(stmt)).scalars().all())

    @classmethod
    async def count(
        cls,
        db: AsyncSession,
        event_type: Any = None,
        since: Any = None,
    ) -> int:
        stmt = select(func.count()).select_from(DBAuditLog)
        if event_type is not None:
            stmt = stmt.where(DBAuditLog.event_type == coerce_event_type(event_type))
        since_ts = _coerce_since(since)
        if since_ts is not None:
            stmt = stmt.where(DBAuditLog.timestamp >= since_ts)
        return int((await db.execute(stmt)).scalar_one() or 0)


def _clamp_limit(limit: int) -> int:
    """Keep pagination sane; protects against ``limit=-1`` style accidents."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 100
    if value <= 0:
        return 100
    return min(value, 5000)


# ---------------------------------------------------------------------------
# IntegrityChecker
# ---------------------------------------------------------------------------


@dataclass
class IntegrityResult:
    """Outcome of :meth:`IntegrityChecker.verify_chain`."""

    intact: bool
    checked: int = 0
    first_broken_id: Optional[str] = None
    first_broken_index: Optional[int] = None
    reason: Optional[str] = None
    anchor: Optional[str] = None
    from_genesis: bool = True
    broken_ids: list[str] = field(default_factory=list)
    #: ``None`` when no anchor row exists; otherwise whether the live chain's
    #: dangling ``prev_hash`` matches the recorded archive anchor.
    anchor_matches_archive: Optional[bool] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "intact": self.intact,
            "checked": self.checked,
            "first_broken_id": self.first_broken_id,
            "first_broken_index": self.first_broken_index,
            "reason": self.reason,
            "anchor": self.anchor,
            "from_genesis": self.from_genesis,
            "broken_ids": list(self.broken_ids),
            "anchor_matches_archive": self.anchor_matches_archive,
            "error": self.error,
        }

    def __bool__(self) -> bool:  # pragma: no cover - ergonomic
        return self.intact


class IntegrityChecker:
    """Walks the chain and detects any edit, fork, or deletion."""

    @classmethod
    async def verify_chain(
        cls,
        db: AsyncSession,
        *,
        stop_on_first: bool = True,
        check_anchor: bool = True,
    ) -> IntegrityResult:
        """Recompute every row's hash and compare against what is stored.

        Returns an :class:`IntegrityResult`.  Never raises for a *broken*
        chain (that is a normal return value); genuine DB errors are captured
        into ``IntegrityResult.error``.
        """
        try:
            stmt = select(DBAuditLog).order_by(
                DBAuditLog.timestamp.asc(), DBAuditLog.id.asc()
            )
            rows: list[DBAuditLog] = list((await db.execute(stmt)).scalars().all())

            if not rows:
                return IntegrityResult(
                    intact=True,
                    checked=0,
                    from_genesis=True,
                    anchor=GENESIS_HASH,
                )

            anchor = rows[0].prev_hash
            from_genesis = anchor == GENESIS_HASH
            broken_ids: list[str] = []
            first_broken_id: Optional[str] = None
            first_broken_index: Optional[int] = None
            reason: Optional[str] = None

            prev_hash = anchor
            for index, row in enumerate(rows):
                recomputed = compute_entry_hash(
                    prev_hash,
                    row.timestamp,
                    row.event_type,
                    row.user_id,
                    row.detail,
                )
                if recomputed != row.entry_hash:
                    broken_ids.append(row.id)
                    if first_broken_id is None:
                        first_broken_id = row.id
                        first_broken_index = index
                        reason = (
                            "entry_hash mismatch at index "
                            f"{index} (id={row.id}): stored={row.entry_hash[:12]}… "
                            f"recomputed={recomputed[:12]}…"
                        )
                    if stop_on_first:
                        break
                # Spec: recompute using the previous row's *stored* hash.
                prev_hash = row.entry_hash

            anchor_matches: Optional[bool] = None
            if check_anchor and not from_genesis:
                anchor_matches = await cls._check_anchor(db, anchor)

            intact = not broken_ids
            if intact and anchor_matches is False:
                intact = False
                reason = (
                    "live chain anchor does not match the recorded archive "
                    "anchor; the head of the chain may have been rewritten"
                )

            return IntegrityResult(
                intact=intact,
                checked=len(rows) if stop_on_first and broken_ids else len(rows),
                first_broken_id=first_broken_id,
                first_broken_index=first_broken_index,
                reason=reason,
                anchor=anchor,
                from_genesis=from_genesis,
                broken_ids=broken_ids[:_MAX_BROKEN_REPORTED],
                anchor_matches_archive=anchor_matches,
            )
        except Exception as exc:  # noqa: BLE001 - verification must not explode
            logger.error("audit integrity verification failed: %s", exc, exc_info=True)
            return IntegrityResult(
                intact=False,
                reason="verification error",
                error=str(exc),
            )

    @classmethod
    async def _check_anchor(cls, db: AsyncSession, live_anchor: str) -> Optional[bool]:
        """Compare the live chain's dangling prev_hash with the newest anchor row."""
        try:
            stmt = (
                select(DBAuditChainAnchor)
                .order_by(DBAuditChainAnchor.cutoff_timestamp.desc())
                .limit(1)
            )
            anchor_row = (await db.execute(stmt)).scalars().first()
        except Exception:  # noqa: BLE001 - anchor table may not exist yet
            return None
        if anchor_row is None:
            return None
        return anchor_row.last_archived_hash == live_anchor

    @classmethod
    async def tail_hash(cls, db: AsyncSession) -> str:
        """Current head of the chain (or the genesis seed if empty)."""
        _, last = await _fetch_tail(db)
        return last


# ---------------------------------------------------------------------------
# AuditRetention
# ---------------------------------------------------------------------------


class AuditRetention:
    """Explicit maintenance path -- the only place rows are ever deleted."""

    @classmethod
    async def archive_older_than(
        cls,
        db: AsyncSession,
        days: int = 90,
        *,
        archive_dir: str | os.PathLike = DEFAULT_ARCHIVE_DIR,
        verify_first: bool = True,
        delete_after_archive: bool = True,
        commit: bool = True,
    ) -> dict:
        """Export rows older than ``days`` to JSON, then delete them.

        Returns a summary dict.  On failure returns ``{"ok": False, ...}``
        rather than raising, so a scheduled maintenance job can log and retry.
        """
        try:
            cutoff = time.time() - (float(days) * 86400.0)

            if verify_first:
                integrity = await IntegrityChecker.verify_chain(db)
                if not integrity.intact:
                    logger.error(
                        "refusing to archive: audit chain is broken (%s)",
                        integrity.reason,
                    )
                    return {
                        "ok": False,
                        "archived": 0,
                        "deleted": 0,
                        "error": "chain integrity check failed",
                        "integrity": integrity.as_dict(),
                    }

            stmt = (
                select(DBAuditLog)
                .where(DBAuditLog.timestamp < cutoff)
                .order_by(DBAuditLog.timestamp.asc())
            )
            rows: list[DBAuditLog] = list((await db.execute(stmt)).scalars().all())

            if not rows:
                return {
                    "ok": True,
                    "archived": 0,
                    "deleted": 0,
                    "archive_file": None,
                    "cutoff_timestamp": cutoff,
                    "cutoff_iso": _iso(cutoff),
                }

            records = [row.to_dict() for row in rows]
            last_hash = rows[-1].entry_hash
            first_hash = rows[0].entry_hash

            out_dir = Path(archive_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = _archive_path(out_dir, datetime.now(timezone.utc))

            payload = {
                "archive_version": 1,
                "generated_at": _iso(time.time()),
                "cutoff_timestamp": cutoff,
                "cutoff_iso": _iso(cutoff),
                "rows_archived": len(records),
                "first_entry_hash": first_hash,
                "last_entry_hash": last_hash,
                "prev_hash_of_first": rows[0].prev_hash,
                "records": records,
            }

            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)

            deleted = 0
            if delete_after_archive:
                await db.execute(
                    DBAuditLog.__table__.delete().where(DBAuditLog.timestamp < cutoff)
                )

                anchor = DBAuditChainAnchor(
                    created_at=time.time(),
                    cutoff_timestamp=cutoff,
                    last_archived_hash=last_hash,
                    archive_file=str(path),
                    rows_archived=len(records),
                )
                db.add(anchor)

                if commit:
                    await db.commit()
                deleted = len(records)

            logger.info(
                "audit retention: archived %d rows older than %s to %s",
                len(records),
                _iso(cutoff),
                path,
            )

            return {
                "ok": True,
                "archived": len(records),
                "deleted": deleted,
                "archive_file": str(path),
                "cutoff_timestamp": cutoff,
                "cutoff_iso": _iso(cutoff),
                "first_archived_hash": first_hash,
                "last_archived_hash": last_hash,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("audit retention failed: %s", exc, exc_info=True)
            try:
                await db.rollback()
            except Exception:  # noqa: BLE001
                pass
            return {"ok": False, "archived": 0, "deleted": 0, "error": str(exc)}


def _archive_path(archive_dir: Path, when: datetime) -> Path:
    """``logs/audit_archive_<YYYY-MM-DD>.json`` with collision suffixes."""
    base = f"audit_archive_{when.strftime('%Y-%m-%d')}"
    candidate = archive_dir / f"{base}.json"
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        candidate = archive_dir / f"{base}_{i:03d}.json"
        if not candidate.exists():
            return candidate
    return archive_dir / f"{base}_{int(time.time())}.json"  # pragma: no cover


# ---------------------------------------------------------------------------
# AuditReporter
# ---------------------------------------------------------------------------


class AuditReporter:
    """Read-only aggregations suitable for a dashboard or a JSON/Markdown dump."""

    @classmethod
    async def summary(
        cls,
        db: AsyncSession,
        days: int = 7,
        *,
        user_limit: int = 5,
        ip_limit: int = 5,
        failed_login_threshold: int = 2,
    ) -> dict:
        """Aggregate audit activity over the trailing ``days`` window.

        The returned dict is pure JSON-serialisable data -- the caller decides
        whether to render it as JSON, Markdown, or a Slack message.
        """
        now = time.time()
        since = now - (float(days) * 86400.0)

        try:
            by_event_type = await cls._counts_by(db, DBAuditLog.event_type, since)
            by_severity = await cls._counts_by(db, DBAuditLog.severity, since)

            total = sum(by_event_type.values())

            success_count = int(
                (
                    await db.execute(
                        select(func.count())
                        .select_from(DBAuditLog)
                        .where(
                            DBAuditLog.timestamp >= since,
                            DBAuditLog.success.is_(True),
                            DBAuditLog.event_type.not_in(_DEFENSIVE_EVENT_TYPES),
                        )
                    )
                ).scalar_one()
                or 0
            )

            failure_count = int(
                (
                    await db.execute(
                        select(func.count())
                        .select_from(DBAuditLog)
                        .where(
                            DBAuditLog.timestamp >= since,
                            DBAuditLog.success.is_(False),
                            DBAuditLog.event_type.not_in(_DEFENSIVE_EVENT_TYPES),
                        )
                    )
                ).scalar_one()
                or 0
            )

            top_users = await cls._top_values(
                db, DBAuditLog.user_id, since, user_limit
            )
            top_ips = await cls._top_values(db, DBAuditLog.ip, since, ip_limit)

            failed_login_clusters = await cls._failed_login_clusters(
                db, since, failed_login_threshold
            )

            return {
                "window_days": float(days),
                "since": since,
                "since_iso": _iso(since),
                "until": now,
                "until_iso": _iso(now),
                "total_events": total,
                "success_count": success_count,
                "failure_count": failure_count,
                "success_rate": (success_count / total) if total else None,
                "by_event_type": by_event_type,
                "by_severity": by_severity,
                "critical_events": by_severity.get(AuditSeverity.CRITICAL.value, 0),
                "warning_events": by_severity.get(AuditSeverity.WARNING.value, 0),
                "top_users": top_users,
                "top_ips": top_ips,
                "failed_login_clusters": failed_login_clusters,
                "generated_at": _iso(now),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("audit summary failed: %s", exc, exc_info=True)
            return {
                "window_days": float(days),
                "since": since,
                "since_iso": _iso(since),
                "until": now,
                "until_iso": _iso(now),
                "total_events": 0,
                "error": str(exc),
                "generated_at": _iso(now),
            }

    # -- internals ---------------------------------------------------------

    @classmethod
    async def _counts_by(cls, db: AsyncSession, column, since: float) -> dict:
        stmt = (
            select(column, func.count())
            .where(DBAuditLog.timestamp >= since, column.is_not(None))
            .group_by(column)
            .order_by(func.count().desc())
        )
        rows = (await db.execute(stmt)).all()
        return {str(key): int(count) for key, count in rows}

    @classmethod
    async def _top_values(
        cls,
        db: AsyncSession,
        column,
        since: float,
        limit: int,
    ) -> list[dict]:
        count_col = func.count().label("event_count")
        stmt = (
            select(column, count_col)
            .where(DBAuditLog.timestamp >= since, column.is_not(None))
            .group_by(column)
            .order_by(count_col.desc())
            .limit(max(1, int(limit)))
        )
        rows = (await db.execute(stmt)).all()
        return [{"key": str(key), "count": int(count)} for key, count in rows]

    @classmethod
    async def _failed_login_clusters(
        cls,
        db: AsyncSession,
        since: float,
        threshold: int,
    ) -> list[dict]:
        stmt = (
            select(DBAuditLog.user_id, DBAuditLog.ip, DBAuditLog.timestamp)
            .where(
                DBAuditLog.timestamp >= since,
                DBAuditLog.event_type == AuditEventType.LOGIN_FAILED.value,
            )
            .order_by(DBAuditLog.timestamp.asc())
        )
        rows = (await db.execute(stmt)).all()

        by_user: dict[str, dict] = {}
        by_ip: dict[str, dict] = {}

        for user_id, ip, ts in rows:
            if user_id:
                bucket = by_user.setdefault(
                    str(user_id),
                    {"key": str(user_id), "kind": "user", "count": 0,
                     "first": ts, "last": ts, "distinct_ips": set()},
                )
                bucket["count"] += 1
                bucket["last"] = ts
                if ip:
                    bucket["distinct_ips"].add(str(ip))
            if ip:
                bucket = by_ip.setdefault(
                    str(ip),
                    {"key": str(ip), "kind": "ip", "count": 0,
                     "first": ts, "last": ts, "distinct_users": set()},
                )
                bucket["count"] += 1
                bucket["last"] = ts
                if user_id:
                    bucket["distinct_users"].add(str(user_id))

        clusters: list[dict] = []
        for bucket in list(by_user.values()) + list(by_ip.values()):
            if bucket["count"] < max(1, int(threshold)):
                continue
            distinct = bucket.pop("distinct_ips", None)
            if distinct is None:
                distinct = bucket.pop("distinct_users", set())
            clusters.append(
                {
                    **bucket,
                    "first_iso": _iso(bucket["first"]),
                    "last_iso": _iso(bucket["last"]),
                    "distinct_peers": sorted(distinct),
                }
            )

        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters

    @classmethod
    async def render_markdown(cls, db: AsyncSession, days: int = 7) -> str:
        """Convenience renderer so callers can dump straight to a report file."""
        data = await cls.summary(db, days=days)
        lines = [
            f"# Nydra audit summary (last {data.get('window_days')} days)",
            "",
            f"- Window: `{data.get('since_iso')}` → `{data.get('until_iso')}`",
            f"- Total events: **{data.get('total_events', 0)}**",
            f"- Success rate: {data.get('success_rate')}",
            f"- Critical: {data.get('critical_events', 0)}  "
            f"Warning: {data.get('warning_events', 0)}",
            "",
            "## Events by type",
            "",
        ]
        for key, value in (data.get("by_event_type") or {}).items():
            lines.append(f"- `{key}`: {value}")
        lines += ["", "## Top users", ""]
        for entry in data.get("top_users") or []:
            lines.append(f"- `{entry['key']}`: {entry['count']}")
        lines += ["", "## Top IPs", ""]
        for entry in data.get("top_ips") or []:
            lines.append(f"- `{entry['key']}`: {entry['count']}")
        lines += ["", "## Failed-login clusters", ""]
        for cluster in data.get("failed_login_clusters") or []:
            lines.append(
                f"- {cluster['kind']} `{cluster['key']}`: {cluster['count']} "
                f"({cluster['first_iso']} → {cluster['last_iso']})"
            )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


async def log_event(
    db: AsyncSession,
    event_type: Any,
    user_id: Optional[str] = None,
    ip: Optional[str] = None,
    detail: Any = None,
    severity: Any = AuditSeverity.INFO,
    success: Optional[bool] = None,
    session_id: Optional[str] = None,
    request_path: Optional[str] = None,
    **kwargs: Any,
) -> Optional[DBAuditLog]:
    """Thin wrapper around :meth:`AuditLogger.log`.  Never raises."""
    return await AuditLogger.log(
        db,
        event_type,
        user_id=user_id,
        ip=ip,
        detail=detail,
        severity=severity,
        success=success,
        session_id=session_id,
        request_path=request_path,
        **kwargs,
    )


async def get_recent_events(
    db: AsyncSession, limit: int = 50
) -> list[DBAuditLog]:
    """Most recent audit rows, newest first."""
    return await AuditQuery.recent(db, limit=limit)


async def verify_integrity(db: AsyncSession) -> IntegrityResult:
    """Verify the audit chain."""
    return await IntegrityChecker.verify_chain(db)


async def audit_summary(db: AsyncSession, days: int = 7) -> dict:
    """Aggregate audit activity over the trailing window."""
    return await AuditReporter.summary(db, days=days)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


async def create_audit_tables(conn: Any) -> None:
    """Create the audit tables on an existing sync connection.

    Call this from api.py's ``lifespan()`` startup, inside the same
    ``engine.begin()`` block that already creates your other tables::

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await create_audit_tables(conn)   # <-- add this line

    No new engine is created -- the audit tables ride along on the existing one.
    """
    await conn.run_sync(AUDIT_METADATA.create_all)